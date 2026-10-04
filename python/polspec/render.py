"""Rendering a spec as human-readable documentation.

Kept out of `framespec` and `catspec` so those modules describe what a spec
*is* rather than how it is printed. Nothing here is reachable from the
generation or validation paths.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polspec.dtypes import map_entries
from polspec.serialization.dtypes import DTYPE_NAMES
from polspec.tablespec import TableSpec, as_table_spec

if TYPE_CHECKING:
    from polspec.catspec import CatSpec
    from polspec.drift import DriftFinding, DriftReport
    from polspec.foreign_key import ForeignKey

__all__ = [
    "catspec_to_markdown",
    "catspec_to_mermaid",
    "drift_to_markdown",
    "framespec_to_markdown",
    "framespec_to_mermaid",
    "registry_to_mermaid",
]


def _write_if_asked(content: str, path: str | Path | None) -> str:
    """Writes rendered output to `path` when one was given, and returns it either way."""
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return content


def _row(*cells: str) -> str:
    """A Markdown table row. A `|` inside a cell -- in backticks too -- would
    split it, and a newline would end the row, so each is escaped."""
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _cell(text: str) -> str:
    # GitHub's parser reads `\\` as an escaped backslash, so a pipe after an
    # odd run of them would still split: double the run, then escape the pipe.
    escaped = re.sub(r"(\\*)\|", lambda m: m[1] * 2 + "\\|", text)
    return " ".join(escaped.splitlines())


def _code(value: object) -> str:
    """`value` as inline code on one line, fenced by more backticks than any
    run of them inside it -- a name may hold a backtick, or be empty."""
    text = " ".join(str(value).splitlines())
    if not text:
        return "` `"
    fence = "`" * (max(map(len, re.findall("`+", text)), default=0) + 1)
    # CommonMark strips one space from each end of a span, and a backtick at
    # either end would merge with the fence: a space keeps both apart.
    padded = text[0] in "` " or text[-1] in "` "
    return f"{fence} {text} {fence}" if padded else f"{fence}{text}{fence}"


def _describe_dtype(dtype: pl.DataType) -> str:
    """A dtype as it reads in a table cell, with long category lists elided."""
    if isinstance(dtype, pl.Enum):
        categories = list(dtype.categories)
        if len(categories) <= 4:
            return f"Enum({categories})"
        return f"Enum({categories[:3]} + {len(categories) - 3} more)"
    if isinstance(dtype, pl.Categorical):
        registry = getattr(dtype, "categories", None)
        if registry and hasattr(registry, "name") and registry.name():
            return f"Categorical({registry.name()})"
        return "Categorical"
    return str(dtype)


def _describe_domain(cs) -> str:
    """The Domain column of the table: the choices, or the format."""
    value_dtype = cs.value_dtype
    if map_entries(cs.dtype) is not None:
        if cs.list_length is not None:
            return f"{cs.list_length.min}..{cs.list_length.max} entries"
        return "entries"
    if isinstance(value_dtype, pl.Struct):
        described = f"struct of {len(value_dtype.fields)} field(s)"
        if cs.list_length is not None:
            return f"{cs.list_length.min}..{cs.list_length.max} elements, {described}"
        return described
    if cs.format is not None:
        extras = cs.extra_values or {}
        also = f", plus {', '.join(extras)}" if extras else ""
        return f"format {_code(cs.format)}{also}"
    if cs.pattern is not None:
        return f"pattern {_code(cs.pattern)}"
    if cs.list_length is not None:
        return f"{cs.list_length.min}..{cs.list_length.max} elements"
    return _describe_choices(cs.choices)


def _describe_nullable(cs) -> str:
    """The Nullable column: the cell, for a list its elements, and for a
    float value how often it is NaN -- a float's other way of holding no
    number."""
    cell = "Yes" if cs.nullable else "No"
    rate = getattr(cs, "element_null_probability", 0.0)
    described = f"{cell} (elements {rate:.0%})" if rate else cell
    nan = getattr(cs, "nan_probability", 0.0)
    return f"{described}; NaN {nan:.0%}" if nan else described


def _describe_choices(choices) -> str:
    if choices is None:
        return "-"
    values = list(choices)
    if len(values) <= 4:
        return str(values)
    shown = ", ".join(str(c) for c in values[:3])
    return f"[{shown}, ... ({len(values)} total)]"


def _overview_section(spec: TableSpec, doc_title: str) -> list[str]:
    lines = [
        f"# {doc_title}",
        "",
        "## Overview",
        f"- **Schema:** {_code(spec.name)}",
        f"- **Total Columns:** {len(spec.columns)}",
    ]
    if spec.unique_together:
        groups = ", ".join(_code(list(g)) for g in spec.unique_together)
        lines.append(f"- **Composite Unique Keys:** {groups}")
    if spec.checks:
        lines.append(f"- **Custom Invariants / Checks:** {len(spec.checks)} check(s)")
    if spec.foreign_keys:
        lines.append(f"- **Foreign Keys:** {len(spec.foreign_keys)} key(s)")
    return lines


def _columns_section(spec: TableSpec) -> list[str]:
    lines = [
        "",
        "## Columns",
        "",
        "| Column | Type | Nullable | Bounds | Domain / Choices | String Length | Tags | Rules | Unique |",
        "|:---|:---|:---|:---|:---|:---|:---|:---|:---|",
    ]
    for name, cs in spec.columns.items():
        lines.extend(_column_rows(name, cs))
    return lines


def _column_rows(name: str, cs) -> list[str]:
    """A column's row, then a row per struct field beneath it, named by
    path (`point.lat`) and nested as deeply as the dtype does."""
    length = (
        f"[{cs.string_length.min}, {cs.string_length.max}]" if cs.string_length else "-"
    )
    rows = [
        _row(
            _code(name),
            _code(_describe_dtype(cs.dtype)),
            _describe_nullable(cs),
            str(cs.bounds) if cs.bounds else "-",
            _describe_domain(cs),
            length,
            ", ".join(_code(t) for t in cs.tags) if cs.tags else "-",
            f"{len(cs.rules)} rule(s)" if cs.rules else "-",
            "Yes" if cs.unique else "No",
        )
    ]
    value_dtype = cs.value_dtype
    if isinstance(value_dtype, pl.Struct):
        for field in value_dtype.fields:
            rows.extend(_column_rows(f"{name}.{field.name}", cs._field(field.name)))
    return rows


def _constraints_section(spec: TableSpec) -> list[str]:
    """Everything the spec asserts beyond the shape of a single column."""
    columns_with_rules = [(n, c) for n, c in spec.columns.items() if c.rules]
    columns_with_validators = [(n, c) for n, c in spec.columns.items() if c.validators]
    if not (
        spec.checks
        or spec.unique_together
        or spec.foreign_keys
        or columns_with_rules
        or columns_with_validators
    ):
        return []

    lines = ["", "## Constraints & Invariants"]

    if spec.unique_together:
        lines.extend(["", "### Composite Uniqueness"])
        lines.extend(f"- Key: {_code(list(group))}" for group in spec.unique_together)

    if spec.checks:
        lines.extend(["", "### Multi-Column Checks"])
        for check in spec.checks:
            described = (
                f"\n  - *Description:* {check.description}" if check.description else ""
            )
            lines.append(f"- **{_code(check.name)}**: {_code(check.expr)}{described}")

    if spec.foreign_keys:
        lines.extend(["", "### Foreign Keys"])
        for fk in spec.foreign_keys:
            target = spec.name if fk.references == "self" else fk.references
            lines.append(
                f"- **{_code(fk.name)}**: {_code(list(fk.columns))} -> "
                f"{_code(f'{target}.{list(fk.ref_columns)}')}"
            )

    if columns_with_rules:
        lines.extend(["", "### Conditional Rules (`ColRule`)"])
        for name, column in columns_with_rules:
            lines.append(f"- **Column {_code(name)}**:")
            lines.extend(
                f"  {index}. When {_code(rule.when)} -> Choices: {_code(list(rule.choices))}"
                for index, rule in enumerate(column.rules, 1)
            )

    if columns_with_validators:
        lines.extend(["", "### Column Validators"])
        for name, column in columns_with_validators:
            lines.append(f"- **Column {_code(name)}**:")
            for validator in column.validators:
                described = (
                    f" -- {validator.description}" if validator.description else ""
                )
                lines.append(
                    f"  - **{_code(validator.name)}**: {_code(validator.expr)}{described}"
                )

    return lines


def framespec_to_markdown(
    cls: TableSpec | type,
    path: str | Path | None = None,
    *,
    title: str | None = None,
) -> str:
    """Generates a Markdown data dictionary document for this FrameSpec.

    Parameters
    ----------
    path : str | Path | None, optional
        If specified, writes the generated Markdown to this file path.
    title : str | None, optional
        Custom title for the data dictionary. Defaults to the FrameSpec class name.

    Returns
    -------
    str
        The formatted Markdown string.
    """
    spec = as_table_spec(cls)
    lines = [
        *_overview_section(spec, title or spec.name),
        *_columns_section(spec),
        *_constraints_section(spec),
    ]
    return _write_if_asked("\n".join(lines) + "\n", path)


# ---------------------------------------------------------------------------
# Drift reports
# ---------------------------------------------------------------------------


def _drift_rows(findings: Iterable[DriftFinding]) -> list[str]:
    rows = []
    for finding in findings:
        column = ", ".join(_code(c) for c in finding.columns) or "-"
        # The message already names the column; the table has a column for it.
        detail = finding.message
        prefix = f"Column '{finding.columns[0]}': " if len(finding.columns) == 1 else ""
        if detail.startswith(prefix):
            detail = detail[len(prefix) :]
        rows.append(_row(column, _code(finding.code), detail))
    return rows


def drift_to_markdown(report: DriftReport, path: str | Path | None = None) -> str:
    """A `DriftReport` as Markdown -- the shape of a pull-request comment.

    Breaking findings first, under their own heading, so the part a
    reviewer has to read is the part at the top.
    """
    what = (
        f"{_code(report.old)} -> {_code(report.new)}"
        if report.kind == "diff"
        else f"{report.new} against {_code(report.old)}"
    )
    lines = [f"# Drift: {what}", ""]
    if report.unchanged:
        lines.append("No drift.")
    else:
        lines.append(
            f"{len(report.breaking)} breaking, {len(report.compatible)} compatible."
        )
        for heading, findings in (
            ("Breaking", report.breaking),
            ("Compatible", report.compatible),
        ):
            if not findings:
                continue
            lines += [
                "",
                f"## {heading}",
                "",
                "| Column | Change | Detail |",
                "|:---|:---|:---|",
                *_drift_rows(findings),
            ]
    return _write_if_asked("\n".join(lines) + "\n", path)


# Mermaid reads `PK`, `FK` and `UK` as keys, in any case, wherever they stand
# alone -- an attribute so named does not parse.
_MERMAID_KEYS = frozenset({"PK", "FK", "UK"})


def _mermaid_word(name: str, allowed: str) -> str:
    """`name` as one word of Mermaid's ER grammar, in the form every version
    reads: ASCII letters and digits, the `allowed` punctuation, and not
    starting with a digit. Mermaid 11 also takes non-ASCII letters; 10 --
    still what many renderers bundle -- does not."""
    word = "".join(
        c if c.isascii() and (c.isalnum() or c in allowed) else "_" for c in name
    )
    if not word or not (word[0].isalpha() or word[0] == "_"):
        word = f"_{word}"
    return word


# Words Mermaid's ER grammar reserves outside an entity's block, found by
# parsing each with Mermaid 10 and 11: an entity so named does not parse.
_MERMAID_RESERVED = frozenset(
    {
        "class",
        "classDef",
        "end",
        "erDiagram",
        "many",
        "one",
        "style",
        "subgraph",
        "to",
        "u",
    }
)


def _mermaid_name(name: str) -> str:
    word = _mermaid_word(name, "_")
    return f"{word}_" if word in _MERMAID_RESERVED else word


def _attribute_word(name: str) -> str:
    word = _mermaid_word(name, "_-")
    return f"{word}_" if word.upper() in _MERMAID_KEYS else word


def _attribute_names(names: Iterable[str]) -> dict[str, str]:
    """Each column's attribute name: itself where Mermaid can read it, else
    the nearest word it can, told apart from every other by a suffix."""
    names = list(names)
    named = {name: name for name in names if _attribute_word(name) == name}
    taken = set(named)
    for name in names:
        if name in named:
            continue
        word = candidate = _attribute_word(name)
        n = 1
        while candidate in taken:
            n += 1
            candidate = f"{word}_{n}"
        taken.add(candidate)
        named[name] = candidate
    return named


def _entity_lines(spec: TableSpec, entity_name: str) -> list[str]:
    """One `Name { ... }` block: a line per column with its key and notes."""
    unique_count = sum(1 for cs in spec.columns.values() if cs.unique)
    fk_columns: dict[str, ForeignKey] = {}
    for fk in spec.foreign_keys:
        for col in fk.columns:
            fk_columns.setdefault(col, fk)

    attribute_names = _attribute_names(spec.columns)
    lines = [f"    {entity_name} {{"]
    for col_name, cs in spec.columns.items():
        dtype = cs.dtype
        if isinstance(dtype, pl.Enum):
            type_name = "Enum"
        elif isinstance(dtype, pl.Categorical):
            type_name = "Categorical"
        elif isinstance(dtype, pl.Datetime):
            type_name = "Datetime"
        elif isinstance(dtype, pl.Duration):
            type_name = "Duration"
        elif isinstance(dtype, pl.List):
            type_name = "List"
        elif isinstance(dtype, pl.Struct):
            type_name = "Struct"
        elif isinstance(dtype, pl.Array):
            type_name = "Array"
        else:
            type_name = type(dtype).__name__

        # One primary key per entity: a lone unique column is it, several
        # are each a unique key.
        key_label = ""
        if cs.unique:
            key_label = "PK" if unique_count == 1 else "UK"
        elif any(col_name in group for group in spec.unique_together):
            key_label = "UK"
        elif col_name in fk_columns:
            key_label = "FK"

        attribute = attribute_names[col_name]
        # A column renamed for Mermaid keeps its real name, first in the note.
        comments: list[str] = [] if attribute == col_name else [f"name: {col_name!r}"]
        if cs.nullable:
            comments.append("nullable")
        if cs.element_null_probability:
            comments.append(f"element nulls: {cs.element_null_probability:.0%}")
        if cs.nan_probability:
            comments.append(f"NaN: {cs.nan_probability:.0%}")
        if cs.bounds is not None:
            comments.append(f"bounds: {cs.bounds}")
        elif cs.choices is not None:
            ch = list(cs.choices)
            if len(ch) <= 3:
                comments.append(f"choices: [{', '.join(str(c) for c in ch)}]")
            else:
                comments.append(f"choices: [{len(ch)} items]")
        elif cs.format is not None:
            also = f" + {len(cs.extra_values)} extra" if cs.extra_values else ""
            comments.append(f"format: {cs.format}{also}")
        elif cs.pattern is not None:
            comments.append(f"pattern: {cs.pattern}")
        if cs.tags:
            comments.append(f"tags: [{', '.join(cs.tags)}]")
        if cs.string_length is not None:
            comments.append(f"len: [{cs.string_length.min}, {cs.string_length.max}]")
        if isinstance(cs.value_dtype, pl.Struct):
            comments.append(
                f"fields: [{', '.join(f.name for f in cs.value_dtype.fields)}]"
            )

        # A note is one quoted string on one line. A `~` in it, before a long
        # tail, hangs Mermaid 10's lexer -- its rule for generic types like
        # `List~int~` backtracks -- so it is drawn as the look-alike tilde operator.
        noted = ", ".join(comments).replace('"', "'").replace("~", "\N{TILDE OPERATOR}")
        comment_body = " ".join(noted.splitlines())
        comment_str = f' "{comment_body}"' if comments else ""
        key_str = f" {key_label}" if key_label else ""
        lines.append(f"        {type_name} {attribute}{key_str}{comment_str}")
    lines.append("    }")
    return lines


def _relationship_lines(spec: TableSpec, entity_name: str) -> list[str]:
    """One `Parent ||--o{ Child : "key"` line per foreign key."""
    lines: list[str] = []
    for fk in spec.foreign_keys:
        target_name = (
            entity_name if fk.references == "self" else _mermaid_name(fk.references)
        )
        fk_label = fk.name.replace('"', "'") if fk.name else "references"
        lines.append(f'    {target_name} ||--o{{ {entity_name} : "{fk_label}"')
    return lines


def framespec_to_mermaid(
    cls: TableSpec | type,
    path: str | Path | None = None,
    *,
    title: str | None = None,
) -> str:
    """Generates a Mermaid Entity-Relationship (ER) diagram for this FrameSpec.

    Parameters
    ----------
    path : str | Path | None, optional
        If specified, writes the generated Mermaid diagram to this file path.
    title : str | None, optional
        Entity name in the diagram. Defaults to the FrameSpec class name.

    Returns
    -------
    str
        The formatted Mermaid diagram definition.
    """
    spec = as_table_spec(cls)
    entity_name = _mermaid_name(title or spec.name)
    lines = [
        "erDiagram",
        *_entity_lines(spec, entity_name),
        *_relationship_lines(spec, entity_name),
    ]
    return _write_if_asked("\n".join(lines) + "\n", path)


def registry_to_mermaid(
    specs: Iterable[TableSpec | type], path: str | Path | None = None
) -> str:
    """One ER diagram over several specs: every entity, then every key
    between them, so the relationships a single spec cannot see are drawn.
    """
    tables = [as_table_spec(s) for s in specs]
    lines = ["erDiagram"]
    for spec in tables:
        lines.extend(_entity_lines(spec, _mermaid_name(spec.name)))
    for spec in tables:
        lines.extend(_relationship_lines(spec, _mermaid_name(spec.name)))
    return _write_if_asked("\n".join(lines) + "\n", path)


def catspec_to_markdown(
    spec: CatSpec,
    path: str | Path | None = None,
    *,
    title: str | None = None,
) -> str:
    """Generates a Markdown documentation table of this CatSpec registry.

    Parameters
    ----------
    path : str | Path | None, optional
        File destination to write. If None, returns the Markdown string.
    title : str | None, optional
        Custom title for the document. Defaults to 'Categorical & Enum Registry'.

    Returns
    -------
    str
        The formatted Markdown string.
    """
    doc_title = title or "Categorical & Enum Registry"
    lines: list[str] = [
        f"# {doc_title}",
        "",
        "## Summary",
        f"- **Enums:** {len(spec.enums)}",
        f"- **Categoricals:** {len(spec.categoricals)}",
        "",
    ]

    if spec.enums:
        lines.extend(
            [
                "## Enums (`pl.Enum`)",
                "",
                "| Name | Variants Count | Allowed Variants |",
                "|:---|:---|:---|",
            ]
        )
        for k, variants in spec.enums.items():
            var_str = f"[{', '.join(repr(v) for v in variants[:6])}{', ...' if len(variants) > 6 else ''}]"
            lines.append(_row(_code(k), str(len(variants)), _code(var_str)))
        lines.append("")

    if spec.categoricals:
        lines.extend(
            [
                "## Categoricals (`pl.Categorical`)",
                "",
                "| Key | Registry Name | Physical Dtype | Namespace | Domain Choices Pool |",
                "|:---|:---|:---|:---|:---|",
            ]
        )
        for k, cat in spec.categoricals.items():
            phys = DTYPE_NAMES.get(cat.physical(), str(cat.physical()))
            ns = cat.namespace() or "-"
            choices = spec.choices.get(k)
            if choices:
                ch_str = f"[{', '.join(repr(c) for c in choices[:6])}{', ...' if len(choices) > 6 else ''}] ({len(choices)} total)"
            else:
                ch_str = "-"
            lines.append(
                _row(
                    _code(k),
                    _code(cat.name()),
                    _code(phys),
                    _code(ns),
                    _code(ch_str),
                )
            )
        lines.append("")

    content = "\n".join(lines).rstrip() + "\n"
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return content


def catspec_to_mermaid(
    spec: CatSpec,
    path: str | Path | None = None,
    *,
    title: str | None = None,
) -> str:
    """Generates a Mermaid class diagram definition for this CatSpec registry."""
    lines: list[str] = [
        "classDiagram",
    ]
    if title:
        lines.insert(0, f"%% {title}")

    for k, variants in spec.enums.items():
        clean_k = "".join(c if c.isalnum() or c == "_" else "_" for c in k)
        lines.append(f"    class {clean_k} {{")
        lines.append("        <<enumeration>>")
        for v in variants[:10]:
            clean_v = "".join(c if c.isalnum() or c == "_" else "_" for c in str(v))
            lines.append(f"        +{clean_v}")
        if len(variants) > 10:
            lines.append(f"        +... ({len(variants) - 10} more)")
        lines.append("    }")

    for k, cat in spec.categoricals.items():
        clean_k = "".join(c if c.isalnum() or c == "_" else "_" for c in k)
        phys = DTYPE_NAMES.get(cat.physical(), str(cat.physical()))
        lines.append(f"    class {clean_k} {{")
        lines.append(f"        <<categorical: {phys}>>")
        ns = cat.namespace()
        if ns:
            lines.append(f"        +namespace: {ns}")
        choices = spec.choices.get(k)
        if choices:
            for c in choices[:5]:
                clean_c = "".join(c if c.isalnum() or c == "_" else "_" for c in str(c))
                lines.append(f"        +{clean_c}")
            if len(choices) > 5:
                lines.append(f"        +... ({len(choices) - 5} more)")
        lines.append("    }")

    content = "\n".join(lines) + "\n"
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return content
