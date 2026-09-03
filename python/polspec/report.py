"""Rendering a spec as human-readable documentation.

Kept out of `framespec` and `catspec` so those modules describe what a spec
*is* rather than how it is printed. Nothing here is reachable from the
generation or validation paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polspec.serialization.dtypes import DTYPE_NAMES
from polspec.tablespec import TableSpec, as_table_spec

if TYPE_CHECKING:
    from polspec.catspec import CatSpec
    from polspec.foreign_key import ForeignKey


def _write_if_asked(content: str, path: str | Path | None) -> str:
    """Writes rendered output to `path` when one was given, and returns it either way."""
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return content


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
        f"- **Schema:** `{spec.name}`",
        f"- **Total Columns:** {len(spec.columns)}",
    ]
    if spec.unique_together:
        groups = ", ".join(f"`{list(g)}`" for g in spec.unique_together)
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
        length = (
            f"[{cs.string_length.min}, {cs.string_length.max}]"
            if cs.string_length
            else "-"
        )
        lines.append(
            f"| `{name}` "
            f"| `{_describe_dtype(cs.dtype)}` "
            f"| {'Yes' if cs.nullable else 'No'} "
            f"| {str(cs.bounds) if cs.bounds else '-'} "
            f"| {_describe_choices(cs.choices)} "
            f"| {length} "
            f"| {', '.join(f'`{t}`' for t in cs.tags) if cs.tags else '-'} "
            f"| {f'{len(cs.rules)} rule(s)' if cs.rules else '-'} "
            f"| {'Yes' if cs.unique else 'No'} |"
        )
    return lines


def _constraints_section(spec: TableSpec) -> list[str]:
    """Everything the spec asserts beyond the shape of a single column."""
    columns_with_rules = [(n, s) for n, s in spec.columns.items() if s.rules]
    columns_with_validators = [(n, s) for n, s in spec.columns.items() if s.validators]
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
        lines.extend(f"- Key: `{list(group)}`" for group in spec.unique_together)

    if spec.checks:
        lines.extend(["", "### Multi-Column Checks"])
        for check in spec.checks:
            described = (
                f"\n  - *Description:* {check.description}" if check.description else ""
            )
            lines.append(f"- **`{check.name}`**: `{check.expr}`{described}")

    if spec.foreign_keys:
        lines.extend(["", "### Foreign Keys"])
        for fk in spec.foreign_keys:
            target = spec.name if fk.references == "self" else fk.references
            lines.append(
                f"- **`{fk.name}`**: `{list(fk.columns)}` -> "
                f"`{target}.{list(fk.ref_columns)}`"
            )

    if columns_with_rules:
        lines.extend(["", "### Conditional Rules (`ColRule`)"])
        for name, spec in columns_with_rules:
            lines.append(f"- **Column `{name}`**:")
            lines.extend(
                f"  {index}. When `{rule.when}` -> Choices: `{list(rule.choices)}`"
                for index, rule in enumerate(spec.rules, 1)
            )

    if columns_with_validators:
        lines.extend(["", "### Column Validators"])
        for name, spec in columns_with_validators:
            lines.append(f"- **Column `{name}`**:")
            for validator in spec.validators:
                described = (
                    f" -- {validator.description}" if validator.description else ""
                )
                lines.append(
                    f"  - **`{validator.name}`**: `{validator.expr}`{described}"
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
    entity_name = title or spec.name
    entity_name = "".join(c if c.isalnum() or c == "_" else "_" for c in entity_name)

    fk_columns: dict[str, ForeignKey] = {}
    for fk in spec.foreign_keys:
        for col in fk.columns:
            fk_columns.setdefault(col, fk)

    lines: list[str] = [
        "erDiagram",
        f"    {entity_name} {{",
    ]

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

        key_label = ""
        if cs.unique:
            key_label = "PK"
        elif any(col_name in group for group in spec.unique_together):
            key_label = "UK"
        elif col_name in fk_columns:
            key_label = "FK"

        comments: list[str] = []
        if cs.nullable:
            comments.append("nullable")
        if cs.bounds is not None:
            comments.append(f"bounds: {cs.bounds}")
        elif cs.choices is not None:
            ch = list(cs.choices)
            if len(ch) <= 3:
                comments.append(f"choices: [{', '.join(str(c) for c in ch)}]")
            else:
                comments.append(f"choices: [{len(ch)} items]")
        if cs.tags:
            comments.append(f"tags: [{', '.join(cs.tags)}]")
        if cs.string_length is not None:
            comments.append(f"len: [{cs.string_length.min}, {cs.string_length.max}]")

        comment_body = ", ".join(comments).replace('"', "'")
        comment_str = f' "{comment_body}"' if comments else ""
        key_str = f" {key_label}" if key_label else ""
        lines.append(f"        {type_name} {col_name}{key_str}{comment_str}")

    lines.append("    }")

    if spec.foreign_keys:
        for fk in spec.foreign_keys:
            if fk.references == "self":
                target_name = entity_name
            else:
                target_name = "".join(
                    c if c.isalnum() or c == "_" else "_" for c in fk.references
                )
            fk_label = fk.name.replace('"', "'") if fk.name else "references"
            lines.append(f'    {target_name} ||--o{{ {entity_name} : "{fk_label}"')

    content = "\n".join(lines) + "\n"

    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return content


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
        f"- **Enums:** {len(spec._enums)}",
        f"- **Categoricals:** {len(spec._categoricals)}",
        "",
    ]

    if spec._enums:
        lines.extend(
            [
                "## Enums (`pl.Enum`)",
                "",
                "| Name | Variants Count | Allowed Variants |",
                "|:---|:---|:---|",
            ]
        )
        for k, variants in spec._enums.items():
            var_str = f"[{', '.join(repr(v) for v in variants[:6])}{', ...' if len(variants) > 6 else ''}]"
            lines.append(f"| `{k}` | {len(variants)} | `{var_str}` |")
        lines.append("")

    if spec._categoricals:
        lines.extend(
            [
                "## Categoricals (`pl.Categorical`)",
                "",
                "| Key | Registry Name | Physical Dtype | Namespace | Domain Choices Pool |",
                "|:---|:---|:---|:---|:---|",
            ]
        )
        for k, cat in spec._categoricals.items():
            phys = DTYPE_NAMES.get(cat.physical(), str(cat.physical()))
            ns = cat.namespace() or "-"
            choices = spec._choices.get(k)
            if choices:
                ch_str = f"[{', '.join(repr(c) for c in choices[:6])}{', ...' if len(choices) > 6 else ''}] ({len(choices)} total)"
            else:
                ch_str = "-"
            lines.append(f"| `{k}` | `{cat.name()}` | `{phys}` | `{ns}` | `{ch_str}` |")
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

    for k, variants in spec._enums.items():
        clean_k = "".join(c if c.isalnum() or c == "_" else "_" for c in k)
        lines.append(f"    class {clean_k} {{")
        lines.append("        <<enumeration>>")
        for v in variants[:10]:
            clean_v = "".join(c if c.isalnum() or c == "_" else "_" for c in str(v))
            lines.append(f"        +{clean_v}")
        if len(variants) > 10:
            lines.append(f"        +... ({len(variants) - 10} more)")
        lines.append("    }")

    for k, cat in spec._categoricals.items():
        clean_k = "".join(c if c.isalnum() or c == "_" else "_" for c in k)
        phys = DTYPE_NAMES.get(cat.physical(), str(cat.physical()))
        lines.append(f"    class {clean_k} {{")
        lines.append(f"        <<categorical: {phys}>>")
        ns = cat.namespace()
        if ns:
            lines.append(f"        +namespace: {ns}")
        choices = spec._choices.get(k)
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
