"""Checking a frame against the claims a spec makes about it.

Two entry points over the same machinery:

- `inspect(spec, df)` returns a `ValidationReport` -- every finding as data,
  with the offending rows reachable lazily -- and never raises for a bad
  frame.
- `validate(spec, df)` inspects, raises `ValidationError` carrying that
  report if anything was found, and otherwise returns the frame with the
  requested structural transformations applied.

What is public here is exported from `polspec` itself.

Internal: not part of the public API.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, overload

import polars as pl
import polars.selectors as cs

from polspec import frames
from polspec._options import accepted_options, options_from
from polspec._text import parse_series, text_parser
from polspec.errors import ValidationError
from polspec.frames import References, to_lazy
from polspec.tablespec import (
    SpecLike,
    TableSpec,
    as_table_spec,
    require_columns,
    resolve_references,
)
from polspec.validation.constraints import (
    Constraint,
    column_constraints,
    foreign_key_findings,
    frame_constraints,
    hierarchy_constraints,
    hierarchy_findings,
    is_dtype_compatible,
)
from polspec.validation.report import (
    FINDING_COLUMN,
    Finding,
    FindingCode,
    ValidationReport,
)

if TYPE_CHECKING:
    from polspec.spec import ColSpec

__all__ = [
    "FINDING_COLUMN",
    "Finding",
    "FindingCode",
    "ValidationError",
    "ValidationOptions",
    "ValidationReport",
    "inspect",
    "validate",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationOptions:
    """Which checks to run and what to do about structural mismatches.

    Grouped rather than passed as ten separate arguments, so adding a check
    does not widen every signature between here and `FrameSpec.validate`.

    Examples
    --------
    >>> spec = TableSpec("Orders", {"total": ColSpec(pl.Float64, bounds=(0, 100))})
    >>> lenient = ValidationOptions(extra_cols="drop", bounds=False)
    >>> validate(spec, pl.DataFrame({"total": [250.0], "note": ["x"]}), options=lenient).columns
    ['total']
    """

    extra_cols: Literal["drop", "allow", "raise"] = "raise"
    missing_cols: Literal["add", "allow", "raise"] = "raise"
    strict_dtypes: bool = False
    rules: bool = True
    derived: bool = True
    validators: bool = True
    unique: bool = True
    checks: bool = True
    foreign_keys: bool = True
    hierarchy: bool = True
    order: bool = True
    rows: bool = True
    pattern: bool = True
    bounds: bool = True
    cast: bool = False
    fill_defaults: bool = False
    parse_text: bool = False
    streaming: bool = False
    schema_only: bool = False

    def __post_init__(self) -> None:
        if self.extra_cols not in ("drop", "allow", "raise"):
            raise ValueError(
                f"extra_cols must be one of 'drop', 'allow', 'raise'; got {self.extra_cols!r}"
            )
        if self.missing_cols not in ("add", "allow", "raise"):
            raise ValueError(
                f"missing_cols must be one of 'add', 'allow', 'raise'; got {self.missing_cols!r}"
            )
        if self.schema_only and self.parse_text:
            raise ValueError(
                "schema_only=True reads no values, and parse_text=True reads "
                "them to decide each column's type; ask for one or the other"
            )


_Options = ValidationOptions


# One spelling for each option, whichever way it is passed: a keyword is the
# field it sets, as `ValidationOptions(checks=False)` and `--skip checks` say
# it. Until 0.23 the switches were spelt `checks` as keywords.
class ValidationKeywords(TypedDict, total=False):
    """`ValidationOptions` as keywords, each by its field's name -- what
    `validate`, `inspect` and the `FrameSpec` facades take. See
    `ValidationOptions` for what each means and its default."""

    extra_cols: Literal["drop", "allow", "raise"]
    missing_cols: Literal["add", "allow", "raise"]
    strict_dtypes: bool
    rules: bool
    derived: bool
    validators: bool
    unique: bool
    checks: bool
    foreign_keys: bool
    hierarchy: bool
    order: bool
    rows: bool
    pattern: bool
    bounds: bool
    cast: bool
    fill_defaults: bool
    parse_text: bool
    streaming: bool
    schema_only: bool


SWITCHES = (
    "rules",
    "derived",
    "validators",
    "unique",
    "checks",
    "foreign_keys",
    "hierarchy",
    "order",
    "rows",
    "pattern",
    "bounds",
)
_ACCEPTED_OPTIONS = accepted_options(ValidationOptions)


def _options_from(
    options_obj: ValidationOptions | None = None, /, **options: Any
) -> ValidationOptions:
    return options_from(
        ValidationOptions,
        options_obj,
        options,
        what="validation",
    )


def _filled(
    lf: pl.LazyFrame, columns: Mapping[str, ColSpec], *, strict: bool
) -> pl.LazyFrame:
    """`fill_defaults=True`: each defaulted column's nulls filled with its
    default, and each absent one added holding it -- before any claim is
    checked, so a null in a column that allows none is judged filled.

    A NaN is a value, and is left. A column whose dtype is already wrong is
    left too: it is a `dtype` finding, and filling it is not this option's
    to decide.
    """
    schema = lf.collect_schema()
    fills = []
    for name, declared in columns.items():
        if declared.default is None:
            continue
        value = pl.lit(declared.default, dtype=declared.dtype)
        if name not in schema:
            fills.append(value.alias(name))
        elif is_dtype_compatible(declared.dtype, schema[name], strict=strict):
            column = frames.column(name)
            fills.append(column.fill_null(value).alias(name))
    return lf.with_columns(fills) if fills else lf


def _parsed_text(
    lf: pl.LazyFrame,
    columns: Mapping[str, ColSpec],
    collect_kwargs: dict[str, Any],
) -> tuple[pl.LazyFrame, dict[str, pl.Series]]:
    """`parse_text=True`: each column that arrived as text where its spec
    declares a type text can hold -- a number, a flag, a date -- read as
    that type, when every present value parses; and, for each column with
    a value that does not, those values.

    The text columns are read once, here, and parsed as `read()` parses a
    file: stripped, with empty text a null. A column left as text is a
    `dtype` finding, with the values that did not parse as its samples.
    """
    schema = lf.collect_schema()
    candidates = [
        name
        for name, declared in columns.items()
        if name in schema
        and schema[name] == pl.String
        and text_parser(declared.dtype) is not None
    ]
    if not candidates:
        return lf, {}
    texts = lf.select(frames.columns(candidates)).collect(**collect_kwargs)
    parsed: list[pl.Series] = []
    unparsed: dict[str, pl.Series] = {}
    for name in candidates:
        values, failed = parse_series(
            texts[name], columns[name].dtype, columns[name].text_format
        )
        if values is None:
            unparsed[name] = failed
        else:
            parsed.append(values)
    return (lf.with_columns(parsed) if parsed else lf), unparsed


def _dtype_finding(
    name: str,
    declared: pl.DataType,
    actual: pl.DataType,
    unparsed: pl.Series | None,
    text_format: str | None = None,
) -> Finding:
    """A column of the wrong dtype -- with, for text `parse_text` could
    not read, how many values did not parse and which."""
    details: dict[str, Any] = {"expected": str(declared), "actual": str(actual)}
    message = f"Column '{name}': expected dtype {declared}, got {actual}"
    if unparsed is None:
        return Finding(
            code="dtype",
            key=f"{name}__dtype",
            message=message,
            columns=(name,),
            details=details,
        )
    samples = tuple(unparsed.unique(maintain_order=True).head(5).to_list())
    details["unparsed"] = len(unparsed)
    return Finding(
        code="dtype",
        key=f"{name}__dtype",
        message=(
            f"{message}: {len(unparsed):,} value(s) do not parse as {declared}"
            + (f" written as '{text_format}'" if text_format is not None else "")
            + f", such as {list(samples)}"
        ),
        columns=(name,),
        count=len(unparsed),
        samples=samples,
        details=details,
    )


def _resolve_foreign_keys(
    spec: TableSpec, references: References
) -> tuple[list[tuple[Any, pl.LazyFrame | None]], list[Finding]]:
    """Pairs each key with its parent frame; a key with no parent is a finding."""
    parents = resolve_references(references, to_lazy)
    resolved: list[tuple[Any, pl.LazyFrame | None]] = []
    findings: list[Finding] = []
    for fk in spec.foreign_keys:
        if fk.references == "self":
            resolved.append((fk, None))
            continue
        parent = parents.get(fk.references)
        if parent is None:
            findings.append(
                Finding(
                    code="foreign_key_unresolved",
                    key=f"fk:{fk.name}",
                    message=(
                        f"ForeignKey {fk.name!r} references {fk.references!r}, but no "
                        "DataFrame for it was supplied via references={...}"
                    ),
                    columns=tuple(fk.columns),
                    details={"target": fk.references},
                )
            )
            continue
        resolved.append((fk, parent))
    return resolved, findings


@dataclass(frozen=True, slots=True)
class _Recheck:
    """Validates a frame again under the same spec, options and parents:
    how `passing_rows()` settles a split, as a value so a report pickles."""

    spec: TableSpec
    options: ValidationOptions
    references: References

    def __call__(self, frame: pl.LazyFrame) -> ValidationReport:
        return inspect(
            self.spec, frame, options=self.options, references=self.references
        )


def inspect(
    spec: SpecLike,
    df: pl.DataFrame | pl.LazyFrame,
    *,
    options: ValidationOptions | None = None,
    references: References = None,
    **option_kwargs: Unpack[ValidationKeywords],
) -> ValidationReport:
    """Everything `spec` has to say about `df`, as a `ValidationReport`.

    Never raises for a frame that fails: every violation is a `Finding` on
    the report, with `report.rows(finding)` and `report.failing_rows()`
    giving the offending rows back lazily. See `validate` for the options.

    Examples
    --------
    >>> spec = TableSpec("Orders", {"total": ColSpec(pl.Float64, bounds=(0, 100))})
    >>> report = inspect(spec, pl.DataFrame({"total": [10.0, 250.0, 30.0]}))
    >>> report.passed
    False
    >>> [(finding.code, finding.count) for finding in report.findings]
    [('bounds', 1)]
    >>> report.failing_rows().collect()["total"].to_list()
    [250.0]
    >>> report.passing_rows().collect()["total"].to_list()
    [10.0, 30.0]
    """
    spec = as_table_spec(spec)
    require_columns(spec)
    opts = _options_from(options, **option_kwargs)
    columns = dict(spec.columns)

    # The in-memory engine unless streaming is asked for. Polars 2.0.0 runs a
    # lazy query on its streaming engine by default, and that engine panics
    # on validation's aggregation over a list column -- `list.eval` beside a
    # `filter` -- for some frames (scripts/deep/polars2_eval_chunks_repro.py).
    # On Polars 1 this is the engine a lazy query already ran on.
    collect_kwargs: dict[str, Any] = {
        "engine": "streaming" if opts.streaming else "in-memory"
    }

    lf, unparsed = _prepared(to_lazy(df), columns, opts, collect_kwargs)
    df_schema = lf.collect_schema()

    findings = _structural_findings(columns, df_schema.names(), opts)
    dtype_findings, constraints = _planned(spec, df_schema, opts, unparsed)
    findings.extend(dtype_findings)
    if opts.schema_only:
        # The schema says everything above; the rest needs the rows.
        return _report(spec, lf, findings, opts, references)
    aggregated, rows = _aggregated_findings(lf, constraints, collect_kwargs)
    findings.extend(aggregated)
    findings.extend(
        _cross_frame_findings(
            spec, lf, df_schema.names(), opts, references, collect_kwargs
        )
    )
    findings = _with_shares(findings, lf, rows, collect_kwargs)
    return _report(spec, lf, findings, opts, references)


def _prepared(
    lf: pl.LazyFrame,
    columns: Mapping[str, ColSpec],
    opts: ValidationOptions,
    collect_kwargs: dict[str, Any],
) -> tuple[pl.LazyFrame, dict[str, pl.Series]]:
    """The frame as the checks see it: text read as its declared types, and
    defaults filled, as the options ask. Also what text did not parse, by
    column, for the `dtype` finding to show."""
    unparsed: dict[str, pl.Series] = {}
    if opts.parse_text:
        # Before filling, so a default fills what was empty text.
        lf, unparsed = _parsed_text(lf, columns, collect_kwargs)
    if opts.fill_defaults:
        lf = _filled(lf, columns, strict=opts.strict_dtypes)
    return lf, unparsed


def _structural_findings(
    columns: Mapping[str, ColSpec],
    df_col_names: Sequence[str],
    opts: ValidationOptions,
) -> list[Finding]:
    """Columns the frame has and the spec does not, and the other way round:
    read from the schema alone."""
    extra = [col for col in df_col_names if col not in columns]
    # An optional column may be absent; one present is held to every claim.
    missing = [
        col for col in columns if col not in df_col_names and not columns[col].optional
    ]

    findings: list[Finding] = []
    if extra and opts.extra_cols == "raise":
        findings.append(
            Finding(
                code="extra_columns",
                key="extra_columns",
                message=f"Extra columns found that are not in schema: {extra}",
                columns=tuple(extra),
                details={"columns": extra},
            )
        )
    if missing and opts.missing_cols == "raise":
        findings.append(
            Finding(
                code="missing_columns",
                key="missing_columns",
                message=f"Missing required columns in DataFrame: {missing}",
                columns=tuple(missing),
                details={"columns": missing},
            )
        )
    return findings


def _planned(
    spec: TableSpec,
    df_schema: pl.Schema,
    opts: ValidationOptions,
    unparsed: Mapping[str, pl.Series],
) -> tuple[list[Finding], list[Constraint]]:
    """Each present column's `dtype` finding, if its dtype is wrong, and every
    claim that one aggregation over the frame can settle."""
    columns = spec.columns
    df_col_names = df_schema.names()
    present = [c for c in columns if c in df_col_names]
    compatible_with = {
        name: is_dtype_compatible(
            columns[name].dtype, df_schema[name], strict=opts.strict_dtypes
        )
        for name in present
    }
    # The columns a claim may read: a declared column of the wrong dtype is
    # a `dtype` finding already, and a rule or check compiled
    # against it is at best noise and at worst an expression Polars refuses,
    # as it refuses one naming a missing column. Each is skipped the same way.
    readable = [c for c in df_col_names if c not in columns or compatible_with[c]]

    findings: list[Finding] = []
    constraints: list[Constraint] = []
    for name in present:
        cs = columns[name]
        actual_dtype = df_schema[name]
        compatible = compatible_with[name]
        if not compatible:
            findings.append(
                _dtype_finding(
                    name, cs.dtype, actual_dtype, unparsed.get(name), cs.text_format
                )
            )
        constraints.extend(
            column_constraints(
                name,
                cs,
                actual_dtype,
                compatible=compatible,
                options=opts,
                df_col_names=readable,
            )
        )
    if opts.hierarchy and spec.hierarchy is not None:
        constraints.extend(hierarchy_constraints(spec.hierarchy, df_col_names))
    constraints.extend(
        frame_constraints(
            spec.unique_together if opts.unique else None,
            spec.checks if opts.checks else None,
            readable,
            order=(spec.order_by, spec.descending) if opts.order else None,
            rows=spec.rows if opts.rows else None,
        )
    )
    return findings, constraints


_ROWS = "__polspec_rows"


def _aggregated_findings(
    lf: pl.LazyFrame, constraints: Sequence[Constraint], collect_kwargs: dict[str, Any]
) -> tuple[list[Finding], int | None]:
    """The constraints' findings, from one aggregation over the frame, and
    the frame's row count, which the same aggregation counts."""
    if not constraints:
        return [], None
    aggregations = [expr for c in constraints for expr in c.aggregations()]
    aggregations.append(pl.len().alias(_ROWS))
    stats = lf.select(aggregations).collect(**collect_kwargs).to_dict(as_series=False)
    found = [f for f in (c.failure(stats) for c in constraints) if f is not None]
    return found, int(stats[_ROWS][0])


def _with_shares(
    findings: list[Finding],
    lf: pl.LazyFrame,
    rows: int | None,
    collect_kwargs: dict[str, Any],
) -> list[Finding]:
    """Each counted finding with its share of the frame's rows. The rows are
    counted once more only when no aggregation counted them already."""
    if not any(f.count is not None for f in findings):
        return findings
    if rows is None:
        rows = int(lf.select(pl.len()).collect(**collect_kwargs).item())
    return [
        dataclasses.replace(f, share=f.count / rows)
        if f.count is not None and rows
        else f
        for f in findings
    ]


def _cross_frame_findings(
    spec: TableSpec,
    lf: pl.LazyFrame,
    df_col_names: Sequence[str],
    opts: ValidationOptions,
    references: References,
    collect_kwargs: dict[str, Any],
) -> list[Finding]:
    """What one aggregation cannot settle: foreign keys, which join to a
    parent, and a hierarchy, which walks the frame's own rows."""
    findings: list[Finding] = []
    if opts.foreign_keys and spec.foreign_keys:
        resolved, unresolved = _resolve_foreign_keys(spec, references)
        findings.extend(unresolved)
        findings.extend(
            foreign_key_findings(lf, spec.name, resolved, df_col_names, collect_kwargs)
        )
    if opts.hierarchy and spec.hierarchy is not None:
        findings.extend(
            hierarchy_findings(
                lf, spec.name, spec.hierarchy, df_col_names, collect_kwargs
            )
        )
    return findings


def _report(
    spec: TableSpec,
    lf: pl.LazyFrame,
    findings: Sequence[Finding],
    opts: ValidationOptions,
    references: References,
) -> ValidationReport:
    recheck = _Recheck(spec, opts, references)

    # Only a claim relating rows to other rows -- an order, a key into the
    # frame's own rows -- can break in the rows left after the failing ones
    # are taken; for any other spec the first split is already settled, and
    # `passing_rows()` costs no second validation. A new claim of that kind
    # belongs here (the passing-rows property in the contract tests says so).
    relates_rows = bool(spec.order_by) or any(
        fk.references == "self" for fk in spec.foreign_keys
    )
    return ValidationReport(
        spec.name,
        tuple(findings),
        lf,
        opts,
        _recheck=recheck if relates_rows else None,
        _min_rows=(spec.rows.min or 0) if opts.rows and spec.rows is not None else 0,
    )


@overload
def validate(
    spec: SpecLike,
    df: pl.DataFrame,
    *,
    options: ValidationOptions | None = None,
    references: References = None,
    **option_kwargs: Unpack[ValidationKeywords],
) -> pl.DataFrame: ...


@overload
def validate(
    spec: SpecLike,
    df: pl.LazyFrame,
    *,
    options: ValidationOptions | None = None,
    references: References = None,
    **option_kwargs: Unpack[ValidationKeywords],
) -> pl.LazyFrame: ...


def validate(
    spec: SpecLike,
    df: pl.DataFrame | pl.LazyFrame,
    *,
    options: ValidationOptions | None = None,
    references: References = None,
    **option_kwargs: Unpack[ValidationKeywords],
) -> pl.DataFrame | pl.LazyFrame:
    """Validates a DataFrame or LazyFrame against `spec`.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        The frame to validate. A LazyFrame comes back as a LazyFrame.
    options : ValidationOptions, optional
        Every option at once, as a value -- useful for passing one setting
        through several calls. Cannot be combined with the keywords below.
    references : mapping
        Parent frames for foreign keys that reference another spec, keyed by
        that spec, its FrameSpec class, or its name. A key with no entry is
        reported as a `foreign_key_unresolved` finding.
    **option_kwargs
        The fields of `ValidationOptions`, one at a time, by their names --
        `checks=False`, `unique=False`. See `ValidationOptions` for what each
        means and what it defaults to; an unknown name raises `TypeError`
        naming the closest match.

    Returns
    -------
    The validated, optionally transformed frame -- a LazyFrame if `df` was one.

    Raises
    ------
    ValidationError
        Carrying a `ValidationReport` of every violation.
    TypeError
        For an option name this does not accept.
    ValueError
        For an accepted option given a value outside its choices.

    Examples
    --------
    >>> spec = TableSpec("Orders", {"total": ColSpec(pl.Float64, bounds=(0, 100))})
    >>> validate(spec, pl.DataFrame({"total": [10.0, 20.0]})).height
    2
    >>> try:
    ...     validate(spec, pl.DataFrame({"total": [250.0]}))
    ... except ValidationError as error:
    ...     print([finding.code for finding in error.report.findings])
    ...
    ['bounds']
    >>> validate(spec, pl.DataFrame({"total": ["10.5"]}), parse_text=True).schema
    Schema({'total': Float64})
    """
    spec = as_table_spec(spec)
    report = inspect(spec, df, options=options, references=references, **option_kwargs)
    report.raise_if_failed()
    return transformed(spec, df, report)


def transformed(
    spec: TableSpec, df: pl.DataFrame | pl.LazyFrame, report: ValidationReport
) -> pl.DataFrame | pl.LazyFrame:
    """The frame behind a passing report, with the report's structural
    options (drop, add, cast, reorder) applied; lazy if `df` was.
    """
    lf = report.frame
    present = lf.collect_schema().names()
    columns = dict(spec.columns)
    extra = [c for c in present if c not in columns]
    missing = [c for c in columns if c not in present]
    result = _apply_transformations(lf, columns, extra, missing, report.options)
    return result if isinstance(df, pl.LazyFrame) else result.collect()


def _apply_transformations(
    lf: pl.LazyFrame,
    columns: dict[str, Any],
    extra: list[str],
    missing: list[str],
    options: ValidationOptions,
) -> pl.LazyFrame:
    """Drops, adds, casts and reorders once validation has passed."""
    if extra and options.extra_cols == "drop":
        # A selector, which `drop` takes, and which reads every name literally.
        lf = lf.drop(cs.by_name(extra))

    if missing and options.missing_cols == "add":
        # A column with a default is added holding it; one without, null.
        lf = lf.with_columns(
            pl.lit(columns[c].default, dtype=columns[c].dtype).alias(c) for c in missing
        )

    schema = lf.collect_schema()
    if options.cast:
        cast_exprs = [
            frames.column(name).cast(spec.dtype)
            for name, spec in columns.items()
            if name in schema and schema[name] != spec.dtype
        ]
        if cast_exprs:
            lf = lf.with_columns(cast_exprs)

    # Declared columns first, in declaration order, then anything extra that
    # survived.
    present = schema.names()
    declared = [c for c in columns if c in present]
    return lf.select(
        frames.columns(declared + [c for c in present if c not in columns])
    )
