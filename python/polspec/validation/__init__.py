"""Checking a frame against the claims a spec makes about it.

Two entry points over the same machinery:

- `inspect(spec, df)` returns a `ValidationReport` -- every finding as data,
  with the offending rows reachable lazily -- and never raises for a bad
  frame.
- `validate(spec, df)` inspects, raises `ValidationError` carrying that
  report if anything was found, and otherwise returns the frame with the
  requested structural transformations applied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, overload

import polars as pl

from polspec.errors import SpecError, ValidationError
from polspec.tablespec import TableSpec, resolve_references
from polspec.validation.constraints import (
    _column_constraints,
    _Constraint,
    _foreign_key_findings,
    _frame_constraints,
    _is_dtype_compatible,
)
from polspec.validation.report import (
    FINDING_COLUMN,
    Finding,
    FindingCode,
    ValidationReport,
)

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

References = Mapping[Any, pl.DataFrame | pl.LazyFrame] | None


@dataclass(frozen=True, slots=True)
class ValidationOptions:
    """Which checks to run and what to do about structural mismatches.

    Grouped rather than passed as ten separate arguments, so adding a check
    does not widen every signature between here and `FrameSpec.validate`.
    """

    extra_cols: Literal["drop", "allow", "raise"] = "raise"
    missing_cols: Literal["add", "allow", "raise"] = "raise"
    strict_dtypes: bool = False
    rules: bool = True
    validators: bool = True
    unique: bool = True
    checks: bool = True
    foreign_keys: bool = True
    cast: bool = False
    streaming: bool = False

    def __post_init__(self) -> None:
        if self.extra_cols not in ("drop", "allow", "raise"):
            raise ValueError(
                f"extra_cols must be one of 'drop', 'allow', 'raise'; got {self.extra_cols!r}"
            )
        if self.missing_cols not in ("add", "allow", "raise"):
            raise ValueError(
                f"missing_cols must be one of 'add', 'allow', 'raise'; got {self.missing_cols!r}"
            )


_Options = ValidationOptions


def _options_from(**options: Any) -> ValidationOptions:
    renamed = {
        "validate_rules": "rules",
        "validate_validators": "validators",
        "validate_unique": "unique",
        "validate_checks": "checks",
        "validate_foreign_keys": "foreign_keys",
    }
    return ValidationOptions(**{renamed.get(k, k): v for k, v in options.items()})


def _to_lazy(frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    return frame.lazy() if isinstance(frame, pl.DataFrame) else frame


def _resolve_foreign_keys(
    spec: TableSpec, references: References
) -> tuple[list[tuple[Any, pl.LazyFrame | None]], list[Finding]]:
    """Pairs each key with its parent frame; a key with no parent is a finding."""
    parents = resolve_references(references, _to_lazy)
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


def inspect(
    spec: TableSpec,
    df: pl.DataFrame | pl.LazyFrame,
    *,
    references: References = None,
    **options: Any,
) -> ValidationReport:
    """Everything `spec` has to say about `df`, as a `ValidationReport`.

    Never raises for a frame that fails: every violation is a `Finding` on
    the report, with `report.rows(finding)` and `report.failing_rows()`
    giving the offending rows back lazily. See `validate` for the options.
    """
    if not spec.columns:
        raise SpecError(f"{spec.name} declares no ColSpec columns")
    opts = _options_from(**options)
    lf = _to_lazy(df)
    columns = dict(spec.columns)

    df_schema = lf.collect_schema()
    df_col_names = df_schema.names()
    extra = [col for col in df_col_names if col not in columns]
    missing = [col for col in columns if col not in df_col_names]

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

    constraints: list[_Constraint] = []
    for name in (c for c in columns if c in df_col_names):
        cs = columns[name]
        actual_dtype = df_schema[name]
        compatible = _is_dtype_compatible(
            cs.dtype, actual_dtype, strict=opts.strict_dtypes
        )
        if not compatible:
            findings.append(
                Finding(
                    code="dtype",
                    key=f"{name}__dtype",
                    message=f"Column '{name}': expected dtype {cs.dtype}, got {actual_dtype}",
                    columns=(name,),
                    details={"expected": str(cs.dtype), "actual": str(actual_dtype)},
                )
            )
        constraints.extend(
            _column_constraints(
                name,
                cs,
                actual_dtype,
                compatible=compatible,
                options=opts,
                df_col_names=df_col_names,
            )
        )
    constraints.extend(
        _frame_constraints(
            spec.unique_together if opts.unique else None,
            spec.checks if opts.checks else None,
            df_col_names,
        )
    )

    collect_kwargs: dict[str, Any] = {"engine": "streaming"} if opts.streaming else {}

    if constraints:
        aggregations = [expr for c in constraints for expr in c.aggregations()]
        stats = (
            lf.select(aggregations).collect(**collect_kwargs).to_dict(as_series=False)
        )
        findings.extend(
            f for f in (c.failure(stats) for c in constraints) if f is not None
        )

    if opts.foreign_keys and spec.foreign_keys:
        resolved, unresolved = _resolve_foreign_keys(spec, references)
        findings.extend(unresolved)
        findings.extend(
            _foreign_key_findings(lf, spec.name, resolved, df_col_names, collect_kwargs)
        )

    return ValidationReport(spec.name, tuple(findings), lf, opts)


@overload
def validate(spec: TableSpec, df: pl.DataFrame, **options: Any) -> pl.DataFrame: ...


@overload
def validate(spec: TableSpec, df: pl.LazyFrame, **options: Any) -> pl.LazyFrame: ...


def validate(
    spec: TableSpec,
    df: pl.DataFrame | pl.LazyFrame,
    *,
    extra_cols: Literal["drop", "allow", "raise"] = "raise",
    missing_cols: Literal["add", "allow", "raise"] = "raise",
    strict_dtypes: bool = False,
    validate_rules: bool = True,
    validate_validators: bool = True,
    validate_unique: bool = True,
    validate_checks: bool = True,
    validate_foreign_keys: bool = True,
    references: References = None,
    cast: bool = False,
    streaming: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Validates a DataFrame or LazyFrame against `spec`.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        The frame to validate. A LazyFrame comes back as a LazyFrame.
    extra_cols : {"drop", "allow", "raise"}
        Columns present in `df` but not declared: raise a ValidationError
        naming them, drop them from the returned frame, or keep them.
    missing_cols : {"add", "allow", "raise"}
        Declared columns absent from `df`: raise, add them as nulls of the
        declared dtype, or skip them.
    strict_dtypes : bool
        Require identical dtypes, rather than accepting a compatible one
        (a narrower integer, a String where an Enum was declared).
    validate_rules, validate_validators, validate_unique, validate_checks,
    validate_foreign_keys : bool
        Switch off individual kinds of check.
    references : mapping
        Parent frames for foreign keys that reference another spec, keyed by
        that spec, its FrameSpec class, or its name. A key with no entry is
        reported as a `foreign_key_unresolved` finding.
    cast : bool
        Cast validated columns to their declared dtype in the returned frame.
    streaming : bool
        Use Polars' streaming engine for the aggregation.

    Returns the validated, optionally transformed frame. Raises
    `ValidationError` carrying a `ValidationReport` of every violation, or
    `ValueError` for an invalid option.
    """
    report = inspect(
        spec,
        df,
        references=references,
        extra_cols=extra_cols,
        missing_cols=missing_cols,
        strict_dtypes=strict_dtypes,
        validate_rules=validate_rules,
        validate_validators=validate_validators,
        validate_unique=validate_unique,
        validate_checks=validate_checks,
        validate_foreign_keys=validate_foreign_keys,
        cast=cast,
        streaming=streaming,
    )
    report.raise_if_failed()
    return _transformed(spec, df, report)


def _transformed(
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
        lf = lf.drop(extra)

    if missing and options.missing_cols == "add":
        lf = lf.with_columns(
            pl.lit(None, dtype=columns[c].dtype).alias(c) for c in missing
        )

    schema = lf.collect_schema()
    if options.cast:
        cast_exprs = [
            pl.col(name).cast(spec.dtype)
            for name, spec in columns.items()
            if name in schema and schema[name] != spec.dtype
        ]
        if cast_exprs:
            lf = lf.with_columns(cast_exprs)

    # Declared columns first, in declaration order, then anything extra that
    # survived.
    present = schema.names()
    declared = [c for c in columns if c in present]
    return lf.select(declared + [c for c in present if c not in columns])
