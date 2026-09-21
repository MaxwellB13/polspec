"""Drift as a report: what changed between two specs, or between a spec and data.

`validate()` answers yes or no. This package answers *what moved*:

    diff(old, new)      # two declarations -- a spec file in a pull request
    drift(spec, df)     # a declaration and a frame -- last night's load

Both produce one `DriftReport`, whose findings carry a `severity`: a
finding is *breaking* when data that satisfied the old side could fail the
new one, and *compatible* otherwise. That is one mechanical rule -- the
question validation would answer -- so a CI gate can fail on `breaking`
and let a widened bound through.

What drift does not measure is what validation already does: uniqueness,
composite keys, foreign keys and checks are pass/fail claims about rows,
not summaries that move. `drift()` reports on columns; `validate()` is
still the verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl

from polspec._options import options_from
from polspec.drift.data import Observed
from polspec.drift.fields import Pair, compare_column, compare_table
from polspec.drift.report import DriftCode, DriftFinding, DriftReport, Severity
from polspec.frames import Frame, to_eager
from polspec.tablespec import TableSpec, as_table_spec, require_columns

__all__ = [
    "DriftCode",
    "DriftFinding",
    "DriftOptions",
    "DriftReport",
    "Severity",
    "diff",
    "drift",
]


@dataclass(frozen=True, slots=True)
class DriftOptions:
    """What counts as drift, said once.

    Parameters
    ----------
    null_rate_tolerance : float, default 0.05
        How far the observed null rate may sit from a nullable column's
        `null_probability` before `null_rate_moved` is reported. Absolute,
        not relative: a relative tolerance is unstable near zero.
    unseen_values : bool, default True
        Whether to report declared `choices` or `Enum` categories the data
        never holds (`cardinality_moved`).
    strict_dtypes : bool, default False
        The same switch as `ValidationOptions.strict_dtypes`, and decided by
        the same function: whether a `dtype_changed` is breaking.
    max_samples : int, default 10
        How many offending values a finding's `details` carry.
    """

    null_rate_tolerance: float = 0.05
    unseen_values: bool = True
    strict_dtypes: bool = False
    max_samples: int = 10

    def __post_init__(self) -> None:
        if not 0.0 <= self.null_rate_tolerance <= 1.0:
            raise ValueError(
                f"null_rate_tolerance must be between 0 and 1, got {self.null_rate_tolerance!r}"
            )
        if self.max_samples < 0:
            raise ValueError(
                f"max_samples must be non-negative, got {self.max_samples!r}"
            )


def _options_from(
    options_obj: DriftOptions | None = None, /, **options: Any
) -> DriftOptions:
    return options_from(DriftOptions, options_obj, options, what="drift")


def _table_finding(
    code: DriftCode,
    severity: Severity,
    key: str,
    message: str,
    columns: tuple[str, ...],
    **details: Any,
) -> DriftFinding:
    return DriftFinding(
        code=code,
        severity=severity,
        key=key,
        message=message,
        columns=columns,
        details=details,
    )


def diff(
    old: TableSpec | type,
    new: TableSpec | type,
    *,
    renames: Mapping[str, str] | None = None,
    options: DriftOptions | None = None,
) -> DriftReport:
    """What changed between two declarations, as a `DriftReport`.

    Parameters
    ----------
    old, new : TableSpec | FrameSpec class
        The two declarations. A spec's name is not compared: the report is
        about what the two say, not what they are called.
    renames : mapping, optional
        Columns renamed between the two, `{old_name: new_name}`. Applied to
        `old` first, through `TableSpec.rename`, so a rename is reported
        as `column_renamed` rather than as a column removed and another
        added. The caller asserts the rename; nothing is guessed from
        similar names.
    options : DriftOptions, optional
        Only `strict_dtypes` is read when diffing two specs.

    A finding is *breaking* when a frame that satisfied `old` could fail
    `new`: a narrowed domain, a dropped nullability, a column added (a frame
    that lacks it fails `missing_cols="raise"`), a constraint added. A
    widened domain or a removed constraint is *compatible*.
    """
    old_spec, new_spec = as_table_spec(old), as_table_spec(new)
    opts = _options_from(options)
    findings: list[DriftFinding] = []

    if renames:
        old_spec = old_spec.rename(renames)
        findings.extend(
            _table_finding(
                "column_renamed",
                "compatible",
                f"column:{before}",
                f"Column '{before}' renamed to '{after}'",
                (before, after),
                old=before,
                new=after,
            )
            for before, after in renames.items()
        )

    old_columns, new_columns = old_spec.columns, new_spec.columns
    for name in old_columns:
        if name not in new_columns:
            findings.append(
                _table_finding(
                    "column_removed",
                    "breaking",
                    f"column:{name}",
                    f"Column '{name}' removed; a frame carrying it fails "
                    "extra_cols='raise'",
                    (name,),
                )
            )
    for name in new_columns:
        if name not in old_columns:
            findings.append(
                _table_finding(
                    "column_added",
                    "breaking",
                    f"column:{name}",
                    f"Column '{name}' added; a frame without it fails "
                    "missing_cols='raise'",
                    (name,),
                )
            )
    for name, declared in old_columns.items():
        if name in new_columns:
            findings.extend(
                compare_column(Pair(name, declared, new_columns[name], opts))
            )

    findings.extend(compare_table(old_spec, new_spec))
    return DriftReport(
        kind="diff",
        old=old_spec.name,
        new=new_spec.name,
        findings=tuple(findings),
        options=opts,
    )


def drift(
    spec: TableSpec | type,
    df: Frame,
    *,
    options: DriftOptions | None = None,
    **option_kwargs: Any,
) -> DriftReport:
    """How `df` has moved relative to what `spec` declares, as a `DriftReport`.

    Parameters
    ----------
    spec : TableSpec | FrameSpec class
        The declaration.
    df : pl.DataFrame | pl.LazyFrame
        The frame. A LazyFrame is collected: every measurement here is a
        summary of the whole column.
    options : DriftOptions, optional
        Every option at once. Cannot be combined with the keywords.
    **option_kwargs
        The fields of `DriftOptions`, one at a time.

    A finding is *breaking* when this frame fails this spec on that column:
    values outside the domain, a bound exceeded, nulls where none are
    allowed, a format not matched. A null rate that moved within a nullable
    column, or declared values the data never holds, is *compatible* -- the
    data still validates; the declaration has stopped describing it well.

    Uniqueness, composite keys, foreign keys and checks are not measured
    here; they are pass/fail claims that `validate()` already reports.
    """
    table = as_table_spec(spec)
    require_columns(table)
    opts = _options_from(options, **option_kwargs)
    frame = to_eager(df)
    findings: list[DriftFinding] = []

    present = frame.columns
    for name in present:
        if name not in table.columns:
            findings.append(
                _table_finding(
                    "column_added",
                    "breaking",
                    f"column:{name}",
                    f"Column '{name}' is in the data but not the spec; it fails "
                    "extra_cols='raise'. Declare it, or drop it",
                    (name,),
                )
            )
    for name, declared in table.columns.items():
        if name not in present:
            findings.append(
                _table_finding(
                    "column_removed",
                    "breaking",
                    f"column:{name}",
                    f"Column '{name}' is declared but not in the data; it fails "
                    "missing_cols='raise'",
                    (name,),
                )
            )
            continue
        observed = Observed.of(frame[name], declared, opts)
        findings.extend(compare_column(Pair(name, declared, observed, opts)))

    return DriftReport(
        kind="drift",
        old=table.name,
        new="DataFrame" if isinstance(df, pl.DataFrame) else "LazyFrame",
        findings=tuple(findings),
        options=opts,
    )
