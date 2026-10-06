"""One checkable claim, measured by counting the rows that violate a mask.

Subclasses supply the mask, the wording and the details; the aliases tying
the two halves together stay private to the instance. Every constraint a
spec implies is collected first and evaluated together, so validating fifty
columns costs one scan rather than fifty.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from polspec import frames
from polspec.dtypes import element_dtype, field_dtypes, map_entries, typed_values
from polspec.validation.report import Finding, FindingCode

MAX_SAMPLES = 5


@dataclass(kw_only=True)
class Constraint:
    """One checkable claim, measured by counting the rows that violate a mask.

    Subclasses supply the mask, the wording and the details; the aliases
    tying the two halves together stay private to the instance.
    """

    key: str
    mask: pl.Expr
    sample_expr: pl.Expr | None = None
    unique_samples: bool = True
    code: FindingCode

    def _alias(self, suffix: str) -> str:
        return f"__val__{self.key}__{suffix}"

    def aggregations(self) -> list[pl.Expr]:
        exprs = [self.mask.sum().alias(self._alias("cnt"))]
        if self.sample_expr is not None:
            samples = self.sample_expr.filter(self.mask)
            if self.unique_samples:
                samples = samples.unique(maintain_order=True)
            exprs.append(
                samples.head(MAX_SAMPLES).implode().alias(self._alias("samples"))
            )
        return exprs

    def failure(self, stats: dict[str, list]) -> Finding | None:
        count = stats[self._alias("cnt")][0]
        if not count:
            return None
        samples = self._samples(stats)
        mask = self.mask
        return Finding(
            code=self.code,
            key=self.key,
            message=self.message(count, samples, stats),
            columns=self.involved(),
            count=int(count),
            samples=tuple(samples),
            details=self.details(stats),
            _locate=lambda lf: lf.filter(mask),
        )

    def _samples(self, stats: dict[str, list]) -> list:
        if self.sample_expr is None:
            return []
        raw = stats[self._alias("samples")][0]
        return list(raw) if raw is not None else []

    def involved(self) -> tuple[str, ...]:
        return ()

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Dtype compatibility
# ---------------------------------------------------------------------------


def is_dtype_compatible(
    expected: pl.DataType, actual: pl.DataType, *, strict: bool
) -> bool:
    """Whether `actual` can stand in for the declared `expected` dtype.

    Strict mode demands the same dtype, with String/Utf8 treated as one. The
    default is permissive about widening, about textual columns arriving as
    String rather than their declared Enum/Categorical, and about a date
    arriving as a datetime or the reverse, since that is how they come back
    from CSV and JSON. A temporal is still held to its kind: see
    `_same_temporal_kind`.
    """
    if strict:
        if expected in (pl.String, pl.Utf8):
            return actual in (pl.String, pl.Utf8)
        return actual == expected

    if expected == pl.Null:
        # Any dtype can hold nothing but nulls -- a text format hands an
        # empty column back as `String` -- and a value is a finding of its
        # own (`_unholdable`), with its rows.
        return True
    if isinstance(expected, pl.Enum):
        return isinstance(actual, pl.Enum) or actual in (
            pl.String,
            pl.Utf8,
            pl.Categorical,
        )
    if isinstance(expected, pl.Categorical) or expected == pl.Categorical:
        return actual in (pl.String, pl.Utf8, pl.Categorical) or isinstance(
            actual, (pl.Enum, pl.Categorical)
        )
    if expected in (pl.String, pl.Utf8):
        return actual in (pl.String, pl.Utf8)
    if expected.is_integer():
        return actual.is_integer()
    if expected.is_float():
        return actual.is_float() or actual.is_integer()
    if expected.is_decimal():
        # A Decimal comes back from CSV and JSON as a float or an integer;
        # another Decimal stands in whatever its precision, since the
        # bounds are checked on the values either way.
        return actual.is_decimal() or actual.is_float() or actual.is_integer()
    if expected.is_temporal():
        return _same_temporal_kind(expected, actual)
    if isinstance(expected, pl.List):
        return isinstance(actual, pl.List) and is_dtype_compatible(
            element_dtype(expected), element_dtype(actual), strict=strict
        )
    if isinstance(expected, pl.Array):
        return (
            isinstance(actual, pl.Array)
            and actual.size == expected.size
            and is_dtype_compatible(
                element_dtype(expected), element_dtype(actual), strict=strict
            )
        )
    if (want_entries := map_entries(expected)) is not None:
        # A map as the list of entries it is: its key and value each
        # compatible, as a list of structs' fields would be.
        got_entries = map_entries(actual)
        return got_entries is not None and is_dtype_compatible(
            want_entries, got_entries, strict=strict
        )
    if isinstance(expected, pl.Struct):
        # The fields by name, each compatible: order is how the data was
        # written, not what it claims, and a field missing or added is a
        # different struct.
        if not isinstance(actual, pl.Struct):
            return False
        want, got = field_dtypes(expected), field_dtypes(actual)
        return want.keys() == got.keys() and all(
            is_dtype_compatible(want[f], got[f], strict=strict) for f in want
        )
    return actual == expected


def _same_temporal_kind(expected: pl.DataType, actual: pl.DataType) -> bool:
    """Whether `actual` holds the same kind of time as the declared `expected`.

    A calendar value -- a `Date` or a `Datetime` -- stands in for either,
    since text formats hand one back as the other; a time of day only for a
    time of day, and a duration only for a duration, whatever the unit. A
    zoned `Datetime` is an instant and a naive one a wall-clock reading, so
    neither stands in for the other; two zones do, being the same instants
    shown differently.
    """
    calendar = (pl.Date, pl.Datetime)
    if isinstance(expected, calendar) or expected in calendar:
        if not (isinstance(actual, calendar) or actual in calendar):
            return False
        if isinstance(expected, pl.Datetime) and isinstance(actual, pl.Datetime):
            return (expected.time_zone is None) == (actual.time_zone is None)
        return True
    if expected == pl.Time:
        return actual == pl.Time
    if isinstance(expected, pl.Duration) or expected == pl.Duration:
        return isinstance(actual, pl.Duration) or actual == pl.Duration
    return actual == expected


def in_values(
    column: pl.Expr,
    values: Sequence[Any],
    declared: pl.DataType,
    actual: pl.DataType,
) -> pl.Expr:
    """Whether each value of a non-textual `column` is one of `values`, asked
    in the declared dtype -- the column as `cast=True` would hand it back.

    `is_in` compares like with like -- strictly so from Polars 2 -- and a
    Python list reaches it at its widest type, a datetime at microseconds or
    a Decimal at full precision; so the values are typed as declared, and a
    column of another dtype that stands in for it is cast to meet them. A
    value the cast cannot make is null here, neither in nor out: that is the
    range check's finding, not this one's. Imploded, so Polars reads the
    values as one set, not row by row.
    """
    if actual != declared:
        column = column.cast(declared, strict=False)
    return column.is_in(typed_values(values, declared).implode())


def as_strings(values: Sequence[Any], dtype: pl.DataType) -> list[str]:
    """`values` as the strings a column of `dtype` holds them as, so a choice
    of `True` on a String column compares as `"true"` -- the same form
    generation produces -- rather than as Python's `str(True)`.
    """
    try:
        return typed_values(values, dtype).cast(pl.String).to_list()
    except Exception:  # noqa: BLE001 - values the dtype cannot hold fall back to str()
        return [str(v) for v in values]


def sample_source(column: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """`column` as the expression its samples are drawn from.

    An `Array` is sampled as the `List` it holds the values of: Polars 1
    panics on `unique(maintain_order=True)` over an `Array` column of more
    than one chunk when the filter before it keeps no row -- which is every
    passing frame read from a file in row groups, or concatenated. Python
    reads both as a list, so the samples are the same values either way.
    """
    return column.arr.to_list() if isinstance(dtype, pl.Array) else column


def struct_of(names: Sequence[str]) -> pl.Expr | None:
    """A struct of the named columns, for sampling multi-column claims."""
    return pl.struct(frames.columns(names)) if names else None
