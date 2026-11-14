"""One checkable claim, measured by counting the rows that violate a mask.

Subclasses supply the mask, the wording and the details; the aliases tying
the two halves together stay private to the instance. Every constraint a
spec implies is collected first and evaluated together, so validating fifty
columns costs one scan rather than fifty.
"""

from __future__ import annotations

import decimal
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec import frames
from polspec.dtypes import (
    dtype_value_limits,
    element_dtype,
    field_dtypes,
    map_entries,
    typed_values,
)
from polspec.validation.report import Finding, FindingCode, RowsWhere

if TYPE_CHECKING:
    from polspec.bound import Bound

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
        return Finding(
            code=self.code,
            key=self.key,
            message=self.message(count, samples, stats),
            columns=self.involved(),
            count=int(count),
            samples=tuple(samples),
            details=self.details(stats),
            _locate=RowsWhere(self.mask),
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
        # A value `declared` cannot hold is the range check's to report,
        # and is null here: a cast to a Decimal too narrow for it raises,
        # `strict=False` or not, rather than making a null.
        unfit = unholdable(column, declared, actual)
        if unfit is not None:
            column = pl.when(unfit[0]).then(None).otherwise(column)
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


def unholdable(
    column: pl.Expr,
    declared: pl.DataType,
    actual: pl.DataType,
    bounds: Bound | None = None,
) -> tuple[pl.Expr, str] | None:
    """A mask for the values `actual` holds and `declared` cannot, and the
    range `declared` holds in words; None when every value fits.

    Only a declared integer or `Decimal` has a range a compatible dtype can
    exceed and a cast then refuse -- and a declared `Null`, which any dtype
    stands in for and which holds no value at all. A float declared narrower
    than it arrives is left alone: past its range a cast makes an infinity,
    not an error. Each literal is typed as the column is, and each side is
    tested only where `actual` reaches past it, so every literal is one
    `actual` holds.

    `bounds` are the declared bounds being checked, if they are. A bound
    must fit its dtype to be declared, so a value past the dtype's range on
    a bounded side is past the bound too and already a `bounds` finding:
    that side is left to it. Most of the time -- a CSV's `Int64` for a
    bounded `Int32` -- that is both sides, and the check costs nothing.
    """
    if declared == pl.Null:
        return None if actual == pl.Null else (column.is_not_null(), "only nulls")
    low: pl.Expr | None = None
    high: pl.Expr | None = None

    if declared.is_integer() and actual.is_integer():
        declared_limits = dtype_value_limits(declared)
        actual_limits = dtype_value_limits(actual)
        if declared_limits is None or actual_limits is None:
            return None
        lo, hi = (int(v) for v in declared_limits)
        holds = f"{lo}..{hi}"
        if actual_limits[0] < lo:
            low = column < pl.lit(lo, dtype=actual)
        if actual_limits[1] > hi:
            high = column > pl.lit(hi, dtype=actual)
    elif isinstance(declared, pl.Decimal):
        precision = declared.precision or 38
        scale = declared.scale or 0
        # A Decimal(p, s) holds magnitudes under 10**(p - s); a value is
        # rounded to s places before that is asked, so half a unit below it
        # is too many.
        limit = 10 ** (precision - scale)
        holds = f"magnitude under {limit}"
        rounding = decimal.Decimal(5).scaleb(-(scale + 1))
        if actual.is_float():
            threshold = float(limit - rounding)
            # A NaN compares greater than any number, so it is caught on the
            # high side: a Decimal cannot hold one either.
            low, high = column <= -threshold, column >= threshold
        elif actual.is_integer():
            actual_limits = dtype_value_limits(actual)
            if actual_limits is None:
                return None
            if actual_limits[0] <= -limit:
                low = column <= pl.lit(-limit, dtype=actual)
            if actual_limits[1] >= limit:
                high = column >= pl.lit(limit, dtype=actual)
        elif isinstance(actual, pl.Decimal):
            actual_scale = actual.scale or 0
            if (actual.precision or 38) - actual_scale <= precision - scale:
                return None
            edge = decimal.Decimal(limit) - (rounding if actual_scale > scale else 0)
            low = column <= pl.lit(-edge, dtype=actual)
            high = column >= pl.lit(edge, dtype=actual)
        else:
            return None
    else:
        return None

    if bounds is not None and bounds.min is not None:
        low = None
    if bounds is not None and bounds.max is not None:
        high = None
    return _either([side for side in (low, high) if side is not None], holds)


def _either(sides: list[pl.Expr], holds: str) -> tuple[pl.Expr, str] | None:
    """`sides` as one mask, with `holds`; None when there is no side to test."""
    if not sides:
        return None
    mask = sides[0]
    for side in sides[1:]:
        mask = mask | side
    return mask, holds
