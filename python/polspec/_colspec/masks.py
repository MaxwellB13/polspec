"""The masks that say which values of a column break a claim: out of its
bounds, not one of its choices, a length outside its range, a value its
dtype cannot hold.

One definition for validation, which builds its findings from them, and
for `claims_refused`, which holds a default or a pinned value to the same
claims where it is declared.

Internal: not part of the public API.
"""

from __future__ import annotations

import decimal
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec.dtypes import dtype_value_limits, typed_values

if TYPE_CHECKING:
    from polspec.bound import Bound


def out_of_bounds(column: pl.Expr, bounds: Bound, actual_dtype: pl.DataType) -> pl.Expr:
    """A mask for values outside `bounds`, testing only the constrained sides.

    An open end is genuinely unconstrained here, unlike at generation time
    where it falls back to a default (see `ColSpec.bounds`).
    """

    def limit(value: Any) -> Any:
        return pl.lit(value).cast(actual_dtype) if actual_dtype.is_temporal() else value

    violations = []
    if bounds.min is not None:
        violations.append(column < limit(bounds.min))
    if bounds.max is not None:
        violations.append(column > limit(bounds.max))
    return violations[0] if len(violations) == 1 else violations[0] | violations[1]


def measure_length(column: pl.Expr, actual_dtype: pl.DataType) -> pl.Expr | None:
    """Length of each value, for the dtypes where that is meaningful."""
    if actual_dtype in (pl.String, pl.Utf8):
        return column.str.len_chars()
    if actual_dtype == pl.Binary:
        return column.bin.size()
    return None


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
        unfit = unholdable_mask(column, declared, actual)
        if unfit is not None:
            column = pl.when(unfit[0]).then(None).otherwise(column)
        column = column.cast(declared, strict=False)
    return column.is_in(typed_values(values, declared).implode())


def unholdable_mask(
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
