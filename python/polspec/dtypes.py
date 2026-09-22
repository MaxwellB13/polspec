"""What each dtype can actually hold.

Distinct from `polspec.constants`, which carries polspec's *default* generation
ranges: these are the hard limits a value must satisfy to survive being stored
in the dtype and read back out into Python. Both the declaration-time bounds
check (`ColSpec`) and the generation-time clamp (`engine`) need them, and
neither module may import the other, so they live here.
"""

from __future__ import annotations

import datetime as dt
import decimal

import polars as pl
from polars.datatypes import DataTypeClass

# A Polars dtype as either its class or an instance: `pl.Int64` and
# `pl.Int64()` compare and hash as one, and the tables below are keyed by
# the class because that is how they read. A constructed ColSpec always
# holds an instance.
type DtypeLike = pl.DataType | DataTypeClass


def element_dtype(dtype: pl.List | pl.Array) -> pl.DataType:
    """The element dtype of a nested dtype, as an instance: polars types
    `inner` as class-or-instance, and a ColSpec instantiates it."""
    inner = dtype.inner
    return inner() if isinstance(inner, type) else inner


def field_dtypes(dtype: pl.Struct) -> dict[str, pl.DataType]:
    """A struct's fields, name to dtype, each as an instance.

    `element_dtype`'s sibling: polars types a field's dtype as
    class-or-instance, and everything downstream compares against the
    instance a `ColSpec` holds.
    """
    return {
        field.name: (field.dtype() if isinstance(field.dtype, type) else field.dtype)
        for field in dtype.fields
    }


# Factor to scale a day/second-denominated range into a Datetime's or
# Duration's own physical time_unit.
_TIME_UNIT_FACTORS = {"ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}

_I64_MIN, _I64_MAX = -(2**63), 2**63 - 1

# The full representable range of each fixed-width integer dtype: the limits a
# user-supplied bound may not exceed. `engine._default_numeric_bounds` reads
# the same table for the range those dtypes generate within when a ColSpec
# declares no bounds of its own -- the 64-bit types being the exception, where
# the full range is not a useful default.
_INT_DTYPE_LIMITS: dict[DtypeLike, tuple[int, int]] = {
    pl.Int8: (-128, 127),
    pl.Int16: (-32_768, 32_767),
    pl.Int32: (-2_147_483_648, 2_147_483_647),
    pl.Int64: (_I64_MIN, _I64_MAX),
    pl.UInt8: (0, 255),
    pl.UInt16: (0, 65_535),
    pl.UInt32: (0, 4_294_967_295),
    pl.UInt64: (0, 2**64 - 1),
}

# Largest finite magnitude each float dtype represents. Exceeding these turns
# into an infinity on the way to Rust, which then panics building a
# distribution over a non-finite range.
_FLOAT_DTYPE_LIMITS: dict[DtypeLike, tuple[float, float]] = {
    pl.Float32: (-3.4028234663852886e38, 3.4028234663852886e38),
    pl.Float64: (-1.7976931348623157e308, 1.7976931348623157e308),
}

# Temporal limits are bounded by Python's own date/datetime/timedelta range as
# well as by i64: a value polars stores but cannot hand back to Python is no
# more useful than one it cannot store at all.
# Naive on purpose: a Datetime column's physical value is an offset from the
# naive UTC epoch whatever its time_zone, and Python's own datetime.min/max
# are naive, so attaching a tzinfo here would shift the limits by an offset
# that has nothing to do with what the dtype stores.
_EPOCH_DATE = dt.date(1970, 1, 1)
_EPOCH_DATETIME = dt.datetime(1970, 1, 1)  # noqa: DTZ001
_DATE_LIMITS = (
    (dt.date.min - _EPOCH_DATE).days,
    (dt.date.max - _EPOCH_DATE).days,
)
_DATETIME_LIMIT_DELTAS = (
    dt.datetime.min - _EPOCH_DATETIME,  # noqa: DTZ901
    dt.datetime.max - _EPOCH_DATETIME,  # noqa: DTZ901
)
_TIMEDELTA_LIMIT_DELTAS = (dt.timedelta.min, dt.timedelta.max)
# Nanoseconds in a day, minus one: the physical range of pl.Time.
_TIME_LIMITS = (0, 86_399_999_999_999)


def _delta_to_unit(delta: dt.timedelta, factor: int) -> int:
    """A timedelta in `factor`-per-second units, exactly and saturating at i64.

    Integer arithmetic throughout: at these magnitudes (~1e19 for the extreme
    timedelta) float multiplication rounds far enough to push the result past
    the very limit being computed.
    """
    total_us = (delta.days * 86_400 + delta.seconds) * 10**6 + delta.microseconds
    scaled = total_us * factor
    # Truncate toward zero so the result never lands outside the true range.
    magnitude = abs(scaled) // 10**6
    value = magnitude if scaled >= 0 else -magnitude
    return max(_I64_MIN, min(_I64_MAX, value))


def _dtype_value_limits(dtype: pl.DataType) -> tuple[float, float] | None:
    """The widest physical range `dtype` holds and still round-trips to Python.

    Returns None for dtypes with no numeric domain (String, Enum, ...), whose
    values are never bounded this way.
    """
    if dtype in _INT_DTYPE_LIMITS:
        return _INT_DTYPE_LIMITS[dtype]
    if dtype in _FLOAT_DTYPE_LIMITS:
        return _FLOAT_DTYPE_LIMITS[dtype]
    if dtype == pl.Date:
        return _DATE_LIMITS
    if dtype == pl.Time:
        return _TIME_LIMITS
    if isinstance(dtype, pl.Decimal):
        # Physical units: a Decimal(p, s) stores an integer of at most p digits.
        widest = 10**dtype.precision - 1
        return -widest, widest
    if isinstance(dtype, pl.Datetime) or dtype == pl.Datetime:
        factor = _TIME_UNIT_FACTORS[getattr(dtype, "time_unit", None) or "us"]
        lo, hi = (_delta_to_unit(d, factor) for d in _DATETIME_LIMIT_DELTAS)
        return lo, hi
    if isinstance(dtype, pl.Duration) or dtype == pl.Duration:
        factor = _TIME_UNIT_FACTORS[getattr(dtype, "time_unit", None) or "us"]
        lo, hi = (_delta_to_unit(d, factor) for d in _TIMEDELTA_LIMIT_DELTAS)
        return lo, hi
    return None


def _bound_endpoint_to_physical(value: object, dtype: pl.DataType) -> float | int:
    """Coerces a Bound endpoint to the physical (int) representation `dtype`
    stores internally, so real `date`/`datetime`/`time`/`timedelta` objects
    can be used as bounds on temporal ColSpecs, matching what `validate()`
    already accepts.
    """
    if isinstance(dtype, pl.Decimal):
        # Exactly, in Python: the physical form is the value times the scale,
        # and Polars would round or refuse an endpoint the column cannot hold
        # before the caller gets to say so.
        scaled = decimal.Decimal(str(value)).scaleb(dtype.scale)
        return int(scaled.to_integral_value())
    if isinstance(value, (int, float)):
        return value
    return pl.Series([value], dtype=dtype).to_physical().item()


def _typed_values(values, dtype: pl.DataType) -> pl.Series:
    """`values` as a Series of `dtype`, so a domain of choices is held in the
    column's own type -- `[1, 2]` on a `String` column is `["1", "2"]`, a
    `datetime` on a `Datetime` column stays a datetime.
    """
    return pl.Series(list(values), dtype=dtype, strict=False)
