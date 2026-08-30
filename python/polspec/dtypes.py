"""What each dtype can actually hold.

Distinct from `polspec.constants`, which carries polspec's *default* generation
ranges: these are the hard limits a value must satisfy to survive being stored
in the dtype and read back out into Python. Both the declaration-time bounds
check (`ColSpec`) and the generation-time clamp (`engine`) need them, and
neither module may import the other, so they live here.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

# Factor to scale a day/second-denominated range into a Datetime's or
# Duration's own physical time_unit.
_TIME_UNIT_FACTORS = {"ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}

_I64_MIN, _I64_MAX = -(2**63), 2**63 - 1

# The full representable range of each fixed-width integer dtype. Note this is
# *not* _INT_DTYPE_BOUNDS, which is the narrower set polspec generates within
# by default; these are the limits a user-supplied bound may not exceed.
_INT_DTYPE_LIMITS: dict[pl.DataType, tuple[int, int]] = {
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
_FLOAT_DTYPE_LIMITS: dict[pl.DataType, tuple[float, float]] = {
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

# Largest magnitude a float64 represents without losing integer precision.
# The spec tuple handed to the Rust engine carries bounds as f64, so a clamp
# beyond this would not survive the trip intact -- it could round *outward*
# past the limit it exists to enforce.
_EXACT_FLOAT_INT = 2**53


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
    if isinstance(dtype, pl.Datetime) or dtype == pl.Datetime:
        factor = _TIME_UNIT_FACTORS[getattr(dtype, "time_unit", None) or "us"]
        return tuple(_delta_to_unit(d, factor) for d in _DATETIME_LIMIT_DELTAS)
    if isinstance(dtype, pl.Duration) or dtype == pl.Duration:
        factor = _TIME_UNIT_FACTORS[getattr(dtype, "time_unit", None) or "us"]
        return tuple(_delta_to_unit(d, factor) for d in _TIMEDELTA_LIMIT_DELTAS)
    return None


def _generation_clamp_limits(dtype: pl.DataType) -> tuple[float, float] | None:
    """`_dtype_value_limits` narrowed to what the f64 spec channel carries exactly.

    Used to clamp a distribution that has no explicit bounds. A declared bound
    is checked against the dtype's true domain instead -- these are narrower on
    purpose, because a clamp that rounds outward on its way to Rust enforces
    nothing. Once bounds reach the engine as integers this can collapse back
    into `_dtype_value_limits`.
    """
    limits = _dtype_value_limits(dtype)
    if limits is None:
        return None
    lo, hi = limits
    return max(lo, -_EXACT_FLOAT_INT), min(hi, _EXACT_FLOAT_INT)


def _bound_endpoint_to_physical(value: object, dtype: pl.DataType) -> float | int:
    """Coerces a Bound endpoint to the physical (int) representation `dtype`
    stores internally, so real `date`/`datetime`/`time`/`timedelta` objects
    can be used as bounds on temporal ColSpecs, matching what `validate()`
    already accepts.
    """
    if isinstance(value, (int, float)):
        return value
    return pl.Series([value], dtype=dtype).to_physical().item()
