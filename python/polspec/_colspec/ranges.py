"""A column's bounds: read into the column's own type, and held to what
the type can represent.

Internal: not part of the public API.
"""

from __future__ import annotations

import datetime as dt
import decimal
import math
from collections.abc import Callable
from typing import Any

import polars as pl

from polspec import spec as colspec_module
from polspec._frozen import FrozenDict
from polspec.bound import Bound, coerce_bound, is_pair
from polspec.dtypes import (
    bound_endpoint_to_physical,
    dtype_value_limits,
    float16_inside,
)
from polspec.errors import PolspecError, SpecError


def normalize_ranges(spec: colspec_module.ColSpec) -> None:
    """Coerces `bounds` and `string_length` to `Bound`, rejecting open lengths."""
    raw: Any = spec.bounds
    if raw is not None and not isinstance(raw, Bound) and not is_pair(raw):
        raise SpecError(f"ColSpec.bounds must be a (min, max) pair, got {raw!r}")
    if raw is not None and isinstance(spec.value_dtype, pl.Decimal):
        raw = decimal_endpoints(raw, spec.value_dtype)
    if raw is not None:
        ends = (raw.min, raw.max) if isinstance(raw, Bound) else tuple(raw)
        raw = temporal_endpoints(spec, *ends)
    object.__setattr__(spec, "bounds", coerce_bound(raw))
    object.__setattr__(
        spec, "string_length", coerce_bound(spec.string_length, "string_length")
    )
    object.__setattr__(
        spec, "list_length", coerce_bound(spec.list_length, "list_length")
    )
    if spec.fields is not None:
        object.__setattr__(spec, "fields", FrozenDict(spec.fields))
    # One internal representation for "unconstrained", so every downstream
    # `if spec.bounds is not None` guard keeps meaning what it says.
    if spec.bounds is not None and spec.bounds.is_open_both:
        object.__setattr__(spec, "bounds", None)
    if spec.string_length is not None and spec.string_length.is_open:
        raise SpecError(
            "ColSpec.string_length requires both endpoints, got "
            f"{spec.string_length!r}. An open end (None) is supported on "
            "ColSpec.bounds only."
        )
    if spec.list_length is not None and spec.list_length.is_open:
        raise SpecError(
            "ColSpec.list_length requires both endpoints, got "
            f"{spec.list_length!r}. An open end (None) is supported on "
            "ColSpec.bounds only."
        )


def temporal_endpoints(
    spec: colspec_module.ColSpec, low: Any, high: Any
) -> tuple[Any, Any]:
    """The endpoints with each one written as text read as the value it
    spells, on a column whose values are dates, datetimes or times -- as
    a hand-written spec file holds them -- and text anywhere else
    refused. A `Decimal`'s text was read already.

    A `datetime` on a `Date` column is refused by name: Python counts a
    datetime as a date, and it would otherwise be reported as outside the
    range a Date can represent.
    """
    dtype = spec.value_dtype
    readers: dict[type, Callable[[str], Any]] = {
        pl.Date: dt.date.fromisoformat,
        pl.Datetime: dt.datetime.fromisoformat,
        pl.Time: dt.time.fromisoformat,
    }
    read = next((r for kind, r in readers.items() if dtype == kind), None)
    ends: list[Any] = []
    for label, endpoint in (("min", low), ("max", high)):
        if dtype == pl.Date and isinstance(endpoint, dt.datetime):
            raise SpecError(
                f"ColSpec.bounds {label} ({endpoint!r}) is a datetime, and a "
                "Date column's bounds are dates: use its .date(), or declare "
                "a Datetime column."
            )
        if not isinstance(endpoint, str) or isinstance(dtype, pl.Decimal):
            ends.append(endpoint)
            continue
        if read is None:
            hint = (
                " Write a timedelta, or {duration: {days: ...}} in a spec file."
                if isinstance(dtype, pl.Duration)
                else ""
            )
            raise SpecError(
                f"ColSpec.bounds {label} ({endpoint!r}) is text; the bounds "
                f"of a {dtype!r} column are its own values.{hint}"
            )
        try:
            ends.append(read(endpoint))
        except (ValueError, PolspecError) as exc:
            raise SpecError(
                f"ColSpec.bounds {label} ({endpoint!r}) is not an ISO "
                f"{type(dtype).__name__.lower()} a {dtype!r} column can read"
            ) from exc
    return ends[0], ends[1]


def decimal_endpoints(raw: Any, dtype: pl.Decimal) -> tuple[Any, Any]:
    """A Decimal column's endpoints as `decimal.Decimal` (an `int` stays
    an `int`; a string, as a spec file writes one, is read exactly), refusing
    one the column's scale cannot carry: Polars would round `1.005` to
    `1.00` or `1.01` without saying which, and a bound that moves when it
    is stored is not the bound declared.
    """
    given = (raw.min, raw.max) if isinstance(raw, Bound) else tuple(raw)
    if len(given) != 2:
        raise SpecError(f"ColSpec.bounds must be a (min, max) pair, got {raw!r}")
    endpoints: list[Any] = []
    for label, endpoint in zip(("min", "max"), given, strict=True):
        if endpoint is None or isinstance(endpoint, int):
            endpoints.append(endpoint)
            continue
        try:
            value = decimal.Decimal(str(endpoint))
        except decimal.InvalidOperation as exc:
            raise SpecError(
                f"ColSpec.bounds {label} ({endpoint!r}) is not a number a "
                f"{dtype!r} column can hold"
            ) from exc
        if not value.is_finite():
            raise SpecError(
                f"ColSpec.bounds {label} must be a finite value, got {endpoint!r}"
            )
        exponent = value.as_tuple().exponent
        if isinstance(exponent, int) and -exponent > dtype.scale:
            raise SpecError(
                f"ColSpec.bounds {label} ({endpoint!r}) has more decimal places "
                f"than {dtype!r} keeps (scale {dtype.scale}); it would be "
                "rounded when stored. Round it yourself, or widen the scale."
            )
        endpoints.append(value)
    return endpoints[0], endpoints[1]


def validate_bounds_dtype_support(spec: colspec_module.ColSpec) -> None:
    if spec.bounds is not None and not (
        spec.value_dtype.is_integer()
        or spec.value_dtype.is_float()
        or spec.value_dtype.is_decimal()
        or spec.value_dtype.is_temporal()
    ):
        raise SpecError(
            f"ColSpec.bounds is only supported for numeric or temporal "
            f"dtypes, got {spec.value_dtype!r}"
        )


def validate_bounds_fit_dtype(spec: colspec_module.ColSpec) -> None:
    """Rejects bounds the dtype cannot represent.

    Generation clamps to these endpoints, so an endpoint outside the
    dtype's domain has no valid interpretation. Caught here rather than at
    generation time because the Rust engine reaches them as a saturated
    cast -- an out-of-range float becomes an infinity, and building a
    distribution over a non-finite range aborts the process.
    """
    if spec.bounds is None:
        return
    limits = dtype_value_limits(spec.value_dtype)
    if limits is None:
        return
    lo_limit, hi_limit = limits
    for label, endpoint in (("min", spec.bounds.min), ("max", spec.bounds.max)):
        if endpoint is None:
            continue  # unconstrained on this side; nothing to fit
        physical = bound_endpoint_to_physical(endpoint, spec.value_dtype)
        if not math.isfinite(physical):
            raise SpecError(
                f"ColSpec.bounds {label} must be a finite value, got {endpoint!r}"
            )
        if not lo_limit <= physical <= hi_limit:
            raise SpecError(
                f"ColSpec.bounds {label} ({endpoint!r}) is outside the range "
                f"{spec.value_dtype!r} can represent [{lo_limit}, {hi_limit}]"
            )
    if spec.value_dtype == pl.Float16:
        validate_bounds_hold_a_half(spec)


def validate_bounds_hold_a_half(spec: colspec_module.ColSpec) -> None:
    """Bounds so close that no half-precision value lies between them.

    Halves near 1.3 are about 0.001 apart, so `(-1.3, -1.2999)` admits
    none: no data could satisfy it, generated or real.
    """
    assert spec.bounds is not None  # noqa: S101 - the caller checked
    low, high = spec.bounds.min, spec.bounds.max
    if low is None or high is None:
        return
    above, below = (
        float16_inside(float(low), up=True),
        float16_inside(float(high), up=False),
    )
    if above > below:
        raise SpecError(
            f"ColSpec.bounds {spec.bounds} hold no Float16 value: the "
            f"nearest halves are {below} and {above}. Widen the bounds, or "
            "declare a wider float."
        )
