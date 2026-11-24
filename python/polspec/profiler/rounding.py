"""How finely a profile writes what it measured: a step to round bounds,
rates and weights to, `auto`, or exact.

Internal: not part of the public API.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext
from typing import Literal

import polars as pl

from polspec.bound import Bound
from polspec.dtypes import dtype_value_limits

# How finely `profile()` writes what it measured: a step to round to -- a
# positive number, `1000` or `0.01` -- or "auto", or None for exactly as
# observed.
type Rounding = float | int | Decimal | Literal["auto"] | None


def step_of(value: object) -> Rounding:
    """A rounding as `profile()` takes it, checked: a positive finite step,
    "auto", or None. Text reads as the command line writes it -- `1000`,
    `1k`, `10k`, `1M`, `0.01`, `2.5e3`, `auto`, `exact`."""
    if value is None:
        return None
    if value == "auto":
        return "auto"
    if value == "exact":
        return None
    if isinstance(value, str):
        text = value.strip()
        scale = {"k": 1_000, "K": 1_000, "m": 1_000_000, "M": 1_000_000}.get(
            text[-1:], 1
        )
        if scale != 1:
            text = text[:-1]
        try:
            value = Decimal(text) * scale
        except ArithmeticError:
            raise ValueError(
                f"a rounding step must be a number such as 1000, 1k or 0.01, "
                f"or 'auto' or 'exact'; got {value!r}"
            ) from None
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"a rounding step must be a number or 'auto'; got {value!r}")
    step = Decimal(repr(value)) if isinstance(value, float) else Decimal(value)
    if not step.is_finite() or step <= 0:
        raise ValueError(f"a rounding step must be positive and finite; got {value!r}")
    return step


def _exact(value: object) -> Decimal:
    """`value` -- a float, an int or a Decimal -- as a Decimal: a float by
    its shortest repr, which converts back to the same float, so rounding
    it cannot cross the value itself; the others exactly."""
    return Decimal(repr(value)) if isinstance(value, float) else Decimal(str(value))


def _auto_step(low: Decimal, high: Decimal) -> Decimal:
    """Two significant figures of the span: `10 ** (floor(log10(span)) - 1)`.
    A constant column's span is its own magnitude."""
    span = high - low if high != low else abs(high)
    if span == 0:
        return Decimal(1)
    return Decimal(1).scaleb(span.adjusted() - 1)


def rounded_bounds(
    bounds: Bound | None, dtype: pl.DataType, rounding: Rounding
) -> Bound | None:
    """`bounds` rounded outward to `rounding`'s step -- the minimum down, the
    maximum up -- and clamped to what `dtype` holds, so every value they
    were measured from stays inside. An integer's step is a whole number;
    a Decimal's is never finer than its scale."""
    if rounding is None or bounds is None or bounds.min is None or bounds.max is None:
        return bounds
    with localcontext(prec=100):
        return _rounded_bounds(bounds, dtype, rounding)


def _rounded_bounds(bounds: Bound, dtype: pl.DataType, rounding: Rounding) -> Bound:
    low, high = _exact(bounds.min), _exact(bounds.max)
    step = _auto_step(low, high) if rounding == "auto" else _exact(rounding)
    if dtype.is_integer():
        step = max(step.to_integral_value(rounding=ROUND_CEILING), Decimal(1))
    if isinstance(dtype, pl.Decimal):
        step = max(step, Decimal(1).scaleb(-(dtype.scale or 0)))
    down = (low / step).to_integral_value(rounding=ROUND_FLOOR) * step
    up = (high / step).to_integral_value(rounding=ROUND_CEILING) * step
    if isinstance(dtype, pl.Decimal):
        top = Decimal(10) ** ((dtype.precision or 38) - (dtype.scale or 0)) - Decimal(
            1
        ).scaleb(-(dtype.scale or 0))
        down, up = max(down, -top), min(up, top)
        quantum = Decimal(1).scaleb(-(dtype.scale or 0))
        return Bound(
            down.quantize(quantum, ROUND_FLOOR), up.quantize(quantum, ROUND_CEILING)
        )
    limits = dtype_value_limits(dtype)
    if limits is not None:
        down, up = max(down, _exact(limits[0])), min(up, _exact(limits[1]))
    if dtype.is_integer():
        return Bound(int(down), int(up))
    return Bound(float(down), float(up))


def rounded_rate(rate: float, rounding: Rounding) -> float:
    """`rate` rounded up to `rounding`'s step -- "auto" keeps two significant
    figures -- and never past 1. A rate of 0 stays 0: it says a thing never
    occurs, which validation holds data to, while a rate above 0 only
    shapes what generates."""
    if rounding is None or rate <= 0:
        return rate
    exact = _exact(rate)  # 17 digits: the default context holds them
    step = (
        Decimal(1).scaleb(exact.adjusted() - 1)
        if rounding == "auto"
        else _exact(rounding)
    )
    return min(
        float((exact / step).to_integral_value(rounding=ROUND_CEILING) * step), 1.0
    )


def rounded_weights(
    weights: tuple[float, ...] | None, rounding: Rounding
) -> tuple[float, ...] | None:
    """`weights` at `rounding`'s precision, as `rounded_rate` writes a rate --
    but each to its nearest step, since a weight is relative and bounds
    nothing the data has to stay inside. A weight above 0 stays above 0, so
    no category the data holds stops being drawn."""
    if weights is None or rounding is None:
        return weights
    return tuple(_rounded_weight(weight, rounding) for weight in weights)


def _rounded_weight(weight: float, rounding: Rounding) -> float:
    if weight <= 0:
        return weight
    exact = _exact(weight)
    step = (
        Decimal(1).scaleb(exact.adjusted() - 1)
        if rounding == "auto"
        else _exact(rounding)
    )
    nearest = (exact / step).to_integral_value(rounding=ROUND_HALF_EVEN) * step
    return float(nearest if nearest > 0 else step)
