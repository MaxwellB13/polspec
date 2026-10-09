"""Fitting a distribution to the values a column holds.

The profiler records a numeric column's extremes, and `generate()` fills
the range between them evenly -- which a log-normal amount, a bell-curved
age or a count with a long tail is not. This names the distribution the
engine can draw that is closest to what was observed, with its parameters,
so a profiled spec generates the shape of its data and not only its range.

Each candidate the engine supports is fitted by the method of moments, then
*drawn*: through `generate()` itself, with the column's observed bounds, so
the clamping at those bounds and the rounding of an integer column are
exactly what generation will do. The candidate whose draws sit closest to
the observed values -- by the two-sample Kolmogorov-Smirnov distance -- wins,
if it beats drawing uniformly by a margin; otherwise the column stays
uniform, as it always was. Pure Polars: no SciPy, no NumPy.

Every value here is *physical*: a date as its day count, a datetime or a
duration as an integer in its unit, a decimal scaled to its integer form --
the units the engine draws in, and so the units its parameters are in.

Internal: not part of the public API.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import polars as pl

from polspec.bound import Bound
from polspec.errors import PolspecError
from polspec.generation import generate
from polspec.spec import ColSpec
from polspec.tablespec import TableSpec

# Values fitted and drawn per candidate: plenty to tell shapes apart, few
# enough that fitting a wide frame costs seconds, not minutes.
SAMPLE = 20_000

# How much closer than uniform a candidate has to come before it is
# declared. A shape that barely beats uniform is noise, and uniform is the
# claim that promises least.
MARGIN = 0.02


@dataclass(frozen=True, slots=True)
class Fit:
    """A distribution, its parameters, and how far its draws sit from the
    observed values (the KS distance, 0 to 1)."""

    distribution: str
    params: dict[str, float]
    distance: float


def fit(
    values: pl.Series,
    bounds: Bound,
    dtype: pl.DataType,
    *,
    seed: int,
) -> Fit | None:
    """The distribution that best describes `values`, or None when drawing
    uniformly between `bounds` describes them as well.

    `values` are the column's non-null physical values; `bounds` are what
    the column will declare, in the column's own terms, since the
    candidates are drawn through a `ColSpec` of `dtype` with them.
    """
    observed = values.cast(pl.Float64).drop_nulls()
    if observed.len() < 50 or observed.n_unique() < 3:
        return None  # too little to tell one shape from another
    if observed.len() > SAMPLE:
        observed = observed.sample(SAMPLE, seed=seed)
    observed = observed.sort()

    def draws(distribution: str | None, params: dict[str, float]) -> pl.Series | None:
        column = ColSpec(
            dtype,
            bounds=bounds,
            distribution=distribution,
            distribution_params=params or None,
        )
        try:
            frame = generate(TableSpec("Fit", {"v": column}), SAMPLE, seed=seed)
        except (PolspecError, ValueError):
            return None  # parameters the engine refuses are not a fit
        return physical(frame["v"]).cast(pl.Float64).sort()

    uniform = draws(None, {})
    if uniform is None:
        return None
    baseline = ks_distance(observed, uniform)

    best: Fit | None = None
    for distribution, params in _candidates(observed, dtype):
        drawn = draws(distribution, params)
        if drawn is None:
            continue
        distance = ks_distance(observed, drawn)
        if best is None or distance < best.distance:
            best = Fit(distribution, params, distance)
    if best is None or best.distance > baseline - MARGIN:
        return None
    return best


def physical(series: pl.Series) -> pl.Series:
    """A column's values in the units `fit` takes and the engine draws in."""
    if isinstance(series.dtype, pl.Decimal):
        return series.cast(pl.Float64) * 10**series.dtype.scale
    if series.dtype.is_temporal():
        return series.to_physical()
    return series


def _number(value: Any) -> float:
    """A Polars statistic as a float -- `mean()` and friends are typed to
    return any literal, though on a float column each is a float."""
    return float(value)


def _candidates(
    observed: pl.Series, dtype: pl.DataType
) -> list[tuple[str, dict[str, float]]]:
    """Each distribution that could describe `observed`, fitted by the
    method of moments. Only shapes the values' support allows: the engine
    draws without a location shift, so a positive-only shape is offered only
    for positive values, and a beta only for values inside [0, 1]."""
    mean, std = _number(observed.mean()), _number(observed.std())
    lo, hi = _number(observed.min()), _number(observed.max())
    if not std > 0 or not math.isfinite(std):
        return []
    var = std * std
    found: list[tuple[str, dict[str, float]]] = [("normal", {"mean": mean, "std": std})]
    # A date, a datetime or a time of day sits far from zero in physical
    # units; only a symmetric shape means anything for it. A duration is a
    # magnitude, like a number.
    positional = dtype.is_temporal() and not isinstance(dtype, pl.Duration)
    if positional:
        return found
    if lo > 0:
        logs = observed.log()
        log_std = _number(logs.std())
        if log_std > 0:
            found.append(("lognormal", {"mean": _number(logs.mean()), "std": log_std}))
        found.append(("gamma", {"shape": mean * mean / var, "scale": var / mean}))
    if lo >= 0 and mean > 0:
        found.append(("exponential", {"rate": 1.0 / mean}))
        if dtype.is_integer():
            found.append(("poisson", {"lambda": mean}))
    if lo >= 0 and hi <= 1 and 0 < mean < 1:
        common = mean * (1 - mean) / var - 1
        if common > 0:
            found.append(
                ("beta", {"alpha": mean * common, "beta": (1 - mean) * common})
            )
    return found


def ks_distance(a: pl.Series, b: pl.Series) -> float:
    """The two-sample Kolmogorov-Smirnov distance: the widest gap between
    the two empirical distribution functions. Ties -- every value of an
    integer column is one -- are grouped, so a step is taken once."""
    steps = (
        pl.concat(
            [
                pl.DataFrame({"v": a, "fa": 1.0 / a.len(), "fb": 0.0}),
                pl.DataFrame({"v": b, "fa": 0.0, "fb": 1.0 / b.len()}),
            ]
        )
        .group_by("v")
        .agg(pl.col("fa").sum(), pl.col("fb").sum())
        .sort("v")
        .select((pl.col("fa").cum_sum() - pl.col("fb").cum_sum()).abs().max())
    )
    return float(steps.item())


def with_fit(column: ColSpec, fitted: Fit) -> ColSpec:
    """`column` declaring the fitted distribution."""
    return dataclasses.replace(
        column, distribution=fitted.distribution, distribution_params=fitted.params
    )
