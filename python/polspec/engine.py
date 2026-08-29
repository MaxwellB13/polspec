from __future__ import annotations

import random

import polars as pl

from polspec import _polspec
from polspec.bound import Bound
from polspec.constants import (
    _DEFAULT_FLOAT_BOUND,
    _DEFAULT_STRING_LEN,
    _DEFAULT_WIDE_INT_BOUND,
    _INT_DTYPE_BOUNDS,
    _MAX_CARTESIAN_ROWS,
)
from polspec.spec import ColSpec, _column_kind, _is_categorical_dtype

# Factor to scale a day/second-denominated default range into a Datetime's or
# Duration's own physical time_unit (default bounds below are expressed in µs).
_TIME_UNIT_FACTORS = {"ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}


def _bound_endpoint_to_physical(value: object, dtype: pl.DataType) -> float | int:
    """Coerces a Bound endpoint to the physical (int) representation `dtype`
    stores internally, so real `date`/`datetime`/`time`/`timedelta` objects
    can be used as bounds on temporal ColSpecs, matching what `validate()`
    already accepts.
    """
    if isinstance(value, (int, float)):
        return value
    return pl.Series([value], dtype=dtype).to_physical().item()


def _resolve_numeric_bounds(spec: ColSpec) -> tuple[float | int, float | int]:
    """Returns the (min, max) an int/float/temporal ColSpec generates within.

    Mirrors the defaulting rule used at generation time: explicit
    `spec.bounds` wins, otherwise a fixed-width int dtype defaults to its own
    range, and anything else falls back to the wide/default bound constants.
    """
    kind = _column_kind(spec.dtype)
    if spec.bounds is not None:
        lo = _bound_endpoint_to_physical(spec.bounds.min, spec.dtype)
        hi = _bound_endpoint_to_physical(spec.bounds.max, spec.dtype)
        return lo, hi
    if kind == "int":
        if spec.dtype in _INT_DTYPE_BOUNDS:
            lo, hi = _INT_DTYPE_BOUNDS[spec.dtype]
            return lo, hi
        if spec.dtype.is_unsigned_integer():
            return 0, _DEFAULT_WIDE_INT_BOUND
        return -_DEFAULT_WIDE_INT_BOUND, _DEFAULT_WIDE_INT_BOUND
    if kind == "float":
        return -_DEFAULT_FLOAT_BOUND, _DEFAULT_FLOAT_BOUND
    if kind == "temporal":
        if spec.dtype == pl.Date:
            return 0, 36525
        if spec.dtype == pl.Time:
            return 0, 86_399_999_999_999
        if isinstance(spec.dtype, pl.Datetime) or spec.dtype == pl.Datetime:
            factor = _TIME_UNIT_FACTORS[getattr(spec.dtype, "time_unit", None) or "us"]
            return 0, 36525 * 86400 * factor
        if isinstance(spec.dtype, pl.Duration) or spec.dtype == pl.Duration:
            factor = _TIME_UNIT_FACTORS[getattr(spec.dtype, "time_unit", None) or "us"]
            return 0, 365 * 86400 * factor
        return 0, _DEFAULT_WIDE_INT_BOUND
    raise TypeError(f"{spec.dtype!r} is not a numeric or temporal dtype")


def _to_rust_spec(name: str, spec: ColSpec) -> tuple:
    """Builds the tuple the Rust extension expects for one column.

    Layout: (name, kind, nullable, null_probability, min, max, categories,
    weights, str_min_len, str_max_len, distribution, distribution_params).
    """
    kind = _column_kind(spec.dtype)
    null_probability = spec.null_probability if spec.nullable else 0.0

    min_bound: float | None = None
    max_bound: float | None = None
    categories: list[str] | None = None
    weights: list[float] | None = (
        [float(w) for w in spec.weights] if spec.weights is not None else None
    )
    str_min_len: int | None = None
    str_max_len: int | None = None
    distribution: str | None = spec.distribution
    distribution_params: dict[str, float] | None = spec.distribution_params

    if spec.choices is not None:
        categories = [str(c) for c in spec.choices]
        kind = "string"
    else:
        if spec.dtype == pl.Int8:
            kind = "int8"
        elif spec.dtype == pl.Int16:
            kind = "int16"
        elif spec.dtype == pl.Int32:
            kind = "int32"
        elif spec.dtype == pl.Int64:
            kind = "int64"
        elif spec.dtype == pl.UInt8:
            kind = "uint8"
        elif spec.dtype == pl.UInt16:
            kind = "uint16"
        elif spec.dtype == pl.UInt32:
            kind = "uint32"
        elif spec.dtype == pl.UInt64:
            kind = "uint64"
        elif spec.dtype == pl.Float32:
            kind = "float32"
        elif spec.dtype == pl.Float64:
            kind = "float64"
        elif spec.dtype == pl.Date:
            kind = "int32"
        elif (
            spec.dtype == pl.Time
            or isinstance(spec.dtype, pl.Datetime)
            or spec.dtype == pl.Datetime
            or isinstance(spec.dtype, pl.Duration)
            or spec.dtype == pl.Duration
        ):
            kind = "int64"

        if spec.dtype.is_integer() or spec.dtype.is_float() or spec.dtype.is_temporal():
            if (
                spec.bounds is not None
                or spec.distribution is None
                or spec.distribution.lower() == "uniform"
            ):
                lo, hi = _resolve_numeric_bounds(spec)
                min_bound, max_bound = float(lo), float(hi)
        elif kind in ("string", "binary"):
            length = spec.string_length or Bound(*_DEFAULT_STRING_LEN)
            str_min_len, str_max_len = int(length.min), int(length.max)
            kind = "string"
        elif kind in ("enum", "categorical"):
            if kind == "enum":
                categories = spec.dtype.categories.to_list()
            kind = "string"

    return (
        name,
        kind,
        spec.nullable,
        null_probability,
        min_bound,
        max_bound,
        categories,
        weights,
        str_min_len,
        str_max_len,
        distribution,
        distribution_params,
    )


def _cast_expr(name: str, spec: ColSpec) -> pl.Expr:
    if spec.choices is not None:
        if spec.dtype in (pl.String, pl.Utf8):
            return pl.col(name)
        # The Rust engine samples from str(choice) values as plain strings;
        # look each one back up to its original typed value instead of
        # relying on pl.cast(), which can't parse e.g. "True"/a datetime
        # repr string into Boolean/Datetime/Duration/Binary.
        mapping = {str(c): c for c in spec.choices}
        return pl.col(name).replace_strict(mapping, return_dtype=spec.dtype)

    kind = _column_kind(spec.dtype)
    if _is_categorical_dtype(spec.dtype):
        return pl.col(name).cast(pl.Categorical)
    if kind == "string" and spec.dtype in (pl.String, pl.Utf8):
        return pl.col(name)
    if spec.dtype in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Boolean,
    ):
        return pl.col(name)
    return pl.col(name).cast(spec.dtype)


def _coverage_values(spec: ColSpec, rng: random.Random) -> list | None:
    """The finite set of representative values a coverage dimension takes.

    Enum/Boolean contribute their whole domain; numeric columns contribute
    one representative each from the negative/zero/positive partitions their
    bounds actually reach (e.g. bounds of 5..100 only reaches "positive").
    Nullable columns also get a `None` entry. Returns None for columns with
    no natural finite domain (String, bare Categorical) -- those are filled
    in with ordinary random generation instead.
    """
    if spec.choices is not None:
        values = list(spec.choices)
        if spec.nullable:
            values.append(None)
        return values

    kind = _column_kind(spec.dtype)
    values: list = []

    if kind == "enum":
        values = list(spec.dtype.categories.to_list())
    elif kind == "bool":
        values = [True, False]
    elif kind in ("int", "temporal"):
        lo, hi = (int(v) for v in _resolve_numeric_bounds(spec))
        if lo < 0:
            values.append(rng.randint(lo, min(hi, -1)))
        if lo <= 0 <= hi:
            values.append(0)
        if hi > 0:
            values.append(rng.randint(max(lo, 1), hi))
    elif kind == "float":
        lo, hi = (float(v) for v in _resolve_numeric_bounds(spec))
        if lo < 0:
            upper = hi if hi < 0 else (lo / 2.0 if lo > -2e-9 else -1e-9)
            values.append(rng.uniform(lo, upper))
        if lo <= 0.0 <= hi:
            values.append(0.0)
        if hi > 0:
            lower = lo if lo > 0 else (hi / 2.0 if hi < 2e-9 else 1e-9)
            values.append(rng.uniform(lower, hi))
    else:
        return None

    if spec.nullable:
        values.append(None)
    return values


def _generate_random(
    columns: dict[str, ColSpec], n: int, seed: int | None
) -> pl.DataFrame:
    if not columns:
        return pl.DataFrame()
    rust_specs = [_to_rust_spec(name, spec) for name, spec in columns.items()]
    raw_df = _polspec.generate_dataframe(rust_specs, n, seed)
    cast_exprs = [_cast_expr(name, spec) for name, spec in columns.items()]
    return raw_df.select(cast_exprs)


def _generate_cartesian(
    columns: dict[str, ColSpec], n: int, seed: int | None
) -> pl.DataFrame:
    """Guarantees coverage: the cartesian product of every Enum/Boolean's
    categories crossed with the negative/zero/positive/null partitions of
    every bounded numeric column, padded with ordinary random rows up to
    `n` if the coverage set doesn't already reach it.
    """
    rng = random.Random(seed)

    coverage_values: dict[str, list] = {}
    filler_columns: dict[str, ColSpec] = {}
    for name, spec in columns.items():
        values = _coverage_values(spec, rng)
        if values is None:
            filler_columns[name] = spec
        else:
            coverage_values[name] = values

    if not coverage_values:
        raise ValueError(
            "method='cartesian' needs at least one Enum, Boolean, or bounded "
            "numeric column to build coverage from"
        )

    coverage_size = 1
    for values in coverage_values.values():
        coverage_size *= len(values)
    if coverage_size > _MAX_CARTESIAN_ROWS:
        breakdown = ", ".join(f"{name}={len(v)}" for name, v in coverage_values.items())
        raise ValueError(
            f"cartesian coverage would need {coverage_size:,} rows ({breakdown}), "
            f"which exceeds the {_MAX_CARTESIAN_ROWS:,}-row safety cap"
        )

    coverage_df: pl.DataFrame | None = None
    for name, values in coverage_values.items():
        dim_df = pl.DataFrame({name: values}, schema={name: columns[name].dtype})
        coverage_df = (
            dim_df if coverage_df is None else coverage_df.join(dim_df, how="cross")
        )

    coverage_n = coverage_df.height

    if filler_columns:
        filler_df = _generate_random(filler_columns, coverage_n, rng.randrange(2**63))
        coverage_df = pl.concat([coverage_df, filler_df], how="horizontal_extend")

    coverage_df = coverage_df.select(list(columns.keys()))

    if coverage_n >= n:
        return coverage_df

    topup_df = _generate_random(columns, n - coverage_n, rng.randrange(2**63))
    return pl.concat([coverage_df, topup_df], how="vertical")
