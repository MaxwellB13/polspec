"""Inferring a spec from data that already exists.

The inverse of generation: given a DataFrame, describe the columns well enough
that `FrameSpec.generate` could produce something like it again.
"""

from __future__ import annotations

import polars as pl

from polspec.bound import Bound
from polspec.constants import _DEFAULT_NULL_PROBABILITY
from polspec.spec import ColSpec, _is_categorical_dtype


def profile_dataframe(
    df: pl.DataFrame,
    *,
    weights: bool = False,
    max_unique_enum: int = 50,
    max_unique: int | None = None,
    calculate_bounds: bool = True,
    bounds: bool | None = None,
) -> dict[str, ColSpec]:
    """Infers ColSpec column definitions by profiling an existing DataFrame."""
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame, got {type(df).__name__}")

    if max_unique is not None:
        max_unique_enum = max_unique
    if bounds is not None:
        calculate_bounds = bounds

    return {
        name: _profile_column(
            df[name],
            name,
            total_rows=df.height,
            with_weights=weights,
            max_unique_enum=max_unique_enum,
            calculate_bounds=calculate_bounds,
        )
        for name in df.columns
    }


def _profile_column(
    series: pl.Series,
    name: str,
    *,
    total_rows: int,
    with_weights: bool,
    max_unique_enum: int,
    calculate_bounds: bool,
) -> ColSpec:
    """Describes one column: its nullability, its domain, and its extent."""
    dtype = series.dtype
    non_null = series.drop_nulls()
    nullable, null_probability = _nullability(series, total_rows)

    def spec(**kwargs) -> ColSpec:
        return ColSpec(nullable=nullable, null_probability=null_probability, **kwargs)

    if isinstance(dtype, pl.Enum):
        categories = dtype.categories.to_list()
        return spec(
            dtype=dtype,
            weights=_empirical_weights(non_null, name, categories)
            if with_weights
            else None,
        )

    if dtype in (pl.String, pl.Utf8) or _is_categorical_dtype(dtype):
        return _profile_textual(
            non_null,
            name,
            dtype,
            spec,
            with_weights=with_weights,
            max_unique_enum=max_unique_enum,
            calculate_bounds=calculate_bounds,
        )

    if dtype == pl.Boolean:
        return spec(
            dtype=pl.Boolean,
            weights=_boolean_weights(non_null) if with_weights else None,
        )

    if dtype.is_integer():
        return spec(dtype=dtype, bounds=_extent(non_null, int, calculate_bounds))

    if dtype.is_float():
        return spec(dtype=dtype, bounds=_extent(non_null, float, calculate_bounds))

    if dtype.is_temporal():
        # Temporal bounds are recorded as the physical integer the dtype
        # stores, matching what the generation path expects back.
        physical = non_null.to_physical() if len(non_null) else non_null
        return spec(dtype=dtype, bounds=_extent(physical, int, calculate_bounds))

    if dtype == pl.Binary:
        return spec(
            dtype=pl.Binary,
            string_length=_extent(
                non_null.bin.size() if len(non_null) else non_null,
                int,
                calculate_bounds,
            ),
        )

    # A dtype polspec cannot generate. Recorded faithfully so validation still
    # works; `generate()` is what will object.
    return spec(dtype=dtype)


def _profile_textual(
    non_null: pl.Series,
    name: str,
    dtype: pl.DataType,
    spec,
    *,
    with_weights: bool,
    max_unique_enum: int,
    calculate_bounds: bool,
) -> ColSpec:
    """A String or Categorical column, narrowed to an Enum when it is small enough."""
    n_unique = non_null.n_unique()

    if 0 < n_unique <= max_unique_enum:
        categories = non_null.unique().sort().to_list()
        return spec(
            dtype=pl.Enum(categories),
            weights=_empirical_weights(non_null, name, categories)
            if with_weights
            else None,
        )

    if _is_categorical_dtype(dtype):
        return spec(dtype=pl.Categorical())

    return spec(
        dtype=pl.String,
        string_length=_extent(
            non_null.str.len_chars() if len(non_null) else non_null,
            int,
            calculate_bounds,
        ),
    )


def _nullability(series: pl.Series, total_rows: int) -> tuple[bool, float]:
    """Whether the column holds nulls, and how often."""
    null_count = series.null_count()
    if null_count == 0:
        return False, 0.0
    if total_rows > 0:
        return True, float(null_count / total_rows)
    return True, _DEFAULT_NULL_PROBABILITY


def _extent(values: pl.Series, cast, calculate_bounds: bool) -> Bound | None:
    """The observed [min, max] of `values`, or None when not being measured."""
    if not calculate_bounds or len(values) == 0:
        return None
    return Bound(cast(values.min()), cast(values.max()))


def _empirical_weights(
    non_null: pl.Series, name: str, categories: list
) -> tuple[float, ...] | None:
    """How often each category actually occurs, in `categories` order.

    Categories absent from the data get a weight of 0, so a round-trip through
    `generate()` reproduces the observed mix rather than a uniform one.
    """
    if len(non_null) == 0:
        return None
    counts = non_null.value_counts()
    observed = dict(zip(counts[name].to_list(), counts[counts.columns[1]].to_list()))
    total = float(len(non_null))
    return tuple(observed.get(category, 0) / total for category in categories)


def _boolean_weights(non_null: pl.Series) -> tuple[float, float] | None:
    """The observed [p_false, p_true] split."""
    if len(non_null) == 0:
        return None
    true_count = int(non_null.sum())
    false_count = len(non_null) - true_count
    total = false_count + true_count
    if total == 0:
        return (0.5, 0.5)
    return (false_count / total, true_count / total)
