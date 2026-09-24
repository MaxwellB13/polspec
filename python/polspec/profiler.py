"""Inferring a spec from data that already exists.

The inverse of generation: given a DataFrame, describe the columns well enough
that `FrameSpec.generate` could produce something like it again.
"""

from __future__ import annotations

import dataclasses

import polars as pl

from polspec.bound import Bound
from polspec.constants import _DEFAULT_NULL_PROBABILITY
from polspec.dtypes import field_dtypes
from polspec.spec import ColSpec, _is_categorical_dtype


def profile_dataframe(
    df: pl.DataFrame,
    *,
    weights: bool = False,
    max_unique_enum: int = 50,
    calculate_bounds: bool = True,
) -> dict[str, ColSpec]:
    """Infers ColSpec column definitions by profiling an existing DataFrame."""
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame, got {type(df).__name__}")

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

    if isinstance(dtype, pl.Struct):
        # Described as its fields: each profiled as a column of its own over
        # the structs that are present, so a field's null rate is how often
        # it is null inside one. The dtype is rebuilt from the fields', since
        # profiling may narrow a String field to an Enum as it would a column.
        fields = {
            field: _profile_column(
                non_null.struct.field(field),
                field,
                total_rows=len(non_null),
                with_weights=with_weights,
                max_unique_enum=max_unique_enum,
                calculate_bounds=calculate_bounds,
            )
            for field in field_dtypes(dtype)
        }
        return spec(
            dtype=pl.Struct({field: fs.dtype for field, fs in fields.items()}),
            fields=fields or None,
        )

    if isinstance(dtype, (pl.List, pl.Array)) and isinstance(
        dtype.inner, (pl.List, pl.Array)
    ):
        # A list of lists: nothing a declaration can say describes the inner
        # values, so only the outer list's length is recorded.
        return spec(
            dtype=dtype,
            list_length=_extent(non_null.list.len(), int, calculate_bounds)
            if isinstance(dtype, pl.List)
            else None,
        )

    if isinstance(dtype, (pl.List, pl.Array)):
        # Described as its elements: profile the exploded values as a column
        # of the inner dtype, then put the list's own nullability and length
        # back on top. An element that is a struct brings its fields along.
        elements = _profile_column(
            non_null.explode(empty_as_null=False).drop_nulls(),
            name,
            total_rows=0,
            with_weights=with_weights,
            max_unique_enum=max_unique_enum,
            calculate_bounds=calculate_bounds,
        )
        list_length = None
        if isinstance(dtype, pl.List):
            list_length = _extent(non_null.list.len(), int, calculate_bounds)
        return dataclasses.replace(
            elements,
            dtype=pl.List(elements.dtype)
            if isinstance(dtype, pl.List)
            else pl.Array(elements.dtype, dtype.size),
            nullable=nullable,
            null_probability=null_probability,
            list_length=list_length,
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

    if dtype.is_decimal():
        return spec(
            dtype=dtype, bounds=_extent(non_null, lambda v: v, calculate_bounds)
        )

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

    # A dtype with nothing to measure (a `Null` column, an `Object`): recorded
    # faithfully so validation still reads it.
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
    observed = dict(
        zip(counts[name].to_list(), counts[counts.columns[1]].to_list(), strict=True)
    )
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
