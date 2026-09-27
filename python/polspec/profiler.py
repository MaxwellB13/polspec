"""Inferring a spec from data that already exists.

The inverse of generation: given a DataFrame, describe the columns well enough
that `FrameSpec.generate` could produce something like it again.

By default a column is described by its dtype, its null rate, its extremes
and -- for a low-cardinality text column -- its categories. Three opt-in
steps describe more, for when the spec is going to generate a stand-in for
the data (`polspec.synthesize`):

- `shape=True` fits the distribution a numeric or temporal column's values
  follow (`polspec.shape`), so the stand-in is not uniform where the data
  is skewed;
- `detect_unique=True` declares an all-distinct integer or text column
  unique, so a key stays a key;
- `replace=` names columns whose *values* must not be carried into the
  spec: a text column among them is never narrowed to an `Enum` of the
  values it holds, and carries no weights.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import polars as pl

from polspec.bound import Bound
from polspec.constants import DEFAULT_NULL_PROBABILITY
from polspec.dtypes import MAP, field_dtypes, map_entries
from polspec.shape import fit, physical, with_fit
from polspec.spec import ColSpec, is_categorical_dtype

# The fewest distinct values that make an all-distinct column a key. Ten
# rows of distinct values are ten rows, not evidence of a key.
KEY_MIN_ROWS = 100


@dataclass(frozen=True, slots=True)
class _Options:
    """How much to infer, read by every column, however deeply nested."""

    weights: bool
    max_unique_enum: int
    calculate_bounds: bool
    shape: bool = False
    detect_unique: bool = False
    seed: int = 0


def profile_dataframe(
    df: pl.DataFrame,
    *,
    weights: bool = False,
    max_unique_enum: int = 50,
    calculate_bounds: bool = True,
    shape: bool = False,
    detect_unique: bool = False,
    replace: Sequence[str] = (),
    seed: int = 0,
) -> dict[str, ColSpec]:
    """Infers ColSpec column definitions by profiling an existing DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The data to describe.
    weights : bool, default False
        Record each category's observed frequency, for categorical, enum and
        boolean columns.
    max_unique_enum : int, default 50
        A string or categorical column with at most this many distinct values
        becomes an `Enum` of them.
    calculate_bounds : bool, default True
        Record the observed extremes of numeric and temporal columns, and the
        length range of text and binary ones.
    shape : bool, default False
        Fit the distribution each numeric or temporal column follows, so it
        generates that shape rather than uniformly between its extremes.
        Needs `calculate_bounds`.
    detect_unique : bool, default False
        Declare an integer or text column `unique=True` when every one of at
        least a hundred values is distinct.
    replace : sequence of str, default ()
        Columns whose values must not be carried into the spec: a text
        column named here is described by its lengths alone -- never as an
        `Enum` of the values it holds -- and carries no weights.
    seed : int, default 0
        Seeds what fitting a shape samples, so a profile is the same every run.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame, got {type(df).__name__}")
    replaced = set(replace)
    unknown = sorted(replaced - set(df.columns))
    if unknown:
        raise ValueError(
            f"replace= names {unknown}, which the data does not have. Its "
            f"columns are {df.columns}."
        )
    for name in sorted(replaced):
        if isinstance(df.schema[name], pl.Enum):
            raise ValueError(
                f"replace= names {name!r}, an Enum: its categories are its "
                "dtype, so they are in the spec however it is profiled. Cast "
                "it to String first to have its values replaced."
            )

    options = _Options(
        weights=weights,
        max_unique_enum=max_unique_enum,
        calculate_bounds=calculate_bounds,
        shape=shape and calculate_bounds,
        detect_unique=detect_unique,
        seed=seed,
    )
    return {
        name: _profile_column(
            df[name],
            name,
            total_rows=df.height,
            options=options,
            imitate=name not in replaced,
            top=True,
        )
        for name in df.columns
    }


def _profile_column(
    series: pl.Series,
    name: str,
    *,
    total_rows: int,
    options: _Options,
    imitate: bool = True,
    top: bool = False,
) -> ColSpec:
    """Describes one column: its nullability, its domain, and its extent.

    `imitate` is False for a column in `replace=`, whose values are not to be
    carried over; `top` is False for a struct's field or a list's element,
    which cannot be declared unique.
    """
    dtype = series.dtype
    non_null = series.drop_nulls()
    nullable, null_probability = _nullability(series, total_rows)
    with_weights = options.weights and imitate

    def spec(**kwargs) -> ColSpec:
        return ColSpec(nullable=nullable, null_probability=null_probability, **kwargs)

    def nested(values: pl.Series, label: str, rows: int) -> ColSpec:
        return _profile_column(
            values, label, total_rows=rows, options=options, imitate=imitate
        )

    if isinstance(dtype, pl.Enum):
        categories = dtype.categories.to_list()
        return spec(
            dtype=dtype,
            weights=_empirical_weights(non_null, name, categories)
            if with_weights
            else None,
        )

    if (entries := map_entries(dtype)) is not None:
        # Described as the list of entries it is -- its length, its key and
        # value as the fields of a struct -- then made a map again from the
        # key and value dtypes profiling settled on, which may narrow a
        # String key to an Enum as it would a column.
        as_list = nested(series.cast(entries), name, total_rows)
        parts = as_list.fields or {}
        if set(parts) != {"key", "value"}:
            return spec(dtype=dtype)
        return dataclasses.replace(
            as_list,
            dtype=MAP(parts["key"].dtype, parts["value"].dtype),
            element_null_probability=0.0,
        )

    if isinstance(dtype, pl.Struct):
        # Described as its fields: each profiled as a column of its own over
        # the structs that are present, so a field's null rate is how often
        # it is null inside one. The dtype is rebuilt from the fields', since
        # profiling may narrow a String field to an Enum as it would a column.
        fields = {
            field: nested(non_null.struct.field(field), field, len(non_null))
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
            list_length=_extent(non_null.list.len(), int, options.calculate_bounds)
            if isinstance(dtype, pl.List)
            else None,
        )

    if isinstance(dtype, (pl.List, pl.Array)):
        # Described as its elements: profile the exploded values as a column
        # of the inner dtype, then put the list's own nullability and length
        # back on top. An element that is a struct brings its fields along.
        exploded = non_null.explode(empty_as_null=False)
        element_nulls = exploded.null_count() / len(exploded) if len(exploded) else 0.0
        elements = nested(exploded.drop_nulls(), name, 0)
        list_length = None
        if isinstance(dtype, pl.List):
            list_length = _extent(non_null.list.len(), int, options.calculate_bounds)
        return dataclasses.replace(
            elements,
            dtype=pl.List(elements.dtype)
            if isinstance(dtype, pl.List)
            else pl.Array(elements.dtype, dtype.size),
            nullable=nullable,
            null_probability=null_probability,
            list_length=list_length,
            element_null_probability=element_nulls,
        )

    if dtype in (pl.String, pl.Utf8) or is_categorical_dtype(dtype):
        described = _profile_textual(
            non_null,
            name,
            dtype,
            spec,
            with_weights=with_weights,
            max_unique_enum=options.max_unique_enum if imitate else 0,
            calculate_bounds=options.calculate_bounds,
        )
        return _as_key(described, non_null, options, top)

    if dtype == pl.Boolean:
        return spec(
            dtype=pl.Boolean,
            weights=_boolean_weights(non_null) if with_weights else None,
        )

    if dtype.is_integer():
        described = spec(
            dtype=dtype, bounds=_extent(non_null, int, options.calculate_bounds)
        )
        keyed = _as_key(described, non_null, options, top)
        return keyed if keyed.unique else _shaped(keyed, non_null, options)

    if dtype.is_float():
        described = spec(
            dtype=dtype, bounds=_extent(non_null, float, options.calculate_bounds)
        )
        return _shaped(described, non_null, options)

    if dtype.is_decimal():
        described = spec(
            dtype=dtype,
            bounds=_extent(non_null, lambda v: v, options.calculate_bounds),
        )
        return _shaped(described, non_null, options)

    if dtype.is_temporal():
        # Temporal bounds are recorded as the physical integer the dtype
        # stores, matching what the generation path expects back.
        values = non_null.to_physical() if len(non_null) else non_null
        described = spec(
            dtype=dtype, bounds=_extent(values, int, options.calculate_bounds)
        )
        return _shaped(described, non_null, options)

    if dtype == pl.Binary:
        return spec(
            dtype=pl.Binary,
            string_length=_extent(
                non_null.bin.size() if len(non_null) else non_null,
                int,
                options.calculate_bounds,
            ),
        )

    # A dtype with nothing to measure (a `Null` column, an `Object`): recorded
    # faithfully so validation still reads it.
    return spec(dtype=dtype)


def _as_key(
    column: ColSpec, non_null: pl.Series, options: _Options, top: bool
) -> ColSpec:
    """`column` declared unique, when asked to look for keys and every one
    of enough values is distinct. Only an integer or a text column: a float
    or a timestamp is all distinct by nature, without being a key."""
    if (
        not options.detect_unique
        or not top
        or len(non_null) < KEY_MIN_ROWS
        or non_null.n_unique() != len(non_null)
        or column.weights is not None
    ):
        return column
    return dataclasses.replace(column, unique=True)


def _shaped(column: ColSpec, non_null: pl.Series, options: _Options) -> ColSpec:
    """`column` declaring the distribution its values follow, when asked to
    fit one and one describes them better than uniform does."""
    if not options.shape or column.bounds is None:
        return column
    fitted = fit(physical(non_null), column.bounds, column.dtype, seed=options.seed)
    return with_fit(column, fitted) if fitted is not None else column


def _profile_textual(
    non_null: pl.Series,
    name: str,
    dtype: pl.DataType,
    spec: Callable[..., ColSpec],
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

    if is_categorical_dtype(dtype):
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
    return True, DEFAULT_NULL_PROBABILITY


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
