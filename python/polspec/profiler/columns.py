"""Each column described by its dtype's family: its nulls, extent,
values, weights, shape and whether it is a key.

Internal: not part of the public API.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl

from polspec import frames
from polspec._text import infer as infer_text
from polspec.bound import Bound
from polspec.constants import DEFAULT_NULL_PROBABILITY
from polspec.dtypes import field_dtypes, map_entries
from polspec.profiler.formats import (
    FORMAT_MIN_DISTINCT,
    FORMAT_PROBE_ROWS,
    named_format,
)
from polspec.profiler.options import ProfileOptions
from polspec.profiler.rounding import rounded_bounds, rounded_rate, rounded_weights
from polspec.profiler.shapes import SHAPE_KEY_ROOM, shape_pattern
from polspec.shape import fit, physical, with_fit
from polspec.spec import ColSpec, is_categorical_dtype

# The fewest distinct values that make an all-distinct column a key. Ten
# rows of distinct values are ten rows, not evidence of a key.
KEY_MIN_ROWS = 100


# A text column narrows to an `Enum` of its values only when they repeat --
# each, on average, at least this many times. Forty distinct names in forty
# rows are names, not categories, and an `Enum` of them rejects the next
# file's. Below `max_unique_enum` distinct values *and* repeating: both.
ENUM_MIN_REPEATS = 2


@dataclass(frozen=True, slots=True)
class _Column:
    """One column being profiled, and what every dtype family reads of it.

    `imitate` is False for a column in `replace=`, whose values are not to be
    carried over; `top` is False for a struct's field or a list's element,
    which cannot be declared unique. `infinite` collects each float column
    holding an infinity outside the bounds its finite values gave it, as
    (path, infinities), to be said once, after; a NaN needs no word, since
    `nan_probability` records it.
    """

    series: pl.Series
    name: str
    total_rows: int
    options: ProfileOptions
    infinite: list[tuple[str, int]]
    imitate: bool
    top: bool
    non_null: pl.Series
    nullable: bool
    null_probability: float

    @property
    def dtype(self) -> pl.DataType:
        return self.series.dtype

    @property
    def with_weights(self) -> bool:
        return self.options.weights and self.imitate

    def spec(self, **kwargs: Any) -> ColSpec:
        """A declaration carrying this column's nullability."""
        return ColSpec(
            nullable=self.nullable, null_probability=self.null_probability, **kwargs
        )

    def nested(self, values: pl.Series, label: str, rows: int) -> ColSpec:
        """`values` -- a field's, an element's -- profiled as a column."""
        return profile_column(
            values,
            label,
            total_rows=rows,
            options=self.options,
            infinite=self.infinite,
            imitate=self.imitate,
        )


def profile_column(
    series: pl.Series,
    name: str,
    *,
    total_rows: int,
    options: ProfileOptions,
    infinite: list[tuple[str, int]],
    imitate: bool = True,
    top: bool = False,
) -> ColSpec:
    """Describes one column: its nullability, its domain, and its extent --
    by the first of `_FAMILIES` its dtype belongs to."""
    series = frames.plain(series)  # `name` carries the real name
    if series.dtype == pl.Null:
        # Every value null, however many rows -- an empty frame included.
        return ColSpec(dtype=pl.Null, nullable=True)
    nullable, null_probability = _nullability(series, total_rows)
    column = _Column(
        series=series,
        name=name,
        total_rows=total_rows,
        options=options,
        infinite=infinite,
        imitate=imitate,
        top=top,
        non_null=series.drop_nulls(),
        nullable=nullable,
        null_probability=(
            rounded_rate(null_probability, options.round_rates)
            if nullable
            else null_probability
        ),
    )
    for belongs, describe in _FAMILIES:
        if belongs(column.dtype):
            described = describe(column)
            if described.weights is None or options.round_rates is None:
                return described
            # Frequencies are written at the precision rates are.
            return dataclasses.replace(
                described,
                _weights=rounded_weights(described.weights, options.round_rates),
            )
    # A dtype with nothing to measure (an `Object`): recorded faithfully so
    # validation still reads it.
    return column.spec(dtype=column.dtype)


def _enum(c: _Column) -> ColSpec:
    dtype = c.dtype
    assert isinstance(dtype, pl.Enum)  # noqa: S101 - the family says so
    categories = dtype.categories.to_list()
    return c.spec(
        dtype=c.dtype,
        _weights=_empirical_weights(c.non_null, categories) if c.with_weights else None,
    )


def _map(c: _Column) -> ColSpec:
    # Described as the list of entries it is -- its length, its key and
    # value as the fields of a struct -- then made a map again from the
    # key and value dtypes profiling settled on, which may narrow a String
    # key to an Enum as it would a column.
    entries = map_entries(c.dtype)
    assert entries is not None  # noqa: S101 - the family says so
    as_list = c.nested(c.series.cast(entries), c.name, c.total_rows)
    parts = as_list.fields or {}
    if set(parts) != {"key", "value"}:
        return c.spec(dtype=c.dtype)
    return dataclasses.replace(
        as_list,
        dtype=pl.Map(parts["key"].dtype, parts["value"].dtype),
        element_null_probability=0.0,
    )


def _struct(c: _Column) -> ColSpec:
    # Described as its fields: each profiled as a column of its own over the
    # structs that are present, so a field's null rate is how often it is
    # null inside one. The dtype is rebuilt from the fields', since
    # profiling may narrow a String field to an Enum as it would a column.
    fields = {
        field: c.nested(
            c.non_null.struct.field(field), f"{c.name}.{field}", len(c.non_null)
        )
        for field in field_dtypes(_as_struct(c.dtype))
    }
    return c.spec(
        dtype=pl.Struct({field: fs.dtype for field, fs in fields.items()}),
        fields=fields or None,
    )


def _list_of_lists(c: _Column) -> ColSpec:
    # Nothing a declaration can say describes the inner values, so only
    # the outer list's length is recorded.
    return c.spec(
        dtype=c.dtype,
        list_length=_extent(c.non_null.list.len(), int, c.options.calculate_bounds)
        if isinstance(c.dtype, pl.List)
        else None,
    )


def _list(c: _Column) -> ColSpec:
    # Described as its elements: profile the exploded values as a column of
    # the inner dtype, then put the list's own nullability and length back
    # on top. An element that is a struct brings its fields along.
    dtype = c.dtype
    exploded = c.non_null.explode(empty_as_null=False)
    element_nulls = exploded.null_count() / len(exploded) if len(exploded) else 0.0
    elements = c.nested(exploded.drop_nulls(), c.name, 0)
    list_length = None
    if isinstance(dtype, pl.List):
        list_length = _extent(c.non_null.list.len(), int, c.options.calculate_bounds)
    return dataclasses.replace(
        elements,
        dtype=pl.List(elements.dtype)
        if isinstance(dtype, pl.List)
        else pl.Array(elements.dtype, _as_array(dtype).size),
        nullable=c.nullable,
        null_probability=c.null_probability,
        list_length=list_length,
        element_null_probability=rounded_rate(element_nulls, c.options.round_rates),
    )


def _text(c: _Column) -> ColSpec:
    # Counted once, for the text profile and the key check both: hashing a
    # million strings is most of what profiling one costs.
    distinct = c.non_null.n_unique()
    described = _profile_textual(c, distinct)
    return _as_key(described, c.non_null, c.options, c.top, distinct=distinct)


def _boolean(c: _Column) -> ColSpec:
    return c.spec(
        dtype=pl.Boolean,
        _weights=_boolean_weights(c.non_null) if c.with_weights else None,
    )


def _integer(c: _Column) -> ColSpec:
    extent = _extent(c.non_null, int, c.options.calculate_bounds)
    described = c.spec(
        dtype=c.dtype, bounds=rounded_bounds(extent, c.dtype, c.options.round_bounds)
    )
    keyed = _as_key(described, c.non_null, c.options, c.top)
    return keyed if keyed.unique else _shaped(keyed, c.non_null, c.options)


def _float(c: _Column) -> ColSpec:
    # A NaN is recorded as the share of present values it is; an infinity is
    # no bound -- a spec refuses a non-finite one -- so the extent and the
    # shape are the finite values'.
    non_null = c.non_null
    nans = int(non_null.is_nan().sum())
    finite = non_null.filter(non_null.is_finite())
    described = c.spec(
        dtype=c.dtype,
        bounds=rounded_bounds(
            _extent(finite, float, c.options.calculate_bounds),
            c.dtype,
            c.options.round_bounds,
        ),
        nan_probability=rounded_rate(
            nans / len(non_null) if nans else 0.0, c.options.round_rates
        ),
    )
    infinities = len(non_null) - len(finite) - nans
    if described.bounds is not None and infinities:
        c.infinite.append((c.name, infinities))
    return _shaped(described, finite, c.options)


def _decimal(c: _Column) -> ColSpec:
    described = c.spec(
        dtype=c.dtype,
        bounds=rounded_bounds(
            _extent(c.non_null, lambda v: v, c.options.calculate_bounds),
            c.dtype,
            c.options.round_bounds,
        ),
    )
    return _shaped(described, c.non_null, c.options)


def _temporal(c: _Column) -> ColSpec:
    described = c.spec(
        dtype=c.dtype,
        bounds=_temporal_extent(
            c.non_null,
            c.dtype,
            c.options.calculate_bounds,
            whole_seconds=c.options.round_bounds == "auto",
        ),
    )
    return _shaped(described, c.non_null, c.options)


def _binary(c: _Column) -> ColSpec:
    non_null = c.non_null
    return c.spec(
        dtype=pl.Binary,
        string_length=_extent(
            non_null.bin.size() if len(non_null) else non_null,
            int,
            c.options.calculate_bounds,
        ),
    )


def _as_struct(dtype: pl.DataType) -> pl.Struct:
    assert isinstance(dtype, pl.Struct)  # noqa: S101 - the family says so
    return dtype


def _as_array(dtype: pl.DataType) -> pl.Array:
    assert isinstance(dtype, pl.Array)  # noqa: S101 - the family says so
    return dtype


def _nested_list(dtype: pl.DataType) -> bool:
    return isinstance(dtype, (pl.List, pl.Array)) and isinstance(
        dtype.inner, (pl.List, pl.Array)
    )


# Each dtype family and how it is described, in the order a dtype is tested:
# a map before a list (it is one underneath), a list of lists before a list.
_FAMILIES: tuple[
    tuple[Callable[[pl.DataType], bool], Callable[[_Column], ColSpec]], ...
] = (
    (lambda d: isinstance(d, pl.Enum), _enum),
    (lambda d: map_entries(d) is not None, _map),
    (lambda d: isinstance(d, pl.Struct), _struct),
    (_nested_list, _list_of_lists),
    (lambda d: isinstance(d, (pl.List, pl.Array)), _list),
    (lambda d: d in (pl.String, pl.Utf8) or is_categorical_dtype(d), _text),
    (lambda d: d == pl.Boolean, _boolean),
    (lambda d: d.is_integer(), _integer),
    (lambda d: d.is_float(), _float),
    (lambda d: d.is_decimal(), _decimal),
    (lambda d: d.is_temporal(), _temporal),
    (lambda d: d == pl.Binary, _binary),
)


def _as_key(
    column: ColSpec,
    non_null: pl.Series,
    options: ProfileOptions,
    top: bool,
    *,
    distinct: int | None = None,
) -> ColSpec:
    """`column` declared unique, when asked to look for keys and every one
    of enough values is distinct. Only an integer or a text column: a float
    or a timestamp is all distinct by nature, without being a key.
    `distinct` is the values' distinct count, when the caller has it."""
    if (
        not options.detect_unique
        or not top
        or len(non_null) < KEY_MIN_ROWS
        or column.weights is not None
    ):
        return column
    if distinct is None:
        if _repeats_early(non_null):
            return column
        distinct = non_null.n_unique()
    if distinct != len(non_null):
        return column
    return dataclasses.replace(column, unique=True)


# Values a check for repeats reads before the whole column: most columns
# that are not keys repeat a value long before the end, and say so here.
REPEAT_PROBE_ROWS = 4_096


def _repeats_early(values: pl.Series) -> bool:
    """Whether the first values already hold a repeat -- which settles that
    `values` are not all distinct without hashing all of them."""
    head = values.head(REPEAT_PROBE_ROWS)
    return head.n_unique() != len(head)


def _shaped(column: ColSpec, non_null: pl.Series, options: ProfileOptions) -> ColSpec:
    """`column` declaring the distribution its values follow, when asked to
    fit one and one describes them better than uniform does."""
    if not options.shape or column.bounds is None:
        return column
    fitted = fit(physical(non_null), column.bounds, column.dtype, seed=options.seed)
    return with_fit(column, fitted) if fitted is not None else column


def _profile_textual(c: _Column, distinct: int) -> ColSpec:
    """A String or Categorical column, narrowed to an Enum when it holds few
    enough distinct values and they repeat; `distinct` is the values'
    distinct count. A column in `replace=` never narrows: its values are not
    to be carried over.

    A template format (an email, a UUID) is named before that: its values are
    identifiers, and an Enum of them would carry the source's own. A finite
    format (a country, a currency) only where the column would *not* narrow:
    an Enum of the codes it holds keeps their frequencies, which the format
    -- every code, evenly -- would throw away.
    """
    non_null, dtype, spec, options = c.non_null, c.dtype, c.spec, c.options
    keep_values = c.imitate
    n_unique = distinct
    max_unique_enum = options.max_unique_enum if keep_values else 0
    narrows = 0 < n_unique <= max_unique_enum and (
        n_unique * ENUM_MIN_REPEATS <= len(non_null)
    )

    if (
        options.formats
        and dtype in (pl.String, pl.Utf8)
        and n_unique >= FORMAT_MIN_DISTINCT
    ):
        named = named_format(
            non_null, keep_values=keep_values, finite=not narrows, distinct=n_unique
        )
        if named is not None:
            fmt, extras = named
            return spec(dtype=pl.String, format=fmt, extra_values=extras or None)

    if narrows:
        categories = non_null.unique().sort().to_list()
        return spec(
            dtype=pl.Enum(categories),
            _weights=_empirical_weights(non_null, categories)
            if c.with_weights
            else None,
        )

    if is_categorical_dtype(dtype):
        return spec(dtype=pl.Categorical())

    shaped = None
    if (
        options.patterns
        and keep_values
        and n_unique >= FORMAT_MIN_DISTINCT
        # Numbers, dates or booleans written as text are no code: their
        # description says what they parse as.
        and infer_text(non_null.head(FORMAT_PROBE_ROWS)) is None
    ):
        found = shape_pattern(non_null)
        # Values all distinct may be a key, which a larger stand-in needs
        # more of: its shape must hold SHAPE_KEY_ROOM times as many.
        if found is not None and (
            n_unique < len(non_null) or found[1] >= SHAPE_KEY_ROOM * n_unique
        ):
            shaped = found[0]
    return spec(
        dtype=pl.String,
        string_length=_extent(
            non_null.str.len_chars() if len(non_null) else non_null,
            int,
            options.calculate_bounds,
        ),
        pattern=shaped,
    )


def _nullability(series: pl.Series, total_rows: int) -> tuple[bool, float]:
    """Whether the column holds nulls, and how often."""
    null_count = series.null_count()
    if null_count == 0:
        # Not nullable, so no rate: the default is what a spec written by
        # hand would leave, and a file then says nothing about nulls.
        return False, DEFAULT_NULL_PROBABILITY
    if total_rows > 0:
        return True, float(null_count / total_rows)
    return True, DEFAULT_NULL_PROBABILITY


def _temporal_extent(
    values: pl.Series,
    dtype: pl.DataType,
    calculate_bounds: bool,
    *,
    whole_seconds: bool = False,
) -> Bound | None:
    """The observed extremes of a date, time, datetime or duration column, as
    a person writes them: a `date`, a `time`, a `datetime` -- in the
    column's time zone -- or a `timedelta`.

    Python's types stop at the microsecond, so a nanosecond column's
    extremes round outward to one: the minimum down, the maximum up, and
    every value still inside. Where Python cannot hold the result -- a year
    outside 1 to 9999, a time that would round up to 24:00 -- the bounds
    stay the physical integers the dtype stores, which generate the same.

    `whole_seconds` -- `round_bounds="auto"` -- rounds a datetime, time or
    duration outward to the second first, where that can be held.
    """
    if not calculate_bounds or len(values) == 0:
        return None
    physical_values = values.to_physical()
    extremes = physical_values.to_frame("v").select(
        pl.col("v").min().alias("lo"), pl.col("v").max().alias("hi")
    )
    observed = tuple(int(v) for v in extremes.row(0))
    lo, hi = observed
    unit = "ns" if dtype == pl.Time else getattr(dtype, "time_unit", None)
    per_unit = {"ns": 1, "us": 1_000, "ms": 1_000_000}.get(unit or "", 0)
    candidates = []
    if whole_seconds and per_unit:
        second = 1_000_000_000 // per_unit
        candidates.append((lo // second * second, -(-hi // second) * second))
    if unit == "ns":
        candidates.append((lo // 1000 * 1000, -(-hi // 1000) * 1000))
    else:
        candidates.append((lo, hi))
    for low_physical, high_physical in candidates:
        try:
            low, high = (
                pl.Series([low_physical, high_physical])
                .cast(dtype, strict=True)
                .to_list()
            )
        except (pl.exceptions.PolarsError, OverflowError, ValueError):
            continue
        if low is not None and high is not None:
            return Bound(low, high)
    return Bound(*observed)


def _extent(values: pl.Series, cast, calculate_bounds: bool) -> Bound | None:
    """The observed [min, max] of `values`, or None when not being measured."""
    if not calculate_bounds or len(values) == 0:
        return None
    return Bound(cast(values.min()), cast(values.max()))


def _empirical_weights(
    non_null: pl.Series, categories: list
) -> tuple[float, ...] | None:
    """How often each category actually occurs, in `categories` order.

    Categories absent from the data get a weight of 0, so a round-trip through
    `generate()` reproduces the observed mix rather than a uniform one.
    """
    if len(non_null) == 0:
        return None
    counts = frames.value_counts(non_null)
    observed = dict(
        zip(
            counts[frames.PLAIN].to_list(),
            counts[frames.COUNT].to_list(),
            strict=True,
        )
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
