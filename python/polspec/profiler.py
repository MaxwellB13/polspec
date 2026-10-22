"""Inferring a spec from data that already exists.

The inverse of generation: given a DataFrame, describe the columns well enough
that `FrameSpec.generate` could produce something like it again.

By default a column is described by its dtype, its null rate, its extremes
and -- for a low-cardinality text column whose values repeat -- its
categories. Three opt-in steps describe more, for when the spec is going to
generate a stand-in for the data (`polspec.synthesize`):

- `shape=True` fits the distribution a numeric or temporal column's values
  follow (`polspec.shape`), so the stand-in is not uniform where the data
  is skewed;
- `detect_unique=True` declares an all-distinct integer or text column
  unique, so a key stays a key;
- `replace=` names columns whose *values* must not be carried into the
  spec: a text column among them is never narrowed to an `Enum` of the
  values it holds, and carries no weights.

Internal: not part of the public API.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext
from types import MappingProxyType
from typing import Any, Literal, TypedDict

import polars as pl

from polspec import frames
from polspec._deprecation import deprecated
from polspec._options import options_from
from polspec.bound import Bound
from polspec.constants import DEFAULT_NULL_PROBABILITY
from polspec.dtypes import MAP, dtype_value_limits, field_dtypes, map_entries, map_parts
from polspec.formats import lookup as _lookup_format
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

# A text column is named a `format` only when at least this many distinct
# values have it: three rows of "a@b.co" are a short Enum, not evidence.
FORMAT_MIN_DISTINCT = 20

# The formats a column is tried against, most specific first; the first
# that fits wins. A finite format (a country, a currency) is tried only where
# the column would not narrow to an Enum of the codes it holds, which keeps
# their frequencies -- see `_profile_textual`.
FORMAT_ORDER = (
    "uuid4",
    "ipv6",
    "ipv4",
    "mac",
    "iso_country",
    "iso_currency",
    "email",
    "hostname",
)

# A near miss: a column where nearly every value has the format and the rest
# are a few repeated stand-ins -- `N/A`, `UK (ISO)` -- is that format with
# those values as `extra_values`, at the share each is observed at.
FORMAT_MIN_SHARE = 0.99
FORMAT_MAX_EXTRAS = 5


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


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileOptions:
    """What `profile()` infers from data, and how finely it writes it.

    `profile`, `FrameSpec.from_dataframe` and `synthesize` take these as an
    `options=` object or as the keywords themselves -- one or the other, as
    `validate` takes `ValidationOptions`. `FrameSpec.from_dataframe`
    records less by default: `weights`, `shape`, `detect_unique` and
    `detect_order` off.

    Parameters
    ----------
    replace : sequence of str, default ()
        Columns whose values must not be carried into the spec. A text
        column named here is described by its lengths, never as an `Enum`
        of the values it holds, and carries no weights.
    sample : int, optional
        Profile a random sample of this many rows rather than every row.
    seed : int, default 0
        Seeds the sample and the shape fitting, so a profile is the same
        every run.
    max_unique_enum : int, default 50
        A text column with at most this many distinct values, repeating --
        and not in `replace` -- becomes an `Enum` of them.
    formats : bool, default True
        Name the `format` a text column's values have -- an email, a UUID,
        an IP, a country code -- so its fake values have it too: when at
        least 20 distinct values have one of `email`, `uuid4`, `ipv4`,
        `ipv6`, `mac`, `hostname`, `iso_country` or `iso_currency`, and
        every value does -- or 99% do, and the rest are at most five
        repeated stand-ins, which become its `extra_values`.
    weights : bool, default True
        Record each category's observed frequency, for categorical, enum and
        boolean columns.
    shape : bool, default True
        Fit the distribution each numeric or temporal column follows, so it
        generates that shape rather than uniformly between its extremes.
        Needs `calculate_bounds`.
    detect_unique : bool, default True
        Declare an integer or text column `unique=True` when every one of at
        least a hundred values is distinct.
    calculate_bounds : bool, default True
        Record the observed extremes of numeric and temporal columns, and the
        length range of text and binary ones.
    detect_order : bool, default True
        Declare `order_by` when the rows come sorted by a column of single
        values -- either way, nulls last, with at least 20 distinct values,
        so not by chance -- so a stand-in comes sorted too. The first such
        column, in column order. A sample keeps the source's row order, so
        a sorted file's sample shows it.
    round_bounds : number, "auto" or None, default None
        Write each numeric column's bounds rounded outward to this step --
        `1000`, `0.01` -- the minimum down and the maximum up, within what
        its dtype holds, so the data stays inside them. "auto" keeps two
        significant figures of the column's span (`[3.17, 9871.23]` becomes
        `[0, 9900]`) and rounds a datetime, time or duration outward to the
        second. None writes them exactly as observed.
    round_rates : number, "auto" or None, default None
        Write each null, NaN and list-element null rate rounded up to this
        step, never past 1; "auto" keeps two significant figures. A rate of
        0 stays 0. Category frequencies (`weights`) are written at the same
        precision, each to its nearest step and never to 0. None writes them
        exactly as observed.

    """

    replace: Sequence[str] = ()
    sample: int | None = None
    seed: int = 0
    max_unique_enum: int = 50
    formats: bool = True
    weights: bool = True
    shape: bool = True
    detect_unique: bool = True
    calculate_bounds: bool = True
    detect_order: bool = True
    round_bounds: Rounding = None
    round_rates: Rounding = None

    def __post_init__(self) -> None:
        replace: object = self.replace
        if isinstance(replace, str):
            raise TypeError(
                f"replace takes a sequence of column names, got the string {replace!r}"
            )
        object.__setattr__(self, "replace", tuple(self.replace))
        object.__setattr__(self, "round_bounds", step_of(self.round_bounds))
        object.__setattr__(self, "round_rates", step_of(self.round_rates))


class SynthesizeKeywords(TypedDict, total=False):
    """`ProfileOptions`' fields as `synthesize()` takes them: all but
    `seed`, which there seeds the stand-in as well as its profile."""

    replace: Sequence[str]
    sample: int | None
    max_unique_enum: int
    formats: bool
    weights: bool
    shape: bool
    detect_unique: bool
    calculate_bounds: bool
    detect_order: bool
    round_bounds: Rounding
    round_rates: Rounding


class ProfileKeywords(SynthesizeKeywords, total=False):
    """`ProfileOptions`' fields, as the keywords `profile()` takes them."""

    seed: int


def profile_options(
    options: ProfileOptions | None,
    keywords: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any] | None = None,
) -> ProfileOptions:
    """The options for one call: the object, or the keywords over
    `defaults` -- a caller whose defaults differ from `ProfileOptions`'."""
    if options is None and defaults:
        keywords = {**defaults, **keywords}
    return options_from(ProfileOptions, options, keywords, what="profiling")


# What `profile_columns` -- and so `FrameSpec.from_dataframe` -- records
# unless asked: the shape of the data, not a stand-in for it.
DESCRIBE_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {"weights": False, "shape": False, "detect_unique": False, "detect_order": False}
)


@dataclass(frozen=True, slots=True)
class _Options:
    """How much to infer, read by every column, however deeply nested."""

    weights: bool
    max_unique_enum: int
    calculate_bounds: bool
    shape: bool = False
    detect_unique: bool = False
    formats: bool = True
    seed: int = 0
    # Bounds rounded outward and rates up, to a step or "auto"; None leaves
    # them exactly as observed (`Rounding`).
    round_bounds: Rounding = None
    round_rates: Rounding = None
    # Each float column holding an infinity outside the bounds its finite
    # values gave it, as (path, infinities): said once, after. A NaN needs no
    # word -- the column's `nan_probability` records it.
    infinite: list[tuple[str, int]] = field(default_factory=list)


# Distinct values a column needs before its order is taken to be the
# frame's, and not chance: a short or repetitive column is often sorted by
# accident. The same bar a format clears.
ORDER_MIN_DISTINCT = 20

# Rows of the head order detection reads first, before ranking a column whole.
ORDER_PREFIX_ROWS = 2048


def detect_order(df: pl.DataFrame) -> tuple[tuple[str, ...], bool] | None:
    """The column `df`'s rows come sorted by, and whether high to low: the
    first, in column order, of single values -- at least
    `ORDER_MIN_DISTINCT` of them distinct -- that is sorted one way or the
    other, nulls last. None when no column is.

    Tested exactly as validation tests `order_by` (`frames.out_of_order`),
    so the order recorded is one the frame passes. One column only: a
    per-key order (`["user_id", "ts"]`) is the user's to declare.

    Internal: `profile` and `FrameSpec.from_dataframe` are the ways in.
    """
    candidates = [
        name
        for name, dtype in df.schema.items()
        if not dtype.is_nested() and map_parts(dtype) is None
    ]
    if not candidates or df.height < ORDER_MIN_DISTINCT:
        return None
    # The rows' head first: a prefix of a sorted column is sorted, so this
    # never turns one away, and nearly every column that is not sorted shows
    # it within a few rows -- the full test, a rank of every value both ways,
    # is left to the few that survive.
    head = _sorted_ways(df.head(ORDER_PREFIX_ROWS), candidates)
    survivors = [name for name in candidates if head[name]]
    if not survivors:
        return None
    if df.height > ORDER_PREFIX_ROWS:
        whole = _sorted_ways(df, survivors)
    else:
        whole = {name: head[name] for name in survivors}
    distinct = df.select(
        frames.column(name).drop_nulls().n_unique().alias(name) for name in survivors
    ).row(0, named=True)
    for name in survivors:
        if distinct[name] >= ORDER_MIN_DISTINCT and whole[name]:
            return (name,), whole[name][0]
    return None


def _sorted_ways(df: pl.DataFrame, names: Sequence[str]) -> dict[str, list[bool]]:
    """For each of `names`, the directions -- `False` ascending, `True`
    descending -- its rows come sorted in, ascending first."""
    counts = df.select(
        frames.out_of_order([name], [desc]).sum().alias(f"{i}:{desc}")
        for i, name in enumerate(names)
        for desc in (False, True)
    ).row(0, named=True)
    return {
        name: [desc for desc in (False, True) if counts[f"{i}:{desc}"] == 0]
        for i, name in enumerate(names)
    }


def profile_columns(
    df: pl.DataFrame,
    given: ProfileOptions | None = None,
    /,
    **keywords: Any,
) -> dict[str, ColSpec]:
    """Infers ColSpec column definitions by profiling an existing DataFrame.

    Internal: `polspec.profile` is the public way in, and returns these
    columns as a `TableSpec`. Its defaults record weights, shapes and
    uniqueness, which these leave off.

    `given` is a `ProfileOptions`, or pass its fields as keywords, over
    `DESCRIBE_DEFAULTS`: what a column is, not a stand-in for it.

    Warns
    -----
    UserWarning
        Once per float column holding an infinity. The bounds are its finite
        values', so the spec reports the infinities when the data is
        validated against it. A NaN is recorded as `nan_probability`.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame, got {type(df).__name__}")
    opts = profile_options(given, keywords, defaults=DESCRIBE_DEFAULTS)
    replace = opts.replace
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
        weights=opts.weights,
        max_unique_enum=opts.max_unique_enum,
        calculate_bounds=opts.calculate_bounds,
        shape=opts.shape and opts.calculate_bounds,
        detect_unique=opts.detect_unique,
        formats=opts.formats,
        seed=opts.seed,
        round_bounds=opts.round_bounds,
        round_rates=opts.round_rates,
    )
    columns = {
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
    for where, infinities in options.infinite:
        warnings.warn(
            f"Column '{where}' holds {infinities:,} infinite value(s), which "
            "its bounds -- its finite values' -- leave out, so validating this "
            "data against the spec reports them. Open the bound on that side "
            "(None) if infinities belong there.",
            UserWarning,
            stacklevel=2,
        )
    return columns


@dataclass(frozen=True, slots=True)
class _Column:
    """One column being profiled, and what every dtype family reads of it.

    `imitate` is False for a column in `replace=`, whose values are not to be
    carried over; `top` is False for a struct's field or a list's element,
    which cannot be declared unique.
    """

    series: pl.Series
    name: str
    total_rows: int
    options: _Options
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
        return _profile_column(
            values,
            label,
            total_rows=rows,
            options=self.options,
            imitate=self.imitate,
        )


def _profile_column(
    series: pl.Series,
    name: str,
    *,
    total_rows: int,
    options: _Options,
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
                weights=rounded_weights(described.weights, options.round_rates),
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
        weights=_empirical_weights(c.non_null, categories) if c.with_weights else None,
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
        dtype=MAP(parts["key"].dtype, parts["value"].dtype),
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
    described = _profile_textual(
        c.non_null,
        c.name,
        c.dtype,
        c.spec,
        with_weights=c.with_weights,
        max_unique_enum=c.options.max_unique_enum if c.imitate else 0,
        calculate_bounds=c.options.calculate_bounds,
        formats=c.options.formats,
        keep_values=c.imitate,
    )
    return _as_key(described, c.non_null, c.options, c.top)


def _boolean(c: _Column) -> ColSpec:
    return c.spec(
        dtype=pl.Boolean,
        weights=_boolean_weights(c.non_null) if c.with_weights else None,
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
        c.options.infinite.append((c.name, infinities))
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
    formats: bool = False,
    keep_values: bool = True,
) -> ColSpec:
    """A String or Categorical column, narrowed to an Enum when it holds few
    enough distinct values and they repeat.

    A template format (an email, a UUID) is named before that: its values are
    identifiers, and an Enum of them would carry the source's own. A finite
    format (a country, a currency) only where the column would *not* narrow:
    an Enum of the codes it holds keeps their frequencies, which the format
    -- every code, evenly -- would throw away.
    """
    n_unique = non_null.n_unique()
    narrows = 0 < n_unique <= max_unique_enum and (
        n_unique * ENUM_MIN_REPEATS <= len(non_null)
    )

    if formats and dtype in (pl.String, pl.Utf8) and n_unique >= FORMAT_MIN_DISTINCT:
        named = _named_format(non_null, keep_values=keep_values, finite=not narrows)
        if named is not None:
            fmt, extras = named
            return spec(dtype=pl.String, format=fmt, extra_values=extras or None)

    if narrows:
        categories = non_null.unique().sort().to_list()
        return spec(
            dtype=pl.Enum(categories),
            weights=_empirical_weights(non_null, categories) if with_weights else None,
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


def _named_format(
    non_null: pl.Series, *, keep_values: bool, finite: bool = True
) -> tuple[str, dict[str, float]] | None:
    """The format `non_null`'s values have, and the stand-ins beside it with
    their shares; None when no format fits.

    Asked of the distinct values, weighted by how often each occurs, with
    the check validation uses -- so a spec that names a format accepts the
    data it was named from. A near miss is kept only with `keep_values`:
    its stand-ins are values of the source. A finite format is tried only
    with `finite`.
    """
    counts = frames.value_counts(non_null)
    value, count = frames.PLAIN, frames.COUNT
    total = len(non_null)
    # A format tolerates at most FORMAT_MAX_EXTRAS distinct misses -- none,
    # without `keep_values` -- so a probe of any thousand distinct values
    # holding more rules it out exactly, before free text is checked whole.
    allowed_misses = FORMAT_MAX_EXTRAS if keep_values else 0
    probe = counts.head(1_000)
    for name in FORMAT_ORDER:
        fmt = _lookup_format(name)
        if fmt.is_finite and not finite:
            continue
        fits = fmt.check(frames.column(value))
        if name == "hostname":
            fits = fits & frames.column(value).str.contains(".", literal=True)
        misses = probe.select((~fits.fill_null(False)).sum()).item()
        if misses > allowed_misses:
            continue
        judged = counts.with_columns(fits.fill_null(False).alias("__polspec_fits"))
        passing = judged.filter(pl.col("__polspec_fits"))
        if passing.height < FORMAT_MIN_DISTINCT:
            continue
        passing_rows = int(passing[count].sum())
        if passing_rows == total:
            return name, {}
        # Most frequent first, then by value: `value_counts` keeps no order,
        # and a spec file should not change between runs on the same data.
        failing = judged.filter(~pl.col("__polspec_fits")).sort(
            [count, value], descending=[True, False]
        )
        if (
            keep_values
            and passing_rows >= FORMAT_MIN_SHARE * total
            and failing.height <= FORMAT_MAX_EXTRAS
            and bool((failing[count] >= 2).all())
        ):
            return name, {
                str(extra): int(seen) / total
                for extra, seen in zip(
                    failing[value].to_list(), failing[count].to_list(), strict=True
                )
            }
    return None


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


@deprecated(
    "polspec.profile_dataframe",
    use="polspec.profile(df, weights=False, shape=False, detect_unique=False).columns",
)
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
    formats: bool = True,
) -> dict[str, ColSpec]:
    """The columns `polspec.profile` would describe, as a dict, with every
    switch off by default. `profile` takes the same switches, on by default,
    and returns a `TableSpec`."""
    return profile_columns(
        df,
        weights=weights,
        max_unique_enum=max_unique_enum,
        calculate_bounds=calculate_bounds,
        shape=shape,
        detect_unique=detect_unique,
        replace=replace,
        seed=seed,
        formats=formats,
    )
