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
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext
from types import MappingProxyType
from typing import Any, Literal, TypedDict

import polars as pl

from polspec import frames
from polspec._options import options_from
from polspec._text import infer as infer_text
from polspec._text import stays_text
from polspec.bound import Bound
from polspec.constants import DEFAULT_NULL_PROBABILITY
from polspec.dtypes import MAP, dtype_value_limits, field_dtypes, map_entries, map_parts
from polspec.shape import fit, physical, with_fit
from polspec.spec import ColSpec, is_categorical_dtype
from polspec.string_formats import lookup as _lookup_format
from polspec.tablespec import TableSpec

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
    "uuid7",
    "ulid",
    "ipv6",
    "ipv4",
    "mac",
    "url",
    "iso_country",
    "iso_country3",
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
    patterns : bool, default True
        Write the `pattern` a code-like text column's values share -- an
        order reference, a SKU, a postcode -- so its fake values have that
        shape: each run of digits, capitals or small letters a class of the
        lengths seen (`ORD-004211`, `ORD-918200` give `^ORD-[0-9]{6}$`), and
        a part every value shares kept as written. Only for a column of at
        least 20 distinct values that no format fits and that would not be
        an `Enum`, when every value has one of at most three shapes; for one
        of distinct values, which may be a key, only when the pattern holds
        ten times as many. Not for a column in `replace`, whose shared parts
        are real text, nor for numbers, dates or booleans written as text.
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
    parse_text : bool, default False
        Read a text column as the type its values hold, for a frame whose
        types were switched off (`pl.read_csv(..., infer_schema=False)`):
        `Boolean`, `Int64`, `Float64`, `Date` or `Datetime`, the first that
        every present value parses as exactly. Text is stripped and empty
        text is a null. A column with a value written with a leading zero
        (`007`) stays text, being a code, as does one in `replace`. Left
        False, such a column stays text and its description says what it
        would parse as. Validate the same frame with `parse_text=True` too.


    Examples
    --------
    >>> house = ProfileOptions(round_bounds="auto", replace=["name"])
    >>> frame = pl.DataFrame({"name": ["Ada", "Grace"] * 50, "total": [3.17, 9871.23] * 50})
    >>> profile(frame, options=house)["total"].bounds
    Bound(min=0.0, max=9900.0)
    """

    replace: Sequence[str] = ()
    sample: int | None = None
    seed: int = 0
    max_unique_enum: int = 50
    formats: bool = True
    patterns: bool = True
    weights: bool = True
    shape: bool = True
    detect_unique: bool = True
    calculate_bounds: bool = True
    detect_order: bool = True
    round_bounds: Rounding = None
    round_rates: Rounding = None
    parse_text: bool = False

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
    patterns: bool
    weights: bool
    shape: bool
    detect_unique: bool
    calculate_bounds: bool
    detect_order: bool
    round_bounds: Rounding
    round_rates: Rounding
    parse_text: bool


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

    # A shape is fitted between bounds, so without bounds there is none.
    options = dataclasses.replace(opts, shape=opts.shape and opts.calculate_bounds)

    def profiled(name: str) -> tuple[ColSpec, list[tuple[str, int]]]:
        # Each column notes its infinities in a list of its own, so columns
        # profiled at once share nothing, and the notes come out in column
        # order whichever finished first.
        infinite: list[tuple[str, int]] = []
        column = _profile_column(
            df[name],
            name,
            total_rows=df.height,
            options=options,
            infinite=infinite,
            imitate=name not in replaced,
            top=True,
        )
        return column, infinite

    results = _for_each_column(profiled, df)
    columns = {
        name: column for name, (column, _) in zip(df.columns, results, strict=True)
    }
    infinite = [note for _, notes in results for note in notes]
    for where, infinities in infinite:
        warnings.warn(
            f"Column '{where}' holds {infinities:,} infinite value(s), which "
            "its bounds -- its finite values' -- leave out, so validating this "
            "data against the spec reports them. Open the bound on that side "
            "(None) if infinities belong there.",
            UserWarning,
            stacklevel=2,
        )
    return columns


# Rows a frame needs before its columns are profiled at once: below it, a
# thread costs more than the column it would profile.
PARALLEL_MIN_ROWS = 50_000


def _for_each_column(profile_one: Callable[[str], Any], df: pl.DataFrame) -> list[Any]:
    """`profile_one` over every column of `df`, in column order: on a pool
    of threads for a frame large enough to gain, since Polars releases the
    GIL while it works, and one after another otherwise."""
    names = df.columns
    if len(names) < 2 or df.height < PARALLEL_MIN_ROWS:
        return [profile_one(name) for name in names]
    with ThreadPoolExecutor(max_workers=min(len(names), os.cpu_count() or 1)) as pool:
        return list(pool.map(profile_one, names))


# The start of the description a text column is given when it would parse:
# what `polspec schema infer` looks for to say so on the terminal too.
TEXT_NOTE = "Every value is text that parses as "

# The start of the description of a text column that nearly parses as a
# type, and stays text: what stopped it follows.
STAYS_TEXT_NOTE = "Stays text: "


def text_note(dtype: pl.DataType, text_format: str | None = None) -> str:
    """The description of a text column that would parse as `dtype`,
    written as `text_format` says when it is not the form Polars writes."""
    written = f" ('{text_format}')" if text_format is not None else ""
    return (
        f"{TEXT_NOTE}{type(dtype).__name__}{written}; profile with "
        "parse_text=True (--parse-text) to declare it as one."
    )


@dataclass(frozen=True, slots=True)
class _ReadText:
    """What `typed_text` made of a frame's text: the frame, each column it
    read and the format that read it, each it could have read, and why each
    that nearly parsed stays text."""

    frame: pl.DataFrame
    formats: dict[str, str]
    would: dict[str, tuple[pl.DataType, str | None]]
    stays: dict[str, str]


def typed_text(
    frame: pl.DataFrame, opts: ProfileOptions, *, explain: bool = True
) -> _ReadText:
    """`frame` with each text column read as the type its values hold,
    under `parse_text`, and the format each was read with; without it, the
    type each would be read as. Under `parse_text`, why each column that
    nearly parsed stays text, unless not `explain`.

    A column in `replace` is left as text either way: its values are not
    carried over, and a number's extremes would be.
    """
    parsed: list[pl.Series] = []
    formats: dict[str, str] = {}
    would: dict[str, tuple[pl.DataType, str | None]] = {}
    stays: dict[str, str] = {}
    for name, dtype in frame.schema.items():
        if dtype != pl.String or name in opts.replace:
            continue
        read = infer_text(frame[name])
        if read is None:
            reason = stays_text(frame[name]) if opts.parse_text and explain else None
            if reason is not None:
                stays[name] = reason
            continue
        if opts.parse_text:
            parsed.append(read.values)
            if read.text_format is not None:
                formats[name] = read.text_format
        else:
            would[name] = (read.values.dtype, read.text_format)
    typed = frame.with_columns(parsed) if parsed else frame
    return _ReadText(typed, formats, would, stays)


def profiled_table(frame: pl.DataFrame, opts: ProfileOptions, name: str) -> TableSpec:
    """`frame` described as a `TableSpec`: its text read as `opts` says,
    each column profiled, and the order its rows come in."""
    read = typed_text(frame, opts)
    frame = read.frame
    columns = profile_columns(frame, opts)
    for column, fmt in read.formats.items():
        columns[column] = dataclasses.replace(columns[column], text_format=fmt)
    for column, (dtype, fmt) in read.would.items():
        columns[column] = dataclasses.replace(
            columns[column], description=text_note(dtype, fmt)
        )
    for column, reason in read.stays.items():
        columns[column] = dataclasses.replace(
            columns[column], description=f"{STAYS_TEXT_NOTE}{reason}."
        )
    order = detect_order(frame) if opts.detect_order else None
    order_by, descending = order if order is not None else ((), False)
    return TableSpec(name, columns, order_by=order_by, descending=descending)


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
        return _profile_column(
            values,
            label,
            total_rows=rows,
            options=self.options,
            infinite=self.infinite,
            imitate=self.imitate,
        )


def _profile_column(
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
    # Counted once, for the text profile and the key check both: hashing a
    # million strings is most of what profiling one costs.
    distinct = c.non_null.n_unique()
    described = _profile_textual(c, distinct)
    return _as_key(described, c.non_null, c.options, c.top, distinct=distinct)


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
        named = _named_format(
            non_null, keep_values=keep_values, finite=not narrows, distinct=n_unique
        )
        if named is not None:
            fmt, extras = named
            return spec(dtype=pl.String, format=fmt, extra_values=extras or None)

    if narrows:
        categories = non_null.unique().sort().to_list()
        return spec(
            dtype=pl.Enum(categories),
            weights=_empirical_weights(non_null, categories)
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


# The shapes a code-like column's values may have between them, each an
# alternative of its pattern. More is free text, or a column of mixed codes
# a pattern would only list.
SHAPE_MAX = 3

# The longest run of other characters -- a separator, a prefix -- a shape
# keeps. A longer one is prose.
SHAPE_MAX_LITERAL = 8

# A part of a shape every value of it shares is written as it is when at
# least this many values share it, or when one value has the shape alone (a
# stand-in, `N/A`). A part fewer share is chance, and written as its class.
SHAPE_MIN_SHARED = 10

# How many times its own count of values a key's shape must hold: room for a
# stand-in up to ten times the size.
SHAPE_KEY_ROOM = 10

# Values read for their shapes before any more are.
SHAPE_PROBE_ROWS = 1_000

# How many characters each class holds.
_CLASS_SIZES = {"[0-9]": 10, "[A-Z]": 26, "[a-z]": 26}

# Each character class, and the mark a run of it leaves in a value's shape:
# a control character, so no value's own text is read as one.
_CLASSES = (("[0-9]", "\x01"), ("[A-Z]", "\x02"), ("[a-z]", "\x03"))
_MARKS = r"[\x00-\x03]"

# Characters a regular expression reads as syntax outside a class.
_REGEX_SYNTAX = frozenset("\\.+*?()|[]{}^$")


def shape_pattern(non_null: pl.Series) -> tuple[str, int] | None:
    """The pattern `non_null`'s values share, when every one has one of at
    most `SHAPE_MAX` shapes. A shape is a value's runs of digits, capitals,
    small letters and other characters. Each run of a class is written as
    the class with the lengths seen, or as it is when enough values share it
    (`SHAPE_MIN_SHARED`); each run of other characters as it is. None when
    the values have more shapes than that, or a run of other characters is
    longer than `SHAPE_MAX_LITERAL`. Beside the pattern, how many values it
    holds. Classes, literals and one alternation are always within the
    generator's reach.

    The shapes are read from the first values, and the pattern drawn from
    them; only when a later value fails it are the lengths read from every
    value. A value of no shape the first showed means more shapes.
    """
    # Most columns that are not codes show more shapes in their first values.
    if _shapes(non_null.head(SHAPE_PROBE_ROWS)).n_unique() > SHAPE_MAX:
        return None
    head = non_null.head(FORMAT_PROBE_ROWS)
    if head.str.contains(_MARKS).any():
        return None  # a value holding a mark would be misread
    shapes = _shapes(head).unique().sort()
    if len(shapes) > SHAPE_MAX:
        return None  # most columns that are not codes, at once
    runs = [_shape_runs(shape) for shape in shapes]
    if any(r is None for r in runs):
        return None  # prose, not a code
    shaped = [r for r in runs if r is not None]
    drawn = _shape_drawn(shaped, head.unique())
    if drawn is not None and (
        len(head) == len(non_null) or non_null.str.contains(drawn[0]).all()
    ):
        return drawn
    if non_null.str.contains(_MARKS).any():
        return None
    return _shape_drawn(shaped, non_null.unique(), every=True)


def _shape_drawn(
    shapes: list[list[str]], values: pl.Series, *, every: bool = False
) -> tuple[str, int] | None:
    """The pattern of `values`, each of one of `shapes` (its runs), and how
    many values it holds. With `every`, None when a value has none of them."""
    alternatives = []
    for runs in shapes:
        matcher = (
            "^" + "".join(f"({r}+)" if r in _CLASS_SIZES else r for r in runs) + "$"
        )
        groups = values.str.extract_groups(matcher).struct.unnest().drop_nulls()
        alternatives.append((runs, groups))
    if every and sum(groups.height for _, groups in alternatives) != len(values):
        return None  # a shape the first values did not show
    # The commonest shape first; the shapes come sorted, which a tie keeps.
    alternatives.sort(key=lambda a: -a[1].height)
    written, holds = [], 0
    for runs, groups in alternatives:
        part, held = _shape_alternative(runs, groups)
        written.append(part)
        holds += held
    body = written[0] if len(written) == 1 else f"(?:{'|'.join(written)})"
    return f"^{body}$", holds


def _shapes(values: pl.Series) -> pl.Series:
    """Each value with every run of a class replaced by that class's mark."""
    for cls, mark in _CLASSES:
        values = values.str.replace_all(f"{cls}+", mark)
    return values


def _shape_runs(shape: str) -> list[str] | None:
    """`shape`'s runs: each class, and each run of other characters escaped
    as a regular expression; None when one of those is prose."""
    classes = {mark: cls for cls, mark in _CLASSES}
    runs: list[str] = []
    literal = ""
    for c in [*shape, None]:
        if c is not None and c not in classes:
            literal += c
            continue
        if len(literal) > SHAPE_MAX_LITERAL:
            return None
        if literal:
            runs.append("".join(f"\\{x}" if x in _REGEX_SYNTAX else x for x in literal))
            literal = ""
        if c is not None:
            runs.append(classes[c])
    return runs


def _shape_alternative(runs: list[str], groups: pl.DataFrame) -> tuple[str, int]:
    """The pattern of the values of one shape, from its `runs` and each
    class run's values (`groups`, a column per run in order), and how many
    values it holds."""
    shared_by_enough = groups.height == 1 or groups.height >= SHAPE_MIN_SHARED
    captured = iter(groups.columns)
    parts, holds = [], 1
    for cls in runs:
        if cls not in _CLASS_SIZES:
            parts.append(cls)  # other characters, as they are
            continue
        run = groups[next(captured)]
        lengths = run.str.len_chars()
        low, high = int(lengths.min()), int(lengths.max())  # ty: ignore[invalid-argument-type]
        first = run[0]
        if shared_by_enough and high <= SHAPE_MAX_LITERAL and (run == first).all():
            parts.append(first)  # letters or digits: nothing to escape
            continue
        holds *= sum(_CLASS_SIZES[cls] ** n for n in range(low, high + 1))
        if low == high == 1:
            parts.append(cls)
        else:
            parts.append(f"{cls}{{{low}}}" if low == high else f"{cls}{{{low},{high}}}")
    return "".join(parts), holds


# Rows the format probe reads its distinct values from.
FORMAT_PROBE_ROWS = 20_000


def _mostly_dotted(non_null: pl.Series, total: int) -> bool:
    """Whether enough rows hold a dot for a hostname to be named at all: it
    is named only when its dotted values cover `FORMAT_MIN_SHARE` of the
    rows. Read without hashing a value, since a column of single words
    passes the hostname check and would otherwise be counted whole."""
    dotted = int(non_null.str.contains(".", literal=True).sum())
    return dotted >= FORMAT_MIN_SHARE * total


def _named_format(
    non_null: pl.Series, *, keep_values: bool, finite: bool = True, distinct: int
) -> tuple[str, dict[str, float]] | None:
    """The format `non_null`'s values have, and the stand-ins beside it with
    their shares; None when no format fits.

    Asked of the distinct values, weighted by how often each occurs, with
    the check validation uses -- so a spec that names a format accepts the
    data it was named from. A near miss is kept only with `keep_values`:
    its stand-ins are values of the source. A finite format is tried only
    with `finite`. `distinct` is how many distinct values `non_null` holds.
    """
    value, count = frames.PLAIN, frames.COUNT
    total = len(non_null)
    # A format tolerates at most FORMAT_MAX_EXTRAS distinct misses -- none,
    # without `keep_values` -- so a probe of any thousand distinct values
    # holding more rules it out exactly, before free text is counted whole.
    # The probe is the distinct values the column starts with, so free text
    # is ruled out without counting a single value of it.
    allowed_misses = FORMAT_MAX_EXTRAS if keep_values else 0
    probe = (
        non_null.head(FORMAT_PROBE_ROWS)
        .unique(maintain_order=True)
        .head(1_000)
        .alias(value)
        .to_frame()
    )
    counts: pl.DataFrame | None = None
    for name in FORMAT_ORDER:
        fmt = _lookup_format(name)
        if fmt.is_finite and not finite:
            continue
        # What the format accepts, and what counts as evidence of it. They
        # differ for a hostname: a single word passes the check -- so a
        # column of first names would be hostnames -- and so does a decimal,
        # `0.45` being host `0` in domain `45`. Evidence is a dotted value
        # whose last label is not all digits, as no real hostname's is (RFC
        # 1123). A value that passes is no stand-in, though: the format
        # accepts it, and `extra_values` may hold only what it does not.
        fits = fmt.check(frames.column(value)).fill_null(False)
        evidence = fits
        if name == "hostname":
            text = frames.column(value)
            evidence = (
                fits
                & text.str.contains(".", literal=True)
                & ~text.str.contains(r"\.[0-9]+$")
            )
        misses = probe.select((~fits).sum()).item()
        if misses > allowed_misses:
            continue
        if (
            misses == 0
            and name != "hostname"
            and distinct >= FORMAT_MIN_DISTINCT
            and non_null.alias(value).to_frame().select(fits.all()).item()
        ):
            # Every value fits: the column is the format, with no stand-in,
            # which is what counting every distinct value would conclude.
            # One pass over the column, not a count of it. A hostname needs
            # evidence beyond fitting, so it is counted.
            return name, {}
        if name == "hostname" and not _mostly_dotted(non_null, total):
            continue  # single words: evidence of no format, however many
        if counts is None:
            counts = frames.value_counts(non_null)
        judged = counts.with_columns(
            fits.alias("__polspec_fits"), evidence.alias("__polspec_evidence")
        )
        passing = judged.filter(pl.col("__polspec_evidence"))
        if passing.height < FORMAT_MIN_DISTINCT:
            continue
        passing_rows = int(passing[count].sum())
        # Most frequent first, then by value: `value_counts` keeps no order,
        # and a spec file should not change between runs on the same data.
        failing = judged.filter(~pl.col("__polspec_fits")).sort(
            [count, value], descending=[True, False]
        )
        if failing.height == 0 and passing_rows >= FORMAT_MIN_SHARE * total:
            return name, {}
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
