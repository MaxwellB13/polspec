"""Text that holds another type: what it parses as, and the parsed values.

A frame read with its types switched off -- `pl.read_csv(...,
infer_schema=False)`, a spreadsheet exported as text -- holds its numbers,
flags and dates as strings. `infer` reads one such column back as the
first type every present value parses as exactly:

    Boolean   true / false, in any case
    Int64     integers, none written with a leading zero
    Float64   numbers, none written with a leading zero
    Date      ISO dates, 2024-01-31
    Datetime  ISO datetimes, 2024-01-31T09:30:00 or with a space
    then      dates, datetimes and times written another way, from
              `TEXT_FORMATS`: 30 Sep 2031, 30/09/2031, 30 Sep 2031 14:05,
              2:05 PM ...

Text is stripped first, and empty text is a null. A value written with a
leading zero (`007`, `-01`) keeps its column text: it is a code, and the
zero would be lost. So does an integer too large for an Int64, rather than
become a float that cannot hold it. One parser for profiling and
validating, so the two cannot disagree about what parses.

A format from `TEXT_FORMATS` fits only when every present value parses
with it and writes back as the text it came from, month names in any case;
so `30 Feb 2031` fits nothing. Two formats that fit and read the values
differently -- `01/02/2031`, day-first or month-first -- are no fit at
all: the column stays text, and `stays_text` says why, as it says what
kept a column of nearly numbers or nearly dates as text.

Internal: not part of the public API.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

# A probe of this many present values is parsed before the whole column, so
# free text is ruled out without reading all of it.
PROBE = 1_000

_INTEGER = r"^[+-]?[0-9]+$"
_LEADING_ZERO = r"^[+-]?0[0-9]"
_ISO_DATE = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
_ISO_DATETIME = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}"

Parser = Callable[[pl.Series], pl.Series | None]
ExprParser = Callable[[pl.Expr, Any], pl.Expr]

Temporal = Literal["date", "datetime", "time"]


def _dates() -> list[str]:
    """Dates as people write them, each with its day and month unpadded
    too: `1 Jan 2030` as well as `01 Jan 2030`."""
    written = [
        "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%b %d, %Y", "%B %d, %Y",
        "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d.%m.%Y",
    ]  # fmt: skip
    unpadded = [f.replace("%d", "%-d").replace("%m", "%-m") for f in written]
    return [f for pair in zip(written, unpadded, strict=True) for f in pair]


_TIMES = ["%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%-I:%M:%S %p", "%I:%M %p", "%-I:%M %p"]

# Every non-ISO way of writing a date, a datetime or a time that text is
# read as, in the order tried. ISO forms are `_date` and `_datetime`'s.
TEXT_FORMATS: tuple[tuple[Temporal, str], ...] = (
    *(("date", f) for f in _dates()),
    *(("datetime", f"{d} {t}") for d in ["%Y-%m-%d", *_dates()] for t in _TIMES),
    *(("time", t) for t in _TIMES),
)

# What a column must look like before any of them is tried: digits among
# letters, spaces and the separators dates use. Free text is ruled out here.
_LOOKS_TEMPORAL = r"^[0-9A-Za-z ,./:-]*[0-9][0-9A-Za-z ,./:-]*$"

# A column is "nearly" a type when at least this share of its probe parses
# as it: then `stays_text` names what kept it as text.
NEARLY = 0.9


@dataclass(frozen=True, slots=True)
class Parsed:
    """Text read as the type it holds: the values, and the `TEXT_FORMATS`
    entry that read them -- None for the forms Polars writes itself."""

    values: pl.Series
    text_format: str | None = None


def blank_as_null(text: pl.Series) -> pl.Series:
    """`text` stripped, with empty text a null."""
    stripped = text.str.strip_chars().alias("text")
    blank = pl.col("text") == ""
    return (
        stripped.to_frame()
        .select(pl.when(blank).then(None).otherwise(pl.col("text")))
        .to_series()
        .alias(text.name)
    )


def _all(present: pl.Series, pattern: str) -> bool:
    return bool(present.str.contains(pattern).all())


def _any(present: pl.Series, pattern: str) -> bool:
    return bool(present.str.contains(pattern).any())


def _boolean(values: pl.Series) -> pl.Series | None:
    lower = values.str.to_lowercase()
    if not lower.drop_nulls().is_in(["true", "false"]).all():
        return None
    return lower == "true"


def _integer(values: pl.Series) -> pl.Series | None:
    # Cast first: text that is not integers fails it, in one pass, before
    # any pattern is read.
    parsed = values.cast(pl.Int64, strict=False)
    if parsed.null_count() != values.null_count():
        return None
    present = values.drop_nulls()
    if not _all(present, _INTEGER) or _any(present, _LEADING_ZERO):
        return None
    return parsed


def _float(values: pl.Series) -> pl.Series | None:
    parsed = values.cast(pl.Float64, strict=False)
    if parsed.null_count() != values.null_count():
        return None
    present = values.drop_nulls()
    # Every value an integer: an Int64 would have held them, so these are
    # past what it holds -- a code, which a float would round.
    if _any(present, _LEADING_ZERO) or _all(present, _INTEGER):
        return None
    return parsed


def _date(values: pl.Series) -> pl.Series | None:
    if not _all(values.drop_nulls(), _ISO_DATE):
        return None
    return values.str.to_date("%Y-%m-%d", strict=True)


def _datetime(values: pl.Series) -> pl.Series | None:
    if not _all(values.drop_nulls(), _ISO_DATETIME):
        return None
    return values.str.to_datetime(strict=True)


_IN_ORDER: tuple[Parser, ...] = (_boolean, _integer, _float, _date, _datetime)


def _did_not_parse(exc: BaseException) -> bool:
    """Whether `exc` is a parse failing: Polars' own error, or -- on Polars
    1.39, for text a date format nearly fits (`30 Sep 2031` read with
    `%d %B %Y`) -- a panic, which pyo3 raises as a `BaseException`."""
    return isinstance(exc, pl.exceptions.PolarsError) or (
        type(exc).__name__ == "PanicException"
    )


def _parsed(parse: Parser, values: pl.Series) -> pl.Series | None:
    """`values` parsed, or None where any present one does not parse."""
    try:
        parsed = parse(values)
    except BaseException as exc:
        if not _did_not_parse(exc):
            raise
        return None
    if parsed is None or parsed.null_count() != values.null_count():
        return None
    return parsed


def infer(text: pl.Series) -> Parsed | None:
    """`text` as the first type every present value parses as, named as it
    was; None when there is no such type, or no present value, or two
    formats read the values differently."""
    values = blank_as_null(text)
    present = values.drop_nulls()
    if present.is_empty():
        return None
    probe = present.head(PROBE)
    for parse in _IN_ORDER:
        if _parsed(parse, probe) is None:
            continue
        parsed = _parsed(parse, values)
        if parsed is not None:
            return Parsed(parsed.alias(text.name))
    readings = _readings(values, probe)
    if len(readings) != 1:
        return None
    (fmt, parsed), *_ = readings.values()
    return Parsed(parsed.alias(text.name), fmt)


def _as_temporal(values: pl.Series, kind: Temporal, fmt: str) -> pl.Series:
    """`values` read leniently as `kind` with `fmt`: what does not parse is
    a null."""
    if kind == "date":
        return values.str.to_date(fmt, strict=False)
    if kind == "datetime":
        return values.str.to_datetime(fmt, strict=False)
    return values.str.to_time(fmt, strict=False)


def _lenient(values: pl.Series, kind: Temporal, fmt: str) -> pl.Series | None:
    """`_as_temporal`, or None where the parse fails outright."""
    try:
        return _as_temporal(values, kind, fmt)
    except BaseException as exc:
        if not _did_not_parse(exc):
            raise
        return None


def _written_as(values: pl.Series, kind: Temporal, fmt: str) -> pl.Series | None:
    """`values` read with `fmt`, when every present one parses and writes
    back as itself, a month name in any case; None otherwise."""
    parsed = _lenient(values, kind, fmt)
    if parsed is None or parsed.null_count() != values.null_count():
        return None
    back = parsed.dt.strftime(fmt).str.to_lowercase()
    if not (back == values.str.to_lowercase()).fill_null(True).all():
        return None
    return parsed


def _readings(
    values: pl.Series, probe: pl.Series
) -> dict[tuple[str, int], tuple[str, pl.Series]]:
    """Each distinct reading of `values` a `TEXT_FORMATS` entry gives, keyed
    by the values read, with the first entry to give it. One reading is the
    column's type; two are an ambiguity; none, no fit."""
    if not bool(probe.str.contains(_LOOKS_TEMPORAL).all()):
        return {}
    readings: dict[tuple[str, int], tuple[str, pl.Series]] = {}
    for kind, fmt in _shaped_like(probe, TEXT_FORMATS, share=1.0):
        if _written_as(probe, kind, fmt) is None:
            continue
        parsed = _written_as(values, kind, fmt)
        if parsed is None:
            continue
        key = (kind, int(parsed.hash(seed=0).sum()))
        readings.setdefault(key, (fmt, parsed))
    return readings


# A month as `%b` writes it, and as `%B` does, in any case.
_MONTHS = "(?i:" + "|".join(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"]
) + ")"  # fmt: skip
_MONTH_ABBREVIATIONS = "(?i:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"

# What each directive a format uses looks like in text, for a shape test
# cheap enough to run on every format before any is parsed.
_SHAPES = {
    "%Y": r"\d{4}", "%m": r"\d{2}", "%-m": r"\d{1,2}", "%d": r"\d{2}",
    "%-d": r"\d{1,2}", "%b": _MONTH_ABBREVIATIONS, "%B": _MONTHS,
    "%H": r"\d{2}", "%I": r"\d{2}", "%-I": r"\d{1,2}", "%M": r"\d{2}",
    "%S": r"\d{2}", "%p": r"(?:[AaPp][Mm])",
}  # fmt: skip

# Values of a column a shape is tested on: enough to rule most formats out.
_SHAPE_SAMPLE = 50


@functools.cache
def _shape(fmt: str) -> re.Pattern[str]:
    """`fmt` as a regular expression of the text it writes."""
    pieces = re.split(r"(%-?[A-Za-z])", fmt)
    return re.compile(
        "".join(_SHAPES.get(piece, re.escape(piece)) for piece in pieces if piece)
    )


def _shaped_like(
    probe: pl.Series, formats: Sequence[tuple[Temporal, str]], *, share: float
) -> list[tuple[Temporal, str]]:
    """The `formats` whose shape at least `share` of `probe`'s first values
    have: the only ones worth parsing with."""
    sample = probe.head(_SHAPE_SAMPLE).to_list()
    need = share * len(sample)
    return [
        (kind, fmt)
        for kind, fmt in formats
        if sum(1 for v in sample if _shape(fmt).fullmatch(v)) >= need
    ]


def stays_text(text: pl.Series) -> str | None:
    """Why a column `infer` could not read, and that nearly parses as a
    type, stays text, for its description: two date formats that both fit,
    the values that keep it from being dates or numbers, a code's leading
    zero, or a whole number past an Int64. None when it is not nearly
    anything.

    Nearly a date is most of the values parsing as one; nearly a number is
    most of them looking like one -- digits and the marks numbers are
    written with -- since a column of `1,500,000.00` parses not at all."""
    values = blank_as_null(text)
    present = values.drop_nulls()
    if present.is_empty():
        return None
    probe = present.head(PROBE)
    readings = _readings(values, probe)
    if len(readings) > 1:
        formats = " and ".join(f"'{fmt}'" for fmt, _ in readings.values())
        return (
            f"its values fit {formats}, which read them differently; declare "
            "the column's text_format to say which"
        )
    if _share(probe.str.contains(_LOOKS_TEMPORAL)) >= NEARLY:
        for kind, fmt in _shaped_like(probe, _NEARLY_TEMPORAL, share=NEARLY):
            parsed = _lenient(probe, kind, fmt)
            if parsed is None or _share(parsed.is_not_null()) < NEARLY:
                continue
            whole = _lenient(present, kind, fmt)
            if whole is not None:
                return _stray(present.filter(whole.is_null()), f"a {kind} ('{fmt}')")
    if _share(_looks_numeric(probe)) < NEARLY:
        return None
    stray = present.filter(present.cast(pl.Float64, strict=False).is_null())
    if stray.len():
        return _stray(stray, "a number")
    zeros = present.filter(present.str.contains(_LEADING_ZERO))
    if zeros.len():
        return (
            f"{zeros.len():,} value(s) are written with a leading zero, such "
            f"as {_examples(zeros)}, so the column is read as a code"
        )
    too_big = present.filter(present.cast(pl.Int64, strict=False).is_null())
    return (
        "every value is a whole number and some are too large for an Int64, "
        f"such as {_examples(too_big)}, so the column is read as a code"
    )


# The forms a column nearly of dates is measured against: ISO, then the rest.
_NEARLY_TEMPORAL: tuple[tuple[Temporal, str], ...] = (
    ("date", "%Y-%m-%d"),
    ("datetime", "%Y-%m-%d %H:%M:%S"),
    ("datetime", "%Y-%m-%dT%H:%M:%S"),
    *TEXT_FORMATS,
)

# The ways a number is written that a cast does not read: thousands grouped
# by commas or spaces, a decimal comma, and one number with a currency sign,
# brackets, a percent or a trailing minus. An address, `81.233.41.86`, is
# none of them.
_WRITTEN_NUMBERS = (
    r"^[-+(]?[£$€]?\d{1,3}([, ]\d{3})+(\.\d+)?\)?%?-?$",
    r"^[-+]?\d{1,3}(\.\d{3})*,\d+$",
    r"^[-+(]?[£$€]?\d+(\.\d+)?\)?%?-?$",
)


def _looks_numeric(values: pl.Series) -> pl.Series:
    """Which of `values` are numbers: ones a cast reads, and ones written
    in a way it does not (`_WRITTEN_NUMBERS`)."""
    looks = values.cast(pl.Float64, strict=False).is_not_null()
    for written in _WRITTEN_NUMBERS:
        looks = looks | values.str.contains(written)
    return looks


def _share(mask: pl.Series) -> float:
    """The share of `mask` that is true: 0 for no values."""
    share = mask.mean()
    return float(share) if isinstance(share, int | float) else 0.0


def _examples(values: pl.Series) -> str:
    return str(values.unique(maintain_order=True).head(3).to_list())


def _stray(stray: pl.Series, what: str) -> str:
    return (
        f"{stray.len():,} value(s) do not parse as {what}, such as {_examples(stray)}"
    )


# Reading text as a declared type: what `read()` and `validate(parse_text=True)`
# do with a column the spec declares as something text cannot hold. Each
# parser is lenient -- a value that does not parse is a null -- so a caller
# can say which values did not; a column is parsed only when none did.


def _parse_boolean(text: pl.Expr, _dtype: pl.DataType) -> pl.Expr:
    lower = text.str.to_lowercase()
    return pl.when(lower == "true").then(True).when(lower == "false").then(False)


def _parse_integer(text: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """Integers as an Int64 -- a narrower declared width then judges each
    value as it judges any wider integer standing in -- or as the declared
    dtype where an Int64 cannot hold its range."""
    target = dtype if dtype in _WIDER_THAN_INT64 else pl.Int64
    return pl.when(text.str.contains(_INTEGER)).then(text).cast(target, strict=False)


def parsed_dtype(dtype: pl.DataType) -> pl.DataType:
    """The dtype text declared as `dtype` is read to: an integer of any
    width as an Int64 unless only a wider one holds it, a float as a
    Float64, anything else as itself."""
    if dtype.is_integer():
        return dtype if dtype in _WIDER_THAN_INT64 else pl.Int64()
    if dtype.is_float():
        return pl.Float64()
    return dtype


def _parse_float(text: pl.Expr, _dtype: pl.DataType) -> pl.Expr:
    return text.cast(pl.Float64, strict=False)


_WIDER_THAN_INT64 = tuple(
    getattr(pl, name) for name in ("UInt64", "Int128", "UInt128") if hasattr(pl, name)
)


def _parse_date(text: pl.Expr, _dtype: pl.DataType, fmt: str | None = None) -> pl.Expr:
    return text.str.to_date(fmt, strict=False)


def _parse_time(text: pl.Expr, _dtype: pl.DataType, fmt: str | None = None) -> pl.Expr:
    return text.str.to_time(fmt, strict=False)


def _parse_datetime(
    text: pl.Expr, dtype: pl.DataType, fmt: str | None = None
) -> pl.Expr:
    declared = dtype if isinstance(dtype, pl.Datetime) else pl.Datetime()
    return text.str.to_datetime(
        fmt, time_unit=declared.time_unit, time_zone=declared.time_zone, strict=False
    )


def _parse_decimal(text: pl.Expr, dtype: Any) -> pl.Expr:
    """`text` as `dtype`, exactly: digits with no more decimal places than the
    scale, which is how Polars writes one. Anything else -- `1.255` for a
    scale of 2, which a cast would round -- is a null."""
    exact = text.str.contains(rf"^[+-]?\d+(\.\d{{0,{dtype.scale or 0}}})?$")
    return pl.when(exact).then(text).cast(dtype, strict=False)


# Digits of a second in each time unit.
_SUBSECOND_DIGITS = {"ms": 3, "us": 6, "ns": 9}


def _parse_duration(text: pl.Expr, dtype: Any) -> pl.Expr:
    """`text` as `dtype` from the ISO 8601 form Polars writes: a sign, whole
    seconds and a fraction (`-PT8612.164256S`), or `P0D` for zero. A finer
    fraction than the unit holds, any other form, or a value past the range
    of the unit is a null. Integer arithmetic throughout, so nothing is
    rounded through a float."""
    digits = _SUBSECOND_DIGITS[dtype.time_unit or "us"]
    pattern = rf"^(-?)P(?:0D|T(\d+)(?:\.(\d{{1,{digits}}}))?S)$"
    parts = text.str.extract_groups(pattern)
    whole = parts.struct.field("2").fill_null("0").cast(pl.Int128, strict=False)
    fraction = (
        parts.struct.field("3")
        .fill_null("")
        .str.pad_end(digits, "0")
        .cast(pl.Int128, strict=False)
    )
    magnitude = whole * 10**digits + fraction
    negative = pl.lit(0, dtype=pl.Int128) - magnitude  # Int128 has no negation
    signed = pl.when(parts.struct.field("1") == "-").then(negative).otherwise(magnitude)
    return (
        pl.when(text.str.contains(pattern))
        .then(signed)
        .cast(pl.Int64, strict=False)
        .cast(dtype)
    )


def text_parser(
    dtype: pl.DataType, text_format: str | None = None
) -> Callable[[pl.Expr], pl.Expr] | None:
    """How text is read as `dtype`, leniently; None for a dtype text is
    read as itself, or holds nothing of. A date, datetime or time is read
    with `text_format` when the column declares one, and as Polars infers
    otherwise."""
    parsers: tuple[tuple[Callable[[pl.DataType], bool], ExprParser], ...] = (
        (lambda d: d == pl.Boolean, _parse_boolean),
        (lambda d: d.is_integer(), _parse_integer),
        (lambda d: d.is_float(), _parse_float),
        (lambda d: d == pl.Date, _parse_date),
        (lambda d: d == pl.Time, _parse_time),
        (lambda d: isinstance(d, pl.Datetime) or d == pl.Datetime, _parse_datetime),
        (lambda d: isinstance(d, pl.Decimal), _parse_decimal),
        (lambda d: isinstance(d, pl.Duration), _parse_duration),
    )
    for matches, parse in parsers:
        if matches(dtype):
            if text_format is not None and parse in _TEMPORAL_PARSERS:
                return lambda text, parse=parse: parse(text, dtype, text_format)
            return lambda text, parse=parse: parse(text, dtype)
    return None


_TEMPORAL_PARSERS = (_parse_date, _parse_time, _parse_datetime)


def _series_temporal(
    text: pl.Series, dtype: pl.DataType, fmt: str | None = None
) -> pl.Series | None:
    """`text` as `dtype`, with a value that does not parse as a null; None
    for a dtype that is not a date or time, or text Polars cannot read as
    one at all. Parsed as a Series rather than an expression, which is how
    Polars allows an offset-aware value to be read into a naive column."""
    try:
        if dtype == pl.Date:
            return text.str.to_date(fmt, strict=False)
        if dtype == pl.Time:
            return text.str.to_time(fmt, strict=False)
        if dtype == pl.Datetime:
            declared = dtype if isinstance(dtype, pl.Datetime) else pl.Datetime()
            return text.str.to_datetime(
                fmt,
                time_unit=declared.time_unit,
                time_zone=declared.time_zone,
                strict=False,
            )
    except BaseException as exc:
        if not _did_not_parse(exc):
            raise
        return None


def parse_series(
    text: pl.Series, dtype: pl.DataType, text_format: str | None = None
) -> tuple[pl.Series | None, pl.Series]:
    """`text` read as `dtype`: the parsed values, or None unless every
    present one parses; and the present values that do not, as they were.
    A date, datetime or time is read with `text_format` when given.

    Text is stripped and empty text is a null first. A date or time is read
    as a Series rather than an expression, which is how Polars lets an
    offset-aware value be read into a naive column.
    """
    values = blank_as_null(text)
    nothing = values.clear()
    parse = text_parser(dtype, text_format)
    if parse is None:
        return None, nothing
    parsed = _series_temporal(values, dtype, text_format)
    if parsed is None:
        try:
            parsed = (
                values.alias("text")
                .to_frame()
                .select(parse(pl.col("text")))
                .to_series()
            )
        except BaseException as exc:
            if not _did_not_parse(exc):
                raise
            return None, values.drop_nulls()
    failed = values.filter(parsed.is_null() & values.is_not_null())
    if not failed.is_empty():
        return None, failed
    return parsed.alias(text.name), nothing
