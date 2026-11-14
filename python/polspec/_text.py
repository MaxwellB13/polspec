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

Text is stripped first, and empty text is a null. A value written with a
leading zero (`007`, `-01`) keeps its column text: it is a code, and the
zero would be lost. So does an integer too large for an Int64, rather than
become a float that cannot hold it. One parser for profiling and
validating, so the two cannot disagree about what parses.

Internal: not part of the public API.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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


def _parsed(parse: Parser, values: pl.Series) -> pl.Series | None:
    """`values` parsed, or None where any present one does not parse."""
    try:
        parsed = parse(values)
    except pl.exceptions.PolarsError:
        return None
    if parsed is None or parsed.null_count() != values.null_count():
        return None
    return parsed


def infer(text: pl.Series) -> pl.Series | None:
    """`text` as the first type every present value parses as, named as it
    was; None when there is no such type, or no present value."""
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
            return parsed.alias(text.name)
    return None


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


def _parse_date(text: pl.Expr, _dtype: pl.DataType) -> pl.Expr:
    return text.str.to_date(strict=False)


def _parse_time(text: pl.Expr, _dtype: pl.DataType) -> pl.Expr:
    return text.str.to_time(strict=False)


def _parse_datetime(text: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    declared = dtype if isinstance(dtype, pl.Datetime) else pl.Datetime()
    return text.str.to_datetime(
        time_unit=declared.time_unit, time_zone=declared.time_zone, strict=False
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


def text_parser(dtype: pl.DataType) -> Callable[[pl.Expr], pl.Expr] | None:
    """How text is read as `dtype`, leniently; None for a dtype text is
    read as itself, or holds nothing of."""
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
            return lambda text, parse=parse: parse(text, dtype)
    return None


def _series_temporal(text: pl.Series, dtype: pl.DataType) -> pl.Series | None:
    """`text` as `dtype`, with a value that does not parse as a null; None
    for a dtype that is not a date or time, or text Polars cannot read as
    one at all. Parsed as a Series rather than an expression, which is how
    Polars allows an offset-aware value to be read into a naive column."""
    try:
        if dtype == pl.Date:
            return text.str.to_date(strict=False)
        if dtype == pl.Time:
            return text.str.to_time(strict=False)
        if dtype == pl.Datetime:
            declared = dtype if isinstance(dtype, pl.Datetime) else pl.Datetime()
            return text.str.to_datetime(
                time_unit=declared.time_unit,
                time_zone=declared.time_zone,
                strict=False,
            )
    except pl.exceptions.PolarsError:
        return None


def parse_series(
    text: pl.Series, dtype: pl.DataType
) -> tuple[pl.Series | None, pl.Series]:
    """`text` read as `dtype`: the parsed values, or None unless every
    present one parses; and the present values that do not, as they were.

    Text is stripped and empty text is a null first. A date or time is read
    as a Series rather than an expression, which is how Polars lets an
    offset-aware value be read into a naive column.
    """
    values = blank_as_null(text)
    nothing = values.clear()
    parse = text_parser(dtype)
    if parse is None:
        return None, nothing
    parsed = _series_temporal(values, dtype)
    if parsed is None:
        try:
            parsed = (
                values.alias("text")
                .to_frame()
                .select(parse(pl.col("text")))
                .to_series()
            )
        except pl.exceptions.PolarsError:
            return None, values.drop_nulls()
    failed = values.filter(parsed.is_null() & values.is_not_null())
    if not failed.is_empty():
        return None, failed
    return parsed.alias(text.name), nothing
