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

import polars as pl

# A probe of this many present values is parsed before the whole column, so
# free text is ruled out without reading all of it.
PROBE = 1_000

_INTEGER = r"^[+-]?[0-9]+$"
_LEADING_ZERO = r"^[+-]?0[0-9]"
_ISO_DATE = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
_ISO_DATETIME = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}"

Parser = Callable[[pl.Series], pl.Series | None]


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
    present = values.drop_nulls()
    if not _all(present, _INTEGER) or _any(present, _LEADING_ZERO):
        return None
    return values.cast(pl.Int64, strict=True)


def _float(values: pl.Series) -> pl.Series | None:
    present = values.drop_nulls()
    # Every value an integer: an Int64 would have held them, so these are
    # past what it holds -- a code, which a float would round.
    if _any(present, _LEADING_ZERO) or _all(present, _INTEGER):
        return None
    return values.cast(pl.Float64, strict=True)


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
