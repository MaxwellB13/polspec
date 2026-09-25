"""Reading a data file in the terms a spec declares.

A file someone hands you is read loosely -- Polars' own inference, nothing
forced -- and then only one thing is done in the spec's name: a column it
declares as a date or time that arrived as text is parsed as what it
declares. A CSV or JSON file has no date type, so without that a date column
is `String`, validation reports its dtype and checks nothing else about it.
Everything else is left for `inspect()` to report and `validate(cast=True)`
to type.

The command line reads through the same table, so what `polspec validate`
accepts and what `read()` accepts are one list.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec.tablespec import as_table_spec

if TYPE_CHECKING:
    import os

    from polspec.tablespec import TableSpec


def _read_tsv(source: Any, **options: Any) -> pl.DataFrame:
    options.setdefault("separator", "\t")  # a caller's own separator wins
    return pl.read_csv(source, **options)


# One reader per suffix, Polars' options passed straight through.
_READERS: dict[str, Callable[..., pl.DataFrame]] = {
    ".csv": pl.read_csv,
    ".tsv": _read_tsv,
    ".parquet": pl.read_parquet,
    ".pq": pl.read_parquet,
    ".ndjson": pl.read_ndjson,
    ".jsonl": pl.read_ndjson,
    ".json": pl.read_json,
    ".arrow": pl.read_ipc,
    ".ipc": pl.read_ipc,
    ".feather": pl.read_ipc,
}

# The text formats a CSV reader can be asked to recognise dates in.
_DELIMITED = frozenset({".csv", ".tsv"})


def _reader(path: Path) -> Callable[..., pl.DataFrame]:
    """The reader for `path`'s extension, or a `ValueError` naming the ones
    polspec knows."""
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(
            f"don't know how to read {path.suffix!r} files ({path}). "
            f"Supported: {', '.join(sorted(_READERS))}"
        )
    return reader


def read(
    spec: TableSpec | type, path: str | os.PathLike[str], **reader_options: Any
) -> pl.DataFrame:
    """A data file as a frame, in the terms `spec` declares -- read, not
    validated.

    Parameters
    ----------
    spec : TableSpec | FrameSpec class
        The declaration the file should meet.
    path : str | PathLike
        The file. Its extension picks the reader: `.csv`, `.tsv`,
        `.parquet`/`.pq`, `.ndjson`/`.jsonl`, `.json` or
        `.arrow`/`.ipc`/`.feather` -- the set `polspec validate` reads.
    **reader_options
        Passed to the Polars reader, e.g. `separator=";"` for a CSV.

    Returns
    -------
    pl.DataFrame
        The file as Polars reads it, with one change: each column the spec
        declares as a `Date`, `Datetime` or `Time` that arrived as text is
        parsed as what it declares, when every value in it parses. A column
        holding a value that does not stays text, so validation reports its
        dtype rather than a null that was never in the file. Nothing else is
        cast -- pass the frame to `inspect()` for every problem at once, or to
        `validate(cast=True)` for the typed frame.

    Raises
    ------
    ValueError
        For an extension polspec does not know.
    """
    path = Path(path)
    frame = _reader(path)(path, **reader_options)
    return _parse_declared_temporals(frame, as_table_spec(spec))


def _parse_declared_temporals(df: pl.DataFrame, spec: TableSpec) -> pl.DataFrame:
    """Each column `spec` declares as a `Date`, `Datetime` or `Time` that
    arrived as text, parsed as what it declares -- when every value parses.

    Only declared columns are touched, so a `String` column of date-shaped
    text stays text. A column holding a value that does not parse is left as
    it was read: validation then reports the dtype, which is true, rather
    than a null that was never in the file.
    """
    parsed = []
    for name, column in spec.columns.items():
        if name not in df.columns or df.schema[name] != pl.String:
            continue
        values = _parse_temporal(df[name], column.dtype)
        if values is not None and values.null_count() == df[name].null_count():
            parsed.append(values)
    return df.with_columns(parsed) if parsed else df


def _parse_temporal(text: pl.Series, dtype: pl.DataType) -> pl.Series | None:
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
    return None
