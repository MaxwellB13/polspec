"""Reading a data file in the terms a spec declares.

A file someone hands you is read loosely -- Polars' own inference, nothing
forced -- and then only one thing is done in the spec's name: a column it
declares as a number, a flag, a date, a time, a decimal or a duration that
arrived as text is parsed as what it declares (`polspec._text`). A CSV or JSON file has no such types -- Polars
itself writes a `Decimal` to JSON as `"1.25"` and a `Duration` as
`"PT3600S"` -- so without that the column is `String`, validation reports its
dtype and checks nothing else about it. Everything else is left for
`inspect()` to report and `validate(cast=True)` to type.

The command line reads through the same table, so what `polspec validate`
accepts and what `read()` accepts are one list.

Internal: not part of the public API.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec import frames
from polspec._text import parse_series, parsed_dtype, text_parser
from polspec.dtypes import element_dtype
from polspec.tablespec import as_table_spec

if TYPE_CHECKING:
    import os

    from polspec.dtypes import DtypeLike
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

# The extensions `read`, `read_file` and the command line read.
SUFFIXES: tuple[str, ...] = tuple(_READERS)


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


def read_file(
    path: str | os.PathLike[str], *, infer_dates: bool = False, **reader_options: Any
) -> pl.DataFrame:
    """A data file as Polars reads it, with no spec to read it in the terms
    of: the reader chosen by extension, as `read` and the command line
    choose it. `infer_dates` has a CSV or TSV reader recognise dates itself,
    where there is no declaration to say which columns hold them.
    """
    file = Path(path)
    reader = _reader(file)
    if infer_dates and file.suffix.lower() in _DELIMITED:
        reader_options.setdefault("try_parse_dates", True)
    return reader(file, **reader_options)


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
        declares as a `Boolean`, integer, float, `Date`, `Datetime`, `Time`,
        `Decimal` or `Duration` that arrived as text -- itself, or as the
        elements of a list or the fields of a struct -- is parsed as what it
        declares, when every value in it parses exactly. A column of its own
        is stripped first, with empty text a null, so a file read with
        `infer_schema=False` comes back typed; an integer comes back as an
        `Int64`, and a float as a `Float64`. A `Decimal` is parsed from its digits, with no
        more decimal places than its scale; a `Duration` from the ISO 8601
        text Polars writes (`"PT3600S"`, `"-PT0.5S"`, `"P0D"`). A column
        holding a value that does not parse stays text, so validation reports
        its dtype rather than a null -- or a rounded value -- that was never
        in the file. Nothing else is cast -- pass the frame to `inspect()` for
        every problem at once, or to `validate(cast=True)` for the typed
        frame.

    Raises
    ------
    ValueError
        For an extension polspec does not know.

    Examples
    --------
    >>> import pathlib, tempfile
    >>> folder = pathlib.Path(tempfile.mkdtemp())
    >>> _ = (folder / "orders.csv").write_text("id,placed\\n1,2024-01-31\\n2,2024-02-01\\n")
    >>> spec = TableSpec("Orders", {"id": pl.Int64, "placed": pl.Date})
    >>> read(spec, folder / "orders.csv").schema  # the date parsed, as declared
    Schema({'id': Int64, 'placed': Date})
    """
    path = Path(path)
    frame = _reader(path)(path, **reader_options)
    return parse_declared_text(frame, as_table_spec(spec))


def parse_declared_text(df: pl.DataFrame, spec: TableSpec) -> pl.DataFrame:
    """Each column `spec` declares in a type the file could not hold, read as
    what it declares -- when every value parses exactly.

    Only declared columns are touched, so a `String` column of date-shaped
    text stays text. A column holding a value that does not parse is left as
    it was read: validation then reports the dtype, which is true, rather
    than a null that was never in the file.
    """
    parsed = []
    for name, column in spec.columns.items():
        if name not in df.columns:
            continue
        arrived = df.schema[name]
        if arrived == pl.String:
            values, _ = parse_series(df[name], column.dtype)
            if values is not None:
                parsed.append(values)
            continue
        values = _parse_nested(df, name, column.dtype, arrived)
        if values is not None and _nulls(frames.plain(values)) == _nulls(
            frames.plain(df[name])
        ):
            parsed.append(values)
    return df.with_columns(parsed) if parsed else df


def _parse_nested(
    df: pl.DataFrame, name: str, declared: pl.DataType, arrived: pl.DataType
) -> pl.Series | None:
    """Column `name` read as `declared` wherever the file held it otherwise,
    or None where there is nothing to read -- or it does not read."""
    plan = _arrival(frames.column(name), declared, arrived)
    if plan is None:
        return None
    try:
        return df.select(plan[0].alias(name)).to_series()
    except pl.exceptions.PolarsError:
        return None  # an array whose lists are not all its width


def _nulls(values: pl.Series) -> int:
    """Every null in `values`, at any depth: a value that did not parse is a
    null where the text held something, wherever in a list or struct it was."""
    nulls = values.null_count()
    if isinstance(values.dtype, pl.List):
        nulls += _nulls(values.explode(empty_as_null=False))
    elif isinstance(values.dtype, pl.Array):
        nulls += _nulls(values.arr.explode(empty_as_null=False))
    elif isinstance(values.dtype, pl.Struct):
        nulls += sum(_nulls(values.struct.field(f.name)) for f in values.dtype.fields)
    return nulls


def _arrival(
    expr: pl.Expr, declared: pl.DataType, arrived: pl.DataType
) -> tuple[pl.Expr, pl.DataType] | None:
    """`expr` read as `declared` where a file could only hold it as
    `arrived`, and the dtype that reads to; None where nothing needs reading.

    A JSON file holds a decimal, a duration -- and inside a list or struct, a
    date or time -- as text; a fixed-size array as a list; and a column with
    no value at all as nulls of no type. Each is read back as declared.
    Anything else stays as it arrived, for validation to judge.
    """
    if arrived == pl.Null:
        return expr.cast(declared), declared
    if arrived == pl.String:
        parse = text_parser(declared)
        if parse is None:
            return None
        return parse(expr), parsed_dtype(declared)
    if isinstance(declared, pl.List) and isinstance(arrived, pl.List):
        inner = _arrival(pl.element(), element_dtype(declared), element_dtype(arrived))
        if inner is None:
            return None
        return expr.list.eval(inner[0]), pl.List(inner[1])
    if isinstance(declared, pl.Array) and isinstance(arrived, pl.List):
        inner = _arrival(pl.element(), element_dtype(declared), element_dtype(arrived))
        elements = element_dtype(arrived) if inner is None else inner[1]
        values = expr if inner is None else expr.list.eval(inner[0])
        # The shape, not the element type: nothing else is cast here.
        shaped = pl.Array(elements, declared.size)
        # A null list stays a null array: Polars before 1.43 casts one to an
        # array of nulls.
        rebuilt = (
            pl.when(expr.is_null())
            .then(pl.lit(None, dtype=shaped))
            .otherwise(values.cast(shaped))
        )
        return rebuilt, shaped
    if isinstance(declared, pl.Struct) and isinstance(arrived, pl.Struct):
        declared_fields = {f.name: _instance(f.dtype) for f in declared.fields}
        fields, dtypes, changed = [], [], False
        for field in arrived.fields:
            inner = None
            if field.name in declared_fields:
                inner = _arrival(
                    expr.struct.field(field.name),
                    declared_fields[field.name],
                    _instance(field.dtype),
                )
            if inner is None:
                fields.append(expr.struct.field(field.name))
                dtypes.append(field)
            else:
                fields.append(inner[0].alias(field.name))
                dtypes.append(pl.Field(field.name, inner[1]))
                changed = True
        if not changed:
            return None
        return pl.struct(fields), pl.Struct(dtypes)
    return None


def _instance(dtype: DtypeLike) -> pl.DataType:
    """A struct field's dtype as an instance: polars types it as either."""
    return dtype() if isinstance(dtype, type) else dtype

    return None
