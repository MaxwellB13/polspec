"""Data files: one reader and one writer per suffix, so what `generate`
writes, `validate` and `drift` read back; a suffix in one map and not
the other is a red test. And the notes and warnings a command prints
on the way.
"""

from __future__ import annotations

import contextlib
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polspec.errors import CliError, PolspecError
from polspec.frames import json_casts, sample_rows, text_format_casts
from polspec.profiler import STAYS_TEXT_NOTE, TEXT_NOTE
from polspec.reading import SUFFIXES, parse_declared_text, read_file

if TYPE_CHECKING:
    from polspec.tablespec import TableSpec


# Written through `json_casts`: categories as text, no bytes.
_JSON_SUFFIXES = frozenset({".ndjson", ".jsonl", ".json"})


_DATA_WRITERS = {
    ".csv": pl.DataFrame.write_csv,
    ".tsv": lambda df, p: df.write_csv(p, separator="\t"),
    ".parquet": pl.DataFrame.write_parquet,
    ".pq": pl.DataFrame.write_parquet,
    ".ndjson": pl.DataFrame.write_ndjson,
    ".jsonl": pl.DataFrame.write_ndjson,
    ".json": pl.DataFrame.write_json,
    ".arrow": pl.DataFrame.write_ipc,
    ".ipc": pl.DataFrame.write_ipc,
    ".feather": pl.DataFrame.write_ipc,
}


def _existing(path_text: str, *, what: str = "file") -> Path:
    path = Path(path_text)
    if not path.exists():
        raise CliError(f"no such {what}: {path}")
    return path


def _read_data_file(
    path: Path,
    sample: int | None,
    spec: TableSpec | None = None,
    *,
    infer_dates: bool = False,
) -> pl.DataFrame:
    """A data file as a frame, read in `spec`'s terms when there is one.

    With a spec, this is `polspec.read`. Without one, `infer_dates` has
    Polars recognise dates in a CSV itself -- `schema infer`'s case, where
    there is no declaration to go by.
    """
    if path.suffix.lower() not in SUFFIXES:
        raise CliError(
            f"don't know how to read {path.suffix!r} files ({path}). "
            f"Supported: {', '.join(sorted(SUFFIXES))}"
        )
    try:
        df = read_file(path, infer_dates=infer_dates)
    except Exception as exc:
        raise CliError(f"could not read {path}: {exc}") from exc
    try:
        df = sample_rows(df, sample)
    except (ValueError, PolspecError) as exc:
        raise CliError(str(exc)) from exc
    return parse_declared_text(df, spec) if spec is not None else df


# The suffixes whose files hold every value as text, where a column's
# `text_format` says how a date is written.
_TEXT_SUFFIXES = frozenset({".csv", ".tsv", *_JSON_SUFFIXES})


def _write_data_file(
    df: pl.DataFrame, path: Path, spec: TableSpec | None = None
) -> None:
    """`df` written to `path`, by its suffix; a text file writes `spec`'s
    dates, datetimes and times as their `text_format` says."""
    writer = _DATA_WRITERS.get(path.suffix.lower())
    if writer is None:
        raise CliError(
            f"don't know how to write {path.suffix!r} files ({path}). "
            f"Supported: {', '.join(sorted(_DATA_WRITERS))}"
        )
    if spec is not None and path.suffix.lower() in _TEXT_SUFFIXES:
        df = df.with_columns(text_format_casts(spec.columns))
    if path.suffix.lower() in _JSON_SUFFIXES:
        df = df.with_columns(json_casts(df.schema, error=CliError))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer(df, path)
    except Exception as exc:
        raise CliError(f"could not write {path}: {exc}") from exc


@contextlib.contextmanager
def _warnings_printed() -> Iterator[None]:
    """Prints each warning raised inside as a `warning:` line on stderr --
    a spec file that cannot hold a value, a column profiled around its NaNs
    -- where a Python warning would be a traceback-looking line, or an
    error under `-W error`."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for warning in caught:
        print(f"warning: {warning.message}", file=sys.stderr)


def _note_text(spec: TableSpec) -> None:
    """Says on stderr which text columns would parse as another type, which
    their descriptions say too, and the flag that reads them so; and, under
    --parse-text, why each column that nearly parsed stays text."""
    noted = [
        name
        for name, column in spec.columns.items()
        if (column.description or "").startswith(TEXT_NOTE)
    ]
    if noted:
        print(
            f"note: {len(noted)} text column(s) hold numbers, flags or dates "
            f"throughout: {', '.join(noted)}. --parse-text reads them as those "
            "types; their descriptions say which.",
            file=sys.stderr,
        )
    for name, column in spec.columns.items():
        description = column.description or ""
        if description.startswith(STAYS_TEXT_NOTE):
            reason = description.removeprefix(STAYS_TEXT_NOTE)
            print(f"note: column {name!r} stays text: {reason}", file=sys.stderr)
