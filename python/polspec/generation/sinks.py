"""Streaming generated rows straight to a file.

Every sink generates in batches and writes each as it is produced, so the
whole frame never has to fit in memory. Parquet and Arrow IPC go through
PyArrow's incremental writers (the `arrow` extra); CSV and NDJSON append with
Polars alone.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from polspec.frames import Method, References
from polspec.tablespec import TableSpec, require_columns


@dataclass(frozen=True, slots=True)
class _Run:
    """One sink call, once its arguments have been checked.

    The four generation options are the whole of what a sink passes through,
    and each sink used to write the call out in full. They are still spelled
    out on every public signature -- that is what makes a typo in one a
    failure at the call site rather than a `TypeError` from inside the
    generator, several frames away -- but past `_prepare` there is one of
    these instead of four copies of the same six arguments.
    """

    spec: TableSpec
    path: Path
    n: int
    batch_size: int
    method: Method
    seed: int | None
    references: References

    def batches(self) -> Iterator[pl.DataFrame]:
        """The batch stream this sink writes."""
        from polspec.generation import generate_batches

        return generate_batches(
            self.spec,
            self.n,
            batch_size=self.batch_size,
            method=self.method,
            seed=self.seed,
            references=self.references,
        )

    def empty(self) -> pl.DataFrame | None:
        """A no-row frame for a zero-row call, so the file still carries a schema.

        None for any other call, so a sink that fails part-way through does
        not quietly leave an empty file behind.
        """
        from polspec.generation import generate

        if self.n != 0:
            return None
        return generate(self.spec, 0, references=self.references)


def _prepare(
    spec: TableSpec,
    path: str | Path,
    n: int,
    batch_size: int,
    method: Method,
    seed: int | None,
    references: References,
) -> _Run:
    """The argument checks and directory creation every sink repeats.

    Eager rather than folded into the batch generator, so an invalid call
    fails before any destination file is opened.
    """
    from polspec.generation import _requires_whole_frame

    require_columns(spec)
    _requires_whole_frame(spec, "a sink")
    if n < 0:
        raise ValueError("n must be >= 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return _Run(spec, target, n, batch_size, method, seed, references)


def _sink_arrow(
    batches: Iterator[pl.DataFrame],
    make_writer: Callable[[Any], Any],
    empty_frame: pl.DataFrame | None = None,
) -> None:
    """Streams `batches` through an Arrow writer opened from the first batch.

    Shared by the Parquet and IPC sinks, which differ only in how their
    writer is constructed. Both need a schema before they can open one, and
    both must still leave a valid, schema-bearing file behind when there are
    no rows -- hence `empty_frame`, which the caller supplies only for
    `n == 0` so a mid-stream failure does not quietly produce one.
    """
    writer = None
    try:
        for batch_df in batches:
            table = batch_df.to_arrow()
            if writer is None:
                writer = make_writer(table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
        elif empty_frame is not None:
            make_writer(empty_frame.to_arrow().schema).close()


def sink_parquet(
    spec: TableSpec,
    path: str | Path,
    n: int,
    *,
    batch_size: int = 100_000,
    compression: str = "zstd",
    method: Method = "random",
    seed: int | None = None,
    references: References = None,
    **kwargs: Any,
) -> None:
    """Generates `n` rows and streams them to a Parquet file in batches.

    Extra keyword arguments go to `pyarrow.parquet.ParquetWriter`.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            'pyarrow is required for sink_parquet(). Install it with "polspec[arrow]".'
        ) from exc

    run = _prepare(spec, path, n, batch_size, method, seed, references)
    _sink_arrow(
        run.batches(),
        lambda schema: pq.ParquetWriter(
            str(run.path), schema, compression=compression, **kwargs
        ),
        empty_frame=run.empty(),
    )


def sink_ipc(
    spec: TableSpec,
    path: str | Path,
    n: int,
    *,
    batch_size: int = 100_000,
    compression: str | None = "zstd",
    method: Method = "random",
    seed: int | None = None,
    references: References = None,
    **kwargs: Any,
) -> None:
    """Generates `n` rows and streams them to an Arrow IPC / Feather file in batches.

    Extra keyword arguments go to `pyarrow.ipc.new_file`.
    """
    try:
        from pyarrow import ipc
    except ImportError as exc:
        raise ImportError(
            'pyarrow is required for sink_ipc(). Install it with "polspec[arrow]".'
        ) from exc

    run = _prepare(spec, path, n, batch_size, method, seed, references)
    with open(run.path, "wb") as f:
        _sink_arrow(
            run.batches(),
            lambda schema: ipc.new_file(
                f,
                schema,
                options=ipc.IpcWriteOptions(compression=compression),
                **kwargs,
            ),
            empty_frame=run.empty(),
        )


def sink_csv(
    spec: TableSpec,
    path: str | Path,
    n: int,
    *,
    batch_size: int = 100_000,
    include_header: bool = True,
    method: Method = "random",
    seed: int | None = None,
    references: References = None,
    **kwargs: Any,
) -> None:
    """Generates `n` rows and streams them to a CSV file in batches.

    Extra keyword arguments go to `pl.DataFrame.write_csv`.
    """
    run = _prepare(spec, path, n, batch_size, method, seed, references)
    header_needed = include_header
    with open(run.path, "wb") as f:
        empty = run.empty()
        if empty is not None:
            if include_header:
                empty.write_csv(f, include_header=True, **kwargs)
            return
        for batch_df in run.batches():
            batch_df.write_csv(f, include_header=header_needed, **kwargs)
            header_needed = False


def sink_ndjson(
    spec: TableSpec,
    path: str | Path,
    n: int,
    *,
    batch_size: int = 100_000,
    method: Method = "random",
    seed: int | None = None,
    references: References = None,
    **kwargs: Any,
) -> None:
    """Generates `n` rows and streams them to a newline-delimited JSON file in batches.

    Extra keyword arguments go to `pl.DataFrame.write_ndjson`.
    """
    run = _prepare(spec, path, n, batch_size, method, seed, references)
    with open(run.path, "wb") as f:
        for batch_df in run.batches():
            batch_df.write_ndjson(f, **kwargs)
