"""Streaming generated rows straight to a file.

Every sink generates in batches and writes each as it is produced, so the
whole frame never has to fit in memory. Parquet and Arrow IPC go through
PyArrow's incremental writers (the `arrow` extra); CSV and NDJSON append with
Polars alone.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

import polars as pl

from polspec.errors import SpecError
from polspec.tablespec import TableSpec

Method = Literal["random", "cartesian"]
References = Mapping[Any, pl.DataFrame | pl.LazyFrame] | None


def _batches(spec: TableSpec, n: int, **kwargs: Any) -> Iterator[pl.DataFrame]:
    from polspec.generation import generate_batches

    return generate_batches(spec, n, **kwargs)


def _empty(spec: TableSpec, references: References) -> pl.DataFrame:
    from polspec.generation import generate

    return generate(spec, 0, references=references)


def _prepare(spec: TableSpec, path: str | Path, n: int, batch_size: int) -> Path:
    """The argument checks and directory creation every sink repeats.

    Eager rather than folded into the batch generator, so an invalid call
    fails before any destination file is opened.
    """
    if not spec.columns:
        raise SpecError(f"{spec.name} declares no ColSpec columns")
    if n < 0:
        raise ValueError("n must be >= 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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

    path = _prepare(spec, path, n, batch_size)
    _sink_arrow(
        _batches(
            spec,
            n,
            batch_size=batch_size,
            method=method,
            seed=seed,
            references=references,
        ),
        lambda schema: pq.ParquetWriter(
            str(path), schema, compression=compression, **kwargs
        ),
        empty_frame=_empty(spec, references) if n == 0 else None,
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

    path = _prepare(spec, path, n, batch_size)
    with open(path, "wb") as f:
        _sink_arrow(
            _batches(
                spec,
                n,
                batch_size=batch_size,
                method=method,
                seed=seed,
                references=references,
            ),
            lambda schema: ipc.new_file(
                f,
                schema,
                options=ipc.IpcWriteOptions(compression=compression),
                **kwargs,
            ),
            empty_frame=_empty(spec, references) if n == 0 else None,
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
    path = _prepare(spec, path, n, batch_size)
    header_needed = include_header
    with open(path, "wb") as f:
        if n == 0:
            if include_header:
                _empty(spec, references).write_csv(f, include_header=True, **kwargs)
            return
        for batch_df in _batches(
            spec,
            n,
            batch_size=batch_size,
            method=method,
            seed=seed,
            references=references,
        ):
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
    path = _prepare(spec, path, n, batch_size)
    with open(path, "wb") as f:
        for batch_df in _batches(
            spec,
            n,
            batch_size=batch_size,
            method=method,
            seed=seed,
            references=references,
        ):
            batch_df.write_ndjson(f, **kwargs)
