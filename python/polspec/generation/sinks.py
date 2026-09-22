"""Streaming generated rows straight to a file.

Every sink is `scan(...)` handed to the matching `LazyFrame.sink_*`, so the
rows are generated as polars writes them and the whole frame never has to
fit in memory. What each sink adds over calling that itself is the argument
checking, the parent directory, and a signature that names the options
rather than taking them as `**kwargs` -- a typo fails at the call site
rather than inside a writer several frames away.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec.frames import Method, References
from polspec.tablespec import require_columns

if TYPE_CHECKING:
    from polars._typing import IpcCompression, ParquetCompression

    from polspec.tablespec import TableSpec


def _scan(
    spec: TableSpec,
    path: str | Path,
    n: int,
    batch_size: int,
    method: Method,
    seed: int | None,
    references: References,
) -> tuple[pl.LazyFrame, Path]:
    """The lazy frame a sink writes, and where it writes it.

    The checks are eager rather than left to the scan's own, so an invalid
    call fails before any destination file is opened -- and the parent
    directory is created here, which polars' sinks do not do. A hierarchy is
    refused by name here too, so the message says which verb refused.
    """
    from polspec.generation import _requires_whole_frame, scan

    require_columns(spec)
    _requires_whole_frame(spec, "a sink")
    target = Path(path)
    lf = scan(
        spec,
        n,
        seed=seed,
        batch_size=batch_size,
        method=method,
        references=references,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    return lf, target


def sink_parquet(
    spec: TableSpec,
    path: str | Path,
    n: int,
    *,
    batch_size: int = 100_000,
    compression: ParquetCompression = "zstd",
    method: Method = "random",
    seed: int | None = None,
    references: References = None,
    **kwargs: Any,
) -> None:
    """Generates `n` rows and streams them to a Parquet file in batches.

    Extra keyword arguments go to `pl.LazyFrame.sink_parquet`.
    """
    lf, target = _scan(spec, path, n, batch_size, method, seed, references)
    lf.sink_parquet(target, compression=compression, **kwargs)


def sink_ipc(
    spec: TableSpec,
    path: str | Path,
    n: int,
    *,
    batch_size: int = 100_000,
    compression: IpcCompression | None = "zstd",
    method: Method = "random",
    seed: int | None = None,
    references: References = None,
    **kwargs: Any,
) -> None:
    """Generates `n` rows and streams them to an Arrow IPC / Feather file in batches.

    Extra keyword arguments go to `pl.LazyFrame.sink_ipc`.
    """
    lf, target = _scan(spec, path, n, batch_size, method, seed, references)
    lf.sink_ipc(target, compression=compression, **kwargs)


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

    Extra keyword arguments go to `pl.LazyFrame.sink_csv`.
    """
    lf, target = _scan(spec, path, n, batch_size, method, seed, references)
    lf.sink_csv(target, include_header=include_header, **kwargs)


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

    Extra keyword arguments go to `pl.LazyFrame.sink_ndjson`.
    """
    lf, target = _scan(spec, path, n, batch_size, method, seed, references)
    lf.sink_ndjson(target, **kwargs)
