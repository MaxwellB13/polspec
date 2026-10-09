"""Turning a `TableSpec` into data.

`generate` is the whole pipeline (`pipeline.py`): the Rust engine fills every
column independently (`engine.py`), then the passes in `passes/` -- rules,
foreign keys, a hierarchy, composite keys -- rewrite the finished frame, in
the order `polspec.pass_order` derives from what each reads and writes.
`generate_batches` streams the same pipeline in windows; `scan` is a
`LazyFrame` over it (`scan.py`), and the `sink_*` functions write one out
(`sinks.py`).

Internal: not part of the public API.
"""

from __future__ import annotations

from polspec.generation.pipeline import generate, generate_batches
from polspec.generation.scan import scan
from polspec.generation.sinks import sink_csv, sink_ipc, sink_ndjson, sink_parquet

__all__ = [
    "generate",
    "generate_batches",
    "scan",
    "sink_csv",
    "sink_ipc",
    "sink_ndjson",
    "sink_parquet",
]
