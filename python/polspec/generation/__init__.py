"""Turning a `TableSpec` into data.

`generate` is the whole pipeline: the Rust engine fills every column
independently, then rules and foreign keys are applied as vectorised passes
over the finished frame. `generate_batches` and the `sink_*` functions in
`polspec.generation.sinks` stream the same pipeline in chunks.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping
from typing import Any, Literal, overload

import polars as pl

from polspec.engine import _generate_cartesian, _generate_random
from polspec.errors import SpecError
from polspec.foreign_key import ForeignKey, _apply_foreign_keys
from polspec.generation.sinks import sink_csv, sink_ipc, sink_ndjson, sink_parquet
from polspec.rules import _apply_rules
from polspec.tablespec import TableSpec, resolve_references

__all__ = [
    "generate",
    "generate_batches",
    "sink_csv",
    "sink_ipc",
    "sink_ndjson",
    "sink_parquet",
]

Method = Literal["random", "cartesian"]
References = Mapping[Any, pl.DataFrame | pl.LazyFrame] | None


def _require_columns(spec: TableSpec) -> None:
    if not spec.columns:
        raise SpecError(f"{spec.name} declares no ColSpec columns")


def _check_counts(n: int, batch_size: int | None = None) -> None:
    if n < 0:
        raise ValueError("n must be >= 0")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be > 0")


def _collect(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


@overload
def generate(
    spec: TableSpec,
    n: int,
    *,
    method: Method = "random",
    seed: int | None = None,
    references: References = None,
    lazy: Literal[False] = False,
) -> pl.DataFrame: ...


@overload
def generate(
    spec: TableSpec,
    n: int,
    *,
    method: Method = "random",
    seed: int | None = None,
    references: References = None,
    lazy: Literal[True],
) -> pl.LazyFrame: ...


def generate(
    spec: TableSpec,
    n: int,
    *,
    method: Method = "random",
    seed: int | None = None,
    references: References = None,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Generates a DataFrame (or LazyFrame) matching `spec`.

    method="random" (default): `n` rows, each column drawn independently.

    method="cartesian": guarantees a minimum level of coverage. Builds the
    cartesian product of every Enum/Boolean column's full set of values,
    crossed with the negative/zero/positive/null partitions of every bounded
    numeric column, so every enum combination appears alongside every numeric
    sign/null case. `n` is then a *minimum*: if that coverage set has fewer
    than `n` rows it is padded with random rows; if it has more, all of it is
    kept.

    Any `ColSpec.rules` are applied next, as a vectorised overwrite pass over
    the fully generated frame, regardless of method.

    Any `ForeignKey` the spec declares is then made referentially consistent
    where data for its target is available: self-referencing keys always are,
    sampled from this same frame; a key referencing another spec only is if
    `references` carries an entry for it, keyed by the spec, its class, or
    its name -- otherwise that column is left exactly as freely generated.
    Composite keys are sampled as one joint pick per row; a single-column
    key whose ColSpec is `unique=True` samples without replacement when the
    parent has enough distinct rows to cover `n`.

    lazy=True returns a `pl.LazyFrame` around the generated DataFrame.
    """
    _require_columns(spec)
    _check_counts(n)

    rng = random.Random(seed)
    gen_seed = rng.randrange(2**63)
    columns = dict(spec.columns)
    if method == "random":
        df = _generate_random(columns, n, gen_seed)
    elif method == "cartesian":
        df = _generate_cartesian(columns, n, gen_seed)
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'random' or 'cartesian'")

    res = _apply_rules(df, columns, rng.randrange(2**63))

    if spec.foreign_keys:
        parents = resolve_references(references, _collect)
        resolved: list[tuple[ForeignKey, pl.DataFrame | None]] = []
        for fk in spec.foreign_keys:
            if fk.references == "self":
                resolved.append((fk, res))
            else:
                resolved.append((fk, parents.get(fk.references)))
        res = _apply_foreign_keys(res, columns, resolved, rng.randrange(2**63))

    return res.lazy() if lazy else res


def generate_batches(
    spec: TableSpec,
    n: int,
    *,
    batch_size: int = 100_000,
    method: Method = "random",
    seed: int | None = None,
    references: References = None,
) -> Iterator[pl.DataFrame]:
    """Yields chunks of generated rows without holding all `n` in memory.

    Each batch samples independently, so a `unique=True` foreign-key column
    is only sampled without replacement *within* a batch, not across the
    whole `n`.
    """
    _require_columns(spec)
    _check_counts(n, batch_size)
    if method not in ("random", "cartesian"):
        raise ValueError(f"Unknown method {method!r}; expected 'random' or 'cartesian'")
    if n == 0:
        return

    rng = random.Random(seed)
    rows_remaining = n

    if method == "cartesian":
        first = generate(
            spec,
            min(n, batch_size),
            method="cartesian",
            seed=rng.randrange(2**63),
            references=references,
        )
        # The coverage set can be far larger than batch_size (it is a
        # cross-product, not a row count cap), so slice it before yielding.
        for offset in range(0, first.height, batch_size):
            yield first.slice(offset, batch_size)
        rows_remaining = max(0, n - first.height)

    while rows_remaining > 0:
        current = min(rows_remaining, batch_size)
        yield generate(
            spec,
            current,
            method="random",
            seed=rng.randrange(2**63),
            references=references,
        )
        rows_remaining -= current
