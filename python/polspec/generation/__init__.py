"""Turning a `TableSpec` into data.

`generate` is the whole pipeline: the Rust engine fills every column
independently, then rules and foreign keys are applied as vectorised passes
over the finished frame, in the order `polspec.constraints` derives from what
each pass reads and writes. `generate_batches` and the `sink_*` functions in
`polspec.generation.sinks` stream the same pipeline in chunks.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Literal, overload

import polars as pl

from polspec.constraints import ordered_passes, rewritable_members
from polspec.engine import _generate_cartesian, _generate_random
from polspec.errors import SpecError
from polspec.foreign_key import _apply_foreign_key
from polspec.generation.composite import apply_unique_together
from polspec.generation.sinks import sink_csv, sink_ipc, sink_ndjson, sink_parquet
from polspec.rules import _apply_column_rules
from polspec.spec import ColSpec
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

    `ColSpec.rules` and any `ForeignKey` the spec declares are then applied as
    vectorised passes over the generated frame, regardless of method. Each
    pass sees the frame the passes before it produced, and they run in the
    order their reads and writes imply -- a rule keyed on a foreign-keyed
    column reads the parent's values, not the freely generated ones they
    replaced -- so the result satisfies the same declarations `validate`
    checks it against.

    A foreign key is only made referentially consistent where data for its
    target is available: self-referencing keys always are, sampled from this
    same frame; a key referencing another spec only is if `references`
    carries an entry for it, keyed by the spec, its class, or its name --
    otherwise that column is left exactly as freely generated. Composite keys
    are sampled as one joint pick per row; a single-column key whose ColSpec
    is `unique=True` samples without replacement when the parent has enough
    distinct rows to cover `n`.

    A `unique=True` column is drawn without replacement by the engine itself,
    and a `__unique_together__` group is separated afterwards by resampling
    the rows that repeat a combination. Either refuses, naming the column or
    the group, when the domain is too small to cover `n`.

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

    res = _run_passes(spec, columns, df, references, rng)

    return res.lazy() if lazy else res


def _run_passes(
    spec: TableSpec,
    columns: dict[str, ColSpec],
    df: pl.DataFrame,
    references: References,
    rng: random.Random,
) -> pl.DataFrame:
    """Applies every rule and foreign-key pass, in dependency order.

    Seeds are drawn per pass in *declaration* order, so which order the
    passes end up running in does not change the values any one of them
    samples. A cross-spec key the caller gave no data for still draws its
    seed and is then skipped, so supplying references never reshuffles the
    columns around it.
    """
    runners: dict[str, Callable[[pl.DataFrame], pl.DataFrame]] = {}

    for name, col in columns.items():
        if not col.rules:
            continue
        seed = rng.randrange(2**63)
        runners[f"rules:{name}"] = lambda frame, name=name, col=col, seed=seed: (
            _apply_column_rules(frame, name, col, seed)
        )

    if spec.foreign_keys:
        parents = resolve_references(references, _collect)
        for fk in spec.foreign_keys:
            seed = rng.randrange(2**63)
            if fk.references == "self":
                # The parent is this frame as it stands when the pass runs.
                runners[f"fk:{fk.name}"] = lambda frame, fk=fk, seed=seed: (
                    _apply_foreign_key(frame, columns, fk, frame, seed)
                )
            elif (parent := parents.get(fk.references)) is not None:
                runners[f"fk:{fk.name}"] = (
                    lambda frame, fk=fk, parent=parent, seed=seed: _apply_foreign_key(
                        frame, columns, fk, parent, seed
                    )
                )

    for index, group in enumerate(spec.unique_together):
        seed = rng.randrange(2**63)
        key = f"unique_together:{index}"
        members, writable = tuple(group), rewritable_members(spec, group)
        runners[key] = lambda frame, m=members, w=writable, seed=seed: (
            apply_unique_together(frame, columns, m, w, seed)
        )

    for step in ordered_passes(spec):
        run = runners.get(step.key)
        if run is not None:
            df = run(df)
    return df


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

    Each batch samples independently, so uniqueness only holds *within* a
    batch, not across the whole `n`: that applies to a `unique=True` column,
    a `__unique_together__` group, and a foreign-key column sampled without
    replacement alike.
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
