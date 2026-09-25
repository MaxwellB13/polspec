"""A `LazyFrame` that generates rows on demand.

`generate()` builds the whole frame before it returns. `scan()` hands back
a frame that has not been built: polars asks for the columns and the rows
it actually needs, and only those are generated. So

    Orders.scan(50_000_000, seed=1).sink_parquet("orders.parquet")

streams in bounded memory, and

    Orders.scan(50_000_000, seed=1).select("total").head(5).collect()

generates five rows of one column.

Projecting cannot change what a column holds. Every column is seeded by its
name and every pass by what it is for, so dropping a column's neighbours
leaves it alone: `scan(n, seed=s).select(cols).collect()` is
`scan(n, seed=s).collect().select(cols)`, for every subset. That is what
makes the pushdown free rather than a trade.

Rows come in batches, so a scan carries `generate_batches`' terms: a
`Hierarchy` is refused, uniqueness holds within a batch rather than across
`n`, and a column no pass rewrites holds the rows `generate(n, seed=s)`
would while a ruled or foreign-keyed one is drawn per batch.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec.constraints import passes_of
from polspec.frames import Method, References

if TYPE_CHECKING:
    from polspec.tablespec import TableSpec

DEFAULT_BATCH_SIZE = 100_000


def needed_columns(spec: TableSpec, wanted: frozenset[str]) -> list[str]:
    """`wanted` plus every column its values depend on, in spec order.

    A projection cannot simply drop the rest: a rules column's values depend
    on the columns its `when` reads, a composite key's on every member of the
    group, a foreign key's on the other columns of its key. `passes_of`
    already knows what each pass reads and writes, so the closure is that
    relation followed to a fixed point -- generate these, hand back only what
    was asked for.
    """
    needed = set(wanted)
    passes = passes_of(spec)
    changed = True
    while changed:
        changed = False
        for step in passes:
            if step.writes & needed and not (step.writes | step.reads) <= needed:
                needed |= step.writes | step.reads
                changed = True
    return [name for name in spec.columns if name in needed]


def _predicate_columns(predicate: pl.Expr | None) -> frozenset[str]:
    """The columns a pushed-down predicate reads, so they are generated even
    when the projection does not ask for them."""
    if predicate is None:
        return frozenset()
    return frozenset(predicate.meta.root_names())


def scan(
    spec: TableSpec,
    n: int,
    *,
    seed: int | None = None,
    batch_size: int | None = None,
    method: Method = "random",
    references: References = None,
) -> pl.LazyFrame:
    """A `LazyFrame` of `n` generated rows, produced as they are collected.

    `batch_size` left unset lets polars ask for the size it would like, so a
    sink gets the batches it writes best; setting it pins the size whatever
    polars asks.

    A scan is batched, so it has `generate_batches`' terms: a `unique=True`
    column is unique across the whole scan, and a composite key or a
    foreign key sampled without replacement only within each batch.
    """
    from polspec.generation import (
        _check_counts,
        _requires_whole_frame,
        generate_batches,
    )
    from polspec.tablespec import require_columns

    require_columns(spec)
    _requires_whole_frame(spec, "scan")
    _check_counts(n, batch_size)
    if method not in ("random", "cartesian"):
        raise ValueError(f"Unknown method {method!r}; expected 'random' or 'cartesian'")

    def io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        polars_batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        wanted = list(spec.columns) if with_columns is None else with_columns
        needed = frozenset(wanted) | _predicate_columns(predicate)
        source = spec.select(*needed_columns(spec, needed))
        rows = n if n_rows is None else min(n, n_rows)
        size = batch_size or polars_batch_size or DEFAULT_BATCH_SIZE
        for batch in generate_batches(
            source,
            rows,
            batch_size=size,
            method=method,
            seed=seed,
            references=references,
        ):
            # A predicate filters rows that were drawn; it never narrows the
            # draw. `n` rows are generated and the matching ones kept, which
            # is the only reading that leaves a rate or a `unique` column
            # meaning what it declares -- and the reader has to apply it,
            # since polars hands the predicate over rather than keeping it.
            kept = batch if predicate is None else batch.filter(predicate)
            yield kept.select(wanted)

    register = pl.io.plugins.register_io_source
    return register(
        io_source,
        schema=spec.schema(),
        validate_schema=True,
        # Pure for a given seed: the same rows however many times the plan
        # asks for them, which lets polars de-duplicate a repeated scan.
        is_pure=seed is not None,
        **_explain_labels(register, spec, n, seed, batch_size, method),
    )


def _explain_labels(
    register: Callable[..., pl.LazyFrame],
    spec: TableSpec,
    n: int,
    seed: int | None,
    batch_size: int | None,
    method: Method,
) -> dict[str, Any]:
    """What the scan says about itself in `explain()`, where Polars lets a
    source say anything: `PYTHON[polspec: Orders] SCAN`, and an `INFO:` line
    with how many rows, from which seed, in what batches.

    Polars 2 added `explain_name` and `explain_detail` to
    `register_io_source`; Polars 1 has neither and would refuse them. Whether
    to pass them is read off the function's own signature, not a version
    number, so a Polars that has them gets them and one that does not sees
    exactly the call it always did.
    """
    if not _takes_explain_labels(register):
        return {}
    detail = [
        f"{n:,} rows",
        f"seed={seed}" if seed is not None else "unseeded",
        f"batches of {batch_size:,}" if batch_size else "batches chosen by polars",
    ]
    if method != "random":
        detail.append(f"method={method}")
    return {
        "explain_name": f"polspec: {spec.name}",
        "explain_detail": ", ".join(detail),
    }


@functools.cache
def _takes_explain_labels(register: Callable[..., pl.LazyFrame]) -> bool:
    """Whether `register` accepts both explain labels -- asked once per
    function, since a signature does not change under a running process."""
    try:
        parameters = inspect.signature(register).parameters
    except (TypeError, ValueError):  # a builtin or C function with no signature
        return False
    return "explain_name" in parameters and "explain_detail" in parameters
