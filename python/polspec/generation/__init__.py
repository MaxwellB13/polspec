"""Turning a `TableSpec` into data.

`generate` is the whole pipeline: the Rust engine fills every column
independently, then rules and foreign keys are applied as vectorised passes
over the finished frame, in the order `polspec.constraints` derives from what
each pass reads and writes. `generate_batches` and the `sink_*` functions in
`polspec.generation.sinks` stream the same pipeline in chunks.
"""

from __future__ import annotations

import difflib
import random
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import polars as pl

from polspec.constants import _LARGE_FRAME_BYTES
from polspec.constraints import ordered_passes, rewritable_members
from polspec.engine import _generate_cartesian, _generate_random
from polspec.errors import GenerationError, SpecError
from polspec.foreign_key import _apply_foreign_key
from polspec.frames import Method, References, to_eager
from polspec.generation.composite import apply_unique_together
from polspec.generation.scan import scan
from polspec.generation.seeds import pass_seed
from polspec.generation.sinks import sink_csv, sink_ipc, sink_ndjson, sink_parquet
from polspec.hierarchy import _apply_hierarchy
from polspec.rules import _apply_column_rules
from polspec.spec import ColSpec
from polspec.tablespec import TableSpec, require_columns, resolve_references

__all__ = [
    "generate",
    "generate_batches",
    "scan",
    "sink_csv",
    "sink_ipc",
    "sink_ndjson",
    "sink_parquet",
]


def _check_faults(spec: TableSpec, cycles: int, self_references: int) -> None:
    """Faults are hierarchy-shaped, so a spec without one cannot take them."""
    if cycles < 0 or self_references < 0:
        raise ValueError("cycles and self_references must be >= 0")
    if (cycles or self_references) and spec.hierarchy is None:
        raise SpecError(
            f"{spec.name} declares no __hierarchy__, so there are no links to "
            "damage. cycles= and self_references= describe a parent/child edge "
            "list; declare a Hierarchy, or drop the arguments."
        )


def _requires_whole_frame(spec: TableSpec, verb: str) -> None:
    """Refuses the streaming verbs for a spec whose shape spans the frame.

    A hierarchy is a property of every row at once -- each reference has one
    parent somewhere else in the same frame. Batches are sampled
    independently, so a batched hierarchy would be a pile of unrelated
    fragments rather than a shallow tree, which is a worse answer than saying
    no.
    """
    if spec.hierarchy is not None:
        raise SpecError(
            f"{spec.name} declares a __hierarchy__, which {verb} cannot produce: "
            "every reference points at another row of the same frame, and each "
            "batch is generated on its own. Use generate() for a hierarchy, and "
            "write the frame out yourself."
        )


def _check_counts(n: int, batch_size: int | None = None) -> None:
    if n < 0:
        raise ValueError("n must be >= 0")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be > 0")


def _warn_unused_references(spec: TableSpec, parents: dict[str, Any]) -> None:
    """Warns when a supplied parent went unused *and* a key went unfilled.

    A foreign key whose target has no data is left exactly as freely
    generated, which is documented and deliberate -- `references` is optional.
    What is not deliberate is supplying a parent under a name nothing asked
    for: the caller believes that key was satisfied, generation quietly did
    not satisfy it, and `validate()` then reports the key as unresolved. A
    misspelled spec name is the usual cause.

    Both halves have to hold before this says anything. A `Registry` hands
    every spec the whole set of frames generated so far, most of which any one
    spec has no key for, so an unused parent on its own is ordinary.
    """
    if not parents:
        return
    targets = {fk.references for fk in spec.foreign_keys if fk.references != "self"}
    unfilled = sorted(targets - parents.keys())
    unused = sorted(parents.keys() - targets)
    if not unfilled or not unused:
        return
    hints = []
    for name in unfilled:
        close = difflib.get_close_matches(name, unused, n=1)
        hints.append(f"{name!r}{f' (supplied {close[0]!r}?)' if close else ''}")
    warnings.warn(
        f"{spec.name}: references={{...}} supplied {unused} that no foreign key "
        f"points at, while {', '.join(hints)} went unfilled. Those columns are "
        "generated freely, so validate() will report the key as unresolved. "
        f"{spec.name} references: {sorted(targets)}.",
        stacklevel=3,
    )


def generate(
    spec: TableSpec,
    n: int,
    *,
    method: Method = "random",
    seed: int | None = None,
    references: References = None,
    cycles: int = 0,
    self_references: int = 0,
    max_bytes: int | None = None,
) -> pl.DataFrame:
    """Generates a DataFrame matching `spec`.

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

    A spec declaring a `Hierarchy` has its two link columns rewritten as a
    forest of the declared depth. `cycles` and `self_references` then damage it
    on purpose -- closing that many chains into loops, and pointing that many
    rows at themselves -- which is how a graph walk gets something to fail
    against. Both default to zero, and `validate()` reports whatever they
    injected.

    Before anything is allocated the frame's size is estimated from the
    declaration. Past four gibibytes that is a warning naming the estimate;
    `max_bytes=` makes it a refusal instead, and `max_bytes=0` silences both.

    The whole frame is built before this returns. `scan()` is the lazy
    verb: it generates as the plan is collected, so only the columns and
    rows a plan asks for are made.
    """
    require_columns(spec)
    _check_counts(n)
    _check_faults(spec, cycles, self_references)
    _check_size(spec, n, max_bytes)
    if method not in ("random", "cartesian"):
        raise ValueError(f"Unknown method {method!r}; expected 'random' or 'cartesian'")

    return _window(
        spec,
        n,
        _frame_seed(seed),
        method=method,
        references=references,
        cycles=cycles,
        self_references=self_references,
    )


def _describe_bytes(size: int) -> str:
    """A byte count as the unit a reader thinks in."""
    if size < 1024:
        return f"{size:,} bytes"
    scaled = float(size)
    for unit in ("KiB", "MiB", "GiB"):
        scaled /= 1024
        if scaled < 1024 or unit == "GiB":
            return f"{scaled:,.1f} {unit}"
    raise AssertionError  # pragma: no cover - the loop returns at GiB


def _check_size(spec: TableSpec, n: int, max_bytes: int | None) -> None:
    """Says how large the frame will be before it is allocated.

    The first sign that a spec is too big for the machine is usually the
    machine swapping, and the estimate costs nothing -- it is read off the
    declaration. A warning rather than a refusal by default, because a ten
    gigabyte frame on a machine with sixty-four is a reasonable thing to
    ask for; `max_bytes` is for a caller who wants it refused, and
    `max_bytes=0` for one who wants neither.
    """
    if max_bytes == 0:
        return
    estimate = spec.estimated_size(n)
    limit = _LARGE_FRAME_BYTES if max_bytes is None else max_bytes
    if estimate <= limit:
        return
    advice = (
        "scan() generates it as it is collected, and generate_batches() "
        "in batches; either holds a fraction of that at once."
    )
    if max_bytes is not None:
        raise GenerationError(
            f"{spec.name} at {n:,} rows is an estimated "
            f"{_describe_bytes(estimate)}, over the max_bytes of "
            f"{_describe_bytes(max_bytes)}. {advice}"
        )
    warnings.warn(
        f"{spec.name} at {n:,} rows is an estimated {_describe_bytes(estimate)}, "
        f"which generate() allocates before it returns. {advice} Pass "
        "max_bytes=0 to silence this.",
        stacklevel=3,
    )


def _frame_seed(seed: int | None) -> int:
    """The frame seed: the first draw of the caller's seed, as it always was,
    so every column's values are what they were. Everything else -- each
    column, each pass, each batch -- derives its own seed from this one and
    a stable key."""
    return random.Random(seed).randrange(2**63)


def _window(
    spec: TableSpec,
    n: int,
    frame_seed: int,
    *,
    method: Method,
    references: References,
    row_offset: int = 0,
    cycles: int = 0,
    self_references: int = 0,
) -> pl.DataFrame:
    """Rows `[row_offset, row_offset + n)` of the frame `frame_seed` describes,
    with the passes run over them.

    The columns are a window onto one frame: the engine numbers its chunks
    from the offset, so the same rows come back whatever `n` the call asks
    for. The passes are drawn per window, keyed by the offset, so two
    windows never repeat each other's draws.
    """
    columns = dict(spec.columns)
    if method == "cartesian":
        df = _generate_cartesian(columns, n, frame_seed)
    else:
        df = _generate_random(columns, n, frame_seed, row_offset)
    pass_seed_ = (
        frame_seed if row_offset == 0 else pass_seed(frame_seed, f"window:{row_offset}")
    )
    return _run_passes(
        spec,
        columns,
        df,
        references,
        pass_seed_,
        cycles=cycles,
        self_references=self_references,
    )


def _run_passes(
    spec: TableSpec,
    columns: dict[str, ColSpec],
    df: pl.DataFrame,
    references: References,
    frame_seed: int,
    *,
    cycles: int = 0,
    self_references: int = 0,
) -> pl.DataFrame:
    """Applies every rule and foreign-key pass, in dependency order.

    Each pass is seeded from the frame seed and a key naming what it is for
    -- a rules column's seed name, a foreign key's name, a composite key's
    members -- so neither the order the passes run in nor the columns
    declared around them change the values any one of them samples. A
    column inserted ahead of a rules column leaves it alone; a rules column
    renamed with `seed_name` keeps its rule's draw as well as its own.
    """
    runners: dict[str, Callable[[pl.DataFrame], pl.DataFrame]] = {}

    for name, col in columns.items():
        if not col.rules:
            continue
        seed = pass_seed(frame_seed, f"rules:{col.seed_name or name}")
        runners[f"rules:{name}"] = lambda frame, name=name, col=col, seed=seed: (
            _apply_column_rules(frame, name, col, seed)
        )

    if (hierarchy := spec.hierarchy) is not None:
        seed = pass_seed(frame_seed, "hierarchy")
        runners["hierarchy"] = lambda frame, seed=seed: _apply_hierarchy(
            frame,
            columns,
            hierarchy,
            seed,
            cycles=cycles,
            self_references=self_references,
        )

    if spec.foreign_keys:
        parents = resolve_references(references, to_eager)
        _warn_unused_references(spec, parents)
        for fk in spec.foreign_keys:
            seed = pass_seed(frame_seed, f"fk:{fk.name}")
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
        members, writable = tuple(group), rewritable_members(spec, group)
        seed = pass_seed(frame_seed, f"unique_together:{','.join(sorted(members))}")
        key = f"unique_together:{index}"
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

    Each batch is a window onto the one frame `seed` describes: a column no
    pass rewrites holds, batch by batch, exactly the rows `generate(n, seed=seed)`
    would, whatever `batch_size` is. What is drawn per batch instead --
    deterministic, but not row for row the whole frame's -- is a column with
    rules, a foreign key, a composite key, and a List column's elements.

    Uniqueness only holds *within* a batch, not across the whole `n`: that
    applies to a `unique=True` column, a `__unique_together__` group, and a
    foreign-key column sampled without replacement alike.
    """
    require_columns(spec)
    _requires_whole_frame(spec, "generate_batches")
    _check_counts(n, batch_size)
    if method not in ("random", "cartesian"):
        raise ValueError(f"Unknown method {method!r}; expected 'random' or 'cartesian'")
    if n == 0:
        return

    # A parent frame is the same frame for every batch, so collect it once.
    # Left as given, a `LazyFrame` reference would be collected inside each
    # batch's foreign-key pass -- once per batch rather than once per call,
    # which is the whole parent re-read however many batches there are.
    resolved = resolve_references(references, to_eager)
    if spec.foreign_keys:
        # Once per call, not once per batch: the parents are the same every
        # time round, so the batches below are handed only the keys a foreign
        # key actually points at and find nothing left to report.
        _warn_unused_references(spec, resolved)
        targets = {fk.references for fk in spec.foreign_keys if fk.references != "self"}
        resolved = {k: v for k, v in resolved.items() if k in targets}
    references = resolved or None

    frame_seed = _frame_seed(seed)
    produced = 0

    if method == "cartesian":
        # The coverage set is a property of the spec, not of a window: it is
        # built once, as `generate` builds it, and sliced to the batch size
        # (it is a cross-product, not a row count cap). The padding after it
        # is windowed like a random batch stream.
        first = _window(
            spec,
            min(n, batch_size),
            frame_seed,
            method="cartesian",
            references=references,
        )
        for offset in range(0, first.height, batch_size):
            yield first.slice(offset, batch_size)
        produced = first.height

    while produced < n:
        current = min(n - produced, batch_size)
        yield _window(
            spec,
            current,
            frame_seed,
            method="random",
            references=references,
            row_offset=produced,
        )
        produced += current
