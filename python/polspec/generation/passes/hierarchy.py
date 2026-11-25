"""`Hierarchy` as a pass: the two link columns rewritten as a forest of the
declared shape, and -- when asked -- damaged on purpose with cycles and
self-references for a graph walk to fail against.

Two layouts draw the same shape. The windowed one (the default since 2.1,
`apply_windowed_hierarchy`) draws each row's place in the forest from its
position and the seed alone, so any window of rows is the whole frame's
slice there. The whole one (`apply_hierarchy`, 2.0's) builds the forest
over the whole frame from one generator, and generates 2.0's frame.

Internal: not part of the public API.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import TYPE_CHECKING

import polars as pl

from polspec import frames
from polspec._ffi import column_plan, forest_parents, permuted_at
from polspec._ffi import generate_dataframe as _generate_dataframe
from polspec.errors import GenerationError
from polspec.generation.engine import generate_random, unique_at
from polspec.generation.seeds import pass_seed
from polspec.hierarchy import Hierarchy

if TYPE_CHECKING:
    from polspec.spec import ColSpec


# How far up from a leaf a cycle reaches before closing. Three hops makes a
# loop long enough that a resolver cannot pass it by testing `child == parent`,
# and short enough to fit inside a shallow `max_depth`.
_CYCLE_HOPS = 3


# Candidate leaves tried per cycle before giving up. Cycles must not overlap,
# or the count returned would not be the count asked for.
_CYCLE_ATTEMPTS = 32


def _effective_branching(hierarchy: Hierarchy, n: int) -> float:
    """The mean children per reference this declaration implies for `n` rows.

    Given directly, or solved for when `roots` was fixed instead: the two
    are tied by `n = roots * (b + b**2 + ... + b**max_depth)`, which rises
    monotonically in `b`, so a bisection settles it.
    """
    if hierarchy.branching is not None:
        return hierarchy.branching
    if hierarchy.roots is None:
        return 3.0
    if hierarchy.max_depth == 1:
        return n / hierarchy.roots
    low, high = 1.0, max(2.0, float(n))
    for _ in range(60):
        mid = (low + high) / 2
        if hierarchy.roots * _geometric_sum(mid, hierarchy.max_depth) < n:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _geometric_sum(branching: float, depth: int) -> float:
    """`b + b**2 + ... + b**depth`, the references one root implies."""
    if math.isclose(branching, 1.0):
        return float(depth)
    return branching * (branching**depth - 1) / (branching - 1)


def _level_sizes(hierarchy: Hierarchy, n: int) -> list[int]:
    """How many references sit at each level `1..max_depth`, summing to `n`.

    Sized in a geometric progression so the result looks like a tree rather
    than a caterpillar, then adjusted so every level holds at least one
    reference -- which is what guarantees a chain of exactly `max_depth`.
    """
    depth = hierarchy.max_depth
    if n < depth:
        raise GenerationError(
            f"A hierarchy of max_depth={depth} needs at least {depth} row(s) to "
            f"reach that depth, and {n} were asked for. Generate more rows, or "
            "declare a shallower max_depth."
        )
    branching = _effective_branching(hierarchy, n)
    weights = [branching**k for k in range(1, depth + 1)]
    total = sum(weights)
    sizes = [max(1, int(n * w / total)) for w in weights]

    # Rounding and the floor of one leave the total adrift; settle the
    # difference at the widest level, where one reference either way is least
    # visible in the shape.
    while sum(sizes) > n:
        widest = sizes.index(max(sizes))
        if sizes[widest] == 1:
            break  # every level is minimal; `n >= depth` guarantees this fits
        sizes[widest] -= 1
    if sum(sizes) < n:
        sizes[sizes.index(max(sizes))] += n - sum(sizes)
    return sizes


def _root_count(hierarchy: Hierarchy, n: int, level_sizes: list[int]) -> int:
    """How many ultimate parents the shape implies."""
    if hierarchy.roots is not None:
        return hierarchy.roots
    branching = _effective_branching(hierarchy, n)
    return max(1, round(level_sizes[0] / branching))


def _reference_pool(spec: ColSpec, name: str, count: int, seed: int) -> pl.Series:
    """`count` distinct references, drawn from the child column's declaration.

    The column's own dtype, `string_length` and `choices` decide what a
    reference looks like, so a spec that pins its references to a list of real
    identifiers gets those. Anything describing how often a value *recurs* is
    dropped: these are drawn without replacement, so each appears once.
    """
    pool_spec = dataclasses.replace(
        spec,
        unique=True,
        nullable=False,
        null_probability=0.0,
        _weights=None,
        distribution=None,
        distribution_params=None,
        rules=(),
        validators=(),
    )
    return generate_random({name: pool_spec}, count, seed)[name]


def _random_indices(domain_size: int, count: int, seed: int) -> pl.Series:
    """`count` indices into a domain of `domain_size`, drawn with replacement.

    Through the same engine path a `ColRule` uses to pick from `choices`, so
    one million parents are chosen in Rust rather than in a Python loop.
    """
    plan = column_plan("__idx", "index", n_categories=domain_size)
    return _generate_dataframe([plan], count, seed)["__idx"]


def _cycle_edits(
    levels: list[pl.Series],
    parent_index: list[pl.Series],
    offsets: list[int],
    count: int,
    rng: random.Random,
) -> tuple[list[int], list]:
    """Which rows to repoint, and at what, to close `count` chains into loops.

    Picks a leaf, walks up a few hops to an ancestor, and points that
    ancestor's parent back down at the leaf. Everything between the two is
    then a loop, with the ancestor's remaining subtree hanging off it -- a
    resolver that fails to notice will circle forever.

    A cycle crosses levels, so it cannot be expressed in the index-per-level
    form the tree is built in: the walk up is done on indices, and the edit
    comes back as a row position and the value to write there.

    Ancestors are kept distinct, so no two loops share an edge and the caller
    gets exactly the number it asked for.
    """
    depth = len(parent_index)
    if depth < 2:
        raise GenerationError(
            "cycles= needs a hierarchy of max_depth 2 or more: at max_depth 1 "
            "every row already points straight at an ultimate parent, and the "
            "only loop available is a self-reference. Use self_references=."
        )
    # An ancestor needs a row of its own to repoint, so it cannot be a root:
    # keep it at level 1 or below.
    hops = min(_CYCLE_HOPS, depth - 1)
    ancestor_level = depth - hops
    leaves = levels[depth]

    claimed: set[int] = set()
    rows: list[int] = []
    values: list = []
    for _ in range(count * _CYCLE_ATTEMPTS):
        if len(rows) == count:
            break
        leaf = rng.randrange(len(leaves))
        node = leaf
        for level in range(depth - 1, ancestor_level - 1, -1):
            node = int(parent_index[level][node])
        if node in claimed:
            continue
        claimed.add(node)
        # Block `j` of the edge frame holds the children of level `j + 1`, so
        # the ancestor's own row sits in the block one above its level.
        rows.append(offsets[ancestor_level - 1] + node)
        values.append(leaves[leaf])

    if len(rows) < count:
        raise GenerationError(
            f"Could not build {count} non-overlapping cycle(s): only "
            f"{len(rows)} found a chain of their own. Generate more rows, or "
            "ask for fewer cycles."
        )
    return rows, values


def apply_hierarchy(
    df: pl.DataFrame,
    columns: dict[str, ColSpec],
    hierarchy: Hierarchy,
    seed: int | None,
    *,
    cycles: int = 0,
    self_references: int = 0,
) -> pl.DataFrame:
    """Rewrites the child and parent columns as a forest of the declared depth.

    Both columns are overwritten outright: whatever the engine generated for
    them independently cannot be a hierarchy, and the references have to come
    from one pool for the two columns to join at all.
    """
    n = df.height
    if n == 0:
        return df

    rng = random.Random(seed)
    child, parent = hierarchy.child, hierarchy.parent

    sizes = _level_sizes(hierarchy, n)
    n_roots = _root_count(hierarchy, n, sizes)
    pool = _reference_pool(columns[child], child, n_roots + n, rng.randrange(2**63))

    # Level 0 is the ultimate parents; levels 1.. are everything with a row.
    levels: list[pl.Series] = [pool.slice(0, n_roots)]
    offset = n_roots
    for size in sizes:
        levels.append(pool.slice(offset, size))
        offset += size

    # A parent per reference, as an index into the level above.
    parent_index = [
        _random_indices(len(levels[k - 1]), len(levels[k]), rng.randrange(2**63))
        for k in range(1, len(levels))
    ]

    edges = pl.concat(
        [
            pl.DataFrame(
                {
                    child: levels[k + 1],
                    parent: levels[k].gather(parent_index[k]),
                }
            )
            for k in range(len(parent_index))
        ]
    )

    if cycles:
        # Row positions are block offsets into the frame as just concatenated,
        # so the edits land before anything reorders it.
        offsets: list[int] = []
        running = 0
        for size in sizes:
            offsets.append(running)
            running += size
        rows, values = _cycle_edits(levels, parent_index, offsets, cycles, rng)
        edges = edges.with_columns(
            frames.plain(edges[parent])
            .scatter(pl.Series(rows, dtype=pl.UInt32), pl.Series(values))
            .alias(parent)
        )

    # Nothing downstream should be able to depend on the frame arriving
    # grouped by level.
    edges = edges.sample(n=edges.height, shuffle=True, seed=rng.randrange(2**32))

    if self_references:
        if self_references > edges.height:
            raise GenerationError(
                f"Asked for {self_references} self-reference(s) in a hierarchy "
                f"of {edges.height} row(s)."
            )
        # The frame has just been shuffled, so the leading rows are an
        # arbitrary sample of it.
        rows = pl.Series(range(self_references), dtype=pl.UInt32)
        edges = edges.with_columns(
            frames.plain(edges[parent])
            .scatter(rows, frames.plain(edges[child]).gather(rows))
            .alias(parent)
        )

    return df.with_columns(edges[child].alias(child), edges[parent].alias(parent))


# ---------------------------------------------------------------------------
# The windowed layout
# ---------------------------------------------------------------------------
#
# Slots `0..n-1` number the forest's edges level by level, as the whole
# layout's frame stacks them before its shuffle; references `0..roots+n-1`
# number the ultimate parents and then one reference per slot. Every choice
# is a function of the seed and a position, so a window computes its own
# rows and nothing else:
#
# - the level sizes depend on `n` and the declaration alone (`_level_sizes`);
# - row `i` holds slot `perm(i)`, `perm` a keyed permutation of the slots;
# - a slot's parent is a uniform pick, keyed by the slot, in the level above;
# - a reference's value is the child column's unique draw at its index.


def _windowed_pool_spec(spec: ColSpec) -> ColSpec:
    """The child column as its references are drawn: `_reference_pool`'s
    spec, and no NaN either, since each reference is read at its index."""
    return dataclasses.replace(
        spec,
        unique=True,
        nullable=False,
        null_probability=0.0,
        nan_probability=0.0,
        _weights=None,
        distribution=None,
        distribution_params=None,
        rules=(),
        validators=(),
    )


class _Layout:
    """The forest's levels over `n` slots, and the keyed choices on them."""

    def __init__(self, hierarchy: Hierarchy, n: int, seed: int) -> None:
        self.sizes = _level_sizes(hierarchy, n)
        self.roots = _root_count(hierarchy, n, self.sizes)
        self.starts = [0]
        for size in self.sizes[:-1]:
            self.starts.append(self.starts[-1] + size)
        self._ends = pl.Series(
            [start + size for start, size in zip(self.starts, self.sizes, strict=True)],
            dtype=pl.UInt64,
        )
        self.n = n
        self.rows_key = pass_seed(seed, "rows")
        self._parent_key = pass_seed(seed, "parents")

    def parent_refs(self, slots: pl.Series) -> pl.Series:
        """Each slot's parent, as a reference index: a root's for a slot at
        level 1 -- the first `roots` of them one root each, so every root
        has a child -- and a reference one level up otherwise, picked with
        replacement by the slot and the seed alone."""
        return forest_parents(self._parent_key, slots, self.sizes, self.roots)


def _windowed_cycle_edits(layout: _Layout, count: int, seed: int) -> dict[int, int]:
    """`_cycle_edits` on the windowed layout: which slots to repoint, and at
    which reference, to close `count` chains into loops.

    A function of the seed alone, so every window computes the same set and
    applies the edits that land in its rows. The candidates are leaves in a
    keyed order; each walks up the same hops `_cycle_edits` walks, and the
    first `count` with an ancestor of their own are kept.
    """
    depth = len(layout.sizes)
    if depth < 2:
        raise GenerationError(
            "cycles= needs a hierarchy of max_depth 2 or more: at max_depth 1 "
            "every row already points straight at an ultimate parent, and the "
            "only loop available is a self-reference. Use self_references=."
        )
    hops = min(_CYCLE_HOPS, depth - 1)
    n_leaves = layout.sizes[-1]
    attempts = min(n_leaves, count * _CYCLE_ATTEMPTS)
    order = pl.int_range(0, attempts, dtype=pl.UInt64, eager=True)
    leaves = permuted_at(n_leaves, pass_seed(seed, "cycles"), order) + layout.starts[-1]
    node = leaves
    for _ in range(hops):
        # An ancestor is at level 1 or below, so every hop lands on a slot.
        node = layout.parent_refs(node) - layout.roots
    claimed: dict[int, int] = {}
    for leaf, ancestor in zip(leaves.to_list(), node.to_list(), strict=True):
        if ancestor not in claimed:
            claimed[ancestor] = layout.roots + leaf
            if len(claimed) == count:
                return claimed
    raise GenerationError(
        f"Could not build {count} non-overlapping cycle(s): only "
        f"{len(claimed)} found a chain of their own. Generate more rows, or "
        "ask for fewer cycles."
    )


def apply_windowed_hierarchy(
    df: pl.DataFrame,
    columns: dict[str, ColSpec],
    hierarchy: Hierarchy,
    seed: int,
    *,
    frame_rows: int,
    row_offset: int = 0,
    cycles: int = 0,
    self_references: int = 0,
) -> pl.DataFrame:
    """Rewrites the child and parent columns of rows `[row_offset, row_offset
    + df.height)` as the forest of the declared depth that a frame of
    `frame_rows` rows holds there.

    Each row's place in the forest -- its slot, its parent, both references
    -- is drawn from its position and `seed` alone, so any window is the
    whole frame's slice, and the shape is `apply_hierarchy`'s: the same
    level sizes, a parent picked uniformly in the level above, one chain of
    exactly `max_depth`. Damage is a function of the seed too, so every
    window applies the part of it that lands in its rows.
    """
    m = df.height
    if m == 0:
        return df
    n = frame_rows
    child, parent = hierarchy.child, hierarchy.parent
    layout = _Layout(hierarchy, n, seed)
    pool = _windowed_pool_spec(columns[child])
    pool_seed = pass_seed(seed, "pool")
    # The whole frame's references must fit the child column's space, not
    # only this window's: the last one is drawn first, so the refusal is the
    # same whichever window asks.
    unique_at(
        child, pool, pool_seed, pl.Series([layout.roots + n - 1], dtype=pl.UInt64)
    )

    rows = pl.int_range(row_offset, row_offset + m, dtype=pl.UInt64, eager=True)
    slots = permuted_at(n, layout.rows_key, rows)
    parents = layout.parent_refs(slots)
    children = slots + layout.roots

    if cycles:
        edits = _windowed_cycle_edits(layout, cycles, seed)
        keys = pl.Series(list(edits), dtype=pl.UInt64)
        hit = slots.is_in(keys)
        if hit.any():
            values = pl.Series(list(edits.values()), dtype=pl.UInt64)
            repointed = slots.filter(hit).replace_strict(keys, values)
            parents = parents.scatter(hit.arg_true(), repointed)

    if self_references:
        if self_references > n:
            raise GenerationError(
                f"Asked for {self_references} self-reference(s) in a hierarchy "
                f"of {n} row(s)."
            )
        # The frame's first rows hold arbitrary slots, as the whole layout's
        # leading rows after its shuffle do.
        hit = rows < self_references
        if hit.any():
            parents = parents.scatter(hit.arg_true(), children.filter(hit))

    refs = unique_at(child, pool, pool_seed, pl.concat([children, parents]))
    return df.with_columns(
        refs.slice(0, m).alias(child), refs.slice(m, m).alias(parent)
    )
