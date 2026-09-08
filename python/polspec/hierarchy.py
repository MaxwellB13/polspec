"""`Hierarchy` -- a self-referencing link table, and the pass that builds one.

A link table is an edge list: one column holds a reference, the other holds
the reference it points at, and both draw on the same namespace. A child
pointing at its parent, and that parent pointing at its own parent, are the
same row shape -- which is why one table holds both.

A self-referencing `ForeignKey` almost does this and deliberately stops short:
it fills the child column with values that exist, which leaves the shape of
the result to chance -- a random functional graph, with rows in cycles and
whatever depth the draw happened to give. That is the right behaviour for a
foreign key, whose whole claim is referential integrity. It is not data anyone
can test a graph walk against.

`Hierarchy` declares the shape instead: one parent per reference, so every row
has exactly one ultimate parent, and a chain from an ultimate parent to the
furthest child of at most `max_depth` hops -- with at least one chain reaching
it, so the boundary is always exercised.

Generation can then be asked for the opposite on purpose. `cycles=` and
`self_references=` on `generate()` inject exactly the faults a graph walk has
to survive, and `validate()` reports them, so a test can assert that its own
resolver and polspec agree about what is broken.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from polspec._ffi import column_plan
from polspec._ffi import generate_dataframe as _generate_dataframe
from polspec.errors import GenerationError, SpecError

if TYPE_CHECKING:
    from polspec.spec import ColSpec

# How far up from a leaf a cycle reaches before closing. Three hops makes a
# loop long enough that a resolver cannot pass it by testing `child == parent`,
# and short enough to fit inside a shallow `max_depth`.
_CYCLE_HOPS = 3

# Candidate leaves tried per cycle before giving up. Cycles must not overlap,
# or the count returned would not be the count asked for.
_CYCLE_ATTEMPTS = 32


@dataclass(frozen=True, slots=True)
class Hierarchy:
    """Declares that two columns of a spec form a parent/child edge list.

    Parameters
    ----------
    child : str
        The column holding the lower reference -- the one doing the pointing.
        Each value appears in exactly one row, which is what gives every
        reference a single ultimate parent.
    parent : str
        The column holding the reference being pointed at.
    max_depth : int
        Hops from an ultimate parent to the furthest child. Generation
        guarantees at least one chain of exactly this length and none longer,
        so a walk that stops early and one that runs away are both caught.
    branching : float | None
        Mean children per reference, which is what decides how many ultimate
        parents `n` rows imply. Mutually exclusive with `roots`; one of the two
        must be given, and `branching=3.0` is the default when neither is.
    roots : int | None
        An exact number of ultimate parents, as an alternative to `branching`.

    Notes
    -----
    A row is an edge, so `generate(n)` produces `n` rows. Ultimate parents have
    no row of their own -- nothing to point at -- so `n` edges over `R` roots
    need `n + R` distinct references, drawn from the child column's own
    declaration.

    Examples
    --------
    >>> class Links(FrameSpec):
    ...     PARENT_REF = ColSpec(pl.String)
    ...     CHILD_REF = ColSpec(pl.String)
    ...     __hierarchy__ = Hierarchy(
    ...         child="CHILD_REF", parent="PARENT_REF", max_depth=5
    ...     )
    """

    child: str
    parent: str
    max_depth: int = 1
    branching: float | None = None
    roots: int | None = None

    def __post_init__(self) -> None:
        for label, value in (("child", self.child), ("parent", self.parent)):
            if not isinstance(value, str) or not value:
                raise SpecError(
                    f"Hierarchy.{label} must be a non-empty column name, got {value!r}"
                )
        if self.child == self.parent:
            raise SpecError(
                f"Hierarchy.child and Hierarchy.parent are both {self.child!r}. "
                "A link table needs two columns: one for the reference, one "
                "for the reference it points at."
            )
        if not isinstance(self.max_depth, int) or self.max_depth < 1:
            raise SpecError(
                f"Hierarchy.max_depth must be an integer of at least 1, got "
                f"{self.max_depth!r}"
            )
        if self.branching is not None and self.roots is not None:
            raise SpecError(
                "Hierarchy takes branching or roots, not both: each implies the "
                "other once the row count is known. Give whichever you actually "
                "mean to hold fixed."
            )
        if self.branching is not None and self.branching < 1.0:
            raise SpecError(
                f"Hierarchy.branching must be at least 1.0, got {self.branching}. "
                "Below one, each level would hold fewer references than the one "
                "above and the tree would never reach max_depth."
            )
        if self.roots is not None and (
            not isinstance(self.roots, int) or self.roots < 1
        ):
            raise SpecError(
                f"Hierarchy.roots must be an integer of at least 1, got {self.roots!r}"
            )

    @property
    def columns(self) -> tuple[str, str]:
        """The two columns this declaration writes, child first."""
        return (self.child, self.parent)

    def _effective_branching(self, n: int) -> float:
        """The mean children per reference this declaration implies for `n` rows.

        Given directly, or solved for when `roots` was fixed instead: the two
        are tied by `n = roots * (b + b**2 + ... + b**max_depth)`, which rises
        monotonically in `b`, so a bisection settles it.
        """
        if self.branching is not None:
            return self.branching
        if self.roots is None:
            return 3.0
        if self.max_depth == 1:
            return n / self.roots
        low, high = 1.0, max(2.0, float(n))
        for _ in range(60):
            mid = (low + high) / 2
            if self.roots * _geometric_sum(mid, self.max_depth) < n:
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
    branching = hierarchy._effective_branching(n)
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
    branching = hierarchy._effective_branching(n)
    return max(1, round(level_sizes[0] / branching))


def _reference_pool(spec: ColSpec, name: str, count: int, seed: int) -> pl.Series:
    """`count` distinct references, drawn from the child column's declaration.

    The column's own dtype, `string_length` and `choices` decide what a
    reference looks like, so a spec that pins its references to a list of real
    identifiers gets those. Anything describing how often a value *recurs* is
    dropped: these are drawn without replacement, so each appears once.
    """
    import dataclasses

    from polspec.engine import _generate_random

    pool_spec = dataclasses.replace(
        spec,
        unique=True,
        nullable=False,
        null_probability=0.0,
        weights=None,
        distribution=None,
        distribution_params=None,
        rules=(),
        validators=(),
    )
    return _generate_random({name: pool_spec}, count, seed)[name]


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


def _apply_hierarchy(
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
            edges[parent]
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
            edges[parent].scatter(rows, edges[child].gather(rows)).alias(parent)
        )

    return df.with_columns(edges[child].alias(child), edges[parent].alias(parent))
