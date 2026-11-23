"""Making a `__unique_together__` group distinct.

Each column of a composite key is generated on its own, so nothing stops two
rows landing on the same combination. This pass makes them distinct, one of
two ways.

**One permutation**, when every member it may rewrite has a finite or
bounded space (`engine.joint_space`): row `r` of the whole frame takes the
`π(r)`-th combination of those members' values, `π` keyed by the frame. The
combinations are distinct over the whole frame, and a batch takes the ones
the whole frame takes there -- distinct across batches too. Each member
then takes every value of its space equally often: its weights and
distribution are not kept. *Since 2.0.*

**Redrawing the repeats**, otherwise -- a float or free-text member, one a
rule writes, a cartesian frame whose coverage the members carry: it finds the
rows repeating a combination an earlier row already used, draws fresh values
for them, and repeats until none are left. Only the repeats move, so the
columns' declared shapes survive everywhere the key was already satisfied;
but it sees one window at a time, so the combinations are distinct within a
batch only.

Rows where any member is null are exempt, matching how the composite key is
validated: a null means "no value", and two rows that both lack one are not
two rows sharing a combination.

Internal: not part of the public API.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import polars as pl

from polspec import frames
from polspec._ffi import composite_indices
from polspec.domain import Domain
from polspec.errors import GenerationError
from polspec.generation.engine import generate_random, joint_space

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from polspec.spec import ColSpec

# Each round resamples only the rows still repeating a combination, so the
# work shrinks geometrically. Fifty rounds clears a domain with as little as
# ~10% headroom over the row count; below that, no amount of redrawing is a
# fix and the error says so.
MAX_ROUNDS = 50


def _domain_size(columns: dict[str, ColSpec], members: Sequence[str]) -> int | None:
    """How many distinct combinations the members can take, when that is
    knowable. None when any member draws from a range rather than a set.
    """
    total = 1
    for name in members:
        spec = columns[name]
        if spec.dtype == pl.Boolean:
            size = 2
        else:
            values = Domain.of(spec).values
            if values is None:
                return None
            size = len(values)
        total *= size
    return total


def joint_spaces(
    columns: dict[str, ColSpec], members: list[str], rewritable: frozenset[str]
) -> dict[str, tuple[int, Callable[[pl.Series], pl.Series]]] | None:
    """Each member's space, when every member has one and the pass may
    rewrite them all. A member it may not rewrite -- a foreign key's --
    leaves the rest needing to differ only where it repeats,
    which a permutation of the rest alone would over-ask; the repeats are
    redrawn instead. Sorted by name: a key is its members, not their order."""
    if any(name not in rewritable for name in members):
        return None
    spaces = {}
    for name in sorted(members):
        space = joint_space(columns[name])
        if space is None:
            return None
        spaces[name] = space
    return spaces or None


def _draw_jointly(
    df: pl.DataFrame,
    members: list[str],
    spaces: dict[str, tuple[int, Callable[[pl.Series], pl.Series]]],
    key_seed: int | None,
    row_offset: int,
) -> pl.DataFrame:
    """The rewritable members drawn as one permutation of their combinations:
    distinct across the whole frame, whatever the window. A member's nulls
    stay where its own draw put them -- a row with a null is exempt -- and a
    foreign-keyed member keeps its key, the rest distinct beside it."""
    names = list(spaces)
    sizes = [spaces[name][0] for name in names]
    combinations = math.prod(sizes)
    needed = row_offset + df.height
    if needed > combinations:
        raise GenerationError(
            f"Composite unique key {members} cannot be satisfied: the columns "
            f"take {combinations} distinct combination(s) between them and "
            f"{needed} row(s) need one. Widen one of the columns, or generate "
            "fewer rows."
        )
    seed = random.getrandbits(64) if key_seed is None else key_seed & (2**64 - 1)
    indices = composite_indices(sizes, seed, row_offset, df.height)
    drawn = []
    for name, index in zip(names, indices, strict=True):
        decode = spaces[name][1]
        values = frames.plain(decode(index))
        column = frames.plain(df[name])
        drawn.append(values.zip_with(column.is_not_null(), column).alias(name))
    return df.with_columns(drawn)


def apply_unique_together(
    df: pl.DataFrame,
    columns: dict[str, ColSpec],
    members: Sequence[str],
    rewritable: frozenset[str],
    seed: int | None,
    *,
    key_seed: int | None = None,
    row_offset: int = 0,
    jointly: bool = True,
) -> pl.DataFrame:
    """Makes the combinations of `members` distinct.

    `rewritable` is the subset of members this pass may redraw -- a
    foreign-keyed member is excluded, since replacing its value would undo
    the referential integrity its key just established. `key_seed` keys the
    permutation, by the whole frame, and `row_offset` is where this window
    starts in it; `seed` the redraws. `jointly=False` -- a cartesian frame --
    keeps the members' values and redraws the repeats.
    """
    members = list(members)
    if df.height == 0:
        return df
    if any(columns[m].unique for m in members):
        return df  # a member distinct on its own makes every combination so
    spaces = joint_spaces(columns, members, rewritable) if jointly else None
    if spaces is not None:
        return _draw_jointly(df, members, spaces, key_seed, row_offset)

    present = pl.all_horizontal([frames.column(m).is_not_null() for m in members])
    key = pl.struct(frames.columns(members))
    # `is_first_distinct` leaves one row per combination alone; every later
    # row using it is a repeat to be moved.
    repeats = present & ~key.is_first_distinct()

    def repeated_rows(frame: pl.DataFrame) -> pl.Series:
        return frame.select(repeats).to_series().arg_true()

    rows = repeated_rows(df)
    if rows.len() == 0:
        return df

    size = _domain_size(columns, members)
    needed = df.select(present.sum()).item()
    if size is not None and needed > size:
        raise GenerationError(
            f"Composite unique key {members} cannot be satisfied: the columns "
            f"take {size} distinct combination(s) between them and {needed} "
            f"row(s) need one. Widen one of the columns, or generate fewer rows."
        )
    if not rewritable:
        raise GenerationError(
            f"Composite unique key {members} is violated on {rows.len()} row(s) "
            "and cannot be repaired: every column in it is foreign-keyed, so "
            "changing one would break its key. Drop a foreign key, or add a "
            "column of the spec's own to the composite key."
        )

    resampled = {name: columns[name] for name in members if name in rewritable}
    rng = random.Random(seed)
    for _ in range(MAX_ROUNDS):
        fresh = generate_random(resampled, rows.len(), rng.randrange(2**63))
        df = df.with_columns(
            [
                frames.plain(df[name]).scatter(rows, fresh[name]).alias(name)
                for name in resampled
            ]
        )
        rows = repeated_rows(df)
        if rows.len() == 0:
            return df

    raise GenerationError(
        f"Composite unique key {members} still repeats a combination on "
        f"{rows.len()} row(s) after {MAX_ROUNDS} rounds of resampling "
        f"{sorted(resampled)}. The columns' combined domain is too close to "
        "the row count to separate them by redrawing -- widen one of them, or "
        "generate fewer rows."
    )
