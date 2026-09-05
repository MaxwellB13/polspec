"""Making a `__unique_together__` group distinct.

Each column of a composite key is generated on its own, so nothing stops two
rows landing on the same combination. This is the pass that separates them:
it finds the rows repeating a combination an earlier row already used, draws
fresh values for them, and repeats until none are left.

Only the repeats move. The first row to use a combination keeps it, so the
columns' declared shapes -- weights, bounds, distributions -- survive
everywhere the key was already satisfied, which on a roomy domain is almost
everywhere.

Rows where any member is null are exempt, matching how the composite key is
validated: a null means "no value", and two rows that both lack one are not
two rows sharing a combination.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import polars as pl

from polspec.constraints import Domain
from polspec.errors import GenerationError

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def apply_unique_together(
    df: pl.DataFrame,
    columns: dict[str, ColSpec],
    members: Sequence[str],
    rewritable: frozenset[str],
    seed: int | None,
) -> pl.DataFrame:
    """Resamples the rows repeating a combination of `members`.

    `rewritable` is the subset of members this pass may redraw -- a
    foreign-keyed member is excluded, since replacing its value would undo
    the referential integrity its key just established.
    """
    from polspec.engine import _generate_random

    members = list(members)
    if df.height == 0:
        return df

    present = pl.all_horizontal([pl.col(m).is_not_null() for m in members])
    key = pl.struct([pl.col(m) for m in members])
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
        fresh = _generate_random(resampled, rows.len(), rng.randrange(2**63))
        df = df.with_columns(
            [df[name].scatter(rows, fresh[name]) for name in resampled]
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
