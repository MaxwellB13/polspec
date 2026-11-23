"""Generation that keeps a spec's checks: the first rows that pass them.

A check -- a `__checks__` entry or a column's validator -- is an arbitrary
Polars predicate, so nothing can draw rows that satisfy one. What can be
done is to read rows of the frame the seed describes and keep the ones
every check passes, reading further into the frame until there are enough.

Every row kept is a row of that one frame, so every claim a subset of rows
keeps still holds: a `unique` column stays unique -- across a stream's
batches too, since each batch reads on from where the last stopped -- and
a foreign key still points at its parent. A batch is read as one window, so
a composite key stays distinct within it, as it always is. A claim made
*between* rows does not survive having rows taken out, so a hierarchy and
a key into the spec's own rows are refused, as is a cartesian frame, whose
coverage set is the point.

When every row passes, the rows kept are the rows `checks=False` gives.

Internal: not part of the public API.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import polars as pl

from polspec.checks import Check, failure_mask
from polspec.errors import GenerationError, SpecError
from polspec.frames import Method
from polspec.generation.passes.unique_together import joint_spaces
from polspec.pass_order import rewritable_members
from polspec.tablespec import TableSpec

__all__ = [
    "KEY_ROOM",
    "ChecksTooStrict",
    "check_columns",
    "holds_checks",
    "kept_or_warned",
    "kept_window",
    "reads_in_pieces",
    "resolve_checks",
    "why_unkeepable",
]

# Times a window is read, the first included: a check most rows pass needs
# one or two; one that needs more is rejecting nearly everything.
MAX_ROUNDS = 5

# The smallest share of rows that must pass for a larger window to be
# read: below it, a window would be more than fifty times the rows asked
# for, to keep a check that is all but impossible.
MIN_PASS_RATE = 0.02

# How much more a window reads than the pass rate says, so one re-read
# usually suffices.
_MARGIN = 1.2

# Rows of the frame read per row kept that an open unique integer key is
# given room for, when its default range would not hold `n` rows anyway
# and so is sized to exactly `n`, with none to spare for a rejected row.
KEY_ROOM = 128


def _checks(spec: TableSpec) -> list[tuple[str, Check]]:
    """Every check the frame is held to, as validation names it."""
    found = [(f"check {c.name!r}", c) for c in spec.checks]
    for name, column in spec.columns.items():
        found += [(f"{name!r}'s validator {v.name!r}", v) for v in column.validators]
    return found


def holds_checks(spec: TableSpec) -> bool:
    """Whether `spec` declares anything `checks=True` would keep."""
    return bool(spec.checks) or any(c.validators for c in spec.columns.values())


def check_columns(spec: TableSpec) -> frozenset[str]:
    """Every column a check reads, or a validator is declared on: what a
    projection must still generate, since which rows are kept depends on
    them."""
    names = {name for name, column in spec.columns.items() if column.validators}
    for _, check in _checks(spec):
        names.update(check.expr.meta.root_names())
    return frozenset(names)


class ChecksTooStrict(GenerationError):
    """Checks that reject nearly every row a frame holds: generation cannot
    keep them. `checks=True` raises it; the default generates without them,
    and says so."""


def why_unkeepable(
    spec: TableSpec, method: Method, *, sequence_pins: bool = False
) -> str | None:
    """Why taking out the rows a check rejects would break a claim, or a pin
    -- or None when the checks can be kept."""
    if method == "cartesian":
        return (
            "method='cartesian' builds a coverage set, and taking out the rows "
            "a check rejects would leave combinations uncovered"
        )
    if spec.hierarchy is not None:
        return (
            "its __hierarchy__ links rows to other rows of the frame, and "
            "taking a row out would leave its children without a parent"
        )
    if any(fk.references == "self" for fk in spec.foreign_keys):
        return (
            "a foreign key into its own rows would point at rows that taking "
            "out the failures removes"
        )
    if sequence_pins:
        return (
            "fixed= gives a column all n of its values, in order, and taking "
            "rows out would leave them out of step; pin a single value instead"
        )
    return None


def resolve_checks(
    spec: TableSpec,
    checks: bool | None,
    method: Method,
    *,
    sequence_pins: bool = False,
) -> bool:
    """Whether a call keeps `spec`'s checks in what it generates.

    `checks=True` keeps them, and refuses a spec it cannot keep them for;
    `checks=False` does not keep them. The default, `None`, keeps them
    where it can and, where it cannot, generates as `checks=False` does and
    says so once, naming why. A spec with no checks keeps nothing either
    way: there is nothing to keep.
    """
    if checks is False or not holds_checks(spec):
        return False
    why = why_unkeepable(spec, method, sequence_pins=sequence_pins)
    if why is None:
        return True
    if checks:
        raise SpecError(f"checks=True cannot keep {spec.name}'s checks: {why}.")
    warnings.warn(
        f"{spec.name}'s checks are not kept in what is generated: {why}. The "
        "rows may fail them; pass checks=False to say so and silence this.",
        stacklevel=3,
    )
    return False


def kept_or_warned(
    spec: TableSpec,
    need: int,
    start: int,
    read: Callable[[int, int], pl.DataFrame],
    *,
    strict: bool,
) -> tuple[pl.DataFrame, int] | None:
    """`kept_window`, or -- under the default, when the checks reject nearly
    every row -- None, having said once that they are not kept. `strict`
    (`checks=True`) raises instead."""
    try:
        return kept_window(spec, need, start, read)
    except ChecksTooStrict as exc:
        if strict:
            raise
        warnings.warn(
            f"{spec.name}'s checks are not kept in what is generated: {exc} "
            "The rows may fail them; pass checks=False to say so and silence "
            "this, or checks=True to make it an error.",
            stacklevel=4,
        )
        return None


def reads_in_pieces(spec: TableSpec) -> bool:
    """Whether rows `[a, b)` and `[b, c)` of `spec`'s frame, read apart, are
    its rows `[a, c)`: true when every pass is a window onto the frame --
    rules, foreign keys into other tables, a composite key drawn as one
    permutation -- and false for a composite key repaired per window. (A key
    into the spec's own rows and a hierarchy are unkeepable anyway.)"""
    columns = dict(spec.columns)
    return all(
        joint_spaces(columns, list(group), rewritable_members(spec, group)) is not None
        for group in spec.unique_together
        if not any(columns[m].unique for m in group)
    )


def kept_window(
    spec: TableSpec,
    need: int,
    start: int,
    read: Callable[[int, int], pl.DataFrame],
) -> tuple[pl.DataFrame, int]:
    """`need` rows every check of `spec` passes, from the frame's rows from
    `start` on, and how many of the frame's rows they used.

    `read(offset, rows)` gives rows `[offset, offset + rows)` with the
    passes run over them. The window `[start, start + need)` is read first;
    when too few of its rows pass, a longer window from the same `start`
    is read -- the same rows and more -- by the share that failed. When the
    spec `reads_in_pieces`, only the rows past what was read are read and
    appended, so no row is generated twice; otherwise the longer window is
    read whole.
    """
    failing = [(label, failure_mask(check)) for label, check in _checks(spec)]
    if not failing or need == 0:
        frame = read(start, need)
        return frame, frame.height
    fails = pl.any_horizontal([mask for _, mask in failing]).alias("fails")
    in_pieces = reads_in_pieces(spec)
    # What has been read, as pieces in row order, and which of each piece's
    # rows pass; read whole, there is one piece, replaced each round.
    pieces: list[pl.DataFrame] = []
    passing: list[pl.Series] = []
    rows = need
    for round_ in range(MAX_ROUNDS):
        have = sum(piece.height for piece in pieces)
        try:
            if in_pieces and pieces:
                piece = read(start + have, rows - have)
            else:
                pieces, passing = [], []
                piece = read(start, rows)
        except GenerationError as exc:
            if round_ == 0:
                raise
            raise GenerationError(
                f"Generation read {rows:,} rows of {spec.name} to keep {need:,} "
                f"that pass its checks, and could not: {exc}"
            ) from exc
        pieces.append(piece)
        passing.append(~piece.select(fails).to_series())
        kept = sum(int(mask.sum()) for mask in passing)
        read_so_far = have + piece.height if in_pieces else piece.height
        if kept >= need:
            return _first_passing(pieces, passing, need)
        rate = kept / read_so_far
        if rate < MIN_PASS_RATE:
            break
        rows = math.ceil(need / rate * _MARGIN)
    raise ChecksTooStrict(_too_few(spec, need, pl.concat(pieces), failing))


def _first_passing(
    pieces: list[pl.DataFrame], passing: list[pl.Series], need: int
) -> tuple[pl.DataFrame, int]:
    """The first `need` passing rows of `pieces`, read in order, and how
    many rows of the frame they used. Only the passing rows are copied."""
    kept: list[pl.DataFrame] = []
    left, used = need, 0
    for piece, mask in zip(pieces, passing, strict=True):
        at = mask.arg_true()
        if len(at) >= left:
            # Only the rows up to the last one needed are filtered.
            end = int(at[left - 1]) + 1
            kept.append(piece.head(end).filter(mask.head(end)))
            used += end
            break
        kept.append(piece.filter(mask))
        left -= len(at)
        used += piece.height
    return pl.concat(kept, how="vertical"), used


def _too_few(
    spec: TableSpec,
    need: int,
    frame: pl.DataFrame,
    failing: list[tuple[str, pl.Expr]],
) -> str:
    """Which checks rejected how much of the last window."""
    shares = frame.select(
        mask.mean().alias(str(index)) for index, (_, mask) in enumerate(failing)
    ).row(0)
    rejected = ", ".join(
        f"{label} rejected {share:.1%}"
        for (label, _), share in zip(failing, shares, strict=True)
        if share
    )
    return (
        f"Generation could not keep {need:,} rows of {spec.name} that pass "
        f"its checks: of {frame.height:,} rows read, {rejected}. A check that "
        "rejects nearly every row is one generation cannot keep; narrow the "
        "columns it reads (bounds, choices, rules) or pin them with fixed= so "
        "more rows pass."
    )
