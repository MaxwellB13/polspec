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
from collections.abc import Callable

import polars as pl

from polspec.checks import Check, failure_mask
from polspec.errors import GenerationError, SpecError
from polspec.frames import Method
from polspec.tablespec import TableSpec

__all__ = [
    "KEY_ROOM",
    "check_columns",
    "holds_checks",
    "kept_window",
    "refuse_unkeepable",
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


def refuse_unkeepable(
    spec: TableSpec, method: Method, *, sequence_pins: bool = False
) -> None:
    """Refuses `checks=True` where taking rows out would break a claim, or a
    pin. Only for a spec that has checks: with none, there is nothing to
    keep, and the call is the one `checks=False` makes."""
    if not holds_checks(spec):
        return
    why = None
    if method == "cartesian":
        why = (
            "method='cartesian' builds a coverage set, and taking out the rows "
            "a check rejects would leave combinations uncovered"
        )
    elif spec.hierarchy is not None:
        why = (
            "its __hierarchy__ links rows to other rows of the frame, and "
            "taking a row out would leave its children without a parent"
        )
    elif any(fk.references == "self" for fk in spec.foreign_keys):
        why = (
            "a foreign key into its own rows would point at rows that taking "
            "out the failures removes"
        )
    elif sequence_pins:
        why = (
            "fixed= gives a column all n of its values, in order, and taking "
            "rows out would leave them out of step; pin a single value instead"
        )
    if why is not None:
        raise SpecError(f"checks=True cannot keep {spec.name}'s checks: {why}.")


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
    is read instead -- the same rows and more -- by the share that failed.
    """
    failing = [(label, failure_mask(check)) for label, check in _checks(spec)]
    rows = need
    for round_ in range(MAX_ROUNDS):
        try:
            frame = read(start, rows)
        except GenerationError as exc:
            if round_ == 0:
                raise
            raise GenerationError(
                f"checks=True read {rows:,} rows of {spec.name} to keep {need:,} "
                f"that pass its checks, and could not: {exc}"
            ) from exc
        if not failing or need == 0:
            return frame, frame.height
        passes = ~frame.select(
            pl.any_horizontal([mask for _, mask in failing]).alias("fails")
        ).to_series()
        at = passes.arg_true()
        if len(at) >= need:
            used = at[need - 1] + 1
            return frame.filter(passes).head(need), used
        rate = len(at) / frame.height
        if rate < MIN_PASS_RATE:
            break
        rows = math.ceil(need / rate * _MARGIN)
    raise GenerationError(_too_few(spec, need, frame, failing))


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
        f"checks=True could not keep {need:,} rows of {spec.name} that pass "
        f"its checks: of {frame.height:,} rows read, {rejected}. A check that "
        "rejects nearly every row is one generation cannot keep; narrow the "
        "columns it reads (bounds, choices, rules) or pin them with fixed= so "
        "more rows pass."
    )
