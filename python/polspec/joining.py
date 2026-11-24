"""The spec of a join: `TableSpec.join`, the columns `LazyFrame.join`
gives two frames of these specs, each carrying its side's claims.

Internal: not part of the public API.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Literal, cast

import polars as pl

from polspec.errors import SpecError
from polspec.spec import ColSpec
from polspec.tablespec import TableSpec, renamed_check, require_named

JoinHow = Literal["inner", "left", "right", "full", "semi", "anti", "cross"]
_HOWS = ("inner", "left", "right", "full", "semi", "anti", "cross")


def _key_names(value: str | Sequence[str] | None, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    names = tuple(value)
    if not names or not all(isinstance(n, str) for n in names):
        raise SpecError(f"join {label}= must be a column name or a list of them")
    return names


def _key_is_unique(spec: TableSpec, keys: tuple[str, ...]) -> bool:
    """Whether `keys` hold no value twice in a frame of `spec`, by its
    declaration: one unique column, or a declared composite key -- or a
    superset of either, which can repeat no more than it."""
    key_set = set(keys)
    return any(spec.columns[k].unique for k in keys) or any(
        set(group) <= key_set for group in spec.unique_together
    )


def joined(
    left: TableSpec,
    right: TableSpec,
    *,
    on: str | Sequence[str] | None,
    left_on: str | Sequence[str] | None,
    right_on: str | Sequence[str] | None,
    how: str,
    suffix: str,
    coalesce: bool | None,
    name: str | None,
) -> TableSpec:
    """`TableSpec.join`: see there for what survives."""
    lkeys, rkeys = _join_keys(left, right, on, left_on, right_on, how)

    if how in ("semi", "anti"):
        # Fewer rows, so no count; and Polars promises no join keeps its
        # rows' order unless asked, so no order either.
        kept = dataclasses.replace(left, rows=None, order_by=(), descending=())
        return kept if name is None else kept.with_name(name)

    layout = _join_layout(
        left,
        right,
        lkeys,
        rkeys,
        keyed_on=on is not None,
        how=how,
        suffix=suffix,
        coalesce=coalesce,
    )
    columns, sources = _joined_columns(
        left, right, lkeys, rkeys, layout, how=how, suffix=suffix
    )
    return TableSpec(
        name or f"{left.name}_{right.name}", _with_validators_renamed(columns, sources)
    )


def _join_keys(
    left: TableSpec,
    right: TableSpec,
    on: str | Sequence[str] | None,
    left_on: str | Sequence[str] | None,
    right_on: str | Sequence[str] | None,
    how: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Each side's key columns, checked: a known `how`, keys given one way,
    of the same length, naming columns that exist and share a dtype."""
    if how not in _HOWS:
        raise SpecError(f"join how= must be one of {', '.join(_HOWS)}; got {how!r}")
    if how == "cross":
        if on is not None or left_on is not None or right_on is not None:
            raise SpecError("join how='cross' takes no keys")
        lkeys = rkeys = ()
    elif on is not None:
        if left_on is not None or right_on is not None:
            raise SpecError("join takes on=, or left_on= and right_on=, not both")
        lkeys = rkeys = _key_names(on, "on")
    else:
        lkeys, rkeys = _key_names(left_on, "left_on"), _key_names(right_on, "right_on")
        if not lkeys or len(lkeys) != len(rkeys):
            raise SpecError(
                "join needs on=, or left_on= and right_on= of the same length"
            )
    require_named(left, lkeys, "join on")
    require_named(right, rkeys, "join on")
    for lk, rk in zip(lkeys, rkeys, strict=True):
        ldt, rdt = left.columns[lk].dtype, right.columns[rk].dtype
        if ldt != rdt:
            raise SpecError(
                f"Cannot join {left.name}.{lk} ({ldt!r}) to {right.name}.{rk} "
                f"({rdt!r}): join keys must share a dtype. Cast one side's "
                "column first, with with_columns."
            )
    return lkeys, rkeys


def _join_layout(
    left: TableSpec,
    right: TableSpec,
    lkeys: tuple[str, ...],
    rkeys: tuple[str, ...],
    *,
    keyed_on: bool,
    how: str,
    suffix: str,
    coalesce: bool | None,
) -> dict[str, str]:
    """Each column of the joined frame, in order, and where it came from:
    `"key"`, or `"L:<column>"` / `"R:<column>"`. Read from the installed
    Polars, so a suffixed clash or a coalesced key is named as it will be."""
    paired = not keyed_on and bool(lkeys)

    def run(a: pl.LazyFrame, b: pl.LazyFrame) -> pl.LazyFrame:
        return a.join(
            b,
            on=list(lkeys) if keyed_on else None,
            left_on=list(lkeys) if paired else None,
            right_on=list(rkeys) if paired else None,
            how=cast("Any", how),
            suffix=suffix,
            coalesce=coalesce,
        )

    # The names and their order: an empty join of the two schemas. Then
    # where each came from: a one-row join in which every column is text
    # naming its own origin, and every key one value.
    names = (
        run(pl.LazyFrame(schema=left.schema()), pl.LazyFrame(schema=right.schema()))
        .collect_schema()
        .names()
    )

    def probe(spec: TableSpec, side: str, own_keys: tuple[str, ...]) -> pl.LazyFrame:
        return pl.LazyFrame(
            {c: ["key" if c in own_keys else f"{side}:{c}"] for c in spec.columns}
        )

    row = (
        run(probe(left, "L", lkeys), probe(right, "R", rkeys))
        .collect()
        .row(0, named=True)
    )
    assert list(row) == names  # noqa: S101 - the two joins name alike
    return row


def _joined_columns(
    left: TableSpec,
    right: TableSpec,
    lkeys: tuple[str, ...],
    rkeys: tuple[str, ...],
    layout: dict[str, str],
    *,
    how: str,
    suffix: str,
) -> tuple[dict[str, ColSpec], dict[str, str]]:
    """What each joined column keeps of the column it came from, and that
    column's name. A side whose rows may go unmatched makes its columns
    nullable; a column stays unique only if the other side's key is."""
    left_open = how in ("right", "full")
    right_open = how in ("left", "full")
    cross = how == "cross"
    left_unique = not cross and _key_is_unique(right, rkeys)
    right_unique = not cross and _key_is_unique(left, lkeys)

    def carried(spec: ColSpec, *, nullable: bool, unique: bool) -> ColSpec:
        return dataclasses.replace(
            spec,
            col_name=None,
            seed_name=None,
            optional=False,
            rules=(),
            # What it was derived from may not survive the join, or be renamed
            # by its suffix; the column keeps the values, not the claim.
            derive=None,
            nullable=spec.nullable or nullable,
            unique=spec.unique and unique,
        )

    # A coalesced key is one column per pair: the right side's in a right
    # join, named as it names it, and the left side's otherwise. Uncoalesced,
    # the left key keeps its name and the right key its own, or the suffix.
    key_outs = [out for out, origin in layout.items() if origin == "key"]
    coalesced = len(key_outs) == len(lkeys)
    seen_left: set[str] = set()

    def key_source(out: str) -> tuple[str, int]:
        if coalesced:
            if how == "right":
                return "R", rkeys.index(out)
            return "L", lkeys.index(out)
        if out in lkeys and out not in seen_left:
            seen_left.add(out)
            return "L", lkeys.index(out)
        bare = out if out in rkeys else out.removesuffix(suffix)
        return "R", rkeys.index(bare)

    columns: dict[str, ColSpec] = {}
    sources: dict[str, str] = {}
    for out, origin in layout.items():
        if origin == "key":
            side, index = key_source(out)
            lspec = left.columns[lkeys[index]]
            rspec = right.columns[rkeys[index]]
            if how == "full" and coalesced:
                # Either side's value: a claim holds only if both make it.
                same = dataclasses.replace(lspec, description=None) == (
                    dataclasses.replace(rspec, description=None)
                )
                base = (
                    lspec
                    if same
                    else ColSpec(lspec.dtype, nullable=lspec.nullable or rspec.nullable)
                )
                columns[out] = carried(
                    base, nullable=False, unique=left_unique and right_unique
                )
                sources[out] = lkeys[index]
            elif side == "L":
                columns[out] = carried(lspec, nullable=left_open, unique=left_unique)
                sources[out] = lkeys[index]
            else:
                columns[out] = carried(rspec, nullable=right_open, unique=right_unique)
                sources[out] = rkeys[index]
            continue
        side, source = origin.split(":", 1)
        if side == "L":
            columns[out] = carried(
                left.columns[source], nullable=left_open, unique=left_unique
            )
        else:
            columns[out] = carried(
                right.columns[source], nullable=right_open, unique=right_unique
            )
        sources[out] = source
    return columns, sources


def _with_validators_renamed(
    columns: dict[str, ColSpec], sources: dict[str, str]
) -> dict[str, ColSpec]:
    """A validator names its column; one whose column was renamed by the
    suffix is renamed with it, as `rename` would."""
    for out, spec in list(columns.items()):
        source = sources[out]
        if source != out and spec.validators:
            columns[out] = dataclasses.replace(
                spec,
                validators=tuple(
                    renamed_check(
                        v, {source: out}, f"validator {v.name!r} on {source!r}"
                    )
                    for v in spec.validators
                ),
            )
    return columns
