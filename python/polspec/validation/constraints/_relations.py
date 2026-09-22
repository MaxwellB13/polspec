"""Claims about rows in relation to other rows: foreign keys, which need
their own anti-join rather than a share of the single pass, and a
`Hierarchy`, which is walked.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec.validation.report import Finding, FindingCode

if TYPE_CHECKING:
    from polspec.foreign_key import ForeignKey
    from polspec.hierarchy import Hierarchy

from polspec.validation.constraints._base import MAX_SAMPLES, _Constraint

# ---------------------------------------------------------------------------
# Foreign keys -- each needs its own join, so they cannot share the pass above
# ---------------------------------------------------------------------------


def _foreign_key_findings(
    lf: pl.LazyFrame,
    schema_name: str,
    foreign_keys: Sequence[tuple[ForeignKey, pl.LazyFrame | None]],
    df_col_names: Sequence[str],
    collect_kwargs: dict[str, Any],
) -> list[Finding]:
    findings: list[Finding] = []
    pending: list[tuple[ForeignKey, pl.LazyFrame | None, pl.LazyFrame, Any]] = []
    local_schema = lf.collect_schema()

    for fk, target_lf in foreign_keys:
        local_cols = list(fk.columns)
        ref_cols = list(fk.ref_columns)
        if not all(c in df_col_names for c in local_cols):
            continue  # already reported through missing_cols handling

        parent_lf = target_lf if target_lf is not None else lf
        parent_schema = parent_lf.collect_schema()
        parent_names = parent_schema.names()
        missing_ref = [c for c in ref_cols if c not in parent_names]
        if missing_ref:
            findings.append(
                Finding(
                    code="foreign_key",
                    key=f"fk:{fk.name}",
                    message=(
                        f"ForeignKey '{fk.name}' on {schema_name!r} references columns "
                        f"{missing_ref} not present in the referenced DataFrame"
                    ),
                    columns=tuple(local_cols),
                    details={
                        "target": fk.references,
                        "missing_ref_columns": missing_ref,
                    },
                )
            )
            continue

        key_expr = (
            pl.col(local_cols[0]) if len(local_cols) == 1 else pl.struct(local_cols)
        )
        present = (
            pl.col(local_cols[0]).is_not_null()
            if len(local_cols) == 1
            else pl.all_horizontal([pl.col(c).is_not_null() for c in local_cols])
        )
        # A key may legitimately span dtypes -- a String column referencing
        # an Enum primary key, which declaration allows and generation
        # handles -- and a join across the two would raise instead. Casting
        # the *parent's* keys to the local dtype settles it without touching
        # the frame under validation, so `rows()` returns it as it was. A
        # parent value the local dtype cannot hold becomes null and matches
        # nothing, which is right: the column could never have held it.
        parent_keys = parent_lf.select(
            [
                pl.col(ref_col).cast(local_schema[local_col], strict=False)
                if parent_schema[ref_col] != local_schema[local_col]
                else pl.col(ref_col)
                for local_col, ref_col in zip(local_cols, ref_cols, strict=True)
            ]
        ).unique()

        def orphans_of(
            frame: pl.LazyFrame, _p=present, _k=parent_keys, _l=local_cols, _r=ref_cols
        ) -> pl.LazyFrame:
            return frame.filter(_p).join(_k, left_on=_l, right_on=_r, how="anti")

        stats = orphans_of(lf.select(local_cols)).select(
            pl.len().alias("cnt"),
            key_expr.unique(maintain_order=True)
            .head(MAX_SAMPLES)
            .implode()
            .alias("samples"),
        )
        pending.append((fk, target_lf, stats, orphans_of))

    if pending:
        results = pl.collect_all([p[2] for p in pending], **collect_kwargs)
        for (fk, target_lf, _, orphans_of), stats in zip(pending, results, strict=True):
            count = stats["cnt"][0]
            if not count:
                continue
            raw_samples = stats["samples"][0]
            samples = list(raw_samples) if raw_samples is not None else []
            target_label = "self" if target_lf is None else fk.references
            local_cols = list(fk.columns)
            ref_cols = list(fk.ref_columns)
            findings.append(
                Finding(
                    code="foreign_key",
                    key=f"fk:{fk.name}",
                    message=(
                        f"ForeignKey '{fk.name}' violated ({local_cols} -> "
                        f"{target_label}.{ref_cols}): found {count} row(s) with no "
                        f"matching parent record. Violating samples: {samples}"
                    ),
                    columns=tuple(local_cols),
                    count=int(count),
                    samples=tuple(samples),
                    details={"target": target_label, "ref_columns": ref_cols},
                    _locate=orphans_of,
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Hierarchies -- pointer-chasing, so like foreign keys they run their own joins
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class _SingleParent(_Constraint):
    """Every reference points at one parent, which is what makes the walk
    terminate at a single ultimate parent."""

    column: str
    code: FindingCode = "hierarchy_multi_parent"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': the hierarchy gives each reference one "
            f"parent, but {count} row(s) repeat a reference that already has "
            f"one. Repeated samples: {samples}"
        )


def _hierarchy_constraints(
    hierarchy: Hierarchy, df_col_names: Sequence[str]
) -> list[_Constraint]:
    """The part of a hierarchy that fits the single aggregation pass."""
    if hierarchy.child not in df_col_names:
        return []
    column = pl.col(hierarchy.child)
    return [
        _SingleParent(
            key=f"{hierarchy.child}__single_parent",
            mask=column.is_not_null() & column.is_duplicated(),
            sample_expr=column,
            column=hierarchy.child,
        )
    ]


def _step(walk: pl.DataFrame, by: pl.DataFrame) -> pl.DataFrame:
    """`walk` advanced one hop along `by`, dropping a chain as it ends.

    One column pair throughout: `node` is where the walk started and `anc`
    where it has got to.

    Eager, and deliberately so. Advancing a *lazy* frame along itself nests
    that frame's own plan inside itself, so the doubling below would describe a
    query with two-to-the-rounds join nodes and spend half a minute planning a
    walk over fifty thousand rows. Collecting each round keeps the work
    proportional to the rows still walking, which is the point of the
    algorithm.
    """
    return walk.join(
        by.rename({"node": "_next", "anc": "_anc"}),
        left_on="anc",
        right_on="_next",
        how="inner",
    ).select("node", pl.col("_anc").alias("anc"))


def _deeper_than(edges: pl.DataFrame, hops: int) -> pl.Series:
    """The references whose chain of parents is longer than `hops` edges."""
    walk = edges
    for _ in range(hops):
        if walk.is_empty():
            break
        walk = _step(walk, edges)
    return walk["node"].unique()


def _endless(edges: pl.DataFrame, rounds: int) -> pl.Series:
    """The references whose chain never reaches an ultimate parent.

    Pointer doubling: each round carries `anc` twice as far up, so `rounds`
    of them cover a chain of two-to-the-rounds edges. A chain that reaches a
    root drops out along the way, and whatever outlasts any chain the frame
    could hold is exactly what loops -- a reference on a cycle, or one hanging
    below one.
    """
    ptr = edges
    for _ in range(rounds):
        if ptr.is_empty():
            break
        ptr = _step(ptr, ptr)
    return ptr["node"].unique()


def _hierarchy_findings(
    lf: pl.LazyFrame,
    schema_name: str,
    hierarchy: Hierarchy,
    df_col_names: Sequence[str],
    collect_kwargs: dict[str, Any],
) -> list[Finding]:
    """Cycles and over-deep chains, each found by a bounded walk.

    Bounded is the point. The data this runs against is data someone generated
    *in order* to contain cycles, so a validator that walks until it reaches a
    root is a validator that hangs on its own test fixtures. Depth costs
    `max_depth` joins; the cycle check costs a logarithmic number, because
    pointer doubling covers a chain of length `n` in `log2(n)` steps.
    """
    child, parent = hierarchy.child, hierarchy.parent
    if not all(c in df_col_names for c in (child, parent)):
        return []  # reported through missing_cols instead

    edges = (
        lf.select(pl.col(child).alias("node"), pl.col(parent).alias("anc"))
        .filter(pl.col("node").is_not_null() & pl.col("anc").is_not_null())
        .collect(**collect_kwargs)
    )
    if edges.is_empty():
        return []
    rounds = max(1, math.ceil(math.log2(max(edges.height, 2))) + 1)

    endless = _endless(edges, rounds)
    deep = _deeper_than(edges, hierarchy.max_depth)
    # A reference in a cycle also outruns any depth, so it is reported once,
    # as the cycle it is. `implode` because comparing two Series of one dtype
    # with `is_in` is ambiguous and deprecated: the right-hand side has to say
    # it is one collection rather than a column of values to match row-wise.
    over_deep = deep.filter(~deep.is_in(endless.implode()))

    findings: list[Finding] = []
    for values, code, describe in (
        (endless, "hierarchy_cycle", _cycle_message),
        (over_deep, "hierarchy_depth", _depth_message),
    ):
        if values.is_empty():
            continue
        offenders = values.to_list()
        mask = pl.col(child).is_in(offenders)
        count = int(lf.select(mask.sum()).collect(**collect_kwargs).item())
        samples = offenders[:MAX_SAMPLES]
        findings.append(
            Finding(
                code=code,  # type: ignore[arg-type]
                key=f"hierarchy:{code}",
                message=describe(schema_name, hierarchy, count, samples),
                columns=(child, parent),
                count=count,
                samples=tuple(samples),
                details={
                    "child": child,
                    "parent": parent,
                    "max_depth": hierarchy.max_depth,
                },
                _locate=lambda frame, _m=mask: frame.filter(_m),
            )
        )
    return findings


def _cycle_message(
    schema_name: str, hierarchy: Hierarchy, count: int, samples: list
) -> str:
    return (
        f"Hierarchy on {schema_name!r} ({hierarchy.child} -> {hierarchy.parent}): "
        f"{count} row(s) never reach an ultimate parent, because their chain of "
        f"parents forms a loop. Walking them without a visited set will not "
        f"terminate. Samples: {samples}"
    )


def _depth_message(
    schema_name: str, hierarchy: Hierarchy, count: int, samples: list
) -> str:
    return (
        f"Hierarchy on {schema_name!r} ({hierarchy.child} -> {hierarchy.parent}): "
        f"{count} row(s) sit deeper than the declared max_depth of "
        f"{hierarchy.max_depth}. Samples: {samples}"
    )
