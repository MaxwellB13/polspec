from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from polspec.errors import GenerationError, SpecError
from polspec.spec import ColSpec

if TYPE_CHECKING:
    from polspec.framespec import FrameSpec
    from polspec.tablespec import TableSpec


def _default_fk_name(columns: tuple[str, ...], references: str) -> str:
    return f"fk_{'_'.join(columns)}__{references}"


@dataclass(eq=False, frozen=True, slots=True)
class ForeignKey:
    """Declares referential integrity: one or more columns must only contain
    values that exist in another FrameSpec's (or this same FrameSpec's) columns.

    Parameters
    ----------
    columns : str | Sequence[str]
        The local column(s) that must reference existing parent values.
    references : type[FrameSpec] | TableSpec | str
        The spec this key references -- a `FrameSpec` subclass, a
        `TableSpec`, or a spec's *name* -- or the literal string "self" for a
        self-referencing key (an `employee.manager_id` pointing back at
        `employee.id`). "self" always resolves to whichever spec the key ends
        up declared or inherited on, not the class it was first written in.

        After construction `references` is always a string: the target's
        name. When a spec object was given, it is kept as `target`, so its
        columns can be checked at declaration; a bare name has no `target`
        until a registry resolves it.
    ref_columns : str | Sequence[str] | None, optional
        The referenced column(s) on the target, in the same order as
        `columns`. Defaults to `columns` (same names on both sides).
    name : str | None, optional
        A human-readable identifier. Defaults to a name derived from the
        columns and target.

    Rows where any of `columns` is null are exempt (standard FK semantics --
    a null foreign key means "no reference", not "an invalid one").

    Examples
    --------
    >>> class OrderSpec(FrameSpec):
    ...     customer_id = ColSpec(pl.Int64)
    ...     __foreign_keys__ = [
    ...         ForeignKey("customer_id", references=CustomerSpec, ref_columns="id"),
    ...     ]
    >>> class EmployeeSpec(FrameSpec):
    ...     id = ColSpec(pl.Int64, unique=True)
    ...     manager_id = ColSpec(pl.Int64, nullable=True)
    ...     __foreign_keys__ = [
    ...         ForeignKey("manager_id", references="self", ref_columns="id"),
    ...     ]
    """

    columns: str | Sequence[str]
    references: type[FrameSpec] | TableSpec | str
    ref_columns: str | Sequence[str] | None = None
    name: str | None = None
    target: TableSpec | None = None

    def __post_init__(self) -> None:
        cols = (self.columns,) if isinstance(self.columns, str) else tuple(self.columns)
        if not cols:
            raise SpecError("ForeignKey.columns must not be empty")
        object.__setattr__(self, "columns", cols)

        if self.ref_columns is None:
            ref_cols = cols
        elif isinstance(self.ref_columns, str):
            ref_cols = (self.ref_columns,)
        else:
            ref_cols = tuple(self.ref_columns)
        if len(ref_cols) != len(cols):
            raise SpecError(
                f"ForeignKey.ref_columns ({ref_cols}) must have the same length as "
                f"columns ({cols})"
            )
        object.__setattr__(self, "ref_columns", ref_cols)

        from polspec.tablespec import TableSpec  # local: tablespec imports this module

        ref = self.references
        target: TableSpec | None = self.target
        if isinstance(ref, TableSpec):
            target, ref = ref, ref.name
        elif isinstance(ref, type) and isinstance(
            getattr(ref, "spec", None), TableSpec
        ):
            target, ref = ref.spec, ref.spec.name
        elif not isinstance(ref, str) or not ref:
            raise SpecError(
                "ForeignKey.references must be a FrameSpec subclass, a TableSpec, "
                f"a spec name, or the literal string 'self', got {self.references!r}"
            )
        if ref == "self":
            target = None
        object.__setattr__(self, "references", ref)
        object.__setattr__(self, "target", target)

        if self.name is None:
            object.__setattr__(self, "name", _default_fk_name(cols, ref))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ForeignKey):
            return False
        return (
            self.name == other.name
            and self.columns == other.columns
            and self.ref_columns == other.ref_columns
            and self.references == other.references
        )

    def __hash__(self) -> int:
        return hash((self.name, self.columns, self.ref_columns, self.references))


def _apply_foreign_keys(
    df: pl.DataFrame,
    columns: dict[str, ColSpec],
    foreign_keys: Sequence[tuple[ForeignKey, pl.DataFrame | None]],
    seed: int | None,
) -> pl.DataFrame:
    """Overwrites non-null ForeignKey column values with values actually
    sampled from the referenced parent frame's key columns, so generated
    data satisfies referential integrity by construction.

    Only FKs paired with an actual parent DataFrame are touched -- the
    second tuple element is None for any FK generation wasn't given data
    for (e.g. a cross-spec reference the caller didn't pass via
    `FrameSpec.generate(references=...)`), and that column is left exactly
    as its own ColSpec freely generated, same as before this existed.

    Already-null local values are left untouched (they already reflect the
    column's declared null_probability); only non-null values are replaced.
    Composite keys are sampled as one joint pick per row from the parent, so
    multi-column keys stay internally consistent with each other. A
    single-column key whose ColSpec is `unique=True` is sampled without
    replacement when the parent has enough distinct rows to cover every row
    of `df` (a clean one-to-one relationship); otherwise -- or for composite
    keys -- sampling is with replacement, the common many-to-one case.
    """
    if df.height == 0 or not foreign_keys:
        return df
    rng = random.Random(seed)
    n = df.height
    exprs: list[pl.Expr] = []

    for fk, parent_df in foreign_keys:
        if parent_df is None:
            continue
        local_cols = list(fk.columns)
        ref_cols = list(fk.ref_columns)

        parent_keys = (
            parent_df.select(ref_cols).drop_nulls().unique(maintain_order=True)
        )
        if parent_keys.height == 0:
            raise GenerationError(
                f"ForeignKey '{fk.name}' cannot generate values: the referenced "
                f"parent has no non-null rows for columns {ref_cols}"
            )

        wants_unique = len(local_cols) == 1 and columns[local_cols[0]].unique
        with_replacement = not (wants_unique and parent_keys.height >= n)
        sampled_rows = parent_keys.sample(
            n=n, with_replacement=with_replacement, seed=rng.randrange(2**63)
        )

        for local_col, ref_col in zip(local_cols, ref_cols, strict=True):
            sampled_col = sampled_rows[ref_col].cast(df.schema[local_col])
            exprs.append(
                pl.when(pl.col(local_col).is_not_null())
                .then(pl.lit(sampled_col))
                .otherwise(pl.col(local_col))
                .alias(local_col)
            )

    return df.with_columns(exprs) if exprs else df
