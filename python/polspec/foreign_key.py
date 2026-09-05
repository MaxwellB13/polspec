from __future__ import annotations

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

    Notes
    -----
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


def _apply_foreign_key(
    df: pl.DataFrame,
    columns: dict[str, ColSpec],
    fk: ForeignKey,
    parent_df: pl.DataFrame,
    seed: int | None,
) -> pl.DataFrame:
    """Overwrites `fk`'s non-null local values with values drawn from `parent_df`,
    so generated data satisfies referential integrity by construction.

    Already-null local values are left untouched (they already reflect the
    column's declared null_probability); only non-null values are replaced.
    Composite keys are sampled as one joint pick per row, so multi-column
    keys stay internally consistent with each other. A single-column key
    whose ColSpec is `unique=True` is sampled without replacement, and a
    parent with fewer distinct rows than `df` refuses rather than repeating
    one: the column promises distinct values and every one of them has to
    come from the parent. Otherwise -- the common many-to-one case, and every
    composite key -- sampling is with replacement.

    `parent_df` for a self-referencing key is the frame as it stands when
    this pass runs, so a key reading a column another key rewrote draws from
    the values that column ended up with.
    """
    if df.height == 0:
        return df
    local_cols = list(fk.columns)
    ref_cols = list(fk.ref_columns)

    parent_keys = parent_df.select(ref_cols).drop_nulls().unique(maintain_order=True)
    if parent_keys.height == 0:
        raise GenerationError(
            f"ForeignKey '{fk.name}' cannot generate values: the referenced "
            f"parent has no non-null rows for columns {ref_cols}"
        )

    wants_unique = len(local_cols) == 1 and columns[local_cols[0]].unique
    if wants_unique and parent_keys.height < df.height:
        raise GenerationError(
            f"ForeignKey '{fk.name}' fills the unique column "
            f"'{local_cols[0]}', but the referenced parent offers only "
            f"{parent_keys.height} distinct value(s) for {df.height} row(s). "
            "Every value has to come from the parent and no two may repeat, "
            "so generate more parent rows, or fewer of these."
        )
    with_replacement = not wants_unique
    sampled_rows = parent_keys.sample(
        n=df.height, with_replacement=with_replacement, seed=seed
    )

    exprs = []
    for local_col, ref_col in zip(local_cols, ref_cols, strict=True):
        sampled_col = sampled_rows[ref_col].cast(df.schema[local_col])
        exprs.append(
            pl.when(pl.col(local_col).is_not_null())
            .then(pl.lit(sampled_col))
            .otherwise(pl.col(local_col))
            .alias(local_col)
        )
    return df.with_columns(exprs)
