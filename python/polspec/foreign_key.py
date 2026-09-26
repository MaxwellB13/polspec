"""`ForeignKey` -- referential integrity between specs, declared.

The pass that fills a key's column from its parent is
`polspec.generation.passes.foreign_keys`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from polspec.constants import FILLED_IN
from polspec.errors import SpecError

if TYPE_CHECKING:
    from polspec.framespec import FrameSpec
    from polspec.tablespec import TableSpec


def default_fk_name(columns: tuple[str, ...], references: str) -> str:
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

    # Annotated with what a constructed ForeignKey holds: tuples of names, and
    # the referenced spec's *name* whatever the constructor was handed. The
    # accepted forms are spelled out in the `__init__` below, which exists
    # only for type checkers.
    columns: tuple[str, ...]
    references: str
    ref_columns: tuple[str, ...] = FILLED_IN
    name: str = FILLED_IN
    target: TableSpec | None = None

    if TYPE_CHECKING:

        def __init__(
            self,
            columns: str | Sequence[str],
            references: type[FrameSpec] | TableSpec | str,
            ref_columns: str | Sequence[str] | None = None,
            name: str | None = None,
            target: TableSpec | None = None,
        ) -> None: ...

    def __post_init__(self) -> None:
        given_cols: Any = self.columns
        cols = (given_cols,) if isinstance(given_cols, str) else tuple(given_cols)
        if not cols:
            raise SpecError("ForeignKey.columns must not be empty")
        object.__setattr__(self, "columns", cols)

        given_ref_cols: Any = self.ref_columns
        if given_ref_cols is None:
            ref_cols = cols
        elif isinstance(given_ref_cols, str):
            ref_cols = (given_ref_cols,)
        else:
            ref_cols = tuple(given_ref_cols)
        if len(ref_cols) != len(cols):
            raise SpecError(
                f"ForeignKey.ref_columns ({ref_cols}) must have the same length as "
                f"columns ({cols})"
            )
        object.__setattr__(self, "ref_columns", ref_cols)

        from polspec.tablespec import TableSpec  # local: tablespec imports this module

        ref: Any = self.references
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

        name: Any = self.name
        if name is None:
            object.__setattr__(self, "name", default_fk_name(cols, ref))

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
