"""`Hierarchy` -- a self-referencing link table, declared.

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
resolver and polspec agree about what is broken. The pass that builds the
shape, and breaks it, is `polspec.generation.passes.hierarchy`.

Internal: not part of the public API.
"""

from __future__ import annotations

from dataclasses import dataclass

from polspec.errors import SpecError


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
