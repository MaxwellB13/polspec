"""`ColRule` -- a column's values, restricted on the rows a condition matches.

The declaration, and the check that a rule's choices are distinct. The pass
that applies it to a generated frame is `polspec.generation.passes.rules`.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from polspec.dtypes import typed_values
from polspec.errors import SpecError
from polspec.expr import Pred


def reject_duplicate_choices(
    choices: tuple, label: str, dtype: pl.DataType | None = None
) -> None:
    """Rejects a choice that appears twice.

    With a `dtype`, "twice" means twice *once cast to that dtype*: `1` and
    `"1"` on a String column are one value. Without one, as written. Weights
    are positional, so a repeated choice would quietly double its share.
    """
    if dtype is None:
        seen: set = set()
        dupes = []
        for c in choices:
            if c in seen:
                dupes.append(c)
            seen.add(c)
        if dupes:
            raise SpecError(f"{label} contains duplicate values {dupes}")
        return
    try:
        typed = typed_values(choices, dtype)
    except Exception:  # noqa: BLE001 - the domain check reports what the dtype cannot hold
        return
    if typed.n_unique() == len(typed):
        return
    dupes = typed.filter(typed.is_duplicated()).unique(maintain_order=True).to_list()
    raise SpecError(
        f"{label} contains values that are the same once cast to {dtype}: {dupes}"
    )


@dataclass(frozen=True, slots=True, eq=False)
class ColRule:
    """Restricts a column's generated values on rows where `when` matches.

    Applied as a pass over the generated frame: rows where `when` matches get
    a value resampled uniformly (or according to `weights`) from `choices`
    instead of whatever was freely generated for them. Multiple rules on the
    *same* column are checked in declaration order, first match wins (like
    SQL CASE/WHEN).

    `when` is evaluated against the frame as it stands when the rule runs,
    and the passes run in dependency order: a rule keyed on a column that
    another rule or a foreign key rewrites sees the rewritten values -- the
    same values validation checks the rule against. Two columns whose rules
    each read what the other writes have no such order and are rejected at
    declaration.

    `when` is a predicate built with `polspec.col()`, not an arbitrary polars
    expression, so that every rule can round-trip through a spec file:

        ColRule(when=col("region") == "UK", choices=["RoyalMail"])
        ColRule(when=col("region").is_in(["US", "EU"]) & (col("qty") > 10), choices=["UPS"])
    """

    when: Pred
    choices: tuple
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.when, Pred):
            hint = ""
            if isinstance(self.when, dict) and "column" in self.when:
                hint = (
                    " The one-column dict was removed: write "
                    f"col({self.when['column']!r}) == ... instead. A spec file "
                    "written by an earlier version still loads -- its "
                    "conditions are converted as the file migrates."
                )
            raise SpecError(
                "ColRule.when must be a predicate built with col(), got "
                f"{type(self.when).__name__}.{hint}"
            )
        if not self.when.root_names():
            raise SpecError(
                f"ColRule.when must reference at least one column, got {self.when!r}"
            )
        if isinstance(self.choices, dict):
            if self.weights is not None:
                raise SpecError(
                    "Cannot specify both a dict for choices and an explicit weights parameter"
                )
            object.__setattr__(
                self, "weights", tuple(float(w) for w in self.choices.values())
            )
            object.__setattr__(self, "choices", tuple(self.choices.keys()))
        else:
            object.__setattr__(self, "choices", tuple(self.choices))
            if self.weights is not None:
                object.__setattr__(
                    self, "weights", tuple(float(w) for w in self.weights)
                )
        if not self.choices:
            raise SpecError("ColRule.choices must not be empty")
        reject_duplicate_choices(self.choices, "ColRule.choices")
        if self.weights is not None:
            if len(self.weights) != len(self.choices):
                raise SpecError(
                    f"Length of weights ({len(self.weights)}) must match length of choices ({len(self.choices)})"
                )
            if any(w < 0 for w in self.weights):
                raise SpecError("Weights must all be non-negative")
            if sum(self.weights) <= 0:
                raise SpecError("Sum of weights must be positive")

    def __eq__(self, other: object) -> bool:
        # `when` is a Pred, whose `==` builds a predicate; compare its data form.
        if not isinstance(other, ColRule):
            return False
        return (
            self.when.equals(other.when)
            and self.choices == other.choices
            and self.weights == other.weights
        )

    def __hash__(self) -> int:
        return hash((self.when, self.choices, self.weights))

    def _expr(self) -> pl.Expr:
        return self.when.to_expr()
