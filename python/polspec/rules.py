from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from polspec._ffi import column_plan
from polspec._ffi import generate_dataframe as _generate_dataframe
from polspec.dtypes import _typed_values
from polspec.errors import SpecError
from polspec.expr import Pred, col

if TYPE_CHECKING:
    from polspec.spec import ColSpec

# Condition operations ColRule.when accepts.
_CONDITION_OPS = (
    "equals",
    "not_equals",
    "in",
    "not_in",
    "lt",
    "lte",
    "le",
    "gt",
    "gte",
    "ge",
    "between",
    "is_null",
    "is_not_null",
)


def _validate_condition(condition: dict) -> None:
    if not isinstance(condition, dict) or "column" not in condition:
        raise SpecError(
            "ColRule.when must be a dict like {'column': 'enum_1', 'in': ['A', 'B']} "
            f"(supported condition keys: {', '.join(_CONDITION_OPS)})"
        )
    ops_present = [op for op in _CONDITION_OPS if op in condition]
    if len(ops_present) != 1:
        raise SpecError(
            f"ColRule.when for column {condition['column']!r} must have exactly one of "
            f"{_CONDITION_OPS}, got {ops_present}"
        )
    if "between" in condition:
        b = condition["between"]
        if not (isinstance(b, (list, tuple)) and len(b) == 2 and b[0] <= b[1]):
            raise SpecError(
                f"ColRule.when 'between' condition requires a 2-element sequence [min, max] where min <= max, got {b!r}"
            )
    if "in" in condition and not isinstance(condition["in"], (list, tuple, set)):
        raise SpecError(
            f"ColRule.when 'in' condition requires a collection, got {type(condition['in']).__name__}"
        )
    if "not_in" in condition and not isinstance(
        condition["not_in"], (list, tuple, set)
    ):
        raise SpecError(
            f"ColRule.when 'not_in' condition requires a collection, got {type(condition['not_in']).__name__}"
        )


def _reject_duplicate_choices(
    choices: tuple, label: str, dtype: pl.DataType | None = None
) -> None:
    """Rejects a choice that appears twice.

    With a `dtype`, "twice" means twice *once cast to that dtype*: `1` and
    `"1"` on a String column are one value. Without one, as written. Weights
    are positional, so a repeated choice would quietly double its share.
    """
    if dtype is None:
        seen: set = set()
        dupes = [c for c in choices if c in seen or seen.add(c)]
        if dupes:
            raise SpecError(f"{label} contains duplicate values {dupes}")
        return
    try:
        typed = _typed_values(choices, dtype)
    except Exception:  # noqa: BLE001 - the domain check reports what the dtype cannot hold
        return
    if typed.n_unique() == len(typed):
        return
    dupes = typed.filter(typed.is_duplicated()).unique(maintain_order=True).to_list()
    raise SpecError(
        f"{label} contains values that are the same once cast to {dtype}: {dupes}"
    )


def _condition_to_pred(condition: dict) -> Pred:
    """The predicate a legacy `{"column": ..., <op>: ...}` condition means."""
    _validate_condition(condition)
    column = col(condition["column"])
    if "equals" in condition:
        return column == condition["equals"]
    if "not_equals" in condition:
        return column != condition["not_equals"]
    if "in" in condition:
        return column.is_in(list(condition["in"]))
    if "not_in" in condition:
        return ~column.is_in(list(condition["not_in"]))
    if "lt" in condition:
        return column < condition["lt"]
    if "lte" in condition or "le" in condition:
        return column <= condition.get("lte", condition.get("le"))
    if "gt" in condition:
        return column > condition["gt"]
    if "gte" in condition or "ge" in condition:
        return column >= condition.get("gte", condition.get("ge"))
    if "between" in condition:
        lo, hi = condition["between"]
        return column.is_between(lo, hi)
    if "is_null" in condition:
        return column.is_null() if condition["is_null"] else column.is_not_null()
    if "is_not_null" in condition:
        return column.is_not_null() if condition["is_not_null"] else column.is_null()
    raise SpecError(f"Unrecognized condition: {condition}")


@dataclass(frozen=True, slots=True, eq=False)
class ColRule:
    """Restricts a column's generated values on rows where `when` matches.

    Applied as a final vectorized pass after normal generation: rows where
    `when` matches get a value resampled uniformly (or according to `weights`)
    from `choices` instead of whatever was freely generated for them. `when`
    is evaluated against the fully-generated, freely-sampled DataFrame -- never
    against another rule's output -- so rules on different columns are
    independent of declaration order. Multiple rules on the *same* column are
    checked in declaration order, first match wins (like SQL CASE/WHEN).

    `when` is a predicate built with `polspec.col()`, not an arbitrary polars
    expression, so that every rule can round-trip through a spec file:

        ColRule(when=col("region") == "UK", choices=["RoyalMail"])
        ColRule(when=col("region").is_in(["US", "EU"]) & (col("qty") > 10), choices=["UPS"])

    The original one-column dict form is still accepted and converted:

        {"column": "enum_1", "equals": "A"}       {"column": "enum_1", "in": ["A", "B"]}
        {"column": "enum_1", "not_equals": "A"}   {"column": "enum_1", "not_in": ["A", "B"]}
        {"column": "n", "lt" | "le" | "gt" | "ge": 5}   {"column": "n", "between": [1, 5]}
        {"column": "n", "is_null": True}          {"column": "n", "is_not_null": True}
    """

    when: Pred | dict
    choices: tuple
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.when, dict):
            object.__setattr__(self, "when", _condition_to_pred(self.when))
        elif not isinstance(self.when, Pred):
            raise SpecError(
                "ColRule.when must be a predicate built with col(), or a dict like "
                "{'column': 'enum_1', 'in': ['A', 'B']}, got "
                f"{type(self.when).__name__}"
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
        _reject_duplicate_choices(self.choices, "ColRule.choices")
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


def _sample_choices(
    choices: tuple,
    n: int,
    seed: int,
    weights: tuple[float, ...] | None = None,
    dtype: pl.DataType | None = None,
) -> pl.Series:
    """n values drawn (with replacement) from `choices` according to `weights`,
    typed as `dtype` when one is given.
    """
    domain = (
        _typed_values(choices, dtype) if dtype is not None else pl.Series(list(choices))
    )
    if n == 0:
        return domain.clear()
    if len(choices) == 1:
        return domain.gather(pl.repeat(0, n, dtype=pl.UInt32, eager=True))
    plan = column_plan(
        "__idx",
        "index",
        n_categories=len(choices),
        weights=[float(w) for w in weights] if weights is not None else None,
    )
    idx = _generate_dataframe([plan], n, seed)["__idx"]
    return domain.gather(idx)


def _apply_rules(
    df: pl.DataFrame, columns: dict[str, ColSpec], seed: int | None
) -> pl.DataFrame:
    """Overwrites values on rows matched by each column's ColRules.

    A vectorised pass over the already-generated DataFrame: for a column
    with rules, rows matching a rule's `when` get a value resampled from
    that rule's `choices` (first matching rule wins); everything else keeps
    its freely-generated value. Only as many values as there are matched
    rows are sampled, and scattered into place. `when` expressions always
    see the original freely-generated values, never another rule's output,
    so rules on different columns never need dependency ordering.
    """
    if df.height == 0:
        return df
    rng = random.Random(seed)
    replaced: list[pl.Series] = []
    for name, spec in columns.items():
        if not spec.rules:
            continue
        column = df[name]
        claimed = pl.repeat(False, df.height, dtype=pl.Boolean, eager=True)
        for rule in spec.rules:
            mask = df.select(rule._expr().fill_null(False)).to_series() & ~claimed
            rows = mask.arg_true()
            if rows.len() == 0:
                continue
            fill = _sample_choices(
                rule.choices,
                rows.len(),
                rng.randrange(2**63),
                weights=rule.weights,
                dtype=spec.dtype,
            )
            column = column.scatter(rows, fill)
            claimed = claimed | mask
        replaced.append(column.alias(name))
    return df.with_columns(replaced) if replaced else df
