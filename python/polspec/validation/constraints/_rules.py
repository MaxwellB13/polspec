"""A `ColRule` as a constraint: on the rows its `when` matches and no earlier
rule claimed, the value is one of the rule's choices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec import frames
from polspec.domain import is_textual
from polspec.validation.report import FindingCode

if TYPE_CHECKING:
    from polspec.rules import ColRule
    from polspec.spec import ColSpec


from polspec.validation.constraints._base import Constraint, as_strings, in_values


@dataclass(kw_only=True)
class _RuleHolds(Constraint):
    column: str
    rule: ColRule
    code: FindingCode = "rule"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {
            "when": repr(self.rule.when),
            "choices": list(self.rule.choices),
        }

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': found {count} value(s) violating "
            f"ColRule(when={self.rule.when}, choices={self.rule.choices}). "
            f"Violating samples: {samples}"
        )


def rule_constraints(
    name: str,
    spec: ColSpec,
    actual_dtype: pl.DataType,
    df_col_names: Sequence[str],
) -> list[Constraint]:
    """One constraint per ColRule, respecting first-match-wins ordering.

    Each rule only governs the rows no earlier rule already claimed, matching
    how `apply_column_rules` assigns them at generation time -- including how
    it reads a null condition. A `when` that evaluates to null on a row does
    not match there, so generation folds it to False before both testing it and
    accumulating it into `claimed`. Doing anything else here lets a null
    propagate through `~claimed` and silently excuse every later rule on that
    row, which is a row generation did rewrite and validation would not check.

    A rule whose `when` names a column the frame lacks, or holds as a dtype
    its declaration does not accept, ends the list: the rows it would have
    claimed are unknown, so no later rule can be checked.
    """
    column = frames.column(name)
    constraints: list[Constraint] = []
    claimed = pl.lit(False)

    for index, rule in enumerate(spec.rules):
        if not rule.when.root_names() <= set(df_col_names):
            # Reported through missing_cols. Which rows this rule claimed
            # cannot be known, so neither can the rows every later rule on
            # the column governs: checking them anyway reports a row this
            # rule rewrote as a failure of the next. No finding beats a
            # false one.
            break
        matches = rule._expr().fill_null(False)
        applies = matches & ~claimed
        claimed = claimed | matches

        if is_textual(actual_dtype):
            in_choices = column.cast(pl.String).is_in(
                as_strings(rule.choices, spec.dtype)
            )
            sample_expr = column.cast(pl.String)
        else:
            in_choices = in_values(column, rule.choices, spec.value_dtype, actual_dtype)
            sample_expr = column

        constraints.append(
            _RuleHolds(
                key=f"{name}__rule_{index}",
                mask=applies & column.is_not_null() & ~in_choices,
                sample_expr=sample_expr,
                column=name,
                rule=rule,
            )
        )
    return constraints
