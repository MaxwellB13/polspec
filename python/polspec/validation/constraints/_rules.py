"""A `ColRule` as a constraint: on the rows its `when` matches and no earlier
rule claimed, the value is one of the rule's choices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec.constraints import is_textual as _is_textual
from polspec.validation.report import FindingCode

if TYPE_CHECKING:
    from polspec.rules import ColRule
    from polspec.spec import ColSpec

from polspec.dtypes import _typed_values
from polspec.validation.constraints._base import _as_strings, _Constraint


@dataclass(kw_only=True)
class _RuleHolds(_Constraint):
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


def _rule_constraints(
    name: str,
    spec: ColSpec,
    actual_dtype: pl.DataType,
    df_col_names: Sequence[str],
) -> list[_Constraint]:
    """One constraint per ColRule, respecting first-match-wins ordering.

    Each rule only governs the rows no earlier rule already claimed, matching
    how `_apply_column_rules` assigns them at generation time -- including how
    it reads a null condition. A `when` that evaluates to null on a row does
    not match there, so generation folds it to False before both testing it and
    accumulating it into `claimed`. Doing anything else here lets a null
    propagate through `~claimed` and silently excuse every later rule on that
    row, which is a row generation did rewrite and validation would not check.
    """
    column = pl.col(name)
    constraints: list[_Constraint] = []
    claimed = pl.lit(False)

    for index, rule in enumerate(spec.rules):
        if not rule.when.root_names() <= set(df_col_names):
            continue  # reported through missing_cols instead
        matches = rule._expr().fill_null(False)
        applies = matches & ~claimed
        claimed = claimed | matches

        if _is_textual(actual_dtype):
            in_choices = column.cast(pl.String).is_in(
                _as_strings(rule.choices, spec.dtype)
            )
            sample_expr = column.cast(pl.String)
        else:
            # In the column's own dtype, for the reason `_value_constraints`
            # gives for choices.
            in_choices = column.is_in(
                _typed_values(rule.choices, actual_dtype).implode()
            )
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
