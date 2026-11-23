"""`ColRule` as a pass: the rows a rule's condition matches get a value drawn
from its choices, first matching rule first.

Internal: not part of the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polspec import frames
from polspec._ffi import column_plan
from polspec._ffi import generate_dataframe as _generate_dataframe
from polspec.dtypes import typed_values
from polspec.generation.seeds import pass_seed
from polspec.rules import rule_condition

if TYPE_CHECKING:
    from polspec.spec import ColSpec


def sample_choices(
    choices: tuple,
    n: int,
    seed: int | None,
    weights: tuple[float, ...] | None = None,
    dtype: pl.DataType | None = None,
    row_offset: int = 0,
) -> pl.Series:
    """Rows `[row_offset, row_offset + n)` of a column drawn (with
    replacement) from `choices` according to `weights`, typed as `dtype`
    when one is given: an engine column, so a window of it is the whole
    column's rows there.
    """
    domain = (
        typed_values(choices, dtype) if dtype is not None else pl.Series(list(choices))
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
    idx = _generate_dataframe([plan], n, seed, row_offset)["__idx"]
    return domain.gather(idx)


def apply_column_rules(
    df: pl.DataFrame,
    name: str,
    spec: ColSpec,
    seed: int | None,
    *,
    row_offset: int = 0,
) -> pl.DataFrame:
    """Overwrites `name` on the rows its own ColRules match.

    A vectorised pass over the frame as it stands: a row matching a rule's
    `when` takes that rule's value (first matching rule wins); everything
    else keeps the value it already had. Each rule's values are a column of
    their own, drawn in full from the rule's `choices` -- keyed by `seed`
    and the rule's place, rows `[row_offset, ...)` of the whole frame's --
    and the pass selects from it. So a row's value depends on that row
    alone, not on how many rows matched before it, and a batch holds the
    values the whole frame holds there.

    `when` sees the frame this pass is given, so a rule keyed on a column
    another pass rewrites reads the rewritten values -- the same values
    validation will check the rule against. `polspec.pass_order.order`
    decides which pass runs first.

    A null stays a null. The column's nullability was decided when it was
    drawn, at the declared rate; a rule says what a *value* on a matched
    row is, which is also all validation checks it against.
    """
    if df.height == 0 or not spec.rules:
        return df
    column = frames.plain(df[name])  # renamed back below
    claimed = column.is_null()
    for index, rule in enumerate(spec.rules):
        mask = df.select(rule_condition(rule).fill_null(False)).to_series() & ~claimed
        if not mask.any():
            continue
        values = sample_choices(
            rule.choices,
            df.height,
            None if seed is None else pass_seed(seed, f"rule:{index}"),
            weights=rule.weights,
            dtype=spec.dtype,
            row_offset=row_offset,
        )
        column = column.zip_with(~mask, values)
        claimed = claimed | mask
    return df.with_columns(column.alias(name))
