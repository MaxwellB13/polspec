"""`ColRule` as a pass: the rows a rule's condition matches get a value drawn
from its choices, first matching rule first.

Internal: not part of the public API.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import polars as pl

from polspec import frames
from polspec._ffi import column_plan
from polspec._ffi import generate_dataframe as _generate_dataframe
from polspec.dtypes import typed_values

if TYPE_CHECKING:
    from polspec.spec import ColSpec


def sample_choices(
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
    idx = _generate_dataframe([plan], n, seed)["__idx"]
    return domain.gather(idx)


def apply_column_rules(
    df: pl.DataFrame, name: str, spec: ColSpec, seed: int | None
) -> pl.DataFrame:
    """Overwrites `name` on the rows its own ColRules match.

    A vectorised pass over the frame as it stands: rows matching a rule's
    `when` get a value resampled from that rule's `choices` (first matching
    rule wins); everything else keeps the value it already had. Only as many
    values as there are matched rows are sampled, and scattered into place.

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
    rng = random.Random(seed)
    column = frames.plain(df[name])  # renamed back below
    claimed = column.is_null()
    for rule in spec.rules:
        mask = df.select(rule._expr().fill_null(False)).to_series() & ~claimed
        rows = mask.arg_true()
        if rows.len() == 0:
            continue
        fill = sample_choices(
            rule.choices,
            rows.len(),
            rng.randrange(2**63),
            weights=rule.weights,
            dtype=spec.dtype,
        )
        column = column.scatter(rows, fill)
        claimed = claimed | mask
    return df.with_columns(column.alias(name))
