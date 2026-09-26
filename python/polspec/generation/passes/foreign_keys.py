"""`ForeignKey` as a pass: a key's column overwritten with its parent's keys,
drawn with replacement -- or, for a unique column, as a permutation of them."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from polspec._ffi import permuted_indices
from polspec.errors import GenerationError
from polspec.foreign_key import ForeignKey
from polspec.spec import ColSpec


def _as_u64(seed: int | None) -> int:
    """A seed the permutation takes: an unsigned 64-bit key, drawn at random
    for an unseeded frame as `sample(seed=None)` would be."""
    if seed is None:
        import random

        return random.getrandbits(64)
    return seed & 0xFFFF_FFFF_FFFF_FFFF


def unique_parent_shortfall(
    fk: ForeignKey, columns: Mapping[str, ColSpec], parent_df: pl.DataFrame, rows: int
) -> str | None:
    """Why a unique column `fk` fills cannot be filled for `rows` rows, or
    None when it can -- or when the column is not unique."""
    local_cols = list(fk.columns)
    if len(local_cols) != 1 or not columns[local_cols[0]].unique:
        return None
    ref_cols = list(fk.ref_columns)
    distinct = parent_df.select(ref_cols).drop_nulls().unique().height
    if distinct >= rows:
        return None
    return (
        f"ForeignKey '{fk.name}' fills the unique column '{local_cols[0]}', but "
        f"the referenced parent offers only {distinct} distinct value(s) for "
        f"{rows} row(s). Every value has to come from the parent and no two may "
        "repeat, so generate more parent rows, or fewer of these."
    )


def apply_foreign_key(
    df: pl.DataFrame,
    columns: dict[str, ColSpec],
    fk: ForeignKey,
    parent_df: pl.DataFrame,
    seed: int | None,
    *,
    row_offset: int = 0,
    key_seed: int | None = None,
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

    Without replacement is a permutation of the parent's keys: the row at
    `row_offset + i` of the whole frame takes the key at `π(row_offset + i)`,
    with `π` keyed by `key_seed`. So a batch's rows take the keys the whole
    frame's rows take there, and the column is unique across batches, not
    only within one. With replacement, the draw is the window's own, from
    `seed`.

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
    if wants_unique:
        shortfall = unique_parent_shortfall(
            fk, columns, parent_df, row_offset + df.height
        )
        if shortfall is not None:
            raise GenerationError(shortfall)
        picks = permuted_indices(
            parent_keys.height,
            _as_u64(key_seed if key_seed is not None else seed),
            row_offset,
            df.height,
        )
        sampled_rows = parent_keys[picks]
    else:
        sampled_rows = parent_keys.sample(n=df.height, with_replacement=True, seed=seed)

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
