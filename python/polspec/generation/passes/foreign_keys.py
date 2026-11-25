"""`ForeignKey` as a pass: a key's column overwritten with its parent's keys,
drawn with replacement -- or, for a unique column, as a permutation of them.

Internal: not part of the public API.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from polspec import frames
from polspec._ffi import column_plan, permuted_indices
from polspec._ffi import generate_dataframe as _generate_dataframe
from polspec.errors import GenerationError
from polspec.foreign_key import ForeignKey
from polspec.generation.engine import unique_at
from polspec.spec import ColSpec


def apply_windowed_self_key(
    df: pl.DataFrame,
    columns: dict[str, ColSpec],
    fk: ForeignKey,
    *,
    frame_seed: int,
    key_seed: int,
    frame_rows: int,
    row_offset: int = 0,
) -> pl.DataFrame:
    """A key into the spec's own rows, drawn from the whole frame: the row at
    `row_offset + i` takes the referenced column's value at a row of all
    `frame_rows`, picked by `key_seed` and the row alone -- with replacement,
    or, for a unique key, as a permutation of the rows -- and read where it
    sits (`unique_at`), so a batch's children point anywhere in the frame,
    as the whole frame's do. *Since 2.1; 2.0 drew from the batch's own rows.*

    Only for a one-column key whose referenced column `readable_at` reads
    and no other pass rewrites (`windowed_self_key`); `frame_seed` is the
    frame's engine seed, which drew that column.
    """
    m = df.height
    if m == 0:
        return df
    local, ref = fk.columns[0], fk.ref_columns[0]
    if columns[local].unique:
        picks = permuted_indices(frame_rows, _as_u64(key_seed), row_offset, m)
    else:
        plan = column_plan("__pick", "index", n_categories=frame_rows)
        picks = _generate_dataframe([plan], m, key_seed, row_offset)["__pick"]
    values = unique_at(ref, columns[ref], frame_seed, picks.cast(pl.UInt64))
    sampled = frames.plain(values).cast(df.schema[local])
    return df.with_columns(
        pl.when(frames.column(local).is_not_null())
        .then(pl.lit(sampled))
        .otherwise(frames.column(local))
        .alias(local)
    )


def _as_u64(seed: int | None) -> int:
    """A seed the permutation takes: an unsigned 64-bit key, drawn at random
    for an unseeded frame as `sample(seed=None)` would be."""
    if seed is None:
        import random

        return random.getrandbits(64)
    return seed & 0xFFFF_FFFF_FFFF_FFFF


def parent_keys_of(fk: ForeignKey, parent_df: pl.DataFrame) -> pl.DataFrame:
    """The distinct non-null keys `parent_df` offers `fk`, in the order they
    first occur: what the pass draws from.

    Reading them means hashing every parent row, so a caller drawing
    several windows from one parent -- `generate_batches`, a scan, a sink --
    reads them once and hands them to each window.
    """
    keys = parent_df.select(frames.columns(list(fk.ref_columns)))
    # A parent's key is usually unique already -- it is what a key points
    # at. Then every row is its own first occurrence, and the keys are the
    # column as it stands: one count of distinct values says so, at a
    # fraction of what an ordered `unique` costs.
    if keys.width == 1:
        column = keys.to_series()
        distinct = column.null_count() == 0 and column.n_unique() == column.len()
    else:
        distinct = (
            keys.null_count().sum_horizontal().item() == 0
            and keys.n_unique() == keys.height
        )
    if distinct:
        return keys
    return keys.drop_nulls().unique(maintain_order=True)


def unique_parent_shortfall(
    fk: ForeignKey, columns: Mapping[str, ColSpec], distinct: int, rows: int
) -> str | None:
    """Why a unique column `fk` fills cannot be filled for `rows` rows from a
    parent of `distinct` keys, or None when it can -- or when the column is
    not unique."""
    local_cols = list(fk.columns)
    if len(local_cols) != 1 or not columns[local_cols[0]].unique:
        return None
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
    parent_keys: pl.DataFrame | None = None,
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
    only within one. With replacement is a window too, when a `key_seed` is
    given: the row at `row_offset + i` takes the key an engine index column
    keyed by `key_seed` holds there. Without one -- a key into this frame's
    own rows, whose parent is the window itself -- the draw is the window's
    own, from `seed`.

    `parent_df` for a self-referencing key is the frame as it stands when
    this pass runs, so a key reading a column another key rewrote draws from
    the values that column ended up with. `parent_keys`, when given, are
    `parent_keys_of(fk, parent_df)` read once already.
    """
    if df.height == 0:
        return df
    local_cols = list(fk.columns)
    ref_cols = list(fk.ref_columns)

    if parent_keys is None:
        parent_keys = parent_keys_of(fk, parent_df)
    if parent_keys.height == 0:
        raise GenerationError(
            f"ForeignKey '{fk.name}' cannot generate values: the referenced "
            f"parent has no non-null rows for columns {ref_cols}"
        )

    wants_unique = len(local_cols) == 1 and columns[local_cols[0]].unique
    if wants_unique:
        shortfall = unique_parent_shortfall(
            fk, columns, parent_keys.height, row_offset + df.height
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
    elif key_seed is not None:
        plan = column_plan("__pick", "index", n_categories=parent_keys.height)
        picks = _generate_dataframe([plan], df.height, key_seed, row_offset)["__pick"]
        sampled_rows = parent_keys[picks]
    else:
        sampled_rows = parent_keys.sample(n=df.height, with_replacement=True, seed=seed)

    exprs = []
    for local_col, ref_col in zip(local_cols, ref_cols, strict=True):
        sampled_col = frames.plain(sampled_rows[ref_col]).cast(df.schema[local_col])
        exprs.append(
            pl.when(frames.column(local_col).is_not_null())
            .then(pl.lit(sampled_col))
            .otherwise(frames.column(local_col))
            .alias(local_col)
        )
    return df.with_columns(exprs)
