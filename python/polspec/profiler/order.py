"""Whether a frame's rows come sorted by one of its columns.

Internal: not part of the public API.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from polspec import frames
from polspec.dtypes import map_parts

# Distinct values a column needs before its order is taken to be the
# frame's, and not chance: a short or repetitive column is often sorted by
# accident. The same bar a format clears.
ORDER_MIN_DISTINCT = 20


# Rows of the head order detection reads first, before ranking a column whole.
ORDER_PREFIX_ROWS = 2048


def detect_order(df: pl.DataFrame) -> tuple[tuple[str, ...], bool] | None:
    """The column `df`'s rows come sorted by, and whether high to low: the
    first, in column order, of single values -- at least
    `ORDER_MIN_DISTINCT` of them distinct -- that is sorted one way or the
    other, nulls last. None when no column is.

    Tested exactly as validation tests `order_by` (`frames.out_of_order`),
    so the order recorded is one the frame passes. One column only: a
    per-key order (`["user_id", "ts"]`) is the user's to declare.

    Internal: `profile` and `FrameSpec.from_dataframe` are the ways in.
    """
    candidates = [
        name
        for name, dtype in df.schema.items()
        if not dtype.is_nested() and map_parts(dtype) is None
    ]
    if not candidates or df.height < ORDER_MIN_DISTINCT:
        return None
    # The rows' head first: a prefix of a sorted column is sorted, so this
    # never turns one away, and nearly every column that is not sorted shows
    # it within a few rows -- the full test, a rank of every value both ways,
    # is left to the few that survive.
    head = _sorted_ways(df.head(ORDER_PREFIX_ROWS), candidates)
    survivors = [name for name in candidates if head[name]]
    if not survivors:
        return None
    if df.height > ORDER_PREFIX_ROWS:
        whole = _sorted_ways(df, survivors)
    else:
        whole = {name: head[name] for name in survivors}
    distinct = df.select(
        frames.column(name).drop_nulls().n_unique().alias(name) for name in survivors
    ).row(0, named=True)
    for name in survivors:
        if distinct[name] >= ORDER_MIN_DISTINCT and whole[name]:
            return (name,), whole[name][0]
    return None


def _sorted_ways(df: pl.DataFrame, names: Sequence[str]) -> dict[str, list[bool]]:
    """For each of `names`, the directions -- `False` ascending, `True`
    descending -- its rows come sorted in, ascending first."""
    counts = df.select(
        frames.out_of_order([name], [desc]).sum().alias(f"{i}:{desc}")
        for i, name in enumerate(names)
        for desc in (False, True)
    ).row(0, named=True)
    return {
        name: [desc for desc in (False, True) if counts[f"{i}:{desc}"] == 0]
        for i, name in enumerate(names)
    }
