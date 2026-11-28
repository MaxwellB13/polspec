"""A drawn order: a column's values in order, each from its row's position.

A sorted column is a sample's order statistics, and they can be drawn one
row at a time, without the rows around them:

1. **A table of the column, in order.** The engine draws `TABLE_ROWS` values
   of the column as it always would -- its sampler, distribution, bounds,
   clamping and rounding -- and sorts them. A frame of no more valid rows
   than that is its own table: a sorted sample of the column, not a
   stratified one.
2. **A stratified position per row.** Past that, row `j` of `valid` takes
   `q = (j + u_j) / valid`, `u_j` uniform from the seed and `j` alone. The
   strata are disjoint and increasing, so `q` never decreases.
3. **The table at `q`:** interpolated between neighbours for a number, a
   date or a time (in its physical units, so an integer stays one), the
   table value whose share holds `q` for a finite domain -- a boolean,
   `choices`, an `Enum`. Either way the values never decrease.
4. **Nulls come last,** as a sort puts them: the frame's last
   `round(null_probability * n)` rows.

Steps 2 and 3 are the engine's (`src/order.rs`), in parallel; the table is
drawn here, through the ordinary path.

A unique column past the table draws each row from its own slot of the
range instead, `[lo + j*S/valid, lo + (j+1)*S/valid)`, so the values are
distinct as well as increasing; it follows the bounds, not a distribution.

Every value depends on the seed, the frame's row count and the row alone,
so any window of a frame is that frame's rows there. *Since 2.2.*

Internal: not part of the public API.
"""

from __future__ import annotations

import dataclasses

import polars as pl

from polspec._ffi import (
    drawn_float_slots,
    drawn_from_table,
    drawn_indices,
    drawn_int_slots,
)
from polspec.errors import GenerationError
from polspec.generation.engine import (
    as_decimal,
    generate_random,
    resolve_numeric_bounds,
)
from polspec.generation.seeds import pass_seed
from polspec.spec import ColSpec, column_kind
from polspec.tablespec import why_not_drawn

# How many values a column's table holds: 512 KiB of 8-byte values, drawn
# and sorted in about a millisecond, and fine enough that a drawn column's
# quantiles sit within a fraction of a percent of its distribution's.
TABLE_ROWS = 65_536

_NUMERIC = ("int", "float", "temporal", "decimal")


def drawn_values(
    name: str,
    spec: ColSpec,
    seed: int,
    *,
    frame_rows: int,
    row_offset: int,
    rows: int,
    descending: bool = False,
) -> pl.Series:
    """Rows `[row_offset, row_offset + rows)` of `spec`'s column, drawn in
    order over a frame of `frame_rows` rows, nulls last.

    `seed` is the frame's engine seed. `descending` reads the same values
    from the other end, nulls still last.
    """
    reason = why_not_drawn(spec)
    if reason is not None:
        raise GenerationError(f"Column {name!r} cannot be drawn in order: {reason}.")
    valid = frame_rows - _null_count(spec, frame_rows)
    stop = min(row_offset + rows, valid)
    count = max(0, stop - row_offset)
    parts = []
    if count:
        # Descending, the window reads the ascending positions the other way.
        first = row_offset if not descending else valid - row_offset - count
        values = _ascending(name, spec, seed, valid, first, count)
        parts.append(values.reverse() if descending else values)
    if rows - count:
        parts.append(pl.Series(name, [None] * (rows - count), dtype=spec.dtype))
    if not parts:
        return pl.Series(name, [], dtype=spec.dtype)
    return pl.concat(parts).alias(name)


def _null_count(spec: ColSpec, n: int) -> int:
    return round(spec.null_probability * n) if spec.nullable else 0


def _ascending(
    name: str, spec: ColSpec, seed: int, valid: int, first: int, count: int
) -> pl.Series:
    """Ascending positions `[first, first + count)` of `valid` values."""
    if valid <= TABLE_ROWS:
        return _table(name, spec, seed, valid).slice(first, count)
    key = pass_seed(seed, f"order:{name}")
    kind = column_kind(spec.dtype)
    if spec.unique:
        lo, hi = resolve_numeric_bounds(spec)
        if kind == "float":
            drawn = drawn_float_slots(float(lo), float(hi), key, valid, first, count)
            return drawn.cast(spec.dtype).alias(name)
        drawn = drawn_int_slots(int(lo), int(hi), key, valid, first, count)
        return _from_physical(drawn, spec).alias(name)
    if spec.choices is not None or kind not in _NUMERIC:
        table = _table(name, spec, seed, TABLE_ROWS)
        at = drawn_indices(key, valid, first, count, table.len())
        return table.gather(at).alias(name)
    drawn = drawn_from_table(
        _physical_table(name, spec, seed), key, valid, first, count
    )
    if kind == "float":
        return drawn.cast(spec.dtype).alias(name)
    return _from_physical(drawn, spec).alias(name)


# Tables by spec object, not by value -- a spec need not be hashable, and a
# stream hands every window the same one; the spec is held beside its table,
# so its id is not reused while the entry lives.
_TABLES: dict[tuple[int, str, int, int], tuple[ColSpec, pl.Series]] = {}


def _table(name: str, spec: ColSpec, seed: int, size: int) -> pl.Series:
    """`size` values of the column, as the engine draws them, in order."""
    key = (id(spec), name, seed, size)
    hit = _TABLES.get(key)
    if hit is not None and hit[0] is spec:
        return hit[1]
    if len(_TABLES) >= 32:
        _TABLES.clear()
    plain = dataclasses.replace(spec, nullable=False, null_probability=0.0)
    table = generate_random({name: plain}, size, seed)[name].sort()
    _TABLES[key] = (spec, table)
    return table


def _physical_table(name: str, spec: ColSpec, seed: int) -> pl.Series:
    """The table as the engine's order step reads it, in one chunk: Float64
    for a float column, and physical integers otherwise -- Int64, which
    holds every physical value but a 128-bit integer's, a UInt64's past
    2^63 and a Decimal's, which are Int128. Kept beside the table, so a
    stream converts it once."""
    key = (id(spec), name, seed, -1)
    hit = _TABLES.get(key)
    if hit is not None and hit[0] is spec:
        return hit[1]
    table = _table(name, spec, seed, TABLE_ROWS)
    if column_kind(spec.dtype) == "float":
        physical = table.cast(pl.Float64)
    else:
        physical = table.to_physical()
        wide = physical.dtype in (pl.Int128, pl.UInt128, pl.UInt64)
        physical = physical.cast(pl.Int128 if wide else pl.Int64)
    physical = physical.rechunk()
    _TABLES[key] = (spec, physical)
    return physical


def _from_physical(physical: pl.Series, spec: ColSpec) -> pl.Series:
    """Wide physical integers back as the column's own dtype."""
    dtype = spec.dtype
    if isinstance(dtype, pl.Decimal):
        return as_decimal(physical, dtype)
    if column_kind(dtype) == "temporal":
        stored = pl.Series([], dtype=dtype).to_physical().dtype
        return physical.cast(stored).cast(dtype)
    return physical.cast(dtype)
