from __future__ import annotations

import polars as pl

# Bounds baked into the fixed-width integer dtypes themselves. Used as the
# default generation range when a ColSpec doesn't supply its own bounds, so
# generated values always survive the cast back to the requested dtype.
_INT_DTYPE_BOUNDS: dict[pl.DataType, tuple[int, int]] = {
    pl.Int8: (-128, 127),
    pl.Int16: (-32_768, 32_767),
    pl.Int32: (-2_147_483_648, 2_147_483_647),
    pl.UInt8: (0, 255),
    pl.UInt16: (0, 65_535),
    pl.UInt32: (0, 4_294_967_295),
}

_DEFAULT_WIDE_INT_BOUND = 1_000_000
_DEFAULT_FLOAT_BOUND = 1_000_000.0
_DEFAULT_STRING_LEN = (5, 15)
_DEFAULT_NULL_PROBABILITY = 0.1

# Safety cap on method="cartesian": the cross-joined coverage set grows as
# the product of every dimension's cardinality, so a handful of wide enums
# can explode into an unreasonable row count by accident.
_MAX_CARTESIAN_ROWS = 50_000_000
