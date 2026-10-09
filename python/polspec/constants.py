"""Defaults and sentinels the declarations share.

Internal: not part of the public API.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from polspec.dtypes import DtypeLike

# The default of a dataclass field that `__post_init__` always fills in
# (a foreign key's name, a check's name). Typed `Any` so the field itself
# can be annotated with what it holds once constructed, not with `| None`.
FILLED_IN: Any = None

DEFAULT_WIDE_INT_BOUND = 1_000_000
# The engine draws every integer-backed column through an i64.
I64_MAX = 2**63 - 1
DEFAULT_FLOAT_BOUND = 1_000_000.0
DEFAULT_STRING_LEN = (5, 15)
DEFAULT_LIST_LEN = (0, 5)
DEFAULT_NULL_PROBABILITY = 0.1
# The share of present rows each of a format's `extra_values` is drawn on,
# when they are listed without shares.
DEFAULT_EXTRA_SHARE = 0.01

# A frame this size is worth a word before it is allocated. Not a refusal:
# ten gigabytes on a machine with sixty-four is a reasonable thing to ask
# for, and `max_bytes=` is there for a caller who wants it refused.
LARGE_FRAME_BYTES = 4 * 1024**3

# Safety cap on method="cartesian": the cross-joined coverage set grows as
# the product of every dimension's cardinality, so a handful of wide enums
# can explode into an unreasonable row count by accident.
MAX_CARTESIAN_ROWS = 50_000_000

# Max distinct categories a pl.Categories() registry can hold, per physical
# dtype (2**bits - 1). UInt32 (the default physical dtype) is omitted: at
# ~4 billion categories it is never a practical constraint on generation.
CATEGORICAL_PHYSICAL_CAPACITY: dict[DtypeLike, int] = {
    pl.UInt8: 255,
    pl.UInt16: 65_535,
}
