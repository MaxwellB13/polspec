"""How much memory a generated frame will take, from the declaration alone.

`generate()` allocates the whole frame before it returns anything, so the
first sign that a spec is too big for the machine is usually the machine
swapping. This reads the answer off the schema instead: no data, no
sampling, just the width each dtype holds and the lengths the spec declares.

What it measures is the *frame*, not the process. Generation holds working
buffers on top -- most visibly for `Decimal` and `List`, which are assembled
in Polars rather than filled by the engine -- so a peak is higher than this
by a factor that depends on the dtype, and no estimate from a schema can
know it. Against a frame of scalar columns it is within a percent or two.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import polars as pl

from polspec.constants import _DEFAULT_LIST_LEN, _DEFAULT_STRING_LEN
from polspec.dtypes import field_dtypes, map_entries
from polspec.formats import lookup as _lookup_format

if TYPE_CHECKING:
    from polspec.formats import Format
    from polspec.spec import ColSpec
    from polspec.tablespec import TableSpec

# A Polars string or binary value is a 16-byte view. Up to `_INLINE` bytes
# live inside the view; anything longer is the view *and* its bytes in a
# buffer, so a column of short values costs the view alone.
_VIEW_BYTES = 16
_INLINE = 12

_VALIDITY_BYTES = 1 / 8

_FIXED_WIDTHS: dict[object, float] = {
    pl.Boolean: 1 / 8,
    pl.Int8: 1,
    pl.UInt8: 1,
    pl.Int16: 2,
    pl.UInt16: 2,
    pl.Int32: 4,
    pl.UInt32: 4,
    pl.Float32: 4,
    pl.Date: 4,
    pl.Int64: 8,
    pl.UInt64: 8,
    pl.Float64: 8,
    pl.Time: 8,
    pl.Float16: 2,
    pl.Int128: 16,
    pl.UInt128: 16,
}


def estimated_size(spec: TableSpec, n: int) -> int:
    """Bytes `generate(spec, n)` is expected to hold, as whole bytes.

    See the module docstring for what this does and does not count.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    per_row = sum(_column_bytes(column) for column in spec.columns.values())
    return int(per_row * n)


def _column_bytes(spec: ColSpec) -> float:
    """One column's bytes per row: its values, plus a validity bit when the
    column can hold nulls."""
    width = _value_bytes(spec, spec.dtype)
    return width + (_VALIDITY_BYTES if spec.nullable else 0.0)


def _value_bytes(spec: ColSpec, dtype: pl.DataType) -> float:
    """One value of `dtype`, read through what `spec` declares about it."""
    fixed = _FIXED_WIDTHS.get(dtype)
    if fixed is not None:
        return fixed
    if isinstance(dtype, (pl.Datetime, pl.Duration)):
        return 8
    if isinstance(dtype, pl.Decimal):
        return 16  # an i128, whatever the precision
    if isinstance(dtype, pl.Enum):
        return _physical_width(len(dtype.categories))
    if isinstance(dtype, pl.Categorical):
        return float(_FIXED_WIDTHS.get(dtype.categories.physical(), 4))
    if dtype in (pl.String, pl.Utf8, pl.Binary):
        # Every row has a view, null or not; only a present value has bytes.
        return _VIEW_BYTES + _bytes_past_inline(spec) * _present(spec)
    if map_entries(dtype) is not None:
        # A map is held as the list of entries it is, as long as it is drawn.
        return _column_bytes(spec._as_drawn_list())
    if isinstance(dtype, pl.List):
        # An offset per row, plus however many elements the row holds, each
        # read through the declaration generation draws it from. A null list
        # holds none.
        lengths = spec.list_length.closed() if spec.list_length else _DEFAULT_LIST_LEN
        mean_len = (lengths[0] + lengths[1]) / 2
        # An element is a column of its own: a nullable one carries a
        # validity bit, and its content only where it is present.
        element = _column_bytes(spec._element())
        return 8 + mean_len * element * _present(spec)
    if isinstance(dtype, pl.Array):
        return dtype.size * _column_bytes(spec._element())
    if isinstance(dtype, pl.Struct):
        # A struct holds no values of its own: it is its fields, each a
        # column with its own validity when it can be null.
        return sum(_column_bytes(spec._field(name)) for name in field_dtypes(dtype))
    # A dtype with no values to hold (a `Null` column) costs its view at most.
    return _VIEW_BYTES


def _present(spec: ColSpec) -> float:
    """The fraction of rows expected to hold a value rather than a null.

    Only what a present value brings with it scales by this -- a string's
    bytes past its view, a list's elements. Fixed-width values, views,
    offsets, an `Array`'s slots and a struct's fields are allocated on a
    null row too.
    """
    return 1.0 - spec.null_probability if spec.nullable else 1.0


def _physical_width(n_categories: int) -> float:
    """What Polars stores an Enum's codes in: the narrowest that holds them."""
    if n_categories <= 255:
        return 1
    if n_categories <= 65_535:
        return 2
    return 4


def _bytes_past_inline(spec: ColSpec) -> float:
    """The expected bytes a text value costs *beyond* its view.

    A value of twelve bytes or fewer is inside the view and costs nothing
    more; a longer one costs its whole length again. Averaged over the
    lengths the column declares, since that is all a declaration says.
    """
    if spec.choices is not None or _draws_from_a_finite_format(spec):
        # A column drawn from a fixed set is gathered from one Series, and
        # the views point back into its buffer rather than copying it: the
        # bytes are paid once for the column, not once per row.
        return 0.0
    if spec.format is not None:
        return _format_length(_lookup_format(spec.format))
    low, high = (
        spec.string_length.closed() if spec.string_length else _DEFAULT_STRING_LEN
    )
    return _mean_over_inline(range(low, high + 1))


def _draws_from_a_finite_format(spec: ColSpec) -> bool:
    return spec.format is not None and _lookup_format(spec.format).is_finite


def _format_length(fmt: Format) -> float:
    """A format's expected bytes past the view, from the shape it fills.

    A template is a run of parts, each contributing a fixed literal, a
    choice from a set, or between `min` and `max` characters, so the
    expected length is the sum of what each part contributes.
    """
    if fmt.template is None:
        return float(fmt.max_length) if fmt.max_length else 0.0
    expected = 0.0
    for kind, values, low, high in fmt.template:
        if kind == "lit":
            expected += len(values[0]) if values else 0
        elif kind == "one_of":
            expected += sum(len(v) for v in values) / len(values) if values else 0
        else:  # a run of `low`..`high` characters
            expected += (low + high) / 2
    return expected if expected > _INLINE else 0.0


def _mean_over_inline(lengths: Iterable[int]) -> float:
    """The mean of `lengths`, counting the ones that fit in a view as zero."""
    seen = list(lengths)
    return sum(length for length in seen if length > _INLINE) / len(seen)
