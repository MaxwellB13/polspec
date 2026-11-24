"""What a declaration and its checks both read of a dtype and a column.

Internal: not part of the public API.
"""

from __future__ import annotations

import datetime as dt
import decimal
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeGuard

import polars as pl

from polspec.dtypes import (
    dtype_value_limits,
    element_dtype,
    map_entries,
    map_parts,
)
from polspec.errors import SpecError

if TYPE_CHECKING:
    from polspec.spec import ColSpec


def check_description(description: Any, owner: str) -> None:
    """Refuses a description with nothing in it for a reader."""
    if description is None:
        return
    if not isinstance(description, str) or not description.strip():
        raise SpecError(
            f"{owner}.description must be a non-empty string, got {description!r}"
        )


def is_dtype(value: object) -> TypeGuard[pl.DataType | type[pl.DataType]]:
    """Whether `value` is a Polars dtype: an instance, or a class."""
    return isinstance(value, pl.DataType) or (
        isinstance(value, type) and issubclass(value, pl.DataType)
    )


# The Polars dtype each Python type a column is often declared with means.
POLARS_SPELLING: dict[object, str] = {
    int: "pl.Int64",
    float: "pl.Float64",
    str: "pl.String",
    bool: "pl.Boolean",
    bytes: "pl.Binary",
    dt.date: "pl.Date",
    dt.datetime: "pl.Datetime",
    dt.time: "pl.Time",
    dt.timedelta: "pl.Duration",
    decimal.Decimal: "pl.Decimal",
}


def not_a_dtype(value: object) -> str:
    """Why `value` cannot be a column's dtype, and its Polars spelling
    where there is an obvious one."""
    if value is None:
        return "ColSpec needs a dtype, such as pl.Int64."
    if isinstance(value, (Mapping, list, tuple, set)):
        # The values a column holds, written where its dtype goes.
        values = list(value)
        written = (
            f"pl.Enum, choices={dict(value)!r}"
            if isinstance(value, Mapping)
            else f"pl.Enum({values!r})"
        )
        return (
            f"ColSpec.dtype must be a Polars dtype, got the values {values!r}: for a "
            f"column of them, use ColSpec({written})."
        )
    spelling = POLARS_SPELLING.get(value) if isinstance(value, type) else None
    if spelling is None and isinstance(value, str):
        named = getattr(pl, value, None)
        spelling = f"pl.{value}" if is_dtype(named) else None
    shown = getattr(value, "__name__", None) or repr(value)
    suggestion = f"use {spelling}" if spelling else "use one such as pl.Int64"
    return f"ColSpec.dtype must be a Polars dtype, got {shown}: {suggestion}."


def column_kind(dtype: pl.DataType) -> str:
    if dtype.is_integer():
        return "int"
    if dtype.is_float():
        return "float"
    if isinstance(dtype, pl.Decimal):
        return "decimal"
    if dtype.is_temporal():
        return "temporal"
    if dtype == pl.Boolean:
        return "bool"
    if dtype in (pl.String, pl.Utf8):
        return "string"
    if dtype == pl.Binary:
        return "binary"
    if isinstance(dtype, pl.Enum):
        return "enum"
    if is_categorical_dtype(dtype):
        return "categorical"
    if isinstance(dtype, (pl.List, pl.Array)):
        return "list"
    if isinstance(dtype, pl.Struct):
        return "struct"
    if map_parts(dtype) is not None:
        return "map"
    if dtype == pl.Null:
        return "null"
    raise SpecError(f"polspec cannot generate data for dtype {dtype!r}")


def value_dtype_of(dtype: pl.DataType) -> pl.DataType:
    """The dtype a column's *values* have: the element's, for a `List` or
    `Array`; an entry's `{key, value}` struct, for a `Map`; the column's own
    otherwise."""
    if isinstance(dtype, (pl.List, pl.Array)):
        return element_dtype(dtype)
    if (entries := map_entries(dtype)) is not None:
        return element_dtype(entries)
    return dtype


def distinct_values(spec: ColSpec) -> int | None:
    """How many distinct values `spec` can generate, when that is a small,
    knowable number: a finite domain, a Boolean, an integer range. None
    otherwise -- a range of floats, a string of any text."""
    from polspec.domain import Domain

    if spec.value_dtype == pl.Boolean:
        return 2
    values = Domain.of(spec).values
    if values is not None:
        return len(set(values))
    if spec.value_dtype.is_integer():
        limits = dtype_value_limits(spec.value_dtype)
        if limits is None:
            return None
        lo, hi = limits
        if spec.bounds is not None:
            lo = spec.bounds.min if spec.bounds.min is not None else lo
            hi = spec.bounds.max if spec.bounds.max is not None else hi
        return max(0, int(hi) - int(lo) + 1)
    return None


def is_categorical_dtype(dtype: pl.DataType) -> bool:
    return (
        isinstance(dtype, pl.Categorical)
        or dtype == pl.Categorical
        or (isinstance(dtype, type) and issubclass(dtype, pl.Categorical))
    )
