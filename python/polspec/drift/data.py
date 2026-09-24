"""One column of a frame, measured against the declaration it is compared to.

The declaration decides what is worth measuring: values outside a finite
domain only mean something when a domain is declared, format failures only
when a format is. So an `Observed` is built *from* a `ColSpec`, and carries
just the summary statistics the comparators read -- never the rows.

This measures directly rather than going through `profile_dataframe`. The
profiler describes a frame well enough to generate one like it, which is a
different question: it narrows a small String column to an Enum and records
temporal extremes as physical integers, both of which a comparison would
then have to undo. Asking the frame the comparison's own questions is
shorter than translating the profiler's answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import polars as pl

from polspec.bound import Bound
from polspec.constraints import Domain, is_textual
from polspec.dtypes import field_dtypes
from polspec.formats import lookup as _lookup_format

if TYPE_CHECKING:
    from polspec.drift import DriftOptions
    from polspec.spec import ColSpec


@dataclass(frozen=True, slots=True)
class Observed:
    """What a frame's column holds, in the terms its declaration uses.

    Attributes
    ----------
    dtype : pl.DataType
        The column's actual dtype.
    height : int
        Rows in the frame.
    null_count : int
        Nulls in the column; `null_rate` is the fraction.
    extent : Bound | None
        The observed `[min, max]`, typed (a `date` on a Date column), for a
        column whose declaration carries `bounds`.
    length_extent : Bound | None
        The observed `[min, max]` length, for a String or Binary column
        whose declaration carries `string_length`.
    list_length_extent : Bound | None
        The observed `[min, max]` number of elements, for a List column
        whose declaration carries `list_length`. Every other measurement
        of a List column is over its elements.
    outside : tuple[int, tuple]
        Rows holding a value outside the declared finite domain, and up to
        `max_samples` of those values.
    unseen : tuple
        Declared values the column never holds.
    format_failures : tuple[int, tuple]
        Rows failing the declared `format`, and up to `max_samples` of them.
    fields : dict[str, Observed]
        For a Struct column -- or a List of structs -- each field the
        declaration and the data share, measured against its own
        declaration over the structs that are present. A field's `height`
        is that count, so its null rate is how often it is null inside a
        struct, which is what its `null_probability` claims.
    """

    dtype: pl.DataType
    height: int
    null_count: int
    extent: Bound | None = None
    length_extent: Bound | None = None
    list_length_extent: Bound | None = None
    outside: tuple[int, tuple[Any, ...]] = (0, ())
    unseen: tuple[Any, ...] = ()
    format_failures: tuple[int, tuple[Any, ...]] = (0, ())
    fields: dict[str, Observed] = field(default_factory=dict)

    @property
    def null_rate(self) -> float | None:
        return self.null_count / self.height if self.height else None

    @property
    def has_nulls(self) -> bool:
        return self.null_count > 0

    @classmethod
    def of(
        cls, series: pl.Series, declared: ColSpec, options: DriftOptions
    ) -> Observed:
        """Measures `series` in the terms `declared` uses."""
        values = series.drop_nulls()
        measured: dict[str, Any] = {
            "dtype": series.dtype,
            "height": len(series),
            "null_count": series.null_count(),
        }
        if len(values) == 0:
            return cls(**measured)

        if isinstance(values.dtype, (pl.List, pl.Array)):
            # A List column is measured as its elements; only the length is
            # a property of the list itself.
            if declared.list_length is not None and isinstance(values.dtype, pl.List):
                lengths = values.list.len()
                measured["list_length_extent"] = Bound(
                    int(cast("int", lengths.min())), int(cast("int", lengths.max()))
                )
            values = values.explode(empty_as_null=False).drop_nulls()
            if len(values) == 0:
                return cls(**measured)

        if isinstance(values.dtype, pl.Struct):
            # A struct is its fields: each measured as a column of its own.
            declared_dtype = declared.value_dtype
            shared = (
                field_dtypes(declared_dtype)
                if isinstance(declared_dtype, pl.Struct)
                else {}
            )
            measured["fields"] = {
                name: cls.of(values.struct.field(name), declared._field(name), options)
                for name in field_dtypes(values.dtype)
                if name in shared
            }
            return cls(**measured)

        if declared.bounds is not None and _extent_measurable(values.dtype):
            measured["extent"] = Bound(values.min(), values.max())

        if declared.string_length is not None:
            lengths = _lengths(values)
            if lengths is not None:
                measured["length_extent"] = Bound(
                    int(cast("int", lengths.min())), int(cast("int", lengths.max()))
                )

        domain = Domain.of(declared)
        if domain.values is not None and is_textual(values.dtype) == is_textual(
            declared.value_dtype
        ):
            measured["outside"], measured["unseen"] = _against_domain(
                values, domain, options.max_samples
            )

        if declared.format is not None and values.dtype in (pl.String, pl.Utf8):
            fmt = _lookup_format(declared.format)
            if not fmt.is_finite:  # a finite format is a domain, measured above
                failing = values.filter(
                    ~values.to_frame("v").select(fmt.check(pl.col("v")))["v"]
                )
                measured["format_failures"] = (
                    len(failing),
                    tuple(
                        failing.unique(maintain_order=True).head(options.max_samples)
                    ),
                )
        return cls(**measured)


def _extent_measurable(dtype: pl.DataType) -> bool:
    return dtype.is_numeric() or dtype.is_temporal()


def _lengths(values: pl.Series) -> pl.Series | None:
    if values.dtype in (pl.String, pl.Utf8):
        return values.str.len_chars()
    if values.dtype == pl.Binary:
        return values.bin.size()
    return None


def _against_domain(
    values: pl.Series, domain: Domain, max_samples: int
) -> tuple[tuple[int, tuple[Any, ...]], tuple[Any, ...]]:
    """Rows outside the finite `domain`, and declared values never seen.

    Textual columns compare as the strings they hold, the same way
    validation and `Domain` do, so an Enum category and the String choice
    that spells it are one value.
    """
    declared_values = domain.values or ()
    if is_textual(values.dtype):
        observed = values.cast(pl.String)
        declared = pl.Series(list(declared_values), dtype=pl.String, strict=False)
    else:
        observed = values
        declared = pl.Series(list(declared_values), dtype=values.dtype, strict=False)
    outside_mask = ~observed.is_in(declared.to_list())
    outside_values = observed.filter(outside_mask).unique(maintain_order=True)
    outside = (int(outside_mask.sum()), tuple(outside_values.head(max_samples)))
    present = set(observed.unique().to_list())
    unseen = tuple(
        raw
        for raw, cmp in zip(declared_values, declared.to_list(), strict=True)
        if cmp not in present
    )
    return outside, unseen
