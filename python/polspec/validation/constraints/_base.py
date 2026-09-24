"""One checkable claim, measured by counting the rows that violate a mask.

Subclasses supply the mask, the wording and the details; the aliases tying
the two halves together stay private to the instance. Every constraint a
spec implies is collected first and evaluated together, so validating fifty
columns costs one scan rather than fifty.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from polspec.dtypes import _typed_values, element_dtype, field_dtypes
from polspec.validation.report import Finding, FindingCode

MAX_SAMPLES = 5


@dataclass(kw_only=True)
class _Constraint:
    """One checkable claim, measured by counting the rows that violate a mask.

    Subclasses supply the mask, the wording and the details; the aliases
    tying the two halves together stay private to the instance.
    """

    key: str
    mask: pl.Expr
    sample_expr: pl.Expr | None = None
    unique_samples: bool = True
    code: FindingCode

    def _alias(self, suffix: str) -> str:
        return f"__val__{self.key}__{suffix}"

    def aggregations(self) -> list[pl.Expr]:
        exprs = [self.mask.sum().alias(self._alias("cnt"))]
        if self.sample_expr is not None:
            samples = self.sample_expr.filter(self.mask)
            if self.unique_samples:
                samples = samples.unique(maintain_order=True)
            exprs.append(
                samples.head(MAX_SAMPLES).implode().alias(self._alias("samples"))
            )
        return exprs

    def failure(self, stats: dict[str, list]) -> Finding | None:
        count = stats[self._alias("cnt")][0]
        if not count:
            return None
        samples = self._samples(stats)
        mask = self.mask
        return Finding(
            code=self.code,
            key=self.key,
            message=self.message(count, samples, stats),
            columns=self.involved(),
            count=int(count),
            samples=tuple(samples),
            details=self.details(stats),
            _locate=lambda lf: lf.filter(mask),
        )

    def _samples(self, stats: dict[str, list]) -> list:
        if self.sample_expr is None:
            return []
        raw = stats[self._alias("samples")][0]
        return list(raw) if raw is not None else []

    def involved(self) -> tuple[str, ...]:
        return ()

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Dtype compatibility
# ---------------------------------------------------------------------------


def _is_dtype_compatible(
    expected: pl.DataType, actual: pl.DataType, *, strict: bool
) -> bool:
    """Whether `actual` can stand in for the declared `expected` dtype.

    Strict mode demands the same dtype, with String/Utf8 treated as one. The
    default is permissive about widening and about textual columns arriving as
    String rather than their declared Enum/Categorical, since that is how they
    come back from CSV and JSON.
    """
    if strict:
        if expected in (pl.String, pl.Utf8):
            return actual in (pl.String, pl.Utf8)
        return actual == expected

    if isinstance(expected, pl.Enum):
        return isinstance(actual, pl.Enum) or actual in (
            pl.String,
            pl.Utf8,
            pl.Categorical,
        )
    if isinstance(expected, pl.Categorical) or expected == pl.Categorical:
        return actual in (pl.String, pl.Utf8, pl.Categorical) or isinstance(
            actual, (pl.Enum, pl.Categorical)
        )
    if expected in (pl.String, pl.Utf8):
        return actual in (pl.String, pl.Utf8)
    if expected.is_integer():
        return actual.is_integer()
    if expected.is_float():
        return actual.is_float() or actual.is_integer()
    if expected.is_decimal():
        # A Decimal comes back from CSV and JSON as a float or an integer;
        # another Decimal stands in whatever its precision, since the
        # bounds are checked on the values either way.
        return actual.is_decimal() or actual.is_float() or actual.is_integer()
    if expected.is_temporal():
        return actual.is_temporal()
    if isinstance(expected, pl.List):
        return isinstance(actual, pl.List) and _is_dtype_compatible(
            element_dtype(expected), element_dtype(actual), strict=strict
        )
    if isinstance(expected, pl.Array):
        return (
            isinstance(actual, pl.Array)
            and actual.size == expected.size
            and _is_dtype_compatible(
                element_dtype(expected), element_dtype(actual), strict=strict
            )
        )
    if isinstance(expected, pl.Struct):
        # The fields by name, each compatible: order is how the data was
        # written, not what it claims, and a field missing or added is a
        # different struct.
        if not isinstance(actual, pl.Struct):
            return False
        want, got = field_dtypes(expected), field_dtypes(actual)
        return want.keys() == got.keys() and all(
            _is_dtype_compatible(want[f], got[f], strict=strict) for f in want
        )
    return actual == expected


def _as_strings(values: Sequence[Any], dtype: pl.DataType) -> list[str]:
    """`values` as the strings a column of `dtype` holds them as, so a choice
    of `True` on a String column compares as `"true"` -- the same form
    generation produces -- rather than as Python's `str(True)`.
    """
    try:
        return _typed_values(values, dtype).cast(pl.String).to_list()
    except Exception:  # noqa: BLE001 - values the dtype cannot hold fall back to str()
        return [str(v) for v in values]


def _struct_of(names: Sequence[str]) -> pl.Expr | None:
    """A struct of the named columns, for sampling multi-column claims."""
    return pl.struct([pl.col(n) for n in names]) if names else None
