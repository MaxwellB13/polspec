"""A derived column as a constraint: each value is its expression's.

Equal, for an integer, a decimal, text or a flag; for a float, within a
relative 1e-9 of it, so a pipeline that multiplies in another order -- or in
SQL -- still passes. A null is right exactly where the expression is null,
and NaN where it is NaN.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec import frames
from polspec._colspec.derived import derived_values
from polspec.expr import root_names
from polspec.validation.constraints._base import Constraint
from polspec.validation.report import FindingCode

if TYPE_CHECKING:
    from polspec.expr import Pred
    from polspec.spec import ColSpec

#: How far a derived float may be from its expression, relative to the larger.
RELATIVE_TOLERANCE = 1e-9


@dataclass(kw_only=True)
class _Derived(Constraint):
    column: str
    derive: Pred
    code: FindingCode = "derived"

    def involved(self) -> tuple[str, ...]:
        return (self.column, *sorted(root_names(self.derive) - {self.column}))

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"derive": repr(self.derive)}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': found {count} value(s) not equal to "
            f"{self.derive!r}. Violating samples: {samples}"
        )


def derived_constraint(
    name: str, spec: ColSpec, actual_dtype: pl.DataType, df_col_names: Sequence[str]
) -> list[Constraint]:
    """The derived claim on one column, or none when the frame lacks a
    column it reads -- reported through `missing_cols`, and with nothing to
    compute the value from."""
    if spec.derive is None or not root_names(spec.derive) <= set(df_col_names):
        return []
    column = frames.column(name)
    expected = derived_values(spec)
    if actual_dtype.is_float() or spec.dtype.is_float():
        # NaN and infinity first: a tolerance relative to either is no
        # tolerance (`inf <= inf` holds), so each matches only itself.
        either_nan = column.is_nan() | expected.is_nan()
        either_infinite = column.is_infinite() | expected.is_infinite()
        close = (column - expected).abs() <= RELATIVE_TOLERANCE * pl.max_horizontal(
            column.abs(), expected.abs()
        )
        agrees = (
            pl.when(either_nan)
            .then(column.is_nan() & expected.is_nan())
            .when(either_infinite)
            .then(column == expected)
            .otherwise((column == expected) | close)
        )
        matches = (
            pl.when(column.is_null() | expected.is_null())
            .then(column.is_null() & expected.is_null())
            .otherwise(agrees)
        )
    else:
        matches = column.eq_missing(expected)
    return [
        _Derived(
            key=f"{name}__derived",
            mask=~matches.fill_null(False),
            sample_expr=column,
            column=name,
            derive=spec.derive,
        )
    ]
