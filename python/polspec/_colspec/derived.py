"""A derived column: one computed from others with a `col()` expression.

What a column can declare beside `derive`. The columns it reads, and
whether they can be null, are the table's to check: see
`TableSpec._validate_derived`.

Internal: not part of the public API.
"""

from __future__ import annotations

import polars as pl

from polspec import spec as colspec_module
from polspec._colspec.common import column_kind
from polspec.constants import DEFAULT_NULL_PROBABILITY
from polspec.errors import SpecError
from polspec.expr import Pred, root_names

# The fields that say what values a column holds. A derived column's values
# are its expression's, so each is refused beside it: a bound generation
# cannot keep would fail the spec's own round trip.
_DECIDED_BY_THE_INPUTS = (
    "bounds",
    "choices",
    "rules",
    "distribution",
    "format",
    "extra_values",
    "pattern",
    "string_length",
)


def validate_derive(spec: colspec_module.ColSpec) -> None:
    derive = spec.derive
    if derive is None:
        return
    if isinstance(derive, pl.Expr):
        raise SpecError(
            "ColSpec.derive takes a col() expression, such as "
            "col('qty') * col('price'), not a pl.Expr: a spec file has to be "
            "able to hold it. Write it with polspec.col."
        )
    if not isinstance(derive, Pred):
        raise SpecError(
            f"ColSpec.derive must be a col() expression, got {type(derive).__name__}"
        )
    if not root_names(derive):
        raise SpecError(
            f"ColSpec.derive {derive!r} reads no column; a constant column is "
            "choices=[value]"
        )
    if column_kind(spec.dtype) in ("list", "struct", "map", "null"):
        raise SpecError(f"ColSpec.derive is for a scalar column, not {spec.dtype!r}")
    declared = [
        name for name in _DECIDED_BY_THE_INPUTS if getattr(spec, name) not in (None, ())
    ]
    if spec.unique:
        declared.append("unique")
    if spec.default is not None:
        declared.append("default")
    if spec.nullable and spec.null_probability != DEFAULT_NULL_PROBABILITY:
        declared.append("null_probability")
    if spec.nan_probability:
        declared.append("nan_probability")
    if declared:
        raise SpecError(
            f"ColSpec.derive={derive!r} decides the column's values, so it "
            f"cannot also declare {', '.join(declared)}. A derived value is "
            "null exactly where its expression is: declare nullable=True when "
            "an input can be null."
        )


def derived_values(spec: colspec_module.ColSpec) -> pl.Expr:
    """What `spec.derive` computes, as the column's dtype holds it. A value
    the dtype cannot hold is null here, for validation to report; generation
    casts strictly, and refuses one instead."""
    if spec.derive is None:
        raise ValueError("derived_values needs a derived column")
    return spec.derive.to_expr().cast(spec.dtype, strict=False)
