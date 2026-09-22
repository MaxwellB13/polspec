"""Every claim a spec makes, as a `_Constraint` that produces a `Finding`.

A constraint contributes aggregation expressions to one pass over the frame,
then turns the results back into a `Finding`: a count, a few samples, the
facts that make the message actionable, and a way to locate the offending
rows later. All constraints are collected first and evaluated together, so
validating fifty columns costs one scan rather than fifty. Foreign keys are
the exception: each needs its own anti-join.
"""

from polspec.validation.constraints._base import (
    MAX_SAMPLES,
    _Constraint,
    _is_dtype_compatible,
)
from polspec.validation.constraints._relations import (
    _foreign_key_findings,
    _hierarchy_constraints,
    _hierarchy_findings,
)
from polspec.validation.constraints._table import _frame_constraints
from polspec.validation.constraints._values import _column_constraints

__all__ = [
    "MAX_SAMPLES",
    "_Constraint",
    "_column_constraints",
    "_foreign_key_findings",
    "_frame_constraints",
    "_hierarchy_constraints",
    "_hierarchy_findings",
    "_is_dtype_compatible",
]
