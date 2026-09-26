"""Every claim a spec makes, as a `Constraint` that produces a `Finding`.

A constraint contributes aggregation expressions to one pass over the frame,
then turns the results back into a `Finding`: a count, a few samples, the
facts that make the message actionable, and a way to locate the offending
rows later. All constraints are collected first and evaluated together, so
validating fifty columns costs one scan rather than fifty. Foreign keys are
the exception: each needs its own anti-join.
"""

from polspec.validation.constraints._base import (
    MAX_SAMPLES,
    Constraint,
    is_dtype_compatible,
)
from polspec.validation.constraints._relations import (
    foreign_key_findings,
    hierarchy_constraints,
    hierarchy_findings,
)
from polspec.validation.constraints._table import frame_constraints
from polspec.validation.constraints._values import column_constraints

__all__ = [
    "MAX_SAMPLES",
    "Constraint",
    "column_constraints",
    "foreign_key_findings",
    "frame_constraints",
    "hierarchy_constraints",
    "hierarchy_findings",
    "is_dtype_compatible",
]
