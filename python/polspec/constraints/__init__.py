"""One definition per constraint, for both sides of the spec.

polspec makes claims about a table twice over: generation has to produce
data that satisfies them, validation has to detect data that does not. Where
the two carry their own copy of a claim they drift, and the round-trip
breaks. This package holds the parts they share -- what values a column may
hold (`Domain`), and the order the passes that rewrite a generated frame
have to run in (`Pass`, `order`) -- so both sides read the same declaration.
"""

from __future__ import annotations

from polspec.constraints.domain import Domain
from polspec.constraints.passes import (
    Pass,
    order,
    ordered_passes,
    passes_of,
    rewritable_members,
)

__all__ = [
    "Domain",
    "Pass",
    "order",
    "ordered_passes",
    "passes_of",
    "rewritable_members",
]
