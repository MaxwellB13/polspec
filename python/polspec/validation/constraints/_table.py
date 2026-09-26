"""What the frame as a whole asserts: composite uniqueness and `__checks__`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from polspec.check import Check

from polspec.validation.constraints._base import Constraint, struct_of
from polspec.validation.report import FindingCode


@dataclass(kw_only=True)
class _CompositeUnique(Constraint):
    columns: tuple[str, ...] = ()
    code: FindingCode = "unique_together"

    def involved(self) -> tuple[str, ...]:
        return self.columns

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Composite unique key {list(self.columns)} violated: found {count} "
            f"duplicate row(s). Duplicate samples: {samples}"
        )


@dataclass(kw_only=True)
class _FrameCheck(Constraint):
    check: Check
    code: FindingCode = "check"

    def involved(self) -> tuple[str, ...]:
        return tuple(self.check.expr.meta.root_names())

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"check": self.check.name, "condition": str(self.check.expr)}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        described = f" ({self.check.description})" if self.check.description else ""
        return (
            f"Check '{self.check.name}' failed: found {count} row(s) violating "
            f"condition {self.check.expr}{described}"
        )


def frame_constraints(
    unique_together: Sequence[Sequence[str]] | None,
    checks: Sequence[Check] | None,
    df_col_names: Sequence[str],
) -> list[Constraint]:
    """Constraints spanning several columns rather than belonging to one."""
    constraints: list[Constraint] = []

    for index, group in enumerate(unique_together or ()):
        columns = tuple(group)
        if not all(c in df_col_names for c in columns):
            continue
        key_struct = pl.struct([pl.col(c) for c in columns])
        all_present = pl.all_horizontal([pl.col(c).is_not_null() for c in columns])
        constraints.append(
            _CompositeUnique(
                key=f"composite_{index}",
                mask=all_present & key_struct.is_duplicated(),
                sample_expr=key_struct,
                columns=columns,
            )
        )

    for check in checks or ():
        named = check.expr.meta.root_names()
        if not all(c in df_col_names for c in named):
            # Its mask cannot be evaluated -- Polars would raise, and inspect()
            # never does for a frame that fails. The missing columns are a
            # `missing_columns` finding already, or were allowed to be absent.
            continue
        involved = [c for c in named if c in df_col_names]
        constraints.append(
            _FrameCheck(
                key=f"check:{check.name}",
                mask=check._failure_mask(),
                sample_expr=struct_of(involved),
                check=check,
            )
        )
    return constraints
