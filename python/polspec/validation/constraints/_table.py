"""What the frame as a whole asserts: composite uniqueness and `__checks__`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polspec.check import Check

from polspec.validation.constraints._base import _Constraint, _struct_of
from polspec.validation.constraints._values import _CompositeUnique, _FrameCheck


def _frame_constraints(
    unique_together: Sequence[Sequence[str]] | None,
    checks: Sequence[Check] | None,
    df_col_names: Sequence[str],
) -> list[_Constraint]:
    """Constraints spanning several columns rather than belonging to one."""
    constraints: list[_Constraint] = []

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
        involved = [c for c in check.expr.meta.root_names() if c in df_col_names]
        constraints.append(
            _FrameCheck(
                key=f"check:{check.name}",
                mask=check._failure_mask(),
                sample_expr=_struct_of(involved),
                check=check,
            )
        )
    return constraints
