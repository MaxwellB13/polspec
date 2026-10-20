"""What the frame as a whole asserts: composite uniqueness, `__checks__`,
and the order its rows come in."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from polspec.check import Check

from polspec import frames
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
class _Order(Constraint):
    columns: tuple[str, ...] = ()
    descending: tuple[bool, ...] = ()
    code: FindingCode = "order"

    def involved(self) -> tuple[str, ...]:
        return self.columns

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"order_by": list(self.columns), "descending": list(self.descending)}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Rows out of order by {_described(self.columns, self.descending)}: "
            f"{count} row(s) belong before the row above them. Samples: {samples}"
        )


def _described(columns: Sequence[str], descending: Sequence[bool]) -> str:
    return ", ".join(
        f"{name} (descending)" if desc else name
        for name, desc in zip(columns, descending, strict=True)
    )


def _out_of_order(columns: Sequence[str], descending: Sequence[bool]) -> pl.Expr:
    """Each row that belongs before the row above it, as `DataFrame.sort(by,
    descending, nulls_last=True)` orders rows.

    Each column becomes its dense rank, which orders values exactly as that
    sort does -- NaN, categories and every dtype alike -- with a null ranked
    after every value. The ranks then compare as tuples, the first column
    the major key: a row is out of order when the tuple above it is greater.
    """
    ranks = [
        frames.column(name).rank("dense", descending=desc).fill_null(pl.len() + 1)
        for name, desc in zip(columns, descending, strict=True)
    ]
    later = pl.lit(False)
    for rank in reversed(ranks):
        above = rank.shift(1)
        later = (above > rank) | ((above == rank) & later)
    return later.fill_null(False)


@dataclass(kw_only=True)
class _FrameCheck(Constraint):
    check: Check
    code: FindingCode = "check"

    def involved(self) -> tuple[str, ...]:
        return _named_once(self.check.expr)

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"check": self.check.name, "condition": str(self.check.expr)}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        described = f" ({self.check.description})" if self.check.description else ""
        return (
            f"Check '{self.check.name}' failed: found {count} row(s) violating "
            f"condition {self.check.expr}{described}"
        )


def _named_once(expr: pl.Expr) -> tuple[str, ...]:
    """The columns `expr` reads, each once, in the order it first reads them.

    `meta.root_names()` lists a column once per mention, so a range check --
    `(col("a") > 0) & (col("a") < 10)` -- names `a` twice, and a struct of
    its samples built from that list had two fields called `a`: Polars
    refused it, and validation raised `DuplicateError`.
    """
    return tuple(dict.fromkeys(expr.meta.root_names()))


def frame_constraints(
    unique_together: Sequence[Sequence[str]] | None,
    checks: Sequence[Check] | None,
    df_col_names: Sequence[str],
    *,
    order: tuple[Sequence[str], Sequence[bool]] | None = None,
) -> list[Constraint]:
    """Constraints spanning several columns rather than belonging to one."""
    constraints: list[Constraint] = []

    if order is not None and order[0] and all(c in df_col_names for c in order[0]):
        columns, descending = tuple(order[0]), tuple(order[1])
        constraints.append(
            _Order(
                key="order",
                mask=_out_of_order(columns, descending),
                sample_expr=struct_of(columns),
                columns=columns,
                descending=descending,
            )
        )

    for index, group in enumerate(unique_together or ()):
        columns = tuple(group)
        if not all(c in df_col_names for c in columns):
            continue
        key_struct = pl.struct(frames.columns(columns))
        all_present = pl.all_horizontal(
            [frames.column(c).is_not_null() for c in columns]
        )
        constraints.append(
            _CompositeUnique(
                key=f"composite_{index}",
                mask=all_present & key_struct.is_duplicated(),
                sample_expr=key_struct,
                columns=columns,
            )
        )

    for check in checks or ():
        named = _named_once(check.expr)
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
