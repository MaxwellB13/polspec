from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(eq=False, frozen=True, slots=True)
class Check:
    """A declarative multi-column validation constraint evaluated as a Polars boolean expression.

    Parameters
    ----------
    expr : pl.Expr
        The Polars boolean expression that each row must satisfy (evaluating to True).
    name : str | None, optional
        A human-readable identifier for the check constraint (e.g. 'total_gte_subtotal').
        If omitted, defaults to the string representation of the expression.
    description : str | None, optional
        An optional description detailing the business logic or rationale for this check.
    ignore_nulls : bool, default True
        Whether rows evaluating to null in the check condition are considered valid
        (standard SQL CHECK constraint semantics). If False, null results are treated as failures.

    Examples
    --------
    >>> check = Check(pl.col("total") >= pl.col("subtotal"), name="total_gte_subtotal")
    """

    expr: pl.Expr
    name: str | None = None
    description: str | None = None
    ignore_nulls: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.expr, pl.Expr):
            raise TypeError(
                f"Check expr must be a polars Expr (pl.Expr), got {type(self.expr).__name__}"
            )
        if self.name is None:
            object.__setattr__(self, "name", str(self.expr))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Check):
            return False
        return (
            self.name == other.name
            and str(self.expr) == str(other.expr)
            and self.description == other.description
            and self.ignore_nulls == other.ignore_nulls
        )

    def __hash__(self) -> int:
        return hash((self.name, str(self.expr), self.description, self.ignore_nulls))

    def _failure_mask(self) -> pl.Expr:
        """Returns the Polars boolean mask for rows that violate this check."""
        if self.ignore_nulls:
            return ~self.expr.fill_null(True)
        return ~self.expr.fill_null(False)
