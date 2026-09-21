from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec.constants import _FILLED_IN
from polspec.errors import SpecError
from polspec.expr import Pred


@dataclass(eq=False, frozen=True, slots=True)
class Check:
    """A declarative multi-column validation constraint evaluated as a Polars boolean expression.

    Parameters
    ----------
    expr : pl.Expr | Pred
        The boolean condition each row must satisfy: a Polars expression, or a
        predicate built with `polspec.col()`. A predicate can be written to a
        YAML or Python spec file; a raw expression cannot.
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

    # Annotated with what a constructed Check holds -- an expression and a
    # name -- while the constructor also accepts a predicate and no name; see
    # the `__init__` below, which exists only for type checkers.
    expr: pl.Expr
    name: str = _FILLED_IN
    description: str | None = None
    ignore_nulls: bool = True
    pred: Pred | None = None

    if TYPE_CHECKING:

        def __init__(
            self,
            expr: pl.Expr | Pred,
            name: str | None = None,
            description: str | None = None,
            ignore_nulls: bool = True,
            pred: Pred | None = None,
        ) -> None: ...

    def __post_init__(self) -> None:
        given: Any = self.expr
        if isinstance(given, Pred):
            object.__setattr__(self, "pred", given)
            object.__setattr__(self, "expr", given.to_expr())
        elif not isinstance(given, pl.Expr):
            raise SpecError(
                "Check expr must be a polars Expr or a polspec predicate "
                f"(built with col()), got {type(given).__name__}"
            )
        name: Any = self.name
        if name is None:
            default = repr(self.pred) if self.pred is not None else str(self.expr)
            object.__setattr__(self, "name", default)

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
