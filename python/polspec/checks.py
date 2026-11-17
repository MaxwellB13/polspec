"""`Check` -- a claim about a whole row, made across its columns -- and
`check`, which declares one as a function on a `FrameSpec` class.

Internal: not part of the public API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any, overload

import polars as pl

from polspec.constants import FILLED_IN
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
    # Everything after this is keyword-only: a positional call would freeze
    # the order of the fields, and say nothing about what each argument is.
    _: KW_ONLY
    name: str = FILLED_IN
    description: str | None = None
    ignore_nulls: bool = True
    # The predicate a check was built from, when it was: what lets it be
    # written to a file. Filled in from `expr`, never passed.
    pred: Pred | None = field(default=None, init=False)

    if TYPE_CHECKING:

        def __init__(
            self,
            expr: pl.Expr | Pred,
            *,
            name: str | None = None,
            description: str | None = None,
            ignore_nulls: bool = True,
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


@dataclass(frozen=True, slots=True)
class DeclaredCheck:
    """A function `@check` marked in a `FrameSpec` class body, waiting for
    the class to be built: it becomes a `Check` named after the function.
    Not callable, so nothing mistakes it for a method."""

    function: Callable[[], pl.Expr | Pred]
    name: str | None
    description: str | None
    ignore_nulls: bool

    def as_check(self, attr: str) -> Check:
        """The `Check` this declares, named `name` or else `attr`."""
        expr = self.function()
        if not isinstance(expr, (pl.Expr, Pred)):
            raise SpecError(
                f"@check {attr}() must return a polars Expr or a polspec "
                f"predicate (built with col()), got {type(expr).__name__}"
            )
        return Check(
            expr,
            name=self.name or attr,
            description=self.description,
            ignore_nulls=self.ignore_nulls,
        )


type _CheckFunction = Callable[[], pl.Expr | Pred]


@overload
def check(function: _CheckFunction, /) -> DeclaredCheck: ...


@overload
def check(
    *,
    name: str | None = None,
    description: str | None = None,
    ignore_nulls: bool = True,
) -> Callable[[_CheckFunction], DeclaredCheck]: ...


def check(
    function: _CheckFunction | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    ignore_nulls: bool = True,
) -> DeclaredCheck | Callable[[_CheckFunction], DeclaredCheck]:
    """Declares a function in a `FrameSpec` class body as one of its checks,
    named after the function. *Added in 1.7.*

    The function takes no arguments and returns the condition each row must
    satisfy: a Polars expression, or a predicate built with `polspec.col()`,
    which a spec file can hold. It is called once, when the class is built,
    and the check joins `__checks__` after any listed there, in the order
    the functions are declared. The function's name does not stay on the
    class: the check is `Orders.spec.checks`.

    Parameters
    ----------
    function : callable, optional
        The function, when `@check` is used bare.
    name : str, optional
        The check's name, when it should not be the function's.
    description : str, optional
        What the check means, for a reader: `to_markdown` shows it.
    ignore_nulls : bool, default True
        As `Check` takes it: a row whose condition is null passes.

    Examples
    --------
    >>> class Orders(FrameSpec):
    ...     subtotal = ColSpec(pl.Float64, bounds=(0, 1000))
    ...     total = ColSpec(pl.Float64, bounds=(0, 1100))
    ...
    ...     @check(description="tax never makes a total smaller")
    ...     def total_gte_subtotal():
    ...         return pl.col("total") >= pl.col("subtotal")
    >>> [c.name for c in Orders.spec.checks]
    ['total_gte_subtotal']
    """

    def declare(given: _CheckFunction) -> DeclaredCheck:
        if not callable(given):
            raise TypeError(f"@check decorates a function, got {type(given).__name__}")
        return DeclaredCheck(given, name, description, ignore_nulls)

    return declare if function is None else declare(function)


def failure_mask(check: Check) -> pl.Expr:
    """Returns the Polars boolean mask for rows that violate this check."""
    if check.ignore_nulls:
        return ~check.expr.fill_null(True)
    return ~check.expr.fill_null(False)
