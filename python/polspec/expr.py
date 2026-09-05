"""A small predicate language that survives a trip through a file.

A `Check` or a `ColSpec.validators` entry written as a raw `pl.Expr` can be
evaluated but not written down: Polars' own serialization is not stable
across versions and is not something a person can read. `col()` builds the
same predicates as an explicit tree instead:

    col("total") >= col("subtotal")
    col("email").str.contains("@")
    col("status").is_in(["NEW", "PAID"]) & col("qty").is_between(1, 100)

Every node knows how to become a Polars expression (`to_expr`), a plain
YAML-safe value (`to_data` / `from_data`), and Python source (`to_source`),
and which columns it reads (`root_names`).

Comparison operators build predicates, as they do on `pl.Expr`, so a `Pred`
has no truth value: `bool(col("a") == 1)` raises. Compare two predicates
structurally with `Pred.equals`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from polspec.errors import SpecError

CmpOp = Literal["eq", "ne", "lt", "le", "gt", "ge"]
ArithOp = Literal["add", "sub", "mul", "div"]
StrOp = Literal["contains", "starts_with", "ends_with", "matches"]

_CMP_SYMBOLS: dict[str, str] = {
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
}
_ARITH_SYMBOLS: dict[str, str] = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_SCALARS = (str, bool, int, float, dt.date, dt.datetime, dt.time, dt.timedelta, bytes)


def _wrap(value: Any) -> Pred:
    if isinstance(value, Pred):
        return value
    if value is None or isinstance(value, _SCALARS):
        return Lit(value)
    raise SpecError(
        f"A predicate operand must be a column, another predicate, or a scalar; "
        f"got {type(value).__name__}"
    )


def _source(node: Pred) -> str:
    """Source for an operand, parenthesised when it is itself an operation."""
    text = node.to_source()
    return text if isinstance(node, (Col, Lit)) else f"({text})"


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Pred:
    """Base class of every predicate node. Build one with `col()`."""

    def to_expr(self) -> pl.Expr:
        """This predicate as the Polars expression that evaluates it."""
        raise NotImplementedError

    def to_data(self) -> Any:
        """This predicate as plain data, for writing to a spec file."""
        raise NotImplementedError

    def to_source(self) -> str:
        """This predicate as the `col(...)` Python that would rebuild it."""
        raise NotImplementedError

    def root_names(self) -> set[str]:
        """Every column name this predicate reads."""
        return set()

    def literals(self) -> list[Any]:
        """Every constant this predicate compares against.

        Used to check a rule's operands against the column's own domain.
        """
        return []

    def rename(self, mapping: Mapping[str, str]) -> Pred:
        """The same predicate with its columns renamed by `mapping`."""
        return self

    def equals(self, other: object) -> bool:
        """Structural equality, since `==` builds a predicate."""
        return isinstance(other, Pred) and self.to_data() == other.to_data()

    def __hash__(self) -> int:
        return hash(repr(self.to_data()))

    def __repr__(self) -> str:
        return self.to_source()

    def __bool__(self) -> bool:
        raise TypeError(
            "A predicate has no truth value; `==` builds a predicate. Use "
            "`Pred.equals` for structural comparison."
        )

    # -- comparison ----------------------------------------------------------

    def __eq__(self, other: object) -> Cmp:  # type: ignore[override]
        return Cmp("eq", self, _wrap(other))

    def __ne__(self, other: object) -> Cmp:  # type: ignore[override]
        return Cmp("ne", self, _wrap(other))

    def __lt__(self, other: Any) -> Cmp:
        return Cmp("lt", self, _wrap(other))

    def __le__(self, other: Any) -> Cmp:
        return Cmp("le", self, _wrap(other))

    def __gt__(self, other: Any) -> Cmp:
        return Cmp("gt", self, _wrap(other))

    def __ge__(self, other: Any) -> Cmp:
        return Cmp("ge", self, _wrap(other))

    # -- arithmetic ----------------------------------------------------------

    def __add__(self, other: Any) -> Arith:
        return Arith("add", self, _wrap(other))

    def __radd__(self, other: Any) -> Arith:
        return Arith("add", _wrap(other), self)

    def __sub__(self, other: Any) -> Arith:
        return Arith("sub", self, _wrap(other))

    def __rsub__(self, other: Any) -> Arith:
        return Arith("sub", _wrap(other), self)

    def __mul__(self, other: Any) -> Arith:
        return Arith("mul", self, _wrap(other))

    def __rmul__(self, other: Any) -> Arith:
        return Arith("mul", _wrap(other), self)

    def __truediv__(self, other: Any) -> Arith:
        return Arith("div", self, _wrap(other))

    def __rtruediv__(self, other: Any) -> Arith:
        return Arith("div", _wrap(other), self)

    # -- boolean -------------------------------------------------------------

    def __and__(self, other: Pred) -> And:
        return And((self, _wrap(other)))

    def __or__(self, other: Pred) -> Or:
        return Or((self, _wrap(other)))

    def __invert__(self) -> Not:
        return Not(self)

    # -- methods -------------------------------------------------------------

    def is_in(self, values: Sequence[Any]) -> IsIn:
        """A predicate true where this value is one of `values`."""
        return IsIn(self, tuple(values))

    def is_null(self) -> IsNull:
        """A predicate true where this value is null."""
        return IsNull(self)

    def is_not_null(self) -> Not:
        """A predicate true where this value is present."""
        return Not(IsNull(self))

    def is_between(self, lower: Any, upper: Any) -> Between:
        """A predicate true where this value falls within `[lower, upper]`."""
        return Between(self, _wrap(lower), _wrap(upper))

    @property
    def str(self) -> _StrNamespace:
        return _StrNamespace(self)


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Col(Pred):
    name: str

    def to_expr(self) -> pl.Expr:
        return pl.col(self.name)

    def to_data(self) -> dict[str, str]:
        return {"col": self.name}

    def to_source(self) -> str:
        return f"col({self.name!r})"

    def root_names(self) -> set[str]:
        return {self.name}

    def rename(self, mapping: Mapping[str, str]) -> Pred:
        return Col(mapping.get(self.name, self.name))


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Lit(Pred):
    value: Any

    def to_expr(self) -> pl.Expr:
        return pl.lit(self.value)

    def to_data(self) -> Any:
        # A dict or list would be read back as a node, so wrap those.
        if isinstance(self.value, (dict, list, tuple)):
            return {"lit": self.value}
        return self.value

    def to_source(self) -> str:
        return repr(self.value)

    def literals(self) -> list[Any]:
        return [self.value]


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Cmp(Pred):
    op: CmpOp
    left: Pred
    right: Pred

    def to_expr(self) -> pl.Expr:
        left, right = self.left.to_expr(), self.right.to_expr()
        return {
            "eq": left == right,
            "ne": left != right,
            "lt": left < right,
            "le": left <= right,
            "gt": left > right,
            "ge": left >= right,
        }[self.op]

    def to_data(self) -> dict[str, list[Any]]:
        return {self.op: [self.left.to_data(), self.right.to_data()]}

    def to_source(self) -> str:
        # A literal on the left must stay on the left: Python would reflect
        # `1 < col("a")` into `col("a") > 1`, a different tree.
        left = (
            f"lit({self.left.to_source()})"
            if isinstance(self.left, Lit)
            else _source(self.left)
        )
        return f"{left} {_CMP_SYMBOLS[self.op]} {_source(self.right)}"

    def root_names(self) -> set[str]:
        return self.left.root_names() | self.right.root_names()

    def literals(self) -> list[Any]:
        return self.left.literals() + self.right.literals()

    def rename(self, mapping: Mapping[str, str]) -> Pred:
        return Cmp(self.op, self.left.rename(mapping), self.right.rename(mapping))


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Arith(Pred):
    op: ArithOp
    left: Pred
    right: Pred

    def to_expr(self) -> pl.Expr:
        left, right = self.left.to_expr(), self.right.to_expr()
        return {
            "add": left + right,
            "sub": left - right,
            "mul": left * right,
            "div": left / right,
        }[self.op]

    def to_data(self) -> dict[str, list[Any]]:
        return {self.op: [self.left.to_data(), self.right.to_data()]}

    def to_source(self) -> str:
        return f"{_source(self.left)} {_ARITH_SYMBOLS[self.op]} {_source(self.right)}"

    def root_names(self) -> set[str]:
        return self.left.root_names() | self.right.root_names()

    def literals(self) -> list[Any]:
        return self.left.literals() + self.right.literals()

    def rename(self, mapping: Mapping[str, str]) -> Pred:
        return Arith(self.op, self.left.rename(mapping), self.right.rename(mapping))


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class And(Pred):
    items: tuple[Pred, ...]

    def to_expr(self) -> pl.Expr:
        return pl.all_horizontal([item.to_expr() for item in self.items])

    def to_data(self) -> dict[str, list[Any]]:
        return {"and": [item.to_data() for item in self.items]}

    def to_source(self) -> str:
        return " & ".join(_source(item) for item in self.items)

    def root_names(self) -> set[str]:
        return set().union(*(item.root_names() for item in self.items))

    def literals(self) -> list[Any]:
        return [v for item in self.items for v in item.literals()]

    def rename(self, mapping: Mapping[str, str]) -> Pred:
        return And(tuple(item.rename(mapping) for item in self.items))


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Or(Pred):
    items: tuple[Pred, ...]

    def to_expr(self) -> pl.Expr:
        return pl.any_horizontal([item.to_expr() for item in self.items])

    def to_data(self) -> dict[str, list[Any]]:
        return {"or": [item.to_data() for item in self.items]}

    def to_source(self) -> str:
        return " | ".join(_source(item) for item in self.items)

    def root_names(self) -> set[str]:
        return set().union(*(item.root_names() for item in self.items))

    def literals(self) -> list[Any]:
        return [v for item in self.items for v in item.literals()]

    def rename(self, mapping: Mapping[str, str]) -> Pred:
        return Or(tuple(item.rename(mapping) for item in self.items))


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Not(Pred):
    item: Pred

    def to_expr(self) -> pl.Expr:
        return ~self.item.to_expr()

    def to_data(self) -> dict[str, Any]:
        return {"not": self.item.to_data()}

    def to_source(self) -> str:
        if isinstance(self.item, IsNull):
            return f"{_source(self.item.item)}.is_not_null()"
        return f"~{_source(self.item)}"

    def root_names(self) -> set[str]:
        return self.item.root_names()

    def literals(self) -> list[Any]:
        return self.item.literals()

    def rename(self, mapping: Mapping[str, str]) -> Pred:
        return Not(self.item.rename(mapping))


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class IsIn(Pred):
    item: Pred
    values: tuple[Any, ...]

    def to_expr(self) -> pl.Expr:
        return self.item.to_expr().is_in(list(self.values))

    def to_data(self) -> dict[str, list[Any]]:
        return {"is_in": [self.item.to_data(), list(self.values)]}

    def to_source(self) -> str:
        return f"{_source(self.item)}.is_in({list(self.values)!r})"

    def root_names(self) -> set[str]:
        return self.item.root_names()

    def literals(self) -> list[Any]:
        return self.item.literals() + list(self.values)

    def rename(self, mapping: Mapping[str, str]) -> Pred:
        return IsIn(self.item.rename(mapping), self.values)


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class IsNull(Pred):
    item: Pred

    def to_expr(self) -> pl.Expr:
        return self.item.to_expr().is_null()

    def to_data(self) -> dict[str, Any]:
        return {"is_null": self.item.to_data()}

    def to_source(self) -> str:
        return f"{_source(self.item)}.is_null()"

    def root_names(self) -> set[str]:
        return self.item.root_names()

    def literals(self) -> list[Any]:
        return self.item.literals()

    def rename(self, mapping: Mapping[str, str]) -> Pred:
        return IsNull(self.item.rename(mapping))


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Between(Pred):
    item: Pred
    lower: Pred
    upper: Pred

    def to_expr(self) -> pl.Expr:
        return self.item.to_expr().is_between(
            self.lower.to_expr(), self.upper.to_expr()
        )

    def to_data(self) -> dict[str, list[Any]]:
        return {
            "between": [self.item.to_data(), self.lower.to_data(), self.upper.to_data()]
        }

    def to_source(self) -> str:
        return (
            f"{_source(self.item)}.is_between("
            f"{self.lower.to_source()}, {self.upper.to_source()})"
        )

    def root_names(self) -> set[str]:
        return (
            self.item.root_names() | self.lower.root_names() | self.upper.root_names()
        )

    def literals(self) -> list[Any]:
        return self.item.literals() + self.lower.literals() + self.upper.literals()

    def rename(self, mapping: Mapping[str, str]) -> Pred:
        return Between(
            self.item.rename(mapping),
            self.lower.rename(mapping),
            self.upper.rename(mapping),
        )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class StrPred(Pred):
    """`contains` is a literal substring; `matches` is a regular expression."""

    op: StrOp
    item: Pred
    pattern: str

    def to_expr(self) -> pl.Expr:
        text = self.item.to_expr().str
        return {
            "contains": text.contains(self.pattern, literal=True),
            "starts_with": text.starts_with(self.pattern),
            "ends_with": text.ends_with(self.pattern),
            "matches": text.contains(self.pattern, literal=False),
        }[self.op]

    def to_data(self) -> dict[str, list[Any]]:
        return {f"str_{self.op}": [self.item.to_data(), self.pattern]}

    def to_source(self) -> str:
        return f"{_source(self.item)}.str.{self.op}({self.pattern!r})"

    def root_names(self) -> set[str]:
        return self.item.root_names()

    def literals(self) -> list[Any]:
        return self.item.literals()

    def rename(self, mapping: Mapping[str, str]) -> Pred:
        return StrPred(self.op, self.item.rename(mapping), self.pattern)


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class StrLen(Pred):
    item: Pred

    def to_expr(self) -> pl.Expr:
        return self.item.to_expr().str.len_chars()

    def to_data(self) -> dict[str, Any]:
        return {"str_len": self.item.to_data()}

    def to_source(self) -> str:
        return f"{_source(self.item)}.str.len_chars()"

    def root_names(self) -> set[str]:
        return self.item.root_names()

    def literals(self) -> list[Any]:
        return self.item.literals()

    def rename(self, mapping: Mapping[str, str]) -> Pred:
        return StrLen(self.item.rename(mapping))


class _StrNamespace:
    """`col("x").str` -- the string operations a predicate can carry."""

    __slots__ = ("_item",)

    def __init__(self, item: Pred) -> None:
        self._item = item

    def contains(self, substring: str) -> StrPred:
        """True where the value contains `substring` literally (not a regex)."""
        return StrPred("contains", self._item, substring)

    def starts_with(self, prefix: str) -> StrPred:
        return StrPred("starts_with", self._item, prefix)

    def ends_with(self, suffix: str) -> StrPred:
        return StrPred("ends_with", self._item, suffix)

    def matches(self, pattern: str) -> StrPred:
        """True where the value matches the regular expression `pattern`."""
        return StrPred("matches", self._item, pattern)

    def len_chars(self) -> StrLen:
        return StrLen(self._item)


def col(name: str) -> Col:
    """A reference to a column, the starting point of every predicate."""
    if not isinstance(name, str) or not name:
        raise SpecError(f"col() takes a non-empty column name, got {name!r}")
    return Col(name)


def lit(value: Any) -> Pred:
    """An explicit literal. Scalars beside an operator are wrapped automatically."""
    return _wrap(value)


# ---------------------------------------------------------------------------
# Reading a predicate back from its data form
# ---------------------------------------------------------------------------

_BINARY: dict[str, type[Cmp] | type[Arith]] = {
    **dict.fromkeys(("eq", "ne", "lt", "le", "gt", "ge"), Cmp),
    **dict.fromkeys(("add", "sub", "mul", "div"), Arith),
}
_STR_OPS: dict[str, StrOp] = {
    "str_contains": "contains",
    "str_starts_with": "starts_with",
    "str_ends_with": "ends_with",
    "str_matches": "matches",
}
KNOWN_OPS: frozenset[str] = frozenset(
    {
        *_BINARY,
        *_STR_OPS,
        "col",
        "lit",
        "and",
        "or",
        "not",
        "is_in",
        "is_null",
        "between",
        "str_len",
    }
)


def _bad(data: Any, why: str) -> SpecError:
    return SpecError(f"Cannot read predicate {data!r}: {why}")


def from_data(data: Any) -> Pred:
    """The predicate a `to_data()` value describes."""
    if data is None or isinstance(data, _SCALARS):
        return Lit(data)
    if not isinstance(data, Mapping):
        raise _bad(data, "expected a scalar or a one-key mapping")
    if len(data) != 1:
        raise _bad(data, "a predicate node is a mapping with exactly one key")
    ((op, payload),) = data.items()
    if op == "col":
        return col(payload)
    if op == "lit":
        return Lit(payload)
    if op in _BINARY:
        left, right = _pair(data, payload)
        return _BINARY[op](op, from_data(left), from_data(right))  # type: ignore[arg-type]
    if op in ("and", "or"):
        if (
            not isinstance(payload, Sequence)
            or isinstance(payload, str)
            or len(payload) < 2
        ):
            raise _bad(data, f"'{op}' takes a list of at least two predicates")
        items = tuple(from_data(item) for item in payload)
        return And(items) if op == "and" else Or(items)
    if op == "not":
        return Not(from_data(payload))
    if op == "is_in":
        item, values = _pair(data, payload)
        if not isinstance(values, Sequence) or isinstance(values, str):
            raise _bad(data, "'is_in' takes [predicate, [values...]]")
        return IsIn(from_data(item), tuple(values))
    if op == "is_null":
        return IsNull(from_data(payload))
    if op == "between":
        if not isinstance(payload, Sequence) or len(payload) != 3:
            raise _bad(data, "'between' takes [predicate, lower, upper]")
        item, lower, upper = payload
        return Between(from_data(item), from_data(lower), from_data(upper))
    if op in _STR_OPS:
        item, pattern = _pair(data, payload)
        if not isinstance(pattern, str):
            raise _bad(data, f"'{op}' takes [predicate, pattern]")
        return StrPred(_STR_OPS[op], from_data(item), pattern)
    if op == "str_len":
        return StrLen(from_data(payload))
    raise _bad(data, f"unknown operation {op!r}; known: {', '.join(sorted(KNOWN_OPS))}")


def _pair(data: Any, payload: Any) -> tuple[Any, Any]:
    if (
        not isinstance(payload, Sequence)
        or isinstance(payload, str)
        or len(payload) != 2
    ):
        raise _bad(data, "expected a two-element list")
    return payload[0], payload[1]


def is_predicate_data(value: Any) -> bool:
    """Whether a YAML value is a predicate in data form (as opposed to a legacy rule condition)."""
    return (
        isinstance(value, Mapping)
        and len(value) == 1
        and next(iter(value)) in KNOWN_OPS
    )
