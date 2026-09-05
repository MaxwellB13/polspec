"""What a column's values may be, written down once.

Generation samples from a column's domain and validation checks values
against it. The two only agree if there is one definition of it, and
`Domain` is that definition: the finite set a column draws from, the range
it must fall in, or neither.

Foreign keys are the reason this is a class rather than a helper. A key
overwrites its column with values from a parent, so the parent's domain has
to fit inside the child's -- and whether it does is decidable from the two
declarations alone. `rejects` is that decision, so a contradiction is
reported at declaration instead of surfacing as generated data that fails
its own spec.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec.bound import Bound

if TYPE_CHECKING:
    from polspec.spec import ColSpec


MAX_LISTED = 8


def _listed(values: Sequence[Any]) -> str:
    """`values` for an error message, with a long domain cut short.

    A domain of two is worth spelling out; one of two hundred would bury the
    sentence that explains the problem.
    """
    shown = [repr(v) for v in values[:MAX_LISTED]]
    if len(values) > MAX_LISTED:
        shown.append(f"... {len(values) - MAX_LISTED} more")
    return f"[{', '.join(shown)}]"


def _is_textual(dtype: pl.DataType) -> bool:
    return dtype in (pl.String, pl.Utf8, pl.Categorical) or isinstance(
        dtype, (pl.Enum, pl.Categorical)
    )


@dataclass(frozen=True, slots=True)
class Domain:
    """The values one column is declared to hold.

    `values` is the finite set the column draws from -- its `choices`, an
    `Enum`'s categories, or the two narrowed against each other -- and is
    None for a column with no finite domain. `bounds` is the inclusive
    range, None when unconstrained on both sides. A column may have either,
    both or neither.
    """

    dtype: pl.DataType
    values: tuple[Any, ...] | None = None
    bounds: Bound | None = None

    @classmethod
    def of(cls, spec: ColSpec) -> Domain:
        """The domain a `ColSpec` declares."""
        values: tuple[Any, ...] | None = None
        if isinstance(spec.dtype, pl.Enum):
            categories = spec.dtype.categories.to_list()
            values = (
                tuple(c for c in spec.choices if c in categories)
                if spec.choices is not None
                else tuple(categories)
            )
        elif spec.choices is not None:
            values = tuple(spec.choices)

        bounds = spec.bounds if spec.bounds is not None else None
        if bounds is not None and bounds.is_open_both:
            bounds = None
        return cls(dtype=spec.dtype, values=values, bounds=bounds)

    @property
    def is_open(self) -> bool:
        """True when nothing about this column's values is declared."""
        return self.values is None and self.bounds is None

    def __str__(self) -> str:
        if self.values is not None:
            return f"one of {_listed(self.values)}"
        if self.bounds is not None:
            return f"bounds {self.bounds}"
        return f"any {self.dtype}"

    # ------------------------------------------------------------------
    # Whether one domain's values fit inside another's
    # ------------------------------------------------------------------

    def _comparable(self, values: tuple[Any, ...], dtype: pl.DataType) -> list[Any]:
        """`values` in the form this domain compares them in.

        Textual domains compare by the string a column of that dtype holds,
        so an `Enum` category and the `String` choice that spells it are one
        value. Everything else compares as written.
        """
        if not (_is_textual(self.dtype) and _is_textual(dtype)):
            return list(values)
        try:
            return pl.Series(list(values), dtype=pl.String, strict=False).to_list()
        except Exception:  # noqa: BLE001 - fall back to comparing as written
            return list(values)

    def rejects(self, other: Domain) -> str | None:
        """Why a value drawn from `other` might not be one this domain accepts.

        Returns a phrase naming the mismatch, or None when every value
        `other` can produce fits here. Nothing is guessed: a domain that
        declares nothing accepts everything, and two domains whose values
        cannot be compared are left alone.
        """
        if self.is_open:
            return None
        try:
            return self._rejects(other)
        except (TypeError, ValueError):  # incomparable declarations prove nothing
            return None

    def _rejects(self, other: Domain) -> str | None:
        if self.values is not None:
            mine = set(self._comparable(self.values, self.dtype))
            if other.values is None:
                return f"{other} is not limited to {self}"
            outside = [
                raw
                for raw, cmp in zip(
                    other.values,
                    self._comparable(other.values, other.dtype),
                    strict=True,
                )
                if cmp not in mine
            ]
            if outside:
                return f"{_listed(outside)} is not {self}"
            return None

        bounds = self.bounds
        if bounds is None:  # unreachable: `is_open` already returned
            return None
        lo, hi = bounds.min, bounds.max

        def outside(value: Any) -> bool:
            return (lo is not None and value < lo) or (hi is not None and value > hi)

        if other.values is not None:
            escaping = [v for v in other.values if outside(v)]
            if escaping:
                return f"{_listed(escaping)} falls outside {self}"
            return None
        if other.bounds is None:
            return f"an unbounded {other.dtype} does not fit {self}"
        # An open end on the other side is unbounded in that direction, so it
        # escapes any end this domain constrains.
        omin, omax = other.bounds.min, other.bounds.max
        below = lo is not None and (omin is None or omin < lo)
        above = hi is not None and (omax is None or omax > hi)
        if below or above:
            return f"bounds {other.bounds} do not fit inside {bounds}"
        return None
