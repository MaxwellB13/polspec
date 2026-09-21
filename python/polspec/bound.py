from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from polspec.errors import SpecError


@dataclass(frozen=True, slots=True)
class Bound[T]:
    """An inclusive [min, max] range, used for numeric bounds, temporal ranges, and string lengths.

    Either endpoint may be None, meaning that side is unconstrained -- see
    `ColSpec.bounds`, the only field that accepts an open end.
    """

    min: T | None
    max: T | None

    def __post_init__(self) -> None:
        lo: Any = self.min
        hi: Any = self.max
        if lo is not None and hi is not None and lo > hi:
            raise SpecError(f"Bound min ({lo}) must be <= max ({hi})")

    def closed(self) -> tuple[T, T]:
        """Both endpoints of a bound that has both, as `string_length` does."""
        if self.min is None or self.max is None:
            raise SpecError(f"Bound {self} has an open end")
        return self.min, self.max

    @property
    def is_open(self) -> bool:
        """True when either endpoint is unconstrained."""
        return self.min is None or self.max is None

    @property
    def is_open_both(self) -> bool:
        """True when neither endpoint constrains anything."""
        return self.min is None and self.max is None

    def __str__(self) -> str:
        """A readable range, for error messages and generated documentation."""
        if self.min is None and self.max is None:
            return "unconstrained"
        if self.max is None:
            return f">= {self.min}"
        if self.min is None:
            return f"<= {self.max}"
        return f"[{self.min}, {self.max}]"

    @classmethod
    def _coerce(cls, value: Bound | tuple[Any, Any] | list[Any] | None) -> Bound | None:
        if value is None or isinstance(value, Bound):
            return value
        lo, hi = value
        return cls(lo, hi)
