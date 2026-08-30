from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Bound[T]:
    """An inclusive [min, max] range, used for numeric bounds, temporal ranges, and string lengths.

    Either endpoint may be None, meaning that side is unconstrained -- see
    `ColSpec.bounds`, the only field that accepts an open end.
    """

    min: T | None
    max: T | None

    def __post_init__(self) -> None:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"Bound min ({self.min}) must be <= max ({self.max})")

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
