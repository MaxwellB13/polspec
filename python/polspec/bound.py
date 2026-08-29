from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Bound(Generic[T]):
    """An inclusive [min, max] range, used for numeric bounds, temporal ranges, and string lengths."""

    min: T
    max: T

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise ValueError(f"Bound min ({self.min}) must be <= max ({self.max})")

    @classmethod
    def _coerce(cls, value: Bound | tuple[Any, Any] | list[Any] | None) -> Bound | None:
        if value is None or isinstance(value, Bound):
            return value
        lo, hi = value
        return cls(lo, hi)
