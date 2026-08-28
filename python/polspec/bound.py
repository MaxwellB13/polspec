from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Bound:
    """An inclusive [min, max] range, used for numeric bounds and string lengths."""

    min: float | int
    max: float | int

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise ValueError(f"Bound min ({self.min}) must be <= max ({self.max})")

    @classmethod
    def _coerce(
        cls, value: Bound | tuple[float | int, float | int] | None
    ) -> Bound | None:
        if value is None or isinstance(value, Bound):
            return value
        lo, hi = value
        return cls(lo, hi)
