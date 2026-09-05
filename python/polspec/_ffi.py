"""The only module that touches the Rust extension.

Three things happen here and nowhere else:

- The extension is imported lazily, so `import polspec` -- and with it
  validation, serialization, the registry and the report renderers -- works
  without a built extension. Only generation needs it, and asking for it
  without one raises one actionable `ImportError`.
- A `ColumnPlan` is built from keyword arguments, so callers never assemble a
  positional tuple.
- Everything the engine objects to -- an unknown kind, a weight vector of the
  wrong length, a distribution parameter out of range -- comes back as a
  `ValueError` and is re-raised as `GenerationError`, keeping the promise that
  every complaint about turning a spec into data is a `PolspecError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from polspec.errors import GenerationError

if TYPE_CHECKING:
    import polars as pl

    from polspec._polspec import ColumnPlan

_MISSING = (
    "polspec's Rust extension (polspec._polspec) is not built, so data cannot be "
    "generated. Install polspec from a wheel (`pip install polspec`), or build it "
    "with `uv sync` / `maturin develop` in a checkout. Validation, spec files and "
    "the registry work without it."
)


def _extension() -> Any:
    try:
        from polspec import _polspec
    except ImportError as exc:
        raise ImportError(_MISSING) from exc
    return _polspec


def column_plan(name: str, kind: str, **options: Any) -> ColumnPlan:
    """A `ColumnPlan`, with the engine's own validation errors as `GenerationError`."""
    try:
        return _extension().ColumnPlan(name, kind, **options)
    except ValueError as exc:
        raise GenerationError(str(exc)) from exc


def generate_dataframe(
    plans: Sequence[ColumnPlan], n: int, seed: int | None
) -> pl.DataFrame:
    """Fills every planned column, re-raising the engine's errors as `GenerationError`."""
    try:
        return _extension().generate_dataframe(list(plans), n, seed)
    except ValueError as exc:
        raise GenerationError(str(exc)) from exc


def distribution_params(name: str) -> list[tuple[str, float, bool]]:
    """The engine's `(name, default, must_be_positive)` for one distribution."""
    return _extension().distribution_params(name)


def distributions() -> list[str]:
    """The distributions the engine can sample."""
    return _extension().distributions()


def kinds() -> list[str]:
    """The column kinds a `ColumnPlan` accepts."""
    return _extension().kinds()
