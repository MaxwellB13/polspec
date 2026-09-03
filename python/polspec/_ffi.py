"""The one call into the Rust extension.

Everything the engine can object to -- a non-finite bound, a distribution
parameter out of range -- comes back as a `PyValueError`. Re-raising it as
`GenerationError` keeps the Python side's promise that every complaint about
turning a spec into data is a `PolspecError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polspec import _polspec
from polspec.errors import GenerationError

if TYPE_CHECKING:
    import polars as pl


def generate_dataframe(specs, n: int, seed: int) -> pl.DataFrame:
    """Calls `_polspec.generate_dataframe`, re-raising its errors as `GenerationError`."""
    try:
        return _polspec.generate_dataframe(specs, n, seed)
    except ValueError as exc:
        raise GenerationError(str(exc)) from exc
