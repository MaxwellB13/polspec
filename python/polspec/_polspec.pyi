"""Type stubs for the Rust extension module.

`tests/test_engine.py` asserts the names here match what the built module
exports, so this file cannot silently drift from `src/lib.rs`.
"""

from collections.abc import Mapping, Sequence

import polars as pl

class ColumnPlan:
    """One column's generation instructions, validated at construction.

    `kind` is one of `kinds()`. Numeric kinds take `min`/`max` (an int or a
    float, kept exact) and a `distribution` with canonical `params`; `index`
    takes `n_categories` and optional `weights` and yields `UInt32` indices
    into a domain the caller holds; `bool` takes `weights=[p_false, p_true]`;
    `string` takes `str_min_len`/`str_max_len`.
    """

    def __init__(
        self,
        name: str,
        kind: str,
        *,
        nullable: bool = False,
        null_probability: float = 0.0,
        min: int | float | None = None,
        max: int | float | None = None,
        n_categories: int | None = None,
        weights: Sequence[float] | None = None,
        str_min_len: int | None = None,
        str_max_len: int | None = None,
        distribution: str | None = None,
        params: Mapping[str, float] | None = None,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def kind(self) -> str: ...
    @property
    def nullable(self) -> bool: ...
    @property
    def null_probability(self) -> float: ...
    @property
    def min(self) -> int | float | None: ...
    @property
    def max(self) -> int | float | None: ...
    @property
    def n_categories(self) -> int | None: ...
    @property
    def weights(self) -> list[float] | None: ...
    @property
    def str_min_len(self) -> int: ...
    @property
    def str_max_len(self) -> int: ...
    @property
    def distribution(self) -> str: ...
    @property
    def p_true(self) -> float: ...

def generate_dataframe(
    columns: Sequence[ColumnPlan], n_rows: int, seed: int | None = None
) -> pl.DataFrame:
    """Fills every planned column in parallel; each column's seed derives from `seed` and its name."""

def distribution_params(name: str) -> list[tuple[str, float, bool]]:
    """`(name, default, must_be_positive)` for each canonical parameter of a distribution."""

def distributions() -> list[str]:
    """Every distribution the engine can sample."""

def kinds() -> list[str]:
    """Every column kind a `ColumnPlan` accepts."""
