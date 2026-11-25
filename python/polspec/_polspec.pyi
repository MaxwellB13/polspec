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
    `string` takes `str_min_len`/`str_max_len`; `template` takes `template`,
    a list of `(kind, strings, lo, hi)` parts -- see `polspec.string_formats`.

    `unique` draws without replacement instead, from the same domain; a
    domain too small to cover the row count is refused by name.
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
        unique: bool = False,
        template: Sequence[tuple[str, Sequence[str], int, int]] | None = None,
        seed_name: str | None = None,
        pattern: str | None = None,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def seed_name(self) -> str | None: ...
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
    @property
    def unique(self) -> bool: ...

def generate_dataframe(
    columns: Sequence[ColumnPlan],
    n_rows: int,
    seed: int | None = None,
    row_offset: int = 0,
) -> pl.DataFrame:
    """Fills every planned column in parallel; each column's seed derives from `seed` and its seed name (its name, unless the plan carries `seed_name`). `row_offset` asks for rows `[row_offset, row_offset + n_rows)` of the frame `seed` describes, so a batch is a window onto one frame."""

def list_from_lengths(name: str, lengths: pl.Series, elements: pl.Series) -> pl.Series:
    """A `List` column: `elements` cut into cells by `lengths`, a null length a null cell; the elements keep their dtype. Raises `ValueError` when the lengths do not total the elements."""

def decimal_from_physical(raw: pl.Series, precision: int, scale: int) -> pl.Series:
    """A `Decimal(precision, scale)` column from its physical integers: `v` is `v / 10**scale`. Raises `ValueError` for a value the precision cannot hold."""

def chunk_size() -> int:
    """The rows in one chunk of a column, each filled from its own seed: a window starting on a multiple of it draws no row it does not keep."""

def distribution_params(name: str) -> list[tuple[str, float, bool]]:
    """`(name, default, must_be_positive)` for each canonical parameter of a distribution."""

def distributions() -> list[str]:
    """Every distribution the engine can sample."""

def kinds() -> list[str]:
    """Every column kind a `ColumnPlan` accepts."""

def pattern_status(
    pattern: str,
    lengths: tuple[int, int] | None,
    plain_lengths: tuple[int, int],
    unique: bool = False,
) -> tuple[str, str | None]:
    """How a `pattern` column is generated: `("generate", None)` from the pattern; `("keep", None)` as plain text, every value of which already passes it at `plain_lengths` characters; or `("unreachable", reason)`."""

def permuted_at(domain: int, seed: int, indices: pl.Series) -> pl.Series:
    """The keyed permutation of `[0, domain)` at each of `indices`, as
    `UInt64`: `permuted_indices` at scattered positions."""

def unique_values_at(plan: ColumnPlan, seed: int, rows: pl.Series) -> pl.Series:
    """What the unique column `plan`, with no nulls, holds at `rows` of the
    frame `seed` describes, without drawing the rows between."""

def composite_indices(
    sizes: list[int], seed: int, start: int, n: int
) -> list[pl.Series]:
    """Positions `start..start + n` of the permutation of a product space of members with `sizes` values each, as one `UInt64` index column per member. Raises `ValueError` when the space holds fewer than `start + n` combinations."""

def permuted_indices(domain: int, seed: int, start: int, n: int) -> pl.Series:
    """Positions `start..start + n` of the permutation of `[0, domain)` that `seed` keys, as `UInt64` indices; any window is the whole range's slice. Raises `ValueError` for a window past the domain."""
