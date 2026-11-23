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

import functools
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
    plans: Sequence[ColumnPlan], n: int, seed: int | None, row_offset: int = 0
) -> pl.DataFrame:
    """Fills every planned column, re-raising the engine's errors as
    `GenerationError`. `row_offset` asks for rows `[row_offset, row_offset + n)`
    of the frame `seed` describes."""
    try:
        frame = _extension().generate_dataframe(list(plans), n, seed, row_offset)
    except ValueError as exc:
        raise GenerationError(str(exc)) from exc
    # Polars names a series called "" `column_<i>` on its way into Python, so
    # a column with the empty name came back as another; the engine keeps the
    # plans' order, so each is named for its plan again.
    names = [plan.name for plan in plans]
    if frame.columns != names:
        frame.columns = names
    return frame


def permuted_indices(domain: int, seed: int, start: int, n: int) -> pl.Series:
    """Positions `start..start + n` of the permutation of `[0, domain)` that
    `seed` keys: a window of rows takes the indices the whole range's rows
    take there. A window past the domain is a `GenerationError`."""
    try:
        return _extension().permuted_indices(domain, seed, start, n)
    except ValueError as exc:
        raise GenerationError(str(exc)) from exc


def composite_indices(
    sizes: list[int], seed: int, start: int, n: int
) -> list[pl.Series]:
    """Positions `start..start + n` of the permutation of a composite key's
    combinations -- members of `sizes` values each -- that `seed` keys, as
    one `UInt64` index column per member: distinct over the whole space, and
    a window the whole space's slice. Too few combinations for the rows is a
    `GenerationError`."""
    try:
        return _extension().composite_indices(sizes, seed, start, n)
    except ValueError as exc:
        raise GenerationError(str(exc)) from exc


def list_from_lengths(name: str, lengths: pl.Series, elements: pl.Series) -> pl.Series:
    """A `List` column named `name`: `elements` cut into cells by `lengths`,
    a null length a null cell. The elements keep their dtype."""
    try:
        return _extension().list_from_lengths(name, lengths, elements)
    except ValueError as exc:
        raise GenerationError(str(exc)) from exc


def decimal_from_physical(raw: pl.Series, precision: int, scale: int) -> pl.Series:
    """A `Decimal(precision, scale)` column from its physical integers: the
    integer `v` is the decimal `v / 10**scale`."""
    try:
        return _extension().decimal_from_physical(raw, precision, scale)
    except ValueError as exc:
        raise GenerationError(str(exc)) from exc


@functools.cache
def chunk_size() -> int:
    """The rows in one chunk of an engine column. A window that starts on a
    multiple of it draws exactly the rows it keeps; one that starts inside a
    chunk draws the chunk from its start and drops the head."""
    return _extension().chunk_size()


def distribution_params(name: str) -> list[tuple[str, float, bool]]:
    """The engine's `(name, default, must_be_positive)` for one distribution."""
    return _extension().distribution_params(name)


def distributions() -> list[str]:
    """The distributions the engine can sample."""
    return _extension().distributions()


def kinds() -> list[str]:
    """The column kinds a `ColumnPlan` accepts."""
    return _extension().kinds()


@functools.lru_cache(maxsize=256)
def pattern_status(
    pattern: str,
    lengths: tuple[int, int] | None,
    plain_lengths: tuple[int, int],
    unique: bool = False,
) -> tuple[str, str | None]:
    """How a column with `pattern` is generated: `"generate"`, from the
    pattern; `"keep"`, as the plain text it drew before 1.4, every value of
    which already passes it; or `"unreachable"`, with the reason. Asked once
    per pattern and lengths: the automaton is built in Rust, and kept there.
    A unique column numbers the pattern's values too, which a pattern can
    have too many states and lengths for."""
    return _extension().pattern_status(pattern, lengths, plain_lengths, unique)
