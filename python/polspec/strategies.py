"""`strategy`: frames a spec accepts, as a Hypothesis strategy.

A property test asks a question of many frames -- "is spend never
negative, whatever the orders?" -- and Hypothesis draws them. `strategy`
draws a row count and a seed and calls `generate()`, so every frame meets
its spec by construction, comes from the spec's own distributions, and
costs what generating it costs. Hypothesis shrinks the two numbers: a
failure is reported on the fewest rows that still fail, from seed 0
where it can.

Hypothesis is not a dependency of polspec; it is imported when a strategy
is asked for.

Internal: not part of the public API.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl
    from hypothesis.strategies import SearchStrategy

    from polspec.frames import References
    from polspec.tablespec import SpecLike

# Seeds drawn: any 32-bit one, which is every seed a test is likely to name.
MAX_SEED = (1 << 32) - 1


def strategy(
    spec: SpecLike,
    *,
    min_rows: int = 0,
    max_rows: int = 100,
    references: References = None,
    fixed: Mapping[str, Any] | None = None,
) -> SearchStrategy[pl.DataFrame]:
    """A Hypothesis strategy of frames `spec` accepts.

    Each frame is `generate(spec, n, seed=s)` for a drawn row count `n`
    between `min_rows` and `max_rows` and a drawn seed `s`, so it meets the
    spec by construction and follows its distributions, rules and keys.
    Hypothesis shrinks both numbers: a failing example is reported on the
    fewest rows that fail. The values themselves come from the seed and do
    not shrink one by one.

    Parameters
    ----------
    spec : TableSpec | FrameSpec class
        What every frame meets.
    min_rows, max_rows : int, default 0 and 100
        The range of row counts drawn.
    references : mapping, optional
        Parent frames for `spec`'s foreign keys, as `generate()` takes them.
    fixed : mapping, optional
        Columns pinned in every frame, as `generate(fixed=)` takes them. A
        pin of a sequence has to be as long as every frame, so give scalars
        unless `min_rows` and `max_rows` are equal.

    Raises
    ------
    ImportError
        When Hypothesis is not installed.
    ValueError
        For a row range that is negative or empty.

    Examples
    --------
    >>> from hypothesis import given, settings
    >>> spec = TableSpec("Orders", {"total": ColSpec(pl.Float64, bounds=(0, 100))})
    >>> @settings(max_examples=25, deadline=None)
    ... @given(orders=strategy(spec, max_rows=50))
    ... def test_totals_stay_in_range(orders):
    ...     assert orders.height <= 50
    ...     assert (orders["total"] <= 100).all()
    >>> test_totals_stay_in_range()
    """
    try:
        from hypothesis import strategies as st
    except ImportError:
        raise ImportError(
            "polspec.strategy needs Hypothesis, which polspec does not install: "
            "pip install hypothesis"
        ) from None
    from polspec.generation import generate
    from polspec.tablespec import as_table_spec, require_columns

    if min_rows < 0 or max_rows < min_rows:
        raise ValueError(
            f"min_rows and max_rows must be 0 <= min_rows <= max_rows, got "
            f"{min_rows} and {max_rows}"
        )
    table = as_table_spec(spec)
    require_columns(table)

    def frame(drawn: tuple[int, int]) -> pl.DataFrame:
        n, seed = drawn
        return generate(table, n, seed=seed, references=references, fixed=fixed)

    return st.tuples(
        st.integers(min_value=min_rows, max_value=max_rows),
        st.integers(min_value=0, max_value=MAX_SEED),
    ).map(frame)
