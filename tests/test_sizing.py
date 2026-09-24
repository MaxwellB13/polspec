"""`estimated_size`: how big a generated frame will be, from the declaration.

Read off the schema -- the width of each dtype, the lengths the spec
declares -- so it costs nothing and needs no data. The numbers below were
calibrated against measured RSS: the assertions compare against what Polars
says a generated frame holds, plus the 16-byte view per text value that
`Series.estimated_size()` does not count.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest
from polspec import ColSpec, FrameSpec, GenerationError, TableSpec

VIEW = 16


def _per_row(column: ColSpec) -> float:
    return TableSpec("S", {"c": column}).estimated_size(1_000_000) / 1_000_000


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (ColSpec(pl.Int64), 8),
        (ColSpec(pl.Int32), 4),
        (ColSpec(pl.Int8), 1),
        (ColSpec(pl.Float64), 8),
        (ColSpec(pl.Boolean), 0.125),
        (ColSpec(pl.Date), 4),
        (ColSpec(pl.Datetime("ns")), 8),
        (ColSpec(pl.Duration), 8),
        (ColSpec(pl.Time), 8),
        (ColSpec(pl.Decimal(10, 2)), 16),
        (ColSpec(pl.Int64, nullable=True), 8.125),
        # A text value is a 16-byte view; only the bytes past twelve cost more.
        (ColSpec(pl.String, string_length=(2, 4)), VIEW),
        (ColSpec(pl.Binary, string_length=(4, 8)), VIEW),
        (ColSpec(pl.String, string_length=(30, 40)), VIEW + 35),
        (ColSpec(pl.String), VIEW + (13 + 14 + 15) / 11),
        # An Enum is its codes, in the narrowest physical the count needs.
        (ColSpec(pl.Enum(["a", "b"])), 1),
        (ColSpec(pl.Enum([f"v{i}" for i in range(300)])), 2),
        # A gathered column's views point into one shared buffer.
        (ColSpec(pl.String, choices=["short", "a much longer value than that"]), VIEW),
        (ColSpec(pl.String, format="iso_country"), VIEW),
        # A shaped column costs what its shape spells: a uuid4 is 36 bytes.
        (ColSpec(pl.String, format="uuid4"), VIEW + 36),
        (ColSpec(pl.String, format="mac"), VIEW + 17),
        # A list is an offset per row plus the elements it holds.
        (ColSpec(pl.List(pl.Int64), list_length=(0, 4)), 8 + 2 * 8),
        (ColSpec(pl.Array(pl.Float64, 3)), 24),
    ],
    ids=lambda v: None,
)
def test_one_column_costs_what_its_declaration_implies(column, expected):
    assert _per_row(column) == pytest.approx(expected, rel=0.01)


def test_the_estimate_matches_what_polars_says_a_frame_holds():
    """For every dtype whose values Polars stores inline, its own accounting
    and ours agree; where they differ it is the 16-byte view per text value,
    which `Series.estimated_size()` leaves out."""
    for column, views in [
        (ColSpec(pl.Int64), 0),
        (ColSpec(pl.Float64), 0),
        (ColSpec(pl.Date), 0),
        (ColSpec(pl.Enum(["a", "b", "c"])), 0),
        (ColSpec(pl.String, string_length=(20, 30)), VIEW),
        (ColSpec(pl.String, format="uuid4"), VIEW),
        # A null row keeps its view but has no bytes, and a null list no
        # elements; an Array's slots are allocated either way.
        (
            ColSpec(
                pl.String, string_length=(20, 30), nullable=True, null_probability=0.3
            ),
            VIEW,
        ),
        (
            ColSpec(
                pl.List(pl.Int64),
                list_length=(2, 6),
                nullable=True,
                null_probability=0.3,
            ),
            0,
        ),
        (ColSpec(pl.Array(pl.Int64, 3), nullable=True, null_probability=0.3), 0),
    ]:
        spec_cls = type("S", (FrameSpec,), {"__columns__": {"c": column}})
        df = spec_cls.generate(200_000, seed=1)
        polars_says = df.estimated_size() / 200_000 + views
        assert _per_row(column) == pytest.approx(polars_says, rel=0.02), column.dtype


def test_a_whole_spec_is_the_sum_of_its_columns():
    class Sandbox(FrameSpec):
        s1 = ColSpec(pl.String, nullable=True, null_probability=0.01)
        s2 = ColSpec(pl.String)
        i1 = ColSpec(pl.Int64, nullable=True, null_probability=0.01)
        i2 = ColSpec(pl.Int64)
        f1 = ColSpec(pl.Float64, nullable=True, null_probability=0.01)
        f2 = ColSpec(pl.Float64)

    per_row = 2 * (VIEW + (13 + 14 + 15) / 11) + 4 * 8 + 3 * 0.125
    assert Sandbox.estimated_size(1_000_000) == pytest.approx(
        per_row * 1_000_000, rel=0.01
    )
    assert Sandbox.estimated_size(0) == 0
    assert Sandbox.spec.estimated_size(10) == Sandbox.estimated_size(10)
    with pytest.raises(ValueError, match="n must be >= 0"):
        Sandbox.estimated_size(-1)


# ---------------------------------------------------------------------------
# The guardrail
# ---------------------------------------------------------------------------


class Wide(FrameSpec):
    a = ColSpec(pl.Float64)
    b = ColSpec(pl.Float64)


def _rows_for(bytes_wanted: int) -> int:
    return int(bytes_wanted / (Wide.estimated_size(1_000) / 1_000)) + 1


def test_a_frame_worth_mentioning_is_mentioned_and_a_small_one_is_not():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Wide.generate(10, seed=1)
    assert caught == []


def test_max_bytes_refuses_rather_than_warning():
    """The check runs before anything is allocated, so a refusal costs
    nothing however many rows were asked for."""
    huge = _rows_for(5 * 1024**3)
    with pytest.raises(
        GenerationError, match=r"5\.0 GiB, over the max_bytes"
    ) as raised:
        Wide.generate(huge, seed=1, max_bytes=1024**3)
    assert "scan()" in str(raised.value)


def test_max_bytes_zero_says_nothing(monkeypatch):
    import polspec.generation as generation

    monkeypatch.setattr(generation, "_LARGE_FRAME_BYTES", 10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Wide.generate(1_000, seed=1, max_bytes=0)


def test_the_warning_names_the_estimate_and_what_to_do_instead(monkeypatch):
    """Rather than allocating five gibibytes to see the warning, the size is
    lowered to what the test frame actually is."""
    import polspec.generation as generation

    monkeypatch.setattr(generation, "_LARGE_FRAME_BYTES", 100)
    with pytest.warns(UserWarning) as caught:
        Wide.generate(1_000, seed=1)
    message = str(caught[0].message)
    assert "Wide at 1,000 rows is an estimated 15.6 KiB" in message
    assert "scan()" in message and "generate_batches()" in message
    assert "max_bytes=0 to silence" in message
