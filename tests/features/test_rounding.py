"""Rounded inferences: bounds and rates at a person's precision.

`profile()` -- and `polspec schema infer`, by default -- can write each
numeric column's bounds rounded outward to a step and each rate rounded up,
so an inferred spec reads like a written one. The one promise that cannot
bend: the data a spec was inferred from still passes it.
"""

from __future__ import annotations

import datetime as dt
import warnings
from decimal import Decimal

import polars as pl
import pytest
from cases import COLUMN_CASES
from polspec import (
    ColSpec,
    FrameSpec,
    TableSpec,
    from_yaml,
    generate,
    profile,
    validate,
)
from polspec.bound import Bound
from polspec.profiler import rounded_bounds, rounded_rate, step_of

STEPS = ["auto", 1000, 0.01, 7, 0.3, "1M"]


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("observed", "dtype", "step", "written"),
    [
        # "auto": two significant figures of the span.
        ((18, 87), pl.Int64, "auto", (18, 87)),
        ((3.17, 9871.23), pl.Float64, "auto", (0.0, 9900.0)),
        ((0.99, 4.99), pl.Float64, "auto", (0.9, 5.0)),
        ((1203, 8_712_334), pl.Int64, "auto", (0, 8_800_000)),
        # A step: outward to its multiples.
        ((1203, 8_712_334), pl.Int64, Decimal(1000), (1000, 8_713_000)),
        ((-5.5, 5.5), pl.Float32, Decimal(2), (-6.0, 6.0)),
        # Within what the dtype holds.
        ((3, 251), pl.UInt8, "auto", (0, 255)),
        ((-100, 100), pl.Int8, Decimal(1000), (-128, 127)),
        # An integer's step is whole; a Decimal's no finer than its scale.
        ((3, 9), pl.Int32, Decimal("0.01"), (3, 9)),
        (
            (Decimal("1.23"), Decimal("98.76")),
            pl.Decimal(10, 2),
            Decimal("0.001"),
            (Decimal("1.23"), Decimal("98.76")),
        ),
        (
            (Decimal("1.23"), Decimal("98.76")),
            pl.Decimal(10, 2),
            "auto",
            (Decimal("1.00"), Decimal("99.00")),
        ),
        # A constant column keeps its value.
        ((7, 7), pl.Int32, "auto", (7, 7)),
    ],
)
def test_bounds_round_outward_within_the_dtype(observed, dtype, step, written):
    result = rounded_bounds(Bound(*observed), dtype, step)
    assert (result.min, result.max) == written
    assert result.min <= observed[0] and result.max >= observed[1]


def test_a_float_rounds_without_artifacts():
    """0.1 * 9 is 0.9000000000000001 in floats; the rounding is decimal."""
    assert rounded_bounds(Bound(0.95, 1.0), pl.Float64, Decimal("0.1")) == Bound(
        0.9, 1.0
    )


@pytest.mark.parametrize(
    ("rate", "step", "written"),
    [
        (0.023651, "auto", 0.024),
        (0.00031, "auto", 0.00031),
        (0.1234, "auto", 0.13),
        (0.023651, Decimal("0.01"), 0.03),
        (0.999, Decimal("0.01"), 1.0),
        (0.5, Decimal(3), 1.0),  # never past 1
        (0.0, "auto", 0.0),  # none stays none
        (0.0, Decimal("0.01"), 0.0),
    ],
)
def test_rates_round_up_and_zero_stays_zero(rate, step, written):
    assert rounded_rate(rate, step) == written


@pytest.mark.parametrize(
    ("text", "step"),
    [
        ("1k", Decimal(1000)),
        ("10k", Decimal(10_000)),
        ("1M", Decimal(1_000_000)),
        ("0.01", Decimal("0.01")),
        ("2.5e3", Decimal(2500)),
        (1000, Decimal(1000)),
        (0.01, Decimal("0.01")),
        ("auto", "auto"),
        ("exact", None),
        (None, None),
    ],
)
def test_a_step_reads_as_written(text, step):
    assert step_of(text) == step


@pytest.mark.parametrize("bad", ["banana", "-1", "0", 0, -0.5, "inf", True])
def test_a_step_that_is_not_one_is_refused(bad):
    with pytest.raises(ValueError, match="rounding step"):
        step_of(bad)


# ---------------------------------------------------------------------------
# The promise: a frame passes the spec inferred from it
# ---------------------------------------------------------------------------


def _frame_for(case: str) -> pl.DataFrame:
    spec = TableSpec(f"Case_{case}", {"c": COLUMN_CASES[case]})
    return generate(spec, 300, seed=13)


@pytest.mark.parametrize("step", STEPS, ids=str)
@pytest.mark.parametrize("case", sorted(COLUMN_CASES))
def test_every_frame_passes_the_spec_inferred_from_it(case, step):
    """Every column the catalogue can hold -- every dtype, nested, at its
    limits -- profiled at every step, validates against what it was
    profiled into."""
    frame = _frame_for(case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # an infinity outside the bounds says so
        spec = profile(frame, round_bounds=step, round_rates=step, detect_order=False)
    report = validate(spec, frame, validate_bounds=True)
    assert report.equals(frame)


def test_without_rounding_profile_is_what_it_was():
    frame = _frame_for("float64_lognormal")
    assert profile(frame) == profile(frame, round_bounds=None, round_rates=None)


# ---------------------------------------------------------------------------
# Dates and times
# ---------------------------------------------------------------------------


def test_auto_rounds_a_timestamp_to_the_second_and_a_step_leaves_it():
    frame = pl.DataFrame(
        {
            "at": [
                dt.datetime(2024, 1, 1, 0, 1, 7, 579_288),
                dt.datetime(2024, 4, 25, 17, 21, 3, 293_956),
            ]
            * 30
        }
    )
    auto = profile(frame, round_bounds="auto", detect_order=False)["at"].bounds
    assert (auto.min, auto.max) == (
        dt.datetime(2024, 1, 1, 0, 1, 7),
        dt.datetime(2024, 4, 25, 17, 21, 4),
    )
    stepped = profile(frame, round_bounds=1000, detect_order=False)["at"].bounds
    assert stepped.min == dt.datetime(2024, 1, 1, 0, 1, 7, 579_288)
    validate(profile(frame, round_bounds="auto"), frame)


def test_a_time_that_would_round_to_midnight_keeps_its_microseconds():
    """23:59:59.5 rounded up to the second is 24:00, which no time holds."""
    frame = pl.DataFrame({"t": [dt.time(8, 0), dt.time(23, 59, 59, 500_000)] * 30})
    bounds = profile(frame, round_bounds="auto")["t"].bounds
    assert (bounds.min, bounds.max) == (dt.time(8, 0), dt.time(23, 59, 59, 500_000))
    validate(profile(frame, round_bounds="auto"), frame)


# ---------------------------------------------------------------------------
# Where it is asked for
# ---------------------------------------------------------------------------


def test_from_dataframe_rounds_only_when_asked():
    frame = pl.DataFrame({"amount": [3.17, 9871.23, 512.5] * 20})
    assert FrameSpec.from_dataframe(frame).spec["amount"].bounds == Bound(3.17, 9871.23)
    asked = FrameSpec.from_dataframe(frame, round_bounds="auto")
    assert asked.spec["amount"].bounds == Bound(0.0, 9900.0)


def test_schema_infer_rounds_auto_unless_told_exact(tmp_path):
    from polspec.cli import main

    source = tmp_path / "orders.parquet"
    pl.DataFrame(
        {
            "amount": [3.17, 9871.23, None, 512.5] * 25,
            "qty": pl.Series([3, 251, 40, 7] * 25, dtype=pl.UInt8),
        }
    ).write_parquet(source)
    auto, exact, stepped = (
        tmp_path / f"{n}.yaml" for n in ("auto", "exact", "stepped")
    )
    assert main(["schema", "infer", str(source), "-o", str(auto)]) == 0
    assert (
        main(
            [
                "schema",
                "infer",
                str(source),
                "-o",
                str(exact),
                "--round-bounds",
                "exact",
                "--round-rates",
                "exact",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "schema",
                "infer",
                str(source),
                "-o",
                str(stepped),
                "--round-bounds",
                "1k",
                "--round-rates",
                "0.1",
            ]
        )
        == 0
    )
    assert from_yaml(auto)["amount"].bounds == Bound(0.0, 9900.0)
    assert from_yaml(auto)["qty"].bounds == Bound(0, 255)
    assert from_yaml(exact)["amount"].bounds == Bound(3.17, 9871.23)
    assert from_yaml(exact)["amount"].null_probability == 0.25
    assert from_yaml(stepped)["amount"].bounds == Bound(0.0, 10000.0)
    assert from_yaml(stepped)["amount"].null_probability == 0.3
    frame = pl.read_parquet(source)
    for path in (auto, exact, stepped):
        validate(from_yaml(path), frame)


def test_schema_infer_refuses_a_step_that_is_not_one(tmp_path, capsys):
    from polspec.cli import main

    source = tmp_path / "x.parquet"
    pl.DataFrame({"a": [1, 2]}).write_parquet(source)
    with pytest.raises(SystemExit):
        main(
            [
                "schema",
                "infer",
                str(source),
                "-o",
                str(tmp_path / "x.yaml"),
                "--round-bounds",
                "banana",
            ]
        )
    assert "rounding step" in capsys.readouterr().err


def test_a_column_spec_is_still_a_column_spec():
    """The rounded bounds are ordinary bounds: a spec written from them
    generates and validates like any other."""
    spec = TableSpec("T", {"a": ColSpec(pl.Float64, bounds=(0.0, 9900.0))})
    validate(spec, generate(spec, 100, seed=1))
