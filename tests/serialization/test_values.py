"""Values a YAML spec file has no native form for, and the forms it uses.

YAML holds numbers, text, dates, datetimes and bytes, and neither a time of
day nor a duration, so `to_yaml()` used to crash with PyYAML's own
`RepresenterError` on a `Time` or `Duration` bound, choice or literal. Those
two are now written tagged (`polspec.scalars`); anything else YAML cannot
hold is a `SerializationError` naming where it sits, raised before the file
is opened. `to_python` decides its imports from the source it writes, so a
`Decimal` choice or a time literal imports what it uses.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
import yaml
from polspec import (
    Check,
    ColRule,
    ColSpec,
    FrameSpec,
    SerializationError,
    SpecError,
    TableSpec,
    col,
)
from polspec.scalars import from_plain, is_tagged, to_plain
from polspec.serialization import dump_yaml, from_dict, from_yaml, to_dict, to_python

FIXTURES = Path(__file__).parent / "fixtures"

SHIFTS = TableSpec(
    "Shifts",
    {
        "night": ColSpec(pl.Boolean),
        "starts": ColSpec(
            pl.Time,
            bounds=(dt.time(6), dt.time(22)),
            rules=[ColRule(when=col("night"), choices=[dt.time(22)])],
        ),
        "length": ColSpec(
            pl.Duration("us"),
            choices=[dt.timedelta(hours=8), dt.timedelta(hours=12)],
        ),
    },
    checks=[
        Check(col("starts") != dt.time(12, 0, 0, 500_000), name="not_half_past_noon"),
        Check(
            col("length").is_in([dt.timedelta(hours=8), dt.timedelta(hours=12)]),
            name="standard_lengths",
        ),
    ],
)


def _through_yaml(spec: TableSpec) -> TableSpec:
    return from_dict(yaml.safe_load(yaml.safe_dump(to_dict(spec), sort_keys=False)))


def _through_python(spec: TableSpec, tmp_path: Path) -> TableSpec:
    path = tmp_path / f"{spec.name.lower()}_spec.py"
    to_python(spec, path)
    module_spec = importlib.util.spec_from_file_location(path.stem, path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, spec.name).spec


# ---------------------------------------------------------------------------
# The codec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "plain"),
    [
        (dt.time(12, 30), {"time": "12:30:00"}),
        (dt.time(0, 0, 0, 1), {"time": "00:00:00.000001"}),
        (
            dt.timedelta(days=1, seconds=7200, microseconds=5),
            {"duration": {"days": 1, "seconds": 7200, "microseconds": 5}},
        ),
        (dt.timedelta(0), {"duration": {"days": 0, "seconds": 0, "microseconds": 0}}),
        (
            dt.timedelta(days=-1),
            {"duration": {"days": -1, "seconds": 0, "microseconds": 0}},
        ),
    ],
    ids=["time", "time_micros", "duration", "zero", "negative"],
)
def test_a_time_or_duration_has_one_plain_form(value, plain):
    assert to_plain(value) == plain
    assert is_tagged(plain)
    assert from_plain(yaml.safe_load(yaml.safe_dump(plain))) == value


@pytest.mark.parametrize(
    "value",
    [1, 1.5, "12:30", dt.date(2024, 1, 1), dt.datetime(2024, 1, 1, 9), b"\x00", None],
    ids=str,
)
def test_everything_else_passes_through(value):
    assert to_plain(value) is value
    assert from_plain(value) is value


def test_a_duration_field_left_out_is_zero():
    assert from_plain({"duration": {"seconds": 60}}) == dt.timedelta(minutes=1)


@pytest.mark.parametrize(
    "plain",
    [{"time": 5}, {"duration": {"hours": 1}}, {"time": "x", "duration": {}}],
    ids=["time_not_text", "unknown_field", "two_keys"],
)
def test_what_is_not_a_tagged_form_is_left_alone(plain):
    assert not is_tagged(plain)
    assert from_plain(plain) is plain


# ---------------------------------------------------------------------------
# Every place a spec holds one
# ---------------------------------------------------------------------------


def test_the_golden_file_reads_as_the_spec_it_declares():
    spec = from_yaml(FIXTURES / "spec_v3_temporal.yaml")
    assert spec == SHIFTS
    (rule,) = spec["starts"].rules
    assert rule.choices == (dt.time(22),)


def test_bounds_choices_rules_and_literals_survive_yaml(tmp_path):
    assert _through_yaml(SHIFTS) == SHIFTS
    path = tmp_path / "shifts.yaml"
    FrameSpec.from_spec(SHIFTS).to_yaml(path)
    assert from_yaml(path) == SHIFTS
    text = path.read_text(encoding="utf-8")
    assert "time: '12:00:00.500000'" in text
    assert "duration:" in text


def test_they_survive_python_too(tmp_path):
    assert _through_python(SHIFTS, tmp_path) == SHIFTS


def test_a_reloaded_spec_generates_what_the_original_does():
    original = FrameSpec.from_spec(SHIFTS).generate(200, seed=3)
    reloaded = FrameSpec.from_spec(_through_yaml(SHIFTS)).generate(200, seed=3)
    assert reloaded.equals(original)


@pytest.mark.parametrize(
    "spec",
    [
        TableSpec("D", {"c": ColSpec(pl.Decimal(4, 1), choices=[Decimal("0.5")])}),
        TableSpec(
            "R",
            {
                "g": ColSpec(pl.Boolean),
                "c": ColSpec(
                    pl.Decimal(4, 1),
                    rules=[ColRule(when=col("g"), choices=[Decimal("0.5")])],
                ),
            },
        ),
        TableSpec("E", {"c": ColSpec(pl.Date, choices=[dt.date(2024, 1, 1)])}),
    ],
    ids=["decimal_choice", "decimal_rule_choice", "date_choice"],
)
def test_generated_python_imports_what_it_writes(spec, tmp_path):
    """The imports used to be guessed from the bounds alone, so a `Decimal`
    choice wrote `Decimal('0.5')` into a module that never imported it."""
    assert _through_python(spec, tmp_path) == spec


# ---------------------------------------------------------------------------
# What a file still cannot hold
# ---------------------------------------------------------------------------


def test_a_value_yaml_cannot_hold_is_a_serialization_error_naming_where():
    class Unwritable:
        pass

    with pytest.raises(SerializationError, match=r"columns\.c\.choices\[0\]"):
        dump_yaml({"columns": {"c": {"choices": [Unwritable()]}}}, "odd.yaml")


def test_nothing_is_written_when_the_dump_fails(tmp_path, monkeypatch):
    import polspec.serialization as serialization

    def refuse(data, source):
        raise SerializationError("refused")

    monkeypatch.setattr(serialization, "dump_yaml", refuse)
    target = tmp_path / "never.yaml"
    with pytest.raises(SerializationError):
        serialization.to_yaml(SHIFTS, target)
    assert not target.exists()


def test_a_malformed_time_is_refused_naming_where():
    with pytest.raises(SerializationError, match=r"columns\.c\.choices.*noonish"):
        from_dict(
            {
                "name": "T",
                "columns": {"c": {"dtype": "Time", "choices": [{"time": "noonish"}]}},
            }
        )


def test_a_malformed_duration_literal_is_refused():
    with pytest.raises(SpecError, match="must be integers"):
        from_dict(
            {
                "name": "T",
                "columns": {"c": {"dtype": {"Duration": {"time_unit": "us"}}}},
                "checks": [
                    {
                        "expr": {"ne": [{"col": "c"}, {"duration": {"seconds": 1.5}}]},
                        "name": "n",
                    }
                ],
            }
        )


def test_decimal_choices_are_written_exactly_and_read_back_as_decimals():
    """A Decimal bound was always written as its exact string; a Decimal
    choice was written raw, and `to_yaml()` crashed on it."""
    spec = TableSpec(
        "Prices",
        {
            "cheap": ColSpec(pl.Boolean),
            "price": ColSpec(
                pl.Decimal(10, 2),
                choices=[Decimal("0.00"), Decimal("12.50")],
                rules=[ColRule(when=col("cheap"), choices=[Decimal("0.00")])],
            ),
        },
    )
    text = yaml.safe_dump(to_dict(spec), sort_keys=False)
    assert "- '12.50'" in text
    reloaded = from_dict(yaml.safe_load(text))
    assert reloaded == spec
    assert reloaded["price"].choices == (Decimal("0.00"), Decimal("12.50"))
    assert reloaded["price"].rules[0].choices == (Decimal("0.00"),)
