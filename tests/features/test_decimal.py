"""`pl.Decimal`: generated as its physical integer, scaled back to the type.

The cheapest of the four ungeneratable dtypes to close and the only one
needing no design decision: a Decimal is an integer and a scale, so its
column is drawn through the engine's `int64` kind with the bounds scaled,
and `_finish` divides the scale back in. Validation and drift compare the
values Polars holds, whatever the frame arrived as.
"""

from __future__ import annotations

import json
from decimal import Decimal

import polars as pl
import polspec
import pytest
from helpers import spec_for
from polspec import (
    ColRule,
    ColSpec,
    FrameSpec,
    SpecError,
    TableSpec,
    ValidationError,
    col,
    drift,
    inspect,
)

# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


def test_endpoints_are_held_exactly_and_ints_stay_ints():
    spec = ColSpec(pl.Decimal(10, 2), bounds=(0, "99.99"))
    assert spec.bounds is not None
    assert spec.bounds.min == 0 and isinstance(spec.bounds.min, int)
    assert spec.bounds.max == Decimal("99.99") and isinstance(spec.bounds.max, Decimal)
    # A float is read through its repr, so 0.1 is 0.1 and not 0.1000000000000000055.
    assert ColSpec(pl.Decimal(10, 2), bounds=(0.1, None)).bounds.min == Decimal("0.1")


def test_an_endpoint_finer_than_the_scale_is_refused():
    with pytest.raises(SpecError, match="more decimal places than"):
        ColSpec(pl.Decimal(10, 2), bounds=(0, "1.005"))


def test_an_endpoint_wider_than_the_precision_is_refused():
    with pytest.raises(SpecError, match="outside the range"):
        ColSpec(pl.Decimal(4, 2), bounds=(0, 100))


def test_an_endpoint_that_is_not_a_number_is_refused():
    with pytest.raises(SpecError, match="is not a number"):
        ColSpec(pl.Decimal(4, 2), bounds=("abc", None))
    with pytest.raises(SpecError, match="finite"):
        ColSpec(pl.Decimal(4, 2), bounds=(None, "Infinity"))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generated_values_have_the_declared_type_scale_and_bounds():
    spec_cls = spec_for(ColSpec(pl.Decimal(10, 2), bounds=(Decimal("0.50"), 10)))
    df = spec_cls.generate(500, seed=1)
    assert df.schema["c"] == pl.Decimal(10, 2)
    values = df["c"].to_list()
    assert all(Decimal("0.50") <= v <= 10 for v in values)
    assert all(v.as_tuple().exponent == -2 for v in values)
    assert len(set(values)) > 100, "a Decimal is drawn, not repeated"
    spec_cls.validate(df)


def test_the_default_range_respects_a_narrow_precision():
    df = spec_for(ColSpec(pl.Decimal(4, 2))).generate(1_000, seed=2)
    assert all(Decimal("-99.99") <= v <= Decimal("99.99") for v in df["c"].to_list())


@pytest.mark.parametrize(
    ("dtype", "bounds"),
    [
        (pl.Decimal(38, 10), (0, 10**20)),
        (pl.Decimal(38, 0), (-(10**38) + 1, 10**38 - 1)),
        (pl.Decimal(30, 4), None),
    ],
    ids=str,
)
def test_a_decimal_is_drawn_at_its_full_precision(dtype, bounds):
    """Its physical integer is drawn at 128 bits, so bounds needing more
    than the eighteen digits 64 bits hold generate, inside the bounds."""
    spec_cls = spec_for(ColSpec(dtype, bounds=bounds))
    df = spec_cls.generate(2_000, seed=1)
    spec_cls.validate(df)
    if bounds is not None:
        widest = Decimal(bounds[1]) / 1_000
        assert df["c"].abs().max() > widest


def test_cartesian_coverage_reaches_the_partitions():
    df = spec_for(ColSpec(pl.Decimal(6, 2), bounds=(-5, 5), nullable=True)).generate(
        1, seed=1, method="cartesian"
    )
    values = df["c"].to_list()
    assert None in values and Decimal("0.00") in values
    assert any(v is not None and v < 0 for v in values)
    assert any(v is not None and v > 0 for v in values)


# ---------------------------------------------------------------------------
# Validation and drift
# ---------------------------------------------------------------------------


def test_bounds_are_checked_on_the_values_whatever_the_frame_holds():
    spec_cls = spec_for(ColSpec(pl.Decimal(10, 2), bounds=(0, 100)))
    spec_cls.validate(
        pl.DataFrame({"c": [Decimal("99.99")]}, schema={"c": pl.Decimal(10, 2)})
    )
    spec_cls.validate(pl.DataFrame({"c": [99.5]}))  # a float, as CSV hands one back
    spec_cls.validate(pl.DataFrame({"c": [50]}))
    with pytest.raises(ValidationError, match="out of bounds"):
        spec_cls.validate(pl.DataFrame({"c": [100.01]}))
    with pytest.raises(ValidationError, match="dtype"):
        spec_cls.validate(pl.DataFrame({"c": ["1.0"]}))


def test_drift_measures_the_extent_in_the_columns_own_type():
    spec_cls = spec_for(ColSpec(pl.Decimal(10, 2), bounds=(0, 100)))
    df = spec_cls.generate(100, seed=1).with_columns(
        c=(pl.col("c") + 1000).cast(pl.Decimal(10, 2))
    )
    (finding,) = drift(spec_cls, df).breaking
    assert finding.code == "bounds_exceeded"
    assert finding.details["max_found"] == df["c"].max()
    assert finding.details["above_by"] == df["c"].max() - 100
    json.loads(drift(spec_cls, df).to_json())  # a Decimal serialises


# ---------------------------------------------------------------------------
# Files and profiling
# ---------------------------------------------------------------------------


def test_a_decimal_spec_round_trips_through_yaml_and_python(tmp_path):
    spec_cls = spec_for(ColSpec(pl.Decimal(38, 4), bounds=("-1.5", 2), nullable=True))
    spec_cls.to_yaml(tmp_path / "s.yaml")
    text = (tmp_path / "s.yaml").read_text(encoding="utf-8")
    assert "Decimal:" in text and "'-1.5'" in text, text
    assert FrameSpec.from_yaml(tmp_path / "s.yaml").spec == spec_cls.spec
    spec_cls.to_python(tmp_path / "s.py")
    source = (tmp_path / "s.py").read_text(encoding="utf-8")
    assert "from decimal import Decimal" in source
    assert "pl.Decimal(38, 4)" in source and "Decimal('-1.5')" in source
    namespace: dict = {}
    exec(source, namespace)
    assert namespace[spec_cls.__name__].spec == spec_cls.spec


def test_from_dataframe_declares_a_decimal_with_its_extent():
    df = pl.DataFrame(
        {"c": [Decimal("1.25"), Decimal("9.50")]}, schema={"c": pl.Decimal(6, 2)}
    )
    column = FrameSpec.from_dataframe(df, name="P").spec.columns["c"]
    assert column.dtype == pl.Decimal(6, 2)
    assert column.bounds is not None
    assert (column.bounds.min, column.bounds.max) == (Decimal("1.25"), Decimal("9.50"))


@pytest.mark.parametrize("value", [1e8, -1e8, 99_999_999.996])
def test_a_number_too_large_for_a_decimal_with_choices_is_its_dtype_finding(value):
    """Asking whether such a number is a choice cast it to the Decimal, and
    Polars raises on a cast past a Decimal's precision, `strict=False` or
    not: validation crashed. It is the range check's, as without choices.
    Found by the arrival property on 2026-11-14."""
    declared = pl.Decimal(10, 2)
    for column in (
        ColSpec(declared, choices=[Decimal("0.00")]),
        ColSpec(
            declared, rules=[ColRule(when=col("c") > 0, choices=[Decimal("1.00")])]
        ),
    ):
        report = inspect(TableSpec("T", {"c": column}), pl.DataFrame({"c": [value]}))
        assert [f.code for f in report.findings] == ["dtype"]


def test_a_38_digit_column_profiles_and_validates():
    values = [Decimal("12345678901234567890123456789012.3456"), Decimal("-9.0001")]
    frame = pl.DataFrame({"c": pl.Series(values, dtype=pl.Decimal(38, 4))})
    for rounding in (None, "auto", 1):
        spec = polspec.profile(frame, round_bounds=rounding)
        polspec.validate(spec, frame)


def test_bounds_at_a_decimals_full_precision_are_declared_and_covered():
    """Python's default decimal context keeps 28 digits, and converting a
    bound to its physical integer under it rounded 10**38 - 1 up to 10**38,
    so a Decimal(38, 0) bounded by its own limits was refused where it was
    declared. A cartesian frame covers those bounds exactly too."""
    limit = 10**38 - 1
    spec = TableSpec("T", {"c": ColSpec(pl.Decimal(38, 0), bounds=(-limit, limit))})
    frame = polspec.generate(spec, 10, seed=1, method="cartesian")
    polspec.validate(spec, frame)
    assert (frame["c"] < 0).any() and (frame["c"] > 0).any()
