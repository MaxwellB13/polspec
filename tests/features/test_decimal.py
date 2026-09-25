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
import pytest
from helpers import spec_for
from polspec import ColSpec, FrameSpec, SpecError, ValidationError
from polspec.drift import drift

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


def test_bounds_needing_more_than_eighteen_digits_are_refused_at_generation():
    spec_cls = spec_for(ColSpec(pl.Decimal(38, 10), bounds=(0, 10**20)))
    with pytest.raises(Exception, match="more than 18 significant digits"):
        spec_cls.generate(5, seed=1)


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
