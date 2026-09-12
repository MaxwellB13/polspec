"""`pattern=`: a regular expression a String column must match, on validation only.

The honest half of the split `format=` made: polspec can check any regex
today and can generate a curated set. A pattern is checked like a format
and generated like nothing at all, and that boundary is pinned in
`test_roundtrip.py` beside the other validation-only claims.
"""

from __future__ import annotations

import re
import textwrap

import polars as pl
import pytest
from polspec import (
    CatSpec,
    ColSpec,
    FrameSpec,
    SpecError,
    TableSpec,
    ValidationOptions,
    inspect,
)
from polspec.cli import main
from polspec.drift import diff

SKU = r"^[A-Z]{3}-\d{4}$"


def _spec_for(column: ColSpec) -> type[FrameSpec]:
    return type("Patterned", (FrameSpec,), {"__columns__": {"c": column}})


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


def test_a_pattern_needs_a_string_column():
    with pytest.raises(SpecError, match=r"only supported for pl\.String"):
        ColSpec(pl.Int64, pattern=SKU)


def test_a_pattern_and_a_format_are_two_definitions_of_one_shape():
    with pytest.raises(SpecError, match="both format='email' and pattern"):
        ColSpec(pl.String, format="email", pattern=SKU)


def test_a_pattern_must_be_a_non_empty_string():
    with pytest.raises(SpecError, match="non-empty string, got ''"):
        ColSpec(pl.String, pattern="")
    with pytest.raises(SpecError, match="non-empty string, got 3"):
        ColSpec(pl.String, pattern=3)  # type: ignore[arg-type]


def test_a_pattern_is_compiled_by_polars_not_python():
    """Look-around is valid in Python's `re` and not in Polars' engine; the
    engine that will run the check is the one whose opinion counts."""
    with pytest.raises(SpecError, match="not a valid regular expression"):
        ColSpec(pl.String, pattern="(")
    with pytest.raises(SpecError, match="not a valid regular expression"):
        ColSpec(pl.String, pattern=r"^(?=a)a$")


def test_a_pattern_combines_with_what_it_does_not_contradict():
    spec = ColSpec(
        pl.String,
        pattern=SKU,
        string_length=(8, 8),
        choices=["ABC-1234"],
        nullable=True,
        unique=True,
        tags="sku",
    )
    assert spec.pattern == SKU


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_the_pattern_finding_names_the_pattern_and_samples():
    spec_cls = _spec_for(ColSpec(pl.String, pattern=SKU, nullable=True))
    df = pl.DataFrame({"c": ["ABC-1234", "abc-1234", None, "nope", "abc-1234"]})
    report = spec_cls.inspect(df)
    (finding,) = report.findings
    assert finding.code == "pattern"
    assert finding.count == 3
    assert finding.samples == ("abc-1234", "nope")
    assert dict(finding.details) == {"pattern": SKU}
    assert finding.message == (
        f"Column 'c': found 3 value(s) not matching pattern {SKU!r}. "
        "Invalid samples: ['abc-1234', 'nope']"
    )
    assert report.rows(finding).collect()["c"].to_list() == [
        "abc-1234",
        "nope",
        "abc-1234",
    ]


def test_matching_data_passes():
    spec_cls = _spec_for(ColSpec(pl.String, pattern=SKU))
    spec_cls.validate(pl.DataFrame({"c": ["ABC-1234", "XYZ-0001"]}))


def test_the_check_has_a_switch_like_the_other_validation_only_claims():
    spec_cls = _spec_for(ColSpec(pl.String, pattern=SKU))
    df = pl.DataFrame({"c": ["nope"]})
    assert not spec_cls.inspect(df).passed
    assert spec_cls.inspect(df, validate_pattern=False).passed
    assert spec_cls.validate(df, options=ValidationOptions(pattern=False)).height == 1
    assert inspect(spec_cls.spec, df, validate_pattern=False).passed


def test_a_pattern_is_not_checked_on_a_column_of_the_wrong_dtype():
    spec_cls = _spec_for(ColSpec(pl.String, pattern=SKU))
    report = spec_cls.inspect(pl.DataFrame({"c": [1, 2]}))
    assert [f.code for f in report.findings] == ["dtype"]


# ---------------------------------------------------------------------------
# Everything around it
# ---------------------------------------------------------------------------


def test_a_pattern_survives_a_file_round_trip(tmp_path):
    spec_cls = _spec_for(ColSpec(pl.String, pattern=SKU))
    path = tmp_path / "s.yaml"
    spec_cls.to_yaml(path)
    assert f"pattern: {SKU}" in path.read_text(encoding="utf-8")
    assert FrameSpec.from_yaml(path).spec == spec_cls.spec
    py = tmp_path / "s.py"
    spec_cls.to_python(py)
    assert f"pattern={SKU!r}" in py.read_text(encoding="utf-8")


def test_a_changed_pattern_is_a_compatible_change_not_a_domain_move():
    """Whether one regex contains another is not decided; a change is a change."""
    old = TableSpec("T", {"c": ColSpec(pl.String, pattern=SKU)})
    new = TableSpec("T", {"c": ColSpec(pl.String, pattern=r"^\w+$")})
    (finding,) = diff(old, new).findings
    assert finding.code == "field_changed" and not finding.breaking
    assert finding.details["field"] == "pattern"


def test_reports_show_the_pattern():
    spec_cls = _spec_for(ColSpec(pl.String, pattern=SKU))
    assert f"pattern `{SKU}`" in spec_cls.to_markdown()
    assert f"pattern: {SKU}" in spec_cls.to_mermaid()


def test_retyping_to_a_registry_category_drops_the_pattern_with_a_warning():
    cats = CatSpec(enums={"c": ["ABC-1234"]})
    spec = TableSpec("T", {"c": ColSpec(pl.String, pattern=SKU)})
    with pytest.warns(UserWarning, match=re.escape(f"dropping pattern={SKU!r}")):
        retyped = spec.with_catspec(cats)
    assert retyped.columns["c"].pattern is None


def test_the_generated_test_knows_a_pattern_is_validation_only(tmp_path):
    """`polspec test` disables the check with a comment saying why, the way
    it does for validators, rather than emitting a test it knows fails."""
    source = tmp_path / "specs.py"
    source.write_text(
        textwrap.dedent(
            f"""
            import polars as pl
            from polspec import ColSpec, FrameSpec

            class Skus(FrameSpec):
                sku = ColSpec(pl.String, pattern={SKU!r})
            """
        ),
        encoding="utf-8",
    )
    out = tmp_path / "test_skus.py"
    assert main(["test", str(source), "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "validate_pattern=False" in text
    assert "checked by validation only" in text
