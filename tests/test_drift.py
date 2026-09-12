"""`diff(old, new)`: what changed between two declarations.

The contract under test is the severity rule -- a finding is *breaking*
when a frame that satisfied `old` could fail `new` -- and the registry that
makes the comparison complete: one comparator per field of `ColSpec` and of
`TableSpec`, so a field added to either is one entry here or one red test.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json

import polars as pl
import pytest
from polspec import (
    Check,
    ColRule,
    ColSpec,
    DriftFinding,
    DriftOptions,
    DriftReport,
    ForeignKey,
    FrameSpec,
    Hierarchy,
    SpecError,
    TableSpec,
    col,
)
from polspec.drift import diff
from polspec.drift.fields import FIELD_COMPARATORS, TABLE_COMPARATORS


def _spec(**columns: ColSpec) -> TableSpec:
    return TableSpec("T", columns)


def _one(report: DriftReport, code: str) -> DriftFinding:
    found = report.by_code(code)
    assert len(found) == 1, f"expected one {code}, got {[f.code for f in report]}"
    return found[0]


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


def test_every_colspec_field_has_a_comparator():
    assert set(FIELD_COMPARATORS) == {f.name for f in dataclasses.fields(ColSpec)}


def test_every_tablespec_field_has_a_comparator():
    assert set(TABLE_COMPARATORS) == {f.name for f in dataclasses.fields(TableSpec)}


def test_a_spec_does_not_drift_from_itself():
    spec = _spec(
        a=ColSpec(pl.Int64, bounds=(0, 10), nullable=True, tags="x"),
        b=ColSpec(pl.Enum(["p", "q"]), weights=[1.0, 2.0]),
        c=ColSpec(pl.String, format="email", validators=[col("c").str.contains("@")]),
    )
    report = diff(spec, spec)
    assert report.unchanged and bool(report) and len(report) == 0
    assert str(report) == "No drift: 'T' and 'T'"


def test_a_name_is_not_drift():
    spec = _spec(a=ColSpec(pl.Int64))
    assert diff(spec, spec.with_name("Renamed")).unchanged


def test_accepts_framespec_classes():
    class A(FrameSpec):
        a = ColSpec(pl.Int64)

    class B(FrameSpec):
        a = ColSpec(pl.Int64, nullable=True)

    assert A.diff(B).by_code("nullability_changed")
    assert diff(A, B).old == "A" and diff(A, B).new == "B"


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def test_added_and_removed_columns_are_both_breaking():
    old = _spec(a=ColSpec(pl.Int64), gone=ColSpec(pl.Int64))
    new = _spec(a=ColSpec(pl.Int64), added=ColSpec(pl.Int64, nullable=True))
    report = diff(old, new)
    removed, added = _one(report, "column_removed"), _one(report, "column_added")
    assert removed.breaking and removed.columns == ("gone",)
    assert added.breaking and added.columns == ("added",)
    assert "missing_cols='raise'" in added.message
    assert "extra_cols='raise'" in removed.message


def test_a_declared_rename_is_one_compatible_finding():
    old = _spec(a=ColSpec(pl.Int64, bounds=(0, 5)))
    new = _spec(b=ColSpec(pl.Int64, bounds=(0, 5)))
    assert len(diff(old, new)) == 2  # removed + added, without the hint
    report = diff(old, new, renames={"a": "b"})
    (renamed,) = report.findings
    assert renamed.code == "column_renamed" and not renamed.breaking
    assert renamed.columns == ("a", "b")


def test_a_rename_goes_through_tablespec_rename():
    old = _spec(a=ColSpec(pl.Int64, validators=[col("a") > 0]))
    with pytest.raises(SpecError, match="carries validators"):
        diff(old, old, renames={"a": "b"})


# ---------------------------------------------------------------------------
# dtype and nullability
# ---------------------------------------------------------------------------


def test_dtype_severity_is_whatever_validation_would_say():
    widened = diff(_spec(a=ColSpec(pl.Int32)), _spec(a=ColSpec(pl.Int64)))
    assert not _one(widened, "dtype_changed").breaking
    # The same change under strict dtypes is a failure, so it is breaking.
    strict = diff(
        _spec(a=ColSpec(pl.Int32)),
        _spec(a=ColSpec(pl.Int64)),
        options=DriftOptions(strict_dtypes=True),
    )
    assert _one(strict, "dtype_changed").breaking
    crossed = diff(_spec(a=ColSpec(pl.Int64)), _spec(a=ColSpec(pl.String)))
    assert _one(crossed, "dtype_changed").breaking


def test_string_and_utf8_are_one_dtype():
    assert diff(_spec(a=ColSpec(pl.String)), _spec(a=ColSpec(pl.Utf8))).unchanged


@pytest.mark.parametrize(
    "old, new, breaking",
    [(False, True, False), (True, False, True)],
)
def test_nullability_direction_decides_severity(old, new, breaking):
    report = diff(
        _spec(a=ColSpec(pl.Int64, nullable=old)),
        _spec(a=ColSpec(pl.Int64, nullable=new)),
    )
    finding = _one(report, "nullability_changed")
    assert finding.breaking is breaking
    assert dict(finding.details) == {"old": old, "new": new}


# ---------------------------------------------------------------------------
# The domain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old, new, code",
    [
        (
            ColSpec(pl.Int64, bounds=(0, 10)),
            ColSpec(pl.Int64, bounds=(0, 20)),
            "domain_widened",
        ),
        (
            ColSpec(pl.Int64, bounds=(0, 20)),
            ColSpec(pl.Int64, bounds=(0, 10)),
            "domain_narrowed",
        ),
        (
            ColSpec(pl.Int64, bounds=(0, 10)),
            ColSpec(pl.Int64, bounds=(5, 20)),
            "domain_changed",
        ),
        (ColSpec(pl.Int64), ColSpec(pl.Int64, bounds=(0, 10)), "domain_narrowed"),
        (ColSpec(pl.Int64, bounds=(0, 10)), ColSpec(pl.Int64), "domain_widened"),
        (
            ColSpec(pl.Int64, bounds=(0, None)),
            ColSpec(pl.Int64, bounds=(0, 10)),
            "domain_narrowed",
        ),
        (
            ColSpec(pl.String, choices=["a"]),
            ColSpec(pl.String, choices=["a", "b"]),
            "domain_widened",
        ),
        (
            ColSpec(pl.String, choices=["a", "b"]),
            ColSpec(pl.String, choices=["b", "c"]),
            "domain_changed",
        ),
        (ColSpec(pl.Enum(["a", "b"])), ColSpec(pl.Enum(["a"])), "domain_narrowed"),
        (
            ColSpec(pl.String, format="email"),
            ColSpec(pl.String, format="uuid4"),
            "domain_changed",
        ),
        (ColSpec(pl.String), ColSpec(pl.String, format="email"), "domain_narrowed"),
        (ColSpec(pl.String, format="email"), ColSpec(pl.String), "domain_widened"),
        (
            ColSpec(pl.Date, bounds=(dt.date(2024, 1, 1), dt.date(2025, 1, 1))),
            ColSpec(pl.Date, bounds=(dt.date(2024, 1, 1), dt.date(2026, 1, 1))),
            "domain_widened",
        ),
    ],
)
def test_domain_relation_is_read_from_both_directions(old, new, code):
    report = diff(_spec(a=old), _spec(a=new))
    finding = _one(report, code)
    assert finding.breaking is (code != "domain_widened")
    assert finding.key == "a__domain"


def test_widened_one_way_is_narrowed_the_other():
    a, b = _spec(a=ColSpec(pl.Int64, bounds=(0, 10))), _spec(a=ColSpec(pl.Int64))
    assert diff(a, b).by_code("domain_widened") and diff(b, a).by_code(
        "domain_narrowed"
    )


def test_the_domain_is_one_claim_reported_once():
    """`bounds`, `choices` and `format` share a comparator; it runs once."""
    old = _spec(a=ColSpec(pl.String, choices=["a"]))
    new = _spec(a=ColSpec(pl.String, choices=["a", "b"]))
    assert [f.code for f in diff(old, new)] == ["domain_widened"]


def test_enum_to_string_with_the_same_values_is_only_a_dtype_change():
    old = _spec(a=ColSpec(pl.Enum(["a", "b"])))
    new = _spec(a=ColSpec(pl.String, choices=["a", "b"]))
    assert [f.code for f in diff(old, new)] == ["dtype_changed"]


@pytest.mark.parametrize(
    "old, new, code",
    [
        ((0, 10), (0, 20), "domain_widened"),
        ((0, 20), (0, 10), "domain_narrowed"),
        ((0, 10), (5, 20), "domain_changed"),
        (None, (0, 10), "domain_narrowed"),
        ((0, 10), None, "domain_widened"),
    ],
)
def test_string_length_moves_like_a_domain(old, new, code):
    report = diff(
        _spec(a=ColSpec(pl.String, string_length=old)),
        _spec(a=ColSpec(pl.String, string_length=new)),
    )
    finding = _one(report, code)
    assert finding.key == "a__string_length"
    assert finding.details["field"] == "string_length"


# ---------------------------------------------------------------------------
# Constraints and the fields that only shape generation
# ---------------------------------------------------------------------------


def test_unique_is_a_constraint():
    added = diff(_spec(a=ColSpec(pl.Int64)), _spec(a=ColSpec(pl.Int64, unique=True)))
    assert _one(added, "constraint_added").breaking
    removed = diff(_spec(a=ColSpec(pl.Int64, unique=True)), _spec(a=ColSpec(pl.Int64)))
    assert not _one(removed, "constraint_removed").breaking
    assert removed.findings[0].details["kind"] == "unique"


def test_a_validator_modified_under_the_same_name_is_removed_and_added():
    old = _spec(a=ColSpec(pl.Int64, validators=[Check(col("a") > 0, name="pos")]))
    new = _spec(a=ColSpec(pl.Int64, validators=[Check(col("a") > 1, name="pos")]))
    report = diff(old, new)
    assert [f.code for f in report] == ["constraint_removed", "constraint_added"]
    assert {f.key for f in report} == {"a__validator:pos"}
    assert all(f.details["name"] == "pos" for f in report)


def test_rules_are_matched_by_their_condition():
    rule = ColRule(when=col("b") == "x", choices=[1])
    old = _spec(a=ColSpec(pl.Int64, rules=[rule]), b=ColSpec(pl.String))
    new = _spec(a=ColSpec(pl.Int64), b=ColSpec(pl.String))
    finding = _one(diff(old, new), "constraint_removed")
    assert finding.details["kind"] == "rule"
    assert _one(diff(new, old), "constraint_added").breaking


@pytest.mark.parametrize(
    "field, old, new",
    [
        ("tags", ColSpec(pl.Int64), ColSpec(pl.Int64, tags="pii")),
        ("weights", ColSpec(pl.Boolean), ColSpec(pl.Boolean, weights=[0.2, 0.8])),
        (
            "distribution",
            ColSpec(pl.Float64),
            ColSpec(pl.Float64, distribution="normal"),
        ),
        (
            "distribution_params",
            ColSpec(pl.Float64, distribution="normal"),
            ColSpec(
                pl.Float64, distribution="normal", distribution_params={"mean": 2.0}
            ),
        ),
        (
            "null_probability",
            ColSpec(pl.Int64, nullable=True, null_probability=0.1),
            ColSpec(pl.Int64, nullable=True, null_probability=0.5),
        ),
    ],
)
def test_generation_only_fields_are_compatible_changes(field, old, new):
    report = diff(_spec(a=old), _spec(a=new))
    finding = _one(report, "field_changed")
    assert not finding.breaking
    assert finding.details["field"] == field
    assert finding.key == f"a__{field}"


def test_a_null_rate_on_a_non_nullable_column_means_nothing():
    old = _spec(a=ColSpec(pl.Int64, null_probability=0.0))
    new = _spec(a=ColSpec(pl.Int64, null_probability=0.0))
    assert diff(old, new).unchanged


# ---------------------------------------------------------------------------
# Table-level constraints
# ---------------------------------------------------------------------------


def test_table_constraints_added_are_breaking_and_removed_are_not():
    columns = {
        "a": ColSpec(pl.Int64),
        "b": ColSpec(pl.Int64),
        "k": ColSpec(pl.Int64),
        "ref": ColSpec(pl.Int64),
        "parent": ColSpec(pl.Int64),
    }
    bare = TableSpec("T", columns)
    full = TableSpec(
        "T",
        columns,
        checks=[Check(col("a") >= col("b"), name="a_gte_b")],
        unique_together=[["a", "b"]],
        foreign_keys=[ForeignKey("k", references="self", ref_columns="a")],
        hierarchy=Hierarchy(child="ref", parent="parent", max_depth=2),
    )
    added = diff(bare, full)
    assert {f.details["kind"] for f in added} == {
        "check",
        "unique_together",
        "foreign_key",
        "hierarchy",
    }
    assert all(f.code == "constraint_added" and f.breaking for f in added)
    check = next(f for f in added if f.details["kind"] == "check")
    assert check.columns == ("a", "b") and check.key == "check:a_gte_b"

    removed = diff(full, bare)
    assert all(f.code == "constraint_removed" and not f.breaking for f in removed)


def test_a_check_modified_under_the_same_name_is_removed_and_added():
    columns = {"a": ColSpec(pl.Int64)}
    old = TableSpec("T", columns, checks=[Check(col("a") > 0, name="c")])
    new = TableSpec("T", columns, checks=[Check(col("a") > 1, name="c")])
    assert [f.code for f in diff(old, new)] == [
        "constraint_removed",
        "constraint_added",
    ]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_the_report_is_data():
    old = _spec(a=ColSpec(pl.Int64, bounds=(0, 10)), b=ColSpec(pl.Int64))
    new = _spec(a=ColSpec(pl.Int64), b=ColSpec(pl.Int64, unique=True))
    report = diff(old, new)
    assert not report and not report.unchanged
    assert {f.key for f in report.breaking} == {"b__unique"}
    assert {f.key for f in report.compatible} == {"a__domain"}
    assert set(report.by_column()) == {"a", "b"}
    data = json.loads(report.to_json())
    assert data["kind"] == "diff" and data["breaking"] == 1
    assert [f["code"] for f in data["findings"]] == [
        "domain_widened",
        "constraint_added",
    ]
    assert str(report).startswith("Drift: 1 breaking, 1 compatible, 'T' and 'T'")
    # Breaking first, whatever order they were found in.
    lines = str(report).splitlines()[1:]
    assert lines[0].startswith("  - [breaking]") and lines[1].startswith(
        "  - [compatible]"
    )


def test_markdown_puts_breaking_first(tmp_path):
    old = _spec(a=ColSpec(pl.Int64, bounds=(0, 10)), b=ColSpec(pl.Int64))
    new = _spec(a=ColSpec(pl.Int64), b=ColSpec(pl.Int64, unique=True))
    path = tmp_path / "drift.md"
    text = diff(old, new).to_markdown(path)
    assert path.read_text(encoding="utf-8") == text
    assert text.index("## Breaking") < text.index("## Compatible")
    assert "| `b` | `constraint_added` | unique=True added" in text
    assert diff(old, old).to_markdown().rstrip().endswith("No drift.")


def test_options_must_be_the_right_type():
    with pytest.raises(TypeError, match="must be a DriftOptions"):
        diff(_spec(a=ColSpec(pl.Int64)), _spec(a=ColSpec(pl.Int64)), options={"x": 1})
    with pytest.raises(ValueError, match="null_rate_tolerance"):
        DriftOptions(null_rate_tolerance=2)
