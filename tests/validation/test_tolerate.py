"""`tolerate=`: findings accepted up to a share of the rows.

A tolerated finding stays in the report, marked, and does not fail it; the
most specific tolerance -- a code in a column, a column, a code, every
finding -- is the one that applies, and a finding with no share is never
tolerated.
"""

from __future__ import annotations

import json
import pickle

import polars as pl
import polspec
import pytest
from polspec import (
    Check,
    ColSpec,
    FrameSpec,
    TableSpec,
    ValidationError,
    ValidationOptions,
    col,
)
from polspec.cli import main


class Feed(FrameSpec):
    total = ColSpec(pl.Float64, bounds=(0, 100))
    note = ColSpec(pl.String)
    __checks__ = [Check(col("total") >= 1, name="at_least_one")]


# 1,000 rows: one total out of bounds (0.1%), and twenty null notes (2%).
FRAME = pl.DataFrame(
    {"total": [250.0] + [10.0] * 999, "note": [None] * 20 + ["x"] * 980}
)


def _codes(report) -> dict[str, bool]:
    return {finding.code: finding.tolerated for finding in report.findings}


def test_without_a_tolerance_every_finding_fails():
    report = Feed.inspect(FRAME)
    assert not report.passed
    assert _codes(report) == {"bounds": False, "nullability": False}
    assert report.failures == report.findings and report.tolerated == ()


def test_one_share_tolerates_every_finding_up_to_it():
    assert Feed.inspect(FRAME, tolerate=0.02).passed
    half = Feed.inspect(FRAME, tolerate=0.001)
    assert _codes(half) == {"bounds": True, "nullability": False}
    assert not half.passed


def test_a_share_is_inclusive():
    """`0.001` accepts exactly one row in a thousand."""
    assert _codes(Feed.inspect(FRAME, tolerate=0.001))["bounds"] is True
    assert _codes(Feed.inspect(FRAME, tolerate=0.000999))["bounds"] is False


@pytest.mark.parametrize(
    ("tolerate", "expected"),
    [
        ({"bounds": 0.01}, {"bounds": True, "nullability": False}),
        ({"note": 0.05}, {"bounds": False, "nullability": True}),
        ({("total", "bounds"): 0.01}, {"bounds": True, "nullability": False}),
        ({("note", "nullability"): 0.01}, {"bounds": False, "nullability": False}),
    ],
    ids=["by code", "by column", "by both", "by both, too small"],
)
def test_a_mapping_tolerates_by_code_column_or_both(tolerate, expected):
    assert _codes(Feed.inspect(FRAME, tolerate=tolerate)) == expected


def test_the_most_specific_tolerance_wins():
    """A code in a column over the column, the column over the code."""
    tight = {("total", "bounds"): 0.0, "total": 0.5, "bounds": 0.5}
    assert _codes(Feed.inspect(FRAME, tolerate=tight))["bounds"] is False
    loose = {("total", "bounds"): 0.5, "total": 0.0}
    assert _codes(Feed.inspect(FRAME, tolerate=loose))["bounds"] is True
    column_over_code = {"note": 0.0, "nullability": 0.5}
    assert (
        _codes(Feed.inspect(FRAME, tolerate=column_over_code))["nullability"] is False
    )


def test_a_check_across_columns_takes_the_largest_of_its_columns_tolerances():
    class Paired(FrameSpec):
        a = ColSpec(pl.Int64)
        b = ColSpec(pl.Int64)
        __checks__ = [Check(col("a") <= col("b"), name="ordered")]

    frame = pl.DataFrame({"a": [5] + [1] * 99, "b": [1] * 100})  # 1% fail
    assert Paired.inspect(frame, tolerate={"a": 0.0, "b": 0.01}).passed


def test_validate_raises_only_for_what_no_tolerance_covers():
    assert Feed.validate(FRAME, tolerate=0.02).height == 1_000
    with pytest.raises(ValidationError) as caught:
        Feed.validate(FRAME, tolerate={"bounds": 0.01})
    assert len(caught.value.errors) == 1
    assert "null" in caught.value.errors[0]


def test_a_finding_with_no_share_is_never_tolerated():
    missing = FRAME.drop("note")
    report = Feed.inspect(missing, tolerate=1.0)
    assert not report.passed
    assert _codes(report)["missing_columns"] is False


def test_passing_rows_still_leaves_out_tolerated_rows():
    report = Feed.inspect(FRAME, tolerate=0.02)
    # Row 0 is out of bounds and null; rows 1 to 19 are null.
    assert report.passing_rows().collect().height == 1_000 - 20


def test_a_tolerated_finding_is_reported_and_marked():
    report = Feed.inspect(FRAME, tolerate={"bounds": 0.01})
    text = str(report)
    assert "1 error(s) found, 1 tolerated" in text
    assert "[tolerated]" in text
    assert ", tolerated |" in report._repr_markdown_()
    data = json.loads(report.to_json())
    assert {f["code"]: f["tolerated"] for f in data["findings"]} == {
        "bounds": True,
        "nullability": False,
    }
    everything = str(Feed.inspect(FRAME, tolerate=0.5))
    assert (
        everything.startswith("Validation passed")
        and "2 finding(s) tolerated" in everything
    )


def test_the_options_object_takes_it_and_pickles():
    options = ValidationOptions(
        tolerate={"bounds": 0.01, ("note", "nullability"): 0.05}
    )
    assert Feed.inspect(FRAME, options=options).passed
    assert pickle.loads(pickle.dumps(options)) == options  # noqa: S301 - our own
    assert hash(options) == hash(
        ValidationOptions(tolerate={"bounds": 0.01, ("note", "nullability"): 0.05})
    )


@pytest.mark.parametrize(
    ("tolerate", "error", "match"),
    [
        (2, ValueError, "from 0 to 1, got 2"),
        (-0.1, ValueError, "from 0 to 1"),
        (float("nan"), ValueError, "from 0 to 1"),
        (True, TypeError, "a number from 0 to 1, got bool"),
        ({"bounds": "1%"}, TypeError, "got str"),
        ({("total", "bonds"): 0.1}, ValueError, "Did you mean 'bounds'"),
        ({("total",): 0.1}, TypeError, "a \\(column, code\\) pair"),
        ({1: 0.1}, TypeError, "got int"),
    ],
)
def test_a_tolerance_that_is_not_one_is_refused_when_the_options_are_made(
    tolerate, error, match
):
    with pytest.raises(error, match=match):
        ValidationOptions(tolerate=tolerate)


@pytest.mark.parametrize(
    ("tolerate", "match"),
    [
        ({"totl": 0.1}, "neither a finding code nor a column.*Did you mean 'total'"),
        ({("totl", "bounds"): 0.1}, "names the column 'totl'.*Did you mean 'total'"),
    ],
)
def test_a_key_the_spec_does_not_have_is_refused_by_name(tolerate, match):
    with pytest.raises(ValueError, match=match):
        Feed.inspect(FRAME, tolerate=tolerate)


def test_a_key_that_is_both_a_code_and_a_column_is_refused():
    spec = TableSpec("Steps", {"order": ColSpec(pl.Int64, bounds=(0, 10))})
    frame = pl.DataFrame({"order": [50] + [1] * 99})
    with pytest.raises(ValueError, match="both a finding code and a column"):
        polspec.inspect(spec, frame, tolerate={"order": 0.1})
    assert polspec.inspect(spec, frame, tolerate={("order", "bounds"): 0.01}).passed


def test_a_registry_validates_with_it():
    class Other(FrameSpec):
        total = ColSpec(pl.Float64, bounds=(0, 100))

    registry = polspec.Registry(Feed, Other)
    frames = {"Feed": FRAME, "Other": FRAME.select("total")}
    reports = registry.inspect_all(frames, tolerate=0.02)
    assert all(report.passed for report in reports.values())


# ---------------------------------------------------------------------------
# polspec validate --tolerate
# ---------------------------------------------------------------------------


@pytest.fixture
def files(tmp_path):
    spec, data = tmp_path / "feed.yaml", tmp_path / "feed.csv"
    Feed.to_yaml(spec)
    FRAME.write_csv(data)
    return str(spec), str(data)


@pytest.mark.parametrize(
    ("flags", "status"),
    [
        ([], 1),
        (["--tolerate", "2%"], 0),
        (["--tolerate", "0.001"], 1),
        (["--tolerate", "bounds=0.001", "--tolerate", "note=0.02"], 0),
        (["--tolerate", "total:bounds=0.1%", "--tolerate", "nullability=2%"], 0),
        (["--tolerate", "total:bounds=0.1%"], 1),
    ],
)
def test_the_command_line_tolerates_by_the_same_keys(files, flags, status, capsys):
    assert main(["validate", *files, *flags]) == status
    out = capsys.readouterr().out
    if status == 0 and flags:
        assert "tolerated" in out


@pytest.mark.parametrize(
    ("flags", "match"),
    [
        (["--tolerate", "lots"], "is not a share of the rows"),
        (["--tolerate", "0.1", "--tolerate", "bounds=0.1"], "not both"),
        (["--tolerate", "totl=0.1"], "Did you mean 'total'"),
    ],
)
def test_the_command_line_refuses_a_tolerance_it_cannot_read(
    files, flags, match, capsys
):
    assert main(["validate", *files, *flags]) == 1
    assert match in capsys.readouterr().err
