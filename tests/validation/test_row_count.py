"""`rows=`: how many rows a table may hold. An empty frame breaks no claim
about its rows, so this is the claim that catches one."""

import json

import polars as pl
import pytest
from polspec import (
    Bound,
    ColSpec,
    FrameSpec,
    SpecError,
    TableSpec,
    ValidationError,
    diff,
    from_dict,
    from_yaml,
    generate,
    inspect,
    to_dict,
    to_markdown,
    to_python,
    to_yaml,
    validate,
)
from polspec.cli import main


def run_cli(*args: str) -> int:
    return main(list(args))


COLUMNS = {"a": ColSpec(pl.Int64, bounds=(0, 9))}
NOT_EMPTY = TableSpec("T", COLUMNS, rows=(1, None))
EMPTY = pl.DataFrame({"a": []}, schema={"a": pl.Int64})


def frame(n: int) -> pl.DataFrame:
    return pl.DataFrame({"a": [1] * n}, schema={"a": pl.Int64})


# ---------------------------------------------------------------------------
# Declaring
# ---------------------------------------------------------------------------


def test_rows_is_held_as_a_bound():
    assert NOT_EMPTY.rows == Bound(1, None)
    assert TableSpec("T", COLUMNS, rows=[0, 5]).rows == Bound(0, 5)
    assert TableSpec("T", COLUMNS, rows=Bound(None, 5)).rows == Bound(None, 5)
    assert TableSpec("T", COLUMNS).rows is None


@pytest.mark.parametrize(
    "bad",
    [(None, None), (-1, None), (1.5, None), (True, None), (1, 2, 3), 5, "1"],
)
def test_a_count_that_cannot_be_is_refused(bad):
    with pytest.raises(SpecError, match=r"TableSpec\.rows"):
        TableSpec("T", COLUMNS, rows=bad)


def test_a_reversed_count_is_refused():
    with pytest.raises(SpecError):
        TableSpec("T", COLUMNS, rows=(5, 1))


def test_a_framespec_declares_it_and_a_subclass_is_held_to_it():
    class Orders(FrameSpec):
        __rows__ = (1, None)
        a = ColSpec(pl.Int64)

    class Inherits(Orders):
        pass

    class Clears(Orders):
        __rows__ = None

    class Replaces(Orders):
        __rows__ = (5, 10)

    assert Orders.spec.rows == Bound(1, None)
    assert Inherits.spec.rows == Bound(1, None)
    assert Clears.spec.rows is None
    assert Replaces.spec.rows == Bound(5, 10)


def test_with_rows_replaces_or_clears_it():
    assert NOT_EMPTY.with_rows((2, 3)).rows == Bound(2, 3)
    assert NOT_EMPTY.with_rows(None).rows is None


def test_reshaping_a_spec_keeps_it():
    spec = TableSpec(
        "T", {"a": ColSpec(pl.Int64), "b": ColSpec(pl.Int64)}, rows=(1, None)
    )
    assert spec.select("a").rows == spec.rows
    assert spec.drop("a").rows == spec.rows
    assert spec.rename({"a": "c"}).rows == spec.rows


# ---------------------------------------------------------------------------
# Validating
# ---------------------------------------------------------------------------


def test_an_empty_frame_fails_a_table_that_must_not_be_empty():
    report = inspect(NOT_EMPTY, EMPTY)
    assert [f.code for f in report] == ["row_count"]
    (finding,) = report.findings
    assert finding.key == "row_count"
    assert not finding.row_level
    assert finding.count is None
    assert dict(finding.details) == {"rows": 0, "min": 1, "max": None}


@pytest.mark.parametrize(
    ("n", "passed"), [(1, False), (2, True), (3, True), (4, False)]
)
def test_both_ends_are_inclusive(n, passed):
    assert inspect(TableSpec("T", COLUMNS, rows=(2, 3)), frame(n)).passed is passed


def test_a_lazy_frame_is_judged_the_same():
    assert [f.code for f in inspect(NOT_EMPTY, EMPTY.lazy())] == ["row_count"]
    assert inspect(NOT_EMPTY, frame(1).lazy()).passed


def test_it_works_on_the_streaming_engine():
    report = inspect(NOT_EMPTY, EMPTY.lazy(), streaming=True)
    assert [f.code for f in report] == ["row_count"]


def test_a_spec_with_no_other_claim_is_still_counted():
    spec = TableSpec("T", {"a": ColSpec(pl.Null, nullable=True)}, rows=(1, None))
    assert [f.code for f in inspect(spec, pl.DataFrame({"a": []}))] == ["row_count"]


def test_validate_raises_it():
    with pytest.raises(ValidationError, match="0 row"):
        validate(NOT_EMPTY, EMPTY)


def test_the_switch_turns_it_off():
    assert validate(NOT_EMPTY, EMPTY, rows=False).height == 0


def test_a_spec_without_rows_accepts_an_empty_frame():
    assert inspect(TableSpec("T", COLUMNS), EMPTY).passed


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def test_passing_rows_refuses_a_verdict_on_the_whole_frame():
    with pytest.raises(ValueError, match="row_count"):
        inspect(NOT_EMPTY, EMPTY).passing_rows()


def test_passing_rows_refuses_too_few_rows_to_be_accepted():
    spec = TableSpec("T", COLUMNS, rows=(3, None))
    report = inspect(spec, pl.DataFrame({"a": [1, 2, 100]}))
    assert [f.code for f in report] == ["bounds"]
    with pytest.raises(ValueError, match="at least 3"):
        report.passing_rows()
    assert report.failing_rows(once=True).collect()["a"].to_list() == [100]


def test_passing_rows_gives_what_is_enough():
    spec = TableSpec("T", COLUMNS, rows=(2, None))
    report = inspect(spec, pl.DataFrame({"a": [1, 2, 100]}))
    assert report.passing_rows().collect()["a"].to_list() == [1, 2]


def test_the_switch_lets_passing_rows_through():
    spec = TableSpec("T", COLUMNS, rows=(3, None))
    report = inspect(spec, pl.DataFrame({"a": [1, 2, 100]}), rows=False)
    assert report.passing_rows().collect().height == 2


# ---------------------------------------------------------------------------
# Generating
# ---------------------------------------------------------------------------


def test_generate_makes_the_rows_it_is_asked_for():
    spec = TableSpec("T", COLUMNS, rows=(100, None))
    small = generate(spec, 5, seed=1)
    assert small.height == 5
    assert [f.code for f in inspect(spec, small)] == ["row_count"]
    assert validate(spec, small, rows=False).equals(small)


def test_no_seed_moves():
    plain = generate(TableSpec("T", COLUMNS), 20, seed=1)
    assert generate(NOT_EMPTY, 20, seed=1).equals(plain)


# ---------------------------------------------------------------------------
# Writing it down
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [(1, None), (None, 5), (0, 5)])
def test_spec_files_and_dicts_carry_it(tmp_path, rows):
    spec = TableSpec("T", COLUMNS, rows=rows)
    path = tmp_path / "t.yaml"
    to_yaml(spec, path)
    assert from_yaml(path) == spec
    assert from_dict(to_dict(spec)) == spec
    assert to_dict(spec)["rows"] == list(rows)


def test_a_spec_without_it_writes_no_key():
    assert "rows" not in to_dict(TableSpec("T", COLUMNS))


def test_python_source_carries_it():
    namespace: dict = {}
    exec(to_python(NOT_EMPTY), namespace)
    assert namespace["T"].spec == NOT_EMPTY


def test_markdown_shows_it():
    assert "- **Row Count:** >= 1" in to_markdown(NOT_EMPTY)
    assert "Row Count" not in to_markdown(TableSpec("T", COLUMNS))


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("old", "new", "code", "severity"),
    [
        (None, (1, None), "constraint_added", "breaking"),
        ((1, None), None, "constraint_removed", "compatible"),
        ((1, None), (5, None), "domain_narrowed", "breaking"),
        ((1, None), (1, 10), "domain_narrowed", "breaking"),
        ((None, 10), (1, 10), "domain_narrowed", "breaking"),
        ((5, 10), (1, 10), "domain_widened", "compatible"),
        ((5, 10), (5, None), "domain_widened", "compatible"),
        ((5, 10), (1, 8), "domain_changed", "breaking"),
    ],
)
def test_diff_says_whether_a_new_count_accepts_less(old, new, code, severity):
    report = diff(TableSpec("T", COLUMNS, rows=old), TableSpec("T", COLUMNS, rows=new))
    (finding,) = report.findings
    assert (finding.code, finding.severity) == (code, severity)


def test_diff_says_nothing_of_a_count_unchanged():
    assert not diff(NOT_EMPTY, NOT_EMPTY).findings


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _files(tmp_path, rows, data):
    spec = tmp_path / "t.yaml"
    to_yaml(TableSpec("T", COLUMNS, rows=rows), spec)
    path = tmp_path / "t.csv"
    path.write_text("a\n" + "".join(f"{v}\n" for v in data), encoding="utf-8")
    return spec, path


def test_validate_reports_it_and_skip_turns_it_off(tmp_path, capsys):
    spec, data = _files(tmp_path, (5, None), [1])
    assert run_cli("validate", str(spec), str(data), "--json") == 1
    report = json.loads(capsys.readouterr().out)
    assert [f["code"] for f in report["findings"]] == ["row_count"]
    assert run_cli("validate", str(spec), str(data), "--skip", "rows") == 0


def test_output_says_when_too_few_rows_pass(tmp_path, capsys):
    spec, data = _files(tmp_path, (2, None), [1, 100])
    output = tmp_path / "out.csv"
    assert run_cli("validate", str(spec), str(data), "--output", str(output)) == 1
    assert "at least 2" in capsys.readouterr().err
    assert not output.exists()
