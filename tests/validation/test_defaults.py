"""`default=`, `optional=` and `fill_defaults`: what stands in for a missing
value, and which columns may be missing altogether."""

import datetime as dt
import decimal
import json
import math

import polars as pl
import pytest
from polspec import (
    ColSpec,
    FrameSpec,
    SpecError,
    TableSpec,
    ValidationError,
    diff,
    drift,
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


SPEC = TableSpec(
    "Orders",
    {
        "id": ColSpec(pl.Int64),
        "status": ColSpec(pl.Enum(["open", "closed"]), default="open"),
        "qty": ColSpec(pl.Int64, bounds=(0, 9), default=1),
        "channel": ColSpec(pl.String, optional=True),
    },
)


# ---------------------------------------------------------------------------
# Declaring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "given", "held"),
    [
        (pl.Float64, 1, 1.0),
        (pl.Decimal(10, 2), "1.50", decimal.Decimal("1.50")),
        (pl.Date, dt.date(2026, 1, 1), dt.date(2026, 1, 1)),
        (pl.Time, dt.time(9, 30), dt.time(9, 30)),
        (pl.Boolean, False, False),
        (pl.Int64, 0, 0),
        (pl.Categorical, "x", "x"),
    ],
)
def test_a_default_is_kept_as_the_column_holds_it(dtype, given, held):
    default = ColSpec(dtype, default=given).default
    assert default == held
    assert type(default) is type(held)


@pytest.mark.parametrize(
    ("column", "says"),
    [
        ({"dtype": pl.Int64, "bounds": (0, 10), "default": 11}, "bounds"),
        ({"dtype": pl.String, "choices": ["a"], "default": "b"}, "choices"),
        ({"dtype": pl.String, "format": "email", "default": "x"}, "format"),
        ({"dtype": pl.String, "pattern": "^a", "default": "b"}, "pattern"),
        (
            {"dtype": pl.String, "string_length": (1, 2), "default": "abc"},
            "string_length",
        ),
        ({"dtype": pl.Enum(["a"]), "default": "b"}, "Enum categories"),
        ({"dtype": pl.Int8, "default": 1000}, "cannot be held"),
        ({"dtype": pl.Int64, "default": 1.5}, "cannot be held"),
        ({"dtype": pl.String, "default": 5}, "not text"),
        ({"dtype": pl.Float64, "default": math.nan}, "NaN"),
        ({"dtype": pl.Null, "nullable": True, "default": 1}, "pl.Null"),
    ],
)
def test_a_default_the_column_would_refuse_is_refused(column, says):
    with pytest.raises(SpecError, match=r"ColSpec\.default") as exc:
        ColSpec(**column)
    assert says in str(exc.value)


def test_a_default_is_held_to_the_columns_validators():
    with pytest.raises(SpecError, match="validator"):
        ColSpec(pl.Int64, validators=pl.col("x") % 2 == 0, default=3)
    assert ColSpec(pl.Int64, validators=pl.col("x") % 2 == 0, default=4).default == 4


@pytest.mark.parametrize(
    "dtype",
    [
        pl.List(pl.Int64),
        pl.Array(pl.Int64, 2),
        pl.Struct({"a": pl.Int64}),
    ],
)
def test_a_nested_column_takes_no_default(dtype):
    with pytest.raises(SpecError, match="only a scalar column"):
        ColSpec(dtype, default=1)


@pytest.mark.parametrize("claim", [{"default": 0}, {"optional": True}])
def test_a_struct_field_takes_neither(claim):
    with pytest.raises(SpecError, match="struct field"):
        ColSpec(
            pl.Struct({"a": pl.Int64}),
            fields={"a": ColSpec(pl.Int64, **claim)},
        )


def test_optional_is_a_flag():
    with pytest.raises(SpecError, match=r"ColSpec\.optional"):
        ColSpec(pl.Int64, optional="yes")


def test_a_framespec_declares_them_as_any_column_does():
    class Orders(FrameSpec):
        status = ColSpec(pl.Enum(["open", "closed"]), default="open")
        channel = ColSpec(pl.String, optional=True)

    assert Orders.spec["status"].default == "open"
    assert Orders.spec["channel"].optional


# ---------------------------------------------------------------------------
# Optional columns
# ---------------------------------------------------------------------------


def test_an_absent_optional_column_is_no_finding():
    frame = pl.DataFrame({"id": [1], "status": ["open"], "qty": [1]})
    assert inspect(SPEC, frame).passed


def test_a_present_optional_column_is_checked():
    spec = TableSpec("T", {"a": ColSpec(pl.Int64, bounds=(0, 1), optional=True)})
    assert [f.code for f in inspect(spec, pl.DataFrame({"a": [5]}))] == ["bounds"]


def test_a_required_column_is_still_missing():
    report = inspect(SPEC, pl.DataFrame({"status": ["open"], "qty": [1]}))
    (finding,) = report.findings
    assert finding.code == "missing_columns"
    assert finding.columns == ("id",)


# ---------------------------------------------------------------------------
# Filling
# ---------------------------------------------------------------------------

GAPS = pl.DataFrame(
    {"id": [1, 2], "status": [None, "closed"], "qty": [None, 3]},
    schema={"id": pl.Int64, "status": pl.String, "qty": pl.Int64},
)


def test_without_filling_a_null_is_still_a_finding():
    assert {f.code for f in inspect(SPEC, GAPS)} == {"nullability"}


def test_filling_passes_a_frame_whose_only_faults_are_nulls_with_defaults():
    filled = validate(SPEC, GAPS, fill_defaults=True)
    assert filled["status"].to_list() == ["open", "closed"]
    assert filled["qty"].to_list() == [1, 3]


def test_a_lazy_frame_is_filled_and_stays_lazy():
    filled = validate(SPEC, GAPS.lazy(), fill_defaults=True)
    assert isinstance(filled, pl.LazyFrame)
    assert filled.collect()["qty"].to_list() == [1, 3]


def test_an_absent_defaulted_column_is_added_holding_it():
    filled = validate(SPEC, GAPS.drop("qty"), fill_defaults=True)
    assert filled["qty"].to_list() == [1, 1]
    assert filled.columns == ["id", "status", "qty"]


def test_the_report_is_of_the_filled_frame():
    frame = GAPS.with_columns(id=pl.Series([1, None]))
    report = inspect(SPEC, frame, fill_defaults=True)
    assert [f.code for f in report] == ["nullability"]
    assert report.passing_rows().collect()["status"].to_list() == ["open"]
    assert report.failing_rows().collect()["qty"].to_list() == [3]


def test_a_nan_is_not_filled():
    spec = TableSpec("T", {"x": ColSpec(pl.Float64, default=0.0)})
    frame = pl.DataFrame({"x": [None, math.nan]})
    report = inspect(spec, frame, fill_defaults=True)
    assert [f.code for f in report] == ["nan"]


def test_a_column_of_the_wrong_dtype_is_not_filled():
    frame = GAPS.with_columns(qty=pl.Series([None, "3"]))
    report = inspect(SPEC, frame, fill_defaults=True)
    assert {f.code for f in report} == {"dtype", "nullability"}
    assert report.frame.collect()["qty"].to_list() == [None, "3"]


def test_a_column_without_a_default_is_left_alone():
    frame = GAPS.with_columns(id=pl.Series([None, 2]))
    report = inspect(SPEC, frame, fill_defaults=True)
    assert [(f.code, f.columns) for f in report] == [("nullability", ("id",))]


def test_filling_is_no_seed_change_and_generation_ignores_it():
    plain = TableSpec(
        "Orders",
        {
            "id": ColSpec(pl.Int64),
            "status": ColSpec(pl.Enum(["open", "closed"])),
            "qty": ColSpec(pl.Int64, bounds=(0, 9)),
            "channel": ColSpec(pl.String),
        },
    )
    assert generate(SPEC, 30, seed=1).equals(generate(plain, 30, seed=1))


def test_filling_twice_is_filling_once():
    once = validate(SPEC, GAPS, fill_defaults=True)
    assert validate(SPEC, once, fill_defaults=True).equals(once)


# ---------------------------------------------------------------------------
# missing_cols="add"
# ---------------------------------------------------------------------------


def test_add_uses_the_default_and_null_without_one():
    spec = TableSpec(
        "T",
        {
            "a": ColSpec(pl.Int64),
            "b": ColSpec(pl.Int64, default=7),
            "c": ColSpec(pl.Int64, nullable=True),
        },
    )
    added = validate(spec, pl.DataFrame({"a": [1, 2]}), missing_cols="add")
    assert added["b"].to_list() == [7, 7]
    assert added["c"].to_list() == [None, None]
    assert validate(spec, added).equals(added)


# ---------------------------------------------------------------------------
# Writing it down
# ---------------------------------------------------------------------------

WRITTEN = TableSpec(
    "T",
    {
        "a": ColSpec(pl.Int64, default=0, optional=True),
        "b": ColSpec(pl.Decimal(8, 2), default="2.50"),
        "c": ColSpec(pl.Time, default=dt.time(12, 30)),
        "d": ColSpec(pl.Datetime("us"), default=dt.datetime(2026, 1, 1, 9)),
        "e": ColSpec(pl.Duration("us"), default=dt.timedelta(hours=1)),
    },
)


def test_spec_files_and_dicts_carry_them(tmp_path):
    path = tmp_path / "t.yaml"
    to_yaml(WRITTEN, path)
    assert from_yaml(path) == WRITTEN
    assert from_dict(to_dict(WRITTEN)) == WRITTEN
    columns = to_dict(WRITTEN)["columns"]
    assert columns["a"]["optional"] is True
    assert columns["b"]["default"] == "2.50"


def test_a_spec_without_them_writes_no_key():
    column = to_dict(TableSpec("T", {"a": ColSpec(pl.Int64)}))["columns"]["a"]
    assert "default" not in column
    assert "optional" not in column


def test_a_file_with_a_refused_default_names_where(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(
        "version: 4\nname: T\ncolumns:\n"
        "  a:\n    dtype: Int64\n    bounds: [0, 5]\n    default: 9\n",
        encoding="utf-8",
    )
    with pytest.raises(SpecError, match=r"columns\.a.*ColSpec\.default 9"):
        from_yaml(path)


def test_python_source_carries_them():
    namespace: dict = {}
    exec(to_python(WRITTEN), namespace)
    assert namespace["T"].spec == WRITTEN


def test_markdown_shows_them():
    rows = {
        line.split("|")[1].strip(): line
        for line in to_markdown(SPEC).splitlines()
        if line.startswith("| `")
    }
    assert "No; default `open`" in rows["`status`"]
    assert "No; optional" in rows["`channel`"]


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------


def _one(old: ColSpec, new: ColSpec):
    (finding,) = diff(TableSpec("T", {"a": old}), TableSpec("T", {"a": new})).findings
    return finding.code, finding.severity


def test_diff_a_column_made_required_is_breaking_made_optional_compatible():
    required, optional = ColSpec(pl.Int64), ColSpec(pl.Int64, optional=True)
    assert _one(optional, required) == ("constraint_added", "breaking")
    assert _one(required, optional) == ("constraint_removed", "compatible")


def test_diff_a_changed_default_is_compatible():
    assert _one(ColSpec(pl.Int64), ColSpec(pl.Int64, default=1)) == (
        "field_changed",
        "compatible",
    )


def test_diff_an_optional_column_added_is_compatible():
    old = TableSpec("T", {"a": ColSpec(pl.Int64)})
    report = diff(old, old.with_columns(b=ColSpec(pl.Int64, optional=True)))
    assert report.compatible
    assert [f.code for f in report.findings] == ["column_added"]
    report = diff(old, old.with_columns(b=ColSpec(pl.Int64)))
    assert not report.compatible


def test_drift_says_nothing_of_an_absent_optional_column():
    frame = pl.DataFrame({"id": [1], "status": ["open"], "qty": [1]})
    assert not drift(SPEC, frame).findings


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_fill_defaults_on_the_command_line(tmp_path, capsys):
    spec = tmp_path / "t.yaml"
    to_yaml(SPEC, spec)
    data = tmp_path / "t.csv"
    data.write_text("id,status,qty\n1,,\n2,closed,3\n", encoding="utf-8")
    output = tmp_path / "out.csv"
    assert run_cli("validate", str(spec), str(data), "--json") == 1
    report = json.loads(capsys.readouterr().out)
    assert {f["code"] for f in report["findings"]} == {"nullability"}
    assert (
        run_cli(
            "validate", str(spec), str(data), "--fill-defaults", "--output", str(output)
        )
        == 0
    )
    written = pl.read_csv(output)
    assert written["status"].to_list() == ["open", "closed"]
    assert written["qty"].to_list() == [1, 3]


def test_validate_raises_for_a_null_it_was_not_asked_to_fill():
    with pytest.raises(ValidationError, match="status"):
        validate(SPEC, GAPS)
