"""`polspec validate`: a data file against a spec, read in the spec's terms,
and split into what passed and what did not.
"""

import json

import polars as pl
import pytest
from helpers import run_cli, write_orders_specs
from polspec import ColSpec, FrameSpec

# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_passing_data_exits_zero(tmp_path, capsys):
    source, customers = write_orders_specs(tmp_path)
    data = tmp_path / "orders.parquet"
    pl.DataFrame(
        {"order_id": [1, 2], "customer_id": [1, 3], "total": [5.0, 50.0]}
    ).write_parquet(data)
    code = run_cli(
        "validate",
        source,
        data,
        "--class",
        "Orders",
        "--references",
        f"Customers={customers}",
    )
    assert code == 0
    assert "Validation passed" in capsys.readouterr().out


def test_validate_failing_data_exits_one_and_prints_the_report(tmp_path, capsys):
    source, customers = write_orders_specs(tmp_path)
    data = tmp_path / "orders.csv"
    pl.DataFrame(
        {"order_id": [1, 1], "customer_id": [1, 9], "total": [5.0, 500.0]}
    ).write_csv(data)
    code = run_cli(
        "validate",
        source,
        data,
        "--class",
        "Orders",
        "--references",
        f"Customers={customers}",
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "Validation failed" in out
    assert "out of bounds" in out and "duplicate" in out.lower()
    assert "Customers" in out  # the orphaned customer_id


def test_validate_json_output_is_the_report(tmp_path, capsys):
    source, customers = write_orders_specs(tmp_path)
    data = tmp_path / "orders.parquet"
    pl.DataFrame(
        {"order_id": [1, 2], "customer_id": [1, 2], "total": [5.0, 500.0]}
    ).write_parquet(data)
    code = run_cli(
        "validate",
        source,
        data,
        "--class",
        "Orders",
        "--json",
        "--references",
        f"Customers={customers}",
    )
    assert code == 1
    data = json.loads(capsys.readouterr().out)
    assert data["spec"] == "Orders" and data["passed"] is False
    assert [f["code"] for f in data["findings"]] == ["bounds"]


def test_validate_without_references_reports_the_unresolved_key(tmp_path, capsys):
    source, _ = write_orders_specs(tmp_path)
    data = tmp_path / "orders.parquet"
    pl.DataFrame({"order_id": [1], "customer_id": [1], "total": [5.0]}).write_parquet(
        data
    )
    assert run_cli("validate", source, data, "--class", "Orders") == 1
    assert "no DataFrame for it was supplied" in capsys.readouterr().out


def test_validate_needs_class_when_the_source_defines_several(tmp_path, capsys):
    source, _ = write_orders_specs(tmp_path)
    data = tmp_path / "orders.parquet"
    pl.DataFrame({"id": [1]}).write_parquet(data)
    assert run_cli("validate", source, data) == 1
    assert "pick one with --class" in capsys.readouterr().err


def test_validate_rejects_malformed_references(tmp_path, capsys):
    source, _ = write_orders_specs(tmp_path)
    data = tmp_path / "orders.parquet"
    pl.DataFrame({"id": [1]}).write_parquet(data)
    code = run_cli(
        "validate", source, data, "--class", "Orders", "--references", "nonsense"
    )
    assert code == 1
    assert "NAME=PATH" in capsys.readouterr().err


def test_validate_from_yaml_spec(tmp_path, capsys):
    class Items(FrameSpec):
        sku = ColSpec(pl.String, string_length=(3, 3))

    spec_path = tmp_path / "items.yaml"
    Items.to_yaml(spec_path)
    data = tmp_path / "items.ndjson"
    pl.DataFrame({"sku": ["abc", "toolong"]}).write_ndjson(data)
    assert run_cli("validate", spec_path, data) == 1
    assert "sku" in capsys.readouterr().out
    good = tmp_path / "good.ndjson"
    pl.DataFrame({"sku": ["abc"]}).write_ndjson(good)
    assert run_cli("validate", spec_path, good, "--allow-extra") == 0


# ---------------------------------------------------------------------------
# A text file read in the spec's terms
# ---------------------------------------------------------------------------

TEMPORAL_SPEC = """
import datetime as dt
import polars as pl
from polspec import ColSpec, FrameSpec

class Events(FrameSpec):
    day = ColSpec(pl.Date, bounds=(dt.date(2024, 1, 1), dt.date(2025, 1, 1)))
    at = ColSpec(pl.Datetime("us"))
    zoned = ColSpec(pl.Datetime("ms", "Europe/London"))
    clock = ColSpec(pl.Time)
    ref = ColSpec(pl.String)
"""


def _temporal_spec(tmp_path):
    source = tmp_path / "events.py"
    source.write_text(TEMPORAL_SPEC, encoding="utf-8")
    return source


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".ndjson", ".json"])
def test_a_text_file_polspec_wrote_validates_against_its_spec(tmp_path, capsys, suffix):
    """A text format has no date type, so a Date arrives as text; the CLI
    parses each column the spec declares as a date or time, and the file
    `generate` wrote validates against the spec that wrote it."""
    source = _temporal_spec(tmp_path)
    data = tmp_path / f"events{suffix}"
    assert run_cli("generate", source, "-n", "200", "--seed", "1", "-o", data) == 0
    capsys.readouterr()
    assert run_cli("validate", source, data) == 0, capsys.readouterr().out
    assert run_cli("drift", source, data, "--fail-on", "breaking") == 0


def test_a_date_out_of_bounds_in_a_csv_is_a_bounds_finding(tmp_path, capsys):
    """Parsed, the column's own checks run -- where before the dtype
    finding was the whole answer."""
    source = _temporal_spec(tmp_path)
    data = tmp_path / "events.csv"
    data.write_text(
        "day,at,zoned,clock,ref\n"
        "2023-06-01,2024-01-01T00:00:00,2024-01-01T00:00:00+0000,12:00:00,x\n",
        encoding="utf-8",
    )
    assert run_cli("validate", source, data, "--json") == 1
    findings = json.loads(capsys.readouterr().out)["findings"]
    assert [f["key"] for f in findings] == ["day__bounds"]


def test_a_date_that_does_not_parse_stays_text_and_says_so(tmp_path, capsys):
    """One bad value leaves the column as it was read: the dtype finding is
    true, where parsing it to a null would invent one."""
    source = _temporal_spec(tmp_path)
    data = tmp_path / "events.csv"
    data.write_text(
        "day,at,zoned,clock,ref\n"
        "2024-06-01,2024-01-01T00:00:00,2024-01-01T00:00:00+0000,12:00:00,x\n"
        "not a date,2024-01-01T00:00:00,2024-01-01T00:00:00+0000,12:00:00,x\n",
        encoding="utf-8",
    )
    assert run_cli("validate", source, data, "--json") == 1
    findings = json.loads(capsys.readouterr().out)["findings"]
    assert [(f["code"], f["key"]) for f in findings] == [("dtype", "day__dtype")]


def test_a_string_column_of_date_shaped_text_stays_a_string(tmp_path, capsys):
    source = _temporal_spec(tmp_path)
    data = tmp_path / "events.csv"
    data.write_text(
        "day,at,zoned,clock,ref\n"
        "2024-06-01,2024-01-01T00:00:00,2024-01-01T00:00:00+0000,12:00:00,"
        "2024-06-01\n",
        encoding="utf-8",
    )
    assert run_cli("validate", source, data) == 0, capsys.readouterr().out


def test_schema_infer_reads_a_csv_date_as_a_date(tmp_path):
    data = tmp_path / "events.csv"
    data.write_text("day,n\n2024-01-02,1\n2024-03-04,2\n", encoding="utf-8")
    out = tmp_path / "events.yaml"
    assert run_cli("schema", "infer", data, "-o", out) == 0
    inferred = FrameSpec.from_yaml(out)
    assert inferred.spec.columns["day"].dtype == pl.Date


# ---------------------------------------------------------------------------
# --skip
# ---------------------------------------------------------------------------


def test_validate_skip_turns_off_a_kind_of_check(tmp_path, capsys):
    source = _temporal_spec(tmp_path)
    data = tmp_path / "events.csv"
    data.write_text(
        "day,at,zoned,clock,ref\n"
        "2023-06-01,2024-01-01T00:00:00,2024-01-01T00:00:00+0000,12:00:00,x\n",
        encoding="utf-8",
    )
    assert run_cli("validate", source, data) == 1
    capsys.readouterr()
    assert run_cli("validate", source, data, "--skip", "bounds", "--json") == 0
    report = json.loads(capsys.readouterr().out)
    assert report["findings"] == []


def test_validate_skip_takes_every_switch_and_nothing_else(tmp_path, capsys):
    switches = ["rules", "validators", "unique", "checks", "foreign_keys"]
    switches += ["hierarchy", "order", "rows", "pattern", "bounds"]  # as documented
    source = _temporal_spec(tmp_path)
    data = tmp_path / "events.csv"
    data.write_text(
        "day,at,zoned,clock,ref\n"
        "2024-06-01,2024-01-01T00:00:00,2024-01-01T00:00:00+0000,12:00:00,x\n",
        encoding="utf-8",
    )
    every = [arg for name in switches for arg in ("--skip", name)]
    assert run_cli("validate", source, data, *every) == 0
    with pytest.raises(SystemExit):
        run_cli("validate", source, data, "--skip", "nope")
    assert "invalid choice: 'nope'" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --output and --failing: the file split into what passed and what did not
# ---------------------------------------------------------------------------

SPLIT_SPEC = """
import polars as pl
from polspec import ColSpec, FrameSpec

class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, unique=True)
    status = ColSpec(pl.Enum(["NEW", "PAID"]))
    total = ColSpec(pl.Float64, bounds=(0, 100))
"""


def _split_files(tmp_path, rows: str):
    source = tmp_path / "orders.py"
    source.write_text(SPLIT_SPEC, encoding="utf-8")
    data = tmp_path / "orders.csv"
    data.write_text("order_id,status,total\n" + rows, encoding="utf-8")
    return source, data


def test_output_and_failing_split_the_file(tmp_path, capsys):
    source, data = _split_files(
        tmp_path, "1,NEW,10\n2,LOST,20\n3,PAID,500\n4,PAID,30\n"
    )
    clean, bad = tmp_path / "clean.parquet", tmp_path / "bad.csv"
    code = run_cli("validate", source, data, "--output", clean, "--failing", bad)
    assert code == 1, "the exit status still says the file failed"
    passing = pl.read_parquet(clean)
    assert passing["order_id"].to_list() == [1, 4]
    assert passing.schema["status"] == pl.Enum(["NEW", "PAID"]), "typed as declared"
    failing = pl.read_csv(bad)
    assert sorted(failing["order_id"].to_list()) == [2, 3]
    assert "__polspec_finding" in failing.columns
    err = capsys.readouterr().err
    assert "2 passing row(s)" in err and "2 failing row(s)" in err


def test_failing_holds_each_row_once_so_the_two_files_split_the_input(tmp_path):
    """Before 0.15.1 `--failing` wrote a row once per claim it broke: the
    two files held more rows than the input, and a count of rejected rows
    overcounted."""
    source, data = _split_files(
        tmp_path, "1,NEW,10\n2,LOST,500\n3,PAID,500\n4,PAID,30\n1,LOST,20\n"
    )
    clean, bad = tmp_path / "clean.csv", tmp_path / "bad.csv"
    assert run_cli("validate", source, data, "--output", clean, "--failing", bad) == 1
    passing, failing = pl.read_csv(clean), pl.read_csv(bad)
    assert passing.height + failing.height == 5
    # In the file's order, each row once, every claim it breaks named.
    assert failing["order_id"].to_list() == [1, 2, 3, 1]
    findings = [set(f.split(",")) for f in failing["__polspec_finding"]]
    assert findings[0] == {"order_id__unique"}
    assert findings[1] == {"status__choices", "total__bounds"}
    assert findings[2] == {"total__bounds"}
    assert findings[3] == {"order_id__unique", "status__choices"}


def test_failing_refuses_a_column_named_like_its_own_before_writing(tmp_path, capsys):
    """`--failing` names each row's broken claims in `__polspec_finding`; a
    file with a column of that name is refused before either file is
    written, rather than losing the column."""
    source = tmp_path / "d.py"
    source.write_text(
        "import polars as pl\n"
        "from polspec import ColSpec, FrameSpec\n\n"
        "class D(FrameSpec):\n"
        "    __columns__ = {'__polspec_finding': ColSpec(pl.Int64),"
        " 'x': ColSpec(pl.Int64, bounds=(0, 5))}\n",
        encoding="utf-8",
    )
    data = tmp_path / "d.csv"
    data.write_text("__polspec_finding,x\n1,3\n2,9\n", encoding="utf-8")
    clean, bad = tmp_path / "ok.csv", tmp_path / "bad.csv"
    assert run_cli("validate", source, data, "--output", clean, "--failing", bad) != 0
    assert "column named '__polspec_finding'" in capsys.readouterr().err
    assert not clean.exists() and not bad.exists()


def test_json_stays_one_document_when_files_are_written(tmp_path, capsys):
    source, data = _split_files(tmp_path, "1,NEW,10\n2,LOST,20\n")
    run_cli("validate", source, data, "--json", "--output", tmp_path / "c.parquet")
    out = capsys.readouterr().out
    assert json.loads(out)["passed"] is False


def test_a_passing_file_writes_all_of_it_and_no_failures(tmp_path):
    source, data = _split_files(tmp_path, "1,NEW,10\n2,PAID,20\n")
    clean, bad = tmp_path / "clean.parquet", tmp_path / "bad.parquet"
    assert run_cli("validate", source, data, "--output", clean, "--failing", bad) == 0
    assert pl.read_parquet(clean).height == 2
    assert pl.read_parquet(bad).height == 0


def test_a_structural_finding_writes_neither_file(tmp_path, capsys):
    source, data = _split_files(tmp_path, "1,NEW,10\n")
    data.write_text("order_id,status\n1,NEW\n", encoding="utf-8")  # no total
    clean = tmp_path / "clean.parquet"
    assert run_cli("validate", source, data, "--output", clean) == 1
    assert not clean.exists()
    assert "judge the whole frame" in capsys.readouterr().err


def test_output_is_refused_before_reading_what_it_cannot_write(tmp_path, capsys):
    source, data = _split_files(tmp_path, "1,NEW,10\n")
    assert run_cli("validate", source, data, "--output", tmp_path / "x.xlsx") == 1
    assert "don't know how to write '.xlsx'" in capsys.readouterr().err
    assert (
        run_cli(
            "validate", "--all", tmp_path, tmp_path, "--failing", tmp_path / "b.csv"
        )
        == 1
    )
    assert "do not combine with --all" in capsys.readouterr().err


def test_validate_reports_a_column_a_check_needs_rather_than_a_traceback(
    tmp_path, capsys
):
    """The check cannot run without its column; it used to take the command
    down with Polars' `ColumnNotFoundError`."""
    spec = tmp_path / "totals.yaml"
    spec.write_text(
        "version: 4\n"
        "name: Totals\n"
        "columns:\n"
        "  subtotal: {dtype: Float64}\n"
        "  total: {dtype: Float64}\n"
        "checks:\n"
        "- expr: {ge: [{col: total}, {col: subtotal}]}\n"
        "  name: total_covers_subtotal\n",
        encoding="utf-8",
    )
    data = tmp_path / "totals.csv"
    data.write_text("total\n1.0\n", encoding="utf-8")
    assert run_cli("validate", spec, data) == 1
    assert "Missing required columns" in capsys.readouterr().out
