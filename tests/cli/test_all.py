"""`--all`: a directory of specs against a directory of data files."""

import json

import polars as pl
from helpers import run_cli, write_orders_specs

# ---------------------------------------------------------------------------
# --all: a registry of specs, a directory of data files
# ---------------------------------------------------------------------------


def test_generate_all_writes_one_file_per_spec_parents_first(tmp_path, capsys):
    source, _ = write_orders_specs(tmp_path)
    out = tmp_path / "out"
    assert run_cli("generate", "--all", source, "-n", 20, "-o", out, "--seed", 1) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Wrote 20 row(s) of Customers to")
    assert lines[1].startswith("Wrote 20 row(s) of Orders to")
    customers = pl.read_parquet(out / "Customers.parquet")
    orders = pl.read_parquet(out / "Orders.parquet")
    # The foreign key is threaded: every order's customer exists.
    assert set(orders["customer_id"].to_list()) <= set(customers["id"].to_list())


def test_generate_all_picks_the_format_and_refuses_a_file_output(tmp_path, capsys):
    source, _ = write_orders_specs(tmp_path)
    out = tmp_path / "csv"
    assert (
        run_cli("generate", "--all", source, "-n", 5, "-o", out, "--format", "csv") == 0
    )
    assert (out / "Orders.csv").exists()
    assert (
        run_cli("generate", "--all", source, "-n", 5, "-o", tmp_path / "x.parquet") == 1
    )
    assert "is a directory" in capsys.readouterr().err
    assert (
        run_cli("generate", "--all", source, "-n", 5, "-o", out, "--format", "xlsx")
        == 1
    )
    assert "don't know how to write '.xlsx'" in capsys.readouterr().err


def test_validate_all_reads_the_files_named_after_the_specs(tmp_path, capsys):
    source, _ = write_orders_specs(tmp_path)
    out = tmp_path / "data"
    run_cli("generate", "--all", source, "-n", 20, "-o", out, "--seed", 1)
    assert run_cli("validate", "--all", source, out) == 0
    text = capsys.readouterr().out
    assert "== Customers" in text and "== Orders" in text

    # Break the foreign key: the child sees the parent file as its reference.
    orders = pl.read_parquet(out / "Orders.parquet")
    orders.with_columns(customer_id=pl.lit(999)).write_parquet(out / "Orders.parquet")
    assert run_cli("validate", "--all", source, out) == 1
    assert "customer_id" in capsys.readouterr().out
    run_cli("validate", "--all", source, out, "--json")
    reports = json.loads(capsys.readouterr().out)["reports"]
    assert set(reports) == {"Customers", "Orders"}
    assert reports["Customers"]["passed"] and not reports["Orders"]["passed"]


def test_validate_all_says_which_specs_have_no_file(tmp_path, capsys):
    source, _ = write_orders_specs(tmp_path)
    out = tmp_path / "data"
    out.mkdir()
    assert run_cli("validate", "--all", source, out) == 1
    assert "no data file under" in capsys.readouterr().err
    pl.DataFrame({"id": [1, 2]}).write_parquet(out / "Customers.parquet")
    assert run_cli("validate", "--all", source, out) == 0
    assert "(no data file for: Orders)" in capsys.readouterr().out
    pl.DataFrame({"id": [1, 2]}).write_csv(out / "Customers.csv")
    assert run_cli("validate", "--all", source, out) == 1
    assert "several data files" in capsys.readouterr().err


def test_drift_all_measures_every_spec_against_the_file_named_after_it(
    tmp_path, capsys
):
    source, _ = write_orders_specs(tmp_path)
    out = tmp_path / "data"
    run_cli("generate", "--all", source, "-n", 200, "-o", out, "--seed", 1)
    assert run_cli("drift", "--all", source, out) == 0
    text = capsys.readouterr().out
    assert "== Customers" in text and "== Orders" in text

    moved = pl.read_parquet(out / "Orders.parquet")
    moved.with_columns(total=pl.col("total") + 1_000).write_parquet(
        out / "Orders.parquet"
    )
    assert run_cli("drift", "--all", source, out) == 1
    assert "values escape bounds" in capsys.readouterr().out
    assert run_cli("drift", "--all", source, out, "--fail-on", "none") == 0
    capsys.readouterr()
    run_cli("drift", "--all", source, out, "--json")
    reports = json.loads(capsys.readouterr().out)["reports"]
    assert set(reports) == {"Customers", "Orders"}
    assert reports["Customers"]["unchanged"] and not reports["Orders"]["unchanged"]


def test_drift_all_says_which_specs_have_no_file(tmp_path, capsys):
    source, _ = write_orders_specs(tmp_path)
    out = tmp_path / "data"
    out.mkdir()
    assert run_cli("drift", "--all", source, out) == 1
    assert "no data file under" in capsys.readouterr().err
    pl.DataFrame({"id": [1, 2]}).write_parquet(out / "Customers.parquet")
    assert run_cli("drift", "--all", source, out) == 0
    assert "(no data file for: Orders)" in capsys.readouterr().out
