"""`polspec synthesize`, and the profiling `schema infer` does when asked."""

import datetime as dt

import polars as pl
import pytest
from helpers import run_cli
from polspec import FrameSpec

# ---------------------------------------------------------------------------
# synthesize, and schema infer's opt-in profiling
# ---------------------------------------------------------------------------


def _customers_csv(tmp_path, rows: int = 500):
    path = tmp_path / "customers.csv"
    pl.DataFrame(
        {
            "id": list(range(1, rows + 1)),
            "amount": [round(1.0 + (i * 7919 % 1000) / 10, 2) for i in range(rows)],
            "name": [f"person_{i % 30}" for i in range(rows)],
        }
    ).write_csv(path)
    return path


def test_synthesize_writes_a_fake_file_like_the_real_one(tmp_path, capsys):
    source = _customers_csv(tmp_path)
    output = tmp_path / "fake.parquet"
    spec = tmp_path / "fake.yaml"
    code = run_cli(
        "synthesize",
        source,
        "-o",
        output,
        "-n",
        2000,
        "--seed",
        1,
        "--replace",
        "name",
        "--spec",
        spec,
    )
    assert code == 0
    fake = pl.read_parquet(output)
    real = pl.read_csv(source)
    assert fake.height == 2000 and fake.schema == real.schema
    assert fake["id"].is_unique().all()
    assert not set(fake["name"]) & set(real["name"])
    loaded = FrameSpec.from_yaml(spec)
    assert loaded.column("id").unique and loaded.column("name").dtype == pl.String
    assert "synthesized row(s)" in capsys.readouterr().out


def test_synthesize_refuses_a_replaced_column_the_file_lacks(tmp_path, capsys):
    source = _customers_csv(tmp_path)
    code = run_cli("synthesize", source, "-o", tmp_path / "f.csv", "--replace", "nope")
    assert code == 1
    assert "replace= names ['nope']" in capsys.readouterr().err


def test_schema_infer_profiles_shape_keys_and_replaced_columns_when_asked(tmp_path):
    source = _customers_csv(tmp_path)
    plain, full = tmp_path / "plain.yaml", tmp_path / "full.yaml"
    assert run_cli("schema", "infer", source, "-o", plain) == 0
    assert (
        run_cli(
            "schema",
            "infer",
            source,
            "-o",
            full,
            "--shape",
            "--keys",
            "--replace",
            "name",
        )
        == 0
    )
    before, after = FrameSpec.from_yaml(plain), FrameSpec.from_yaml(full)
    assert not before.column("id").unique and after.column("id").unique
    assert isinstance(before.column("name").dtype, pl.Enum)
    assert after.column("name").dtype == pl.String


def test_validate_reports_a_time_of_day_on_a_date_column_rather_than_a_traceback(
    tmp_path, capsys
):
    """Any temporal used to stand in for any other, so the declared bounds
    were compiled against a `Time` and the command ended in Polars'
    `InvalidOperationError`."""
    spec = tmp_path / "shifts.yaml"
    spec.write_text(
        "version: 3\n"
        "name: Shifts\n"
        "columns:\n"
        "  day: {dtype: Date, bounds: [2020-01-01, 2021-01-01]}\n",
        encoding="utf-8",
    )
    data = tmp_path / "shifts.parquet"
    pl.DataFrame({"day": [dt.time(9), dt.time(17)]}).write_parquet(data)
    assert run_cli("validate", spec, data) == 1
    assert "Column 'day': expected dtype Date, got Time" in capsys.readouterr().out


def test_schema_infer_records_a_columns_nans(tmp_path, capsys):
    """0.13.1 warned that the inferred spec would reject the NaN it was
    inferred from; the spec now records the share, and the file validates."""
    data = tmp_path / "readings.csv"
    pl.DataFrame({"level": [1.5, float("nan"), 3.0, 2.0]}).write_csv(data)
    spec = tmp_path / "readings.yaml"
    assert run_cli("schema", "infer", data, "-o", spec) == 0
    assert "warning" not in capsys.readouterr().err
    assert "nan_probability: 0.25" in spec.read_text(encoding="utf-8")
    assert run_cli("validate", spec, data) == 0


def test_synthesize_accepts_a_file_holding_infinities(tmp_path, capsys):
    """It used to end in `error: ColSpec.bounds max must be a finite value`."""
    source = tmp_path / "rates.parquet"
    pl.DataFrame({"rate": [0.5, float("inf"), 1.5, -float("inf")] * 5}).write_parquet(
        source
    )
    fake = tmp_path / "fake.parquet"
    assert run_cli("synthesize", source, "-o", fake, "-n", "50", "--seed", "1") == 0
    assert "holds 10 infinite value(s)" in capsys.readouterr().err
    assert pl.read_parquet(fake)["rate"].is_finite().all()


def test_a_schema_inferred_from_one_sample_of_names_accepts_another(tmp_path):
    """The case that made an Enum need repeats: forty distinct names became
    an Enum of those forty, and the next file of names failed validation."""
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    pl.DataFrame({"name": [f"person_{i}" for i in range(40)]}).write_csv(first)
    pl.DataFrame({"name": [f"person_{i}" for i in range(40, 80)]}).write_csv(second)
    spec = tmp_path / "people.yaml"
    assert run_cli("schema", "infer", first, "-o", spec) == 0
    assert run_cli("validate", spec, second) == 0


def test_schema_infer_names_a_format_and_can_be_told_not_to(tmp_path):
    data = tmp_path / "people.csv"
    pl.DataFrame({"email": [f"user{i}@example.com" for i in range(40)]}).write_csv(data)
    named, plain = tmp_path / "named.yaml", tmp_path / "plain.yaml"
    assert run_cli("schema", "infer", data, "-o", named) == 0
    assert "format: email" in named.read_text(encoding="utf-8")
    assert run_cli("schema", "infer", data, "-o", plain, "--no-formats") == 0
    assert "format:" not in plain.read_text(encoding="utf-8")


def test_synthesize_writes_values_of_the_sources_format(tmp_path):
    source = tmp_path / "hosts.parquet"
    pl.DataFrame(
        {"ip": [f"10.0.{i}.{j}" for i in range(5) for j in range(10)]}
    ).write_parquet(source)
    fake, spec = tmp_path / "fake.parquet", tmp_path / "fake.yaml"
    assert run_cli("synthesize", source, "-o", fake, "--spec", spec, "--seed", "1") == 0
    assert "format: ipv4" in spec.read_text(encoding="utf-8")
    assert run_cli("validate", spec, fake) == 0


@pytest.mark.parametrize("command", ["schema infer", "synthesize"])
def test_no_patterns_writes_no_shape(tmp_path, command):
    source = tmp_path / "orders.csv"
    pl.DataFrame({"ref": [f"ORD-{i % 30:06d}" for i in range(40)]}).write_csv(source)
    out, spec = tmp_path / "out.csv", tmp_path / "spec.yaml"
    for flags, written in (((), True), (("--no-patterns",), False)):
        if command == "synthesize":
            args = ("synthesize", source, "-o", out, "--spec", spec, *flags)
        else:
            args = ("schema", "infer", source, "-o", spec, *flags)
        assert run_cli(*args) == 0
        text = spec.read_text(encoding="utf-8")
        assert ("pattern: ^ORD-[0-9]{6}$" in text) is written
