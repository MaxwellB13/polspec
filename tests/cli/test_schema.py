"""`polspec schema new` and `polspec schema infer`."""

import polars as pl
from helpers import run_cli, spec_for
from polspec import ColSpec, FrameSpec

# ---------------------------------------------------------------------------
# schema new
# ---------------------------------------------------------------------------


def test_schema_new_writes_a_loadable_spec(tmp_path):
    out = tmp_path / "orders.py"
    assert run_cli("schema", "new", "Orders", "-o", out) == 0

    namespace: dict = {}
    exec(compile(out.read_text(encoding="utf-8"), str(out), "exec"), namespace)
    assert issubclass(namespace["Orders"], FrameSpec)


def test_schema_new_rejects_invalid_identifier(tmp_path, capsys):
    assert run_cli("schema", "new", "not valid", "-o", tmp_path / "x.py") == 1
    assert "error:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# schema infer
# ---------------------------------------------------------------------------


def test_schema_infer_produces_a_generatable_spec(sample_data, tmp_path):
    out = tmp_path / "orders.yaml"
    assert run_cli("schema", "infer", sample_data, "-o", out) == 0
    assert out.exists()

    spec_cls = FrameSpec.from_yaml(out)
    df = spec_cls.generate(50, seed=1)
    spec_cls.validate(df)
    assert set(df.columns) == {"order_id", "status", "total"}


def test_schema_infer_to_a_dash_prints_the_yaml_and_only_it(
    sample_data, tmp_path, capsys
):
    out = tmp_path / "orders.yaml"
    run_cli("schema", "infer", sample_data, "-o", out)
    written = out.read_text(encoding="utf-8")
    capsys.readouterr()

    assert run_cli("schema", "infer", sample_data, "-o", "-") == 0
    printed = capsys.readouterr()
    assert printed.out == written
    assert "Inferred 3 column(s)" in printed.err
    assert not (tmp_path / "-").exists()


def test_schema_infer_custom_name(sample_data, tmp_path):
    out = tmp_path / "orders.yaml"
    run_cli("schema", "infer", sample_data, "-o", out, "--name", "MyOrders")
    assert "name: MyOrders" in out.read_text(encoding="utf-8")


def test_schema_infer_missing_file(tmp_path, capsys):
    assert (
        run_cli("schema", "infer", tmp_path / "nope.csv", "-o", tmp_path / "x.yaml")
        == 1
    )
    assert "no such file" in capsys.readouterr().err


def test_schema_infer_unsupported_extension(tmp_path, capsys):
    bad = tmp_path / "data.xlsx"
    bad.write_text("not real data")
    assert run_cli("schema", "infer", bad, "-o", tmp_path / "x.yaml") == 1
    assert "don't know how to read" in capsys.readouterr().err


def test_schema_infer_samples_rows_from_the_whole_file(sample_data, tmp_path):
    """`--sample 10` used to read the first ten rows: order_id 1..10 of a
    file holding 1..200, so the spec rejected the rest. A sample is drawn
    from the whole file now -- seeded, so the spec is the same every run."""
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    for out in (first, second):
        assert run_cli("schema", "infer", sample_data, "-o", out, "--sample", "10") == 0
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    bounds = FrameSpec.from_yaml(first).spec.columns["order_id"].bounds
    assert bounds.max - bounds.min > 50  # ten rows from 1..200, not 1..10


def test_drift_samples_rows_from_the_whole_file(tmp_path):
    """A file sorted by amount: its head is a moved distribution, the whole
    file is not."""
    spec_cls = spec_for(
        ColSpec(
            pl.Float64,
            bounds=(0, 100),
            distribution="normal",
            distribution_params={"mean": 50, "std": 10},
        )
    )
    data = tmp_path / "sorted.parquet"
    spec_cls.generate(20_000, seed=1).sort("c").write_parquet(data)
    spec = tmp_path / "spec.yaml"
    spec_cls.to_yaml(spec)
    assert run_cli("drift", spec, data, "--sample", "2000", "--fail-on", "any") == 0


def test_a_sample_must_be_a_positive_row_count(sample_data, tmp_path, capsys):
    assert (
        run_cli(
            "schema", "infer", sample_data, "-o", tmp_path / "x.yaml", "--sample", "0"
        )
        == 1
    )
    assert "sample must be a positive row count" in capsys.readouterr().err


def test_schema_infer_py_output_produces_a_generatable_spec(sample_data, tmp_path):
    out = tmp_path / "orders.py"
    assert run_cli("schema", "infer", sample_data, "-o", out) == 0
    assert out.exists()

    namespace: dict = {}
    exec(compile(out.read_text(encoding="utf-8"), str(out), "exec"), namespace)
    spec_cls = namespace["Orders"]
    df = spec_cls.generate(50, seed=1)
    spec_cls.validate(df)
    assert set(df.columns) == {"order_id", "status", "total"}


def test_schema_infer_py_output_is_ruff_formatted(sample_data, tmp_path):
    out = tmp_path / "orders.py"
    run_cli("schema", "infer", sample_data, "-o", out)
    text = out.read_text(encoding="utf-8")
    assert "class Orders(FrameSpec):" in text
    assert "__columns__ = {" in text
