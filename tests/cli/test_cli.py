"""The CLI is thin argument-parsing over FrameSpec methods that already have
their own tests, so these focus on what the CLI adds: reading data files,
templating, and -- the part with real risk -- generating a test file that
actually passes when run.
"""

import datetime as dt
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import polars as pl
import pytest
from helpers import spec_for
from polspec import ColSpec, FrameSpec
from polspec.cli import main


def run_cli(*args: str) -> int:
    return main([str(a) for a in args])


def run_pytest_on(path) -> subprocess.CompletedProcess:
    """Runs pytest on a generated file in a fresh subprocess.

    A subprocess rather than pytest.main(): the generated file imports
    `polspec` itself, and running it in-process would collect it as part of
    this very test session. Rooted in the file's own directory: left to find
    its own root, pytest scanned directories up into the system temp folder,
    where other programs' files come and go -- and a file vanishing mid-scan
    failed collection, now and then.
    """
    folder = Path(path).parent
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(path),
            "-q",
            "-p",
            "no:cacheprovider",
            "--rootdir",
            str(folder),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=folder,
    )


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


@pytest.fixture
def sample_data(tmp_path):
    df = pl.DataFrame(
        {
            "order_id": list(range(1, 201)),
            "status": ["NEW", "PAID", "SHIPPED"] * 66 + ["NEW", "NEW"],
            "total": [round(10.0 + i * 0.5, 2) for i in range(200)],
        }
    )
    path = tmp_path / "orders.parquet"
    df.write_parquet(path)
    return path


def test_schema_infer_produces_a_generatable_spec(sample_data, tmp_path):
    out = tmp_path / "orders.yaml"
    assert run_cli("schema", "infer", sample_data, "-o", out) == 0
    assert out.exists()

    spec_cls = FrameSpec.from_yaml(out)
    df = spec_cls.generate(50, seed=1)
    spec_cls.validate(df)
    assert set(df.columns) == {"order_id", "status", "total"}


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


# ---------------------------------------------------------------------------
# test -- from YAML
# ---------------------------------------------------------------------------


def test_generated_test_from_yaml_actually_passes(sample_data, tmp_path):
    yaml_path = tmp_path / "orders.yaml"
    run_cli("schema", "infer", sample_data, "-o", yaml_path)

    test_path = tmp_path / "test_orders.py"
    assert run_cli("test", yaml_path, "-o", test_path) == 0

    content = test_path.read_text(encoding="utf-8")
    assert "def test_" in content
    assert "cartesian" in content  # both tests present by default

    result = run_pytest_on(test_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_generated_test_respects_rows_seed_and_no_cartesian(sample_data, tmp_path):
    yaml_path = tmp_path / "orders.yaml"
    run_cli("schema", "infer", sample_data, "-o", yaml_path)

    test_path = tmp_path / "test_orders.py"
    run_cli(
        "test",
        yaml_path,
        "-o",
        test_path,
        "--rows",
        "37",
        "--seed",
        "9",
        "--no-cartesian",
    )
    content = test_path.read_text(encoding="utf-8")
    assert "37" in content
    assert "seed=9" in content
    assert "cartesian" not in content

    result = run_pytest_on(test_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_generated_test_validates_the_uniqueness_it_can_now_satisfy(tmp_path):
    """A unique=True column used to fail its own generated test, so the CLI
    switched the check off. Generation draws without replacement now, so the
    generated test asserts it.
    """
    yaml_path = tmp_path / "spec.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """
            name: Narrow
            columns:
              id:
                dtype: Int8
                unique: true
            """
        ),
        encoding="utf-8",
    )
    test_path = tmp_path / "test_narrow.py"
    run_cli("test", yaml_path, "-o", test_path, "--rows", "200")

    content = test_path.read_text(encoding="utf-8")
    # A unique column is generated distinct now, so the generated test
    # validates it like every other constraint rather than switching it off.
    assert "validate_unique=False" not in content

    result = run_pytest_on(test_path)
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# test -- from a .py source
# ---------------------------------------------------------------------------


def test_generated_test_from_python_module(tmp_path):
    spec_path = tmp_path / "my_spec.py"
    spec_path.write_text(
        textwrap.dedent(
            """
            import polars as pl
            from polspec import ColSpec, FrameSpec

            class Widgets(FrameSpec):
                sku = ColSpec(pl.Int64, bounds=(1, 10_000))
                price = ColSpec(pl.Float64, bounds=(0.0, None))
            """
        ),
        encoding="utf-8",
    )
    test_path = tmp_path / "test_widgets.py"
    assert run_cli("test", spec_path, "-o", test_path) == 0

    result = run_pytest_on(test_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_generated_test_from_python_module_with_multiple_classes(tmp_path):
    spec_path = tmp_path / "multi.py"
    spec_path.write_text(
        textwrap.dedent(
            """
            import polars as pl
            from polspec import ColSpec, FrameSpec

            class A(FrameSpec):
                x = ColSpec(pl.Int64, bounds=(0, 10))

            class B(FrameSpec):
                y = ColSpec(pl.String, string_length=(1, 5))
            """
        ),
        encoding="utf-8",
    )
    test_path = tmp_path / "test_multi.py"
    run_cli("test", spec_path, "-o", test_path)
    content = test_path.read_text(encoding="utf-8")
    assert "def test_a_roundtrip" in content
    assert "def test_b_roundtrip" in content

    result = run_pytest_on(test_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_generated_test_class_filter(tmp_path):
    spec_path = tmp_path / "multi.py"
    spec_path.write_text(
        textwrap.dedent(
            """
            import polars as pl
            from polspec import ColSpec, FrameSpec

            class A(FrameSpec):
                x = ColSpec(pl.Int64, bounds=(0, 10))

            class B(FrameSpec):
                y = ColSpec(pl.Int64, bounds=(0, 10))
            """
        ),
        encoding="utf-8",
    )
    test_path = tmp_path / "test_one.py"
    run_cli("test", spec_path, "-o", test_path, "--class", "B")
    content = test_path.read_text(encoding="utf-8")
    assert "test_b_roundtrip" in content
    assert "test_a_roundtrip" not in content


def test_generated_test_skips_cross_spec_foreign_key(tmp_path):
    spec_path = tmp_path / "fk_spec.py"
    spec_path.write_text(
        textwrap.dedent(
            """
            import polars as pl
            from polspec import ColSpec, ForeignKey, FrameSpec

            class Parent(FrameSpec):
                id = ColSpec(pl.Int64, bounds=(1, 100))

            class Child(FrameSpec):
                parent_id = ColSpec(pl.Int64, bounds=(1, 100))
                __foreign_keys__ = [
                    ForeignKey("parent_id", references=Parent, ref_columns="id")
                ]
            """
        ),
        encoding="utf-8",
    )
    test_path = tmp_path / "test_fk.py"
    run_cli("test", spec_path, "-o", test_path, "--class", "Child")
    content = test_path.read_text(encoding="utf-8")
    assert "pytest.mark.skip" in content
    assert "references=" in content

    result = run_pytest_on(test_path)
    assert result.returncode == 0, result.stdout + result.stderr  # skipped, not failed
    assert "1 skipped" in result.stdout


def test_test_command_missing_file(tmp_path, capsys):
    assert run_cli("test", tmp_path / "nope.yaml", "-o", tmp_path / "t.py") == 1
    assert "no such file" in capsys.readouterr().err


def test_test_command_unknown_class(tmp_path, capsys):
    spec_path = tmp_path / "s.py"
    spec_path.write_text(
        "import polars as pl\nfrom polspec import ColSpec, FrameSpec\n"
        "class A(FrameSpec):\n    x = ColSpec(pl.Int64)\n",
        encoding="utf-8",
    )
    assert run_cli("test", spec_path, "-o", tmp_path / "t.py", "--class", "Nope") == 1
    assert "no FrameSpec class" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def _write_yaml_spec(tmp_path, name="Orders"):
    class Orders(FrameSpec):
        order_id = ColSpec(pl.Int64, bounds=(1, None), unique=True)
        status = ColSpec(pl.Enum(["NEW", "PAID"]))
        total = ColSpec(pl.Float64, bounds=(0.0, 100.0))
        placed = ColSpec(pl.Date, nullable=True, null_probability=0.2)

    path = tmp_path / f"{name.lower()}.yaml"
    Orders.to_yaml(path)
    return path


@pytest.mark.parametrize(
    "suffix",
    [".csv", ".tsv", ".parquet", ".pq", ".ndjson", ".jsonl", ".json", ".arrow"],
)
def test_generate_writes_a_file_the_readers_read_back(tmp_path, suffix, capsys):
    spec = _write_yaml_spec(tmp_path)
    out = tmp_path / "out" / f"rows{suffix}"
    assert run_cli("generate", spec, "-n", 40, "-o", out, "--seed", 1) == 0
    assert f"Wrote 40 row(s) of Orders to {out}" in capsys.readouterr().out
    # What generate wrote, the readers read -- through the same suffix map --
    # and the spec accepts: a text format's dates are read back as declared.
    from polspec.cli._io import _read_data_file

    back = _read_data_file(out, None)
    assert back.height == 40
    assert back.columns == ["order_id", "status", "total", "placed"]
    assert run_cli("validate", spec, out) == 0


@pytest.mark.parametrize("suffix", [".ndjson", ".json"])
def test_generate_to_json_writes_categories_as_text_and_refuses_bytes(
    tmp_path, suffix, capsys
):
    """Before 0.15.1 both panicked inside Polars' JSON writer."""
    tagged = tmp_path / "tagged.py"
    tagged.write_text(
        "import polars as pl\nfrom polspec import ColSpec, FrameSpec\n\n"
        "class Tagged(FrameSpec):\n"
        "    tags = ColSpec(pl.List(pl.Enum(['x', 'y'])), list_length=(1, 1))\n",
        encoding="utf-8",
    )
    out = tmp_path / f"tagged{suffix}"
    assert run_cli("generate", tagged, "-n", 500, "-o", out, "--seed", 1) == 0
    assert run_cli("validate", tagged, out) == 0

    blob = tmp_path / "blob.py"
    blob.write_text(
        "import polars as pl\nfrom polspec import ColSpec, FrameSpec\n\n"
        "class Blob(FrameSpec):\n    data = ColSpec(pl.Binary)\n",
        encoding="utf-8",
    )
    capsys.readouterr()
    assert run_cli("generate", blob, "-n", 5, "-o", tmp_path / f"b{suffix}") != 0
    assert "JSON has no bytes" in capsys.readouterr().err


def test_every_reader_has_a_writer():
    from polspec.cli._io import _DATA_WRITERS
    from polspec.reading import SUFFIXES

    assert set(SUFFIXES) == set(_DATA_WRITERS)


def test_generate_is_reproducible_by_seed(tmp_path):
    spec = _write_yaml_spec(tmp_path)
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    run_cli("generate", spec, "-n", 20, "-o", a, "--seed", 7)
    run_cli("generate", spec, "-n", 20, "-o", b, "--seed", 7)
    assert pl.read_parquet(a).equals(pl.read_parquet(b))


def test_generate_threads_references_into_a_foreign_key(tmp_path):
    source, customers = _write_orders_specs(tmp_path)
    out = tmp_path / "orders.parquet"
    code = run_cli(
        "generate",
        source,
        "--class",
        "Orders",
        "-n",
        30,
        "-o",
        out,
        "--references",
        f"Customers={customers}",
        "--seed",
        1,
    )
    assert code == 0
    assert set(pl.read_parquet(out)["customer_id"].to_list()) <= {1, 2, 3}


def test_generate_cartesian(tmp_path):
    spec = _write_yaml_spec(tmp_path)
    out = tmp_path / "c.parquet"
    assert (
        run_cli(
            "generate", spec, "-n", 1, "-o", out, "--method", "cartesian", "--seed", 1
        )
        == 0
    )
    assert set(pl.read_parquet(out)["status"].to_list()) == {"NEW", "PAID"}


def test_generate_refuses_an_unknown_extension_before_generating(tmp_path, capsys):
    spec = _write_yaml_spec(tmp_path)
    assert run_cli("generate", spec, "-n", 5, "-o", tmp_path / "x.xlsx") == 1
    assert "don't know how to write '.xlsx'" in capsys.readouterr().err
    assert run_cli("generate", spec, "-n", -1, "-o", tmp_path / "x.csv") == 1
    assert "must be non-negative" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# diff and drift
# ---------------------------------------------------------------------------


def _write_two_versions(tmp_path):
    class V1(FrameSpec):
        order_id = ColSpec(pl.Int64, bounds=(1, None), unique=True)
        status = ColSpec(pl.Enum(["NEW", "PAID"]))
        total = ColSpec(pl.Float64, bounds=(0.0, 100.0))

    class V2(FrameSpec):
        order_id = ColSpec(pl.Int64, bounds=(1, None), unique=True)
        status = ColSpec(pl.Enum(["NEW", "PAID", "SHIPPED"]))  # widened
        total = ColSpec(pl.Float64, bounds=(0.0, 50.0))  # narrowed

    old, new = tmp_path / "v1.yaml", tmp_path / "v2.yaml"
    V1.to_yaml(old)
    V2.to_yaml(new)
    return old, new


def test_diff_exits_one_on_a_breaking_change(tmp_path, capsys):
    old, new = _write_two_versions(tmp_path)
    assert run_cli("diff", old, new) == 1
    out = capsys.readouterr().out
    assert "Drift: 1 breaking, 2 compatible" in out
    assert "[breaking] Column 'total': domain narrowed" in out


def test_diff_fail_on_decides_the_exit_status(tmp_path):
    old, new = _write_two_versions(tmp_path)
    assert run_cli("diff", old, new, "--fail-on", "none") == 0
    assert run_cli("diff", old, old) == 0
    assert run_cli("diff", old, old, "--fail-on", "any") == 0

    class Wider(FrameSpec):
        order_id = ColSpec(pl.Int64, bounds=(1, None), unique=True)
        status = ColSpec(pl.Enum(["NEW", "PAID"]))
        total = ColSpec(pl.Float64, bounds=(0.0, 500.0))

    wider = tmp_path / "wider.yaml"
    Wider.to_yaml(wider)
    # A widening alone is compatible: exit 0 by default, 1 only under `any`.
    assert run_cli("diff", old, wider) == 0
    assert run_cli("diff", old, wider, "--fail-on", "any") == 1


def test_diff_json_and_markdown(tmp_path, capsys):
    old, new = _write_two_versions(tmp_path)
    run_cli("diff", old, new, "--json")
    data = json.loads(capsys.readouterr().out)
    assert data["kind"] == "diff" and data["breaking"] == 1
    run_cli("diff", old, new, "--markdown")
    text = capsys.readouterr().out
    assert text.startswith("# Drift: `V1` -> `V2`")
    assert "| `total` | `domain_narrowed` |" in text


def test_diff_rename_is_declared_not_guessed(tmp_path, capsys):
    class A(FrameSpec):
        order_id = ColSpec(pl.Int64)

    class B(FrameSpec):
        order_ref = ColSpec(pl.Int64)

    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    A.to_yaml(a)
    B.to_yaml(b)
    assert run_cli("diff", a, b) == 1  # removed + added
    assert run_cli("diff", a, b, "--rename", "order_id=order_ref") == 0
    assert "renamed to 'order_ref'" in capsys.readouterr().out
    assert run_cli("diff", a, b, "--rename", "order_id") == 1
    assert "--rename expects OLD=NEW" in capsys.readouterr().err


def test_diff_strict_dtypes(tmp_path):
    class A(FrameSpec):
        n = ColSpec(pl.Int32)

    class B(FrameSpec):
        n = ColSpec(pl.Int64)

    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    A.to_yaml(a)
    B.to_yaml(b)
    assert run_cli("diff", a, b) == 0
    assert run_cli("diff", a, b, "--strict-dtypes") == 1


def test_drift_on_generated_data_exits_zero(tmp_path, capsys):
    spec = _write_yaml_spec(tmp_path)
    data = tmp_path / "rows.parquet"
    run_cli("generate", spec, "-n", 500, "-o", data, "--seed", 1)
    assert run_cli("drift", spec, data) == 0
    assert "No drift" in capsys.readouterr().out


def test_drift_reports_and_gates_on_moved_data(tmp_path, capsys):
    spec = _write_yaml_spec(tmp_path)
    data = tmp_path / "rows.parquet"
    df = FrameSpec.from_yaml(spec).generate(500, seed=1)
    df.with_columns(total=pl.col("total") + 1_000).write_parquet(data)
    assert run_cli("drift", spec, data) == 1
    out = capsys.readouterr().out
    assert "[breaking] Column 'total': values escape bounds [0.0, 100.0]" in out
    assert run_cli("drift", spec, data, "--fail-on", "none") == 0
    capsys.readouterr()
    run_cli("drift", spec, data, "--json")
    assert (
        json.loads(capsys.readouterr().out)["findings"][0]["code"] == "bounds_exceeded"
    )


def test_drift_options_reach_the_report(tmp_path, capsys):
    spec = _write_yaml_spec(tmp_path)
    data = tmp_path / "rows.parquet"
    df = FrameSpec.from_yaml(spec).generate(500, seed=1)
    # Every order PAID: NEW is never seen, and the null rate is far off.
    df.with_columns(
        status=pl.lit("PAID").cast(df.schema["status"]),
        placed=pl.lit(None, dtype=pl.Date),
    ).write_parquet(data)
    run_cli("drift", spec, data, "--json")
    codes = {f["code"] for f in json.loads(capsys.readouterr().out)["findings"]}
    assert codes == {"cardinality_moved", "null_rate_moved"}
    run_cli(
        "drift", spec, data, "--json", "--no-unseen", "--null-rate-tolerance", "1.0"
    )
    assert json.loads(capsys.readouterr().out)["unchanged"] is True
    assert run_cli("drift", spec, data, "--sample", "10", "--fail-on", "any") == 1


def test_drift_json_and_markdown_are_exclusive(tmp_path, capsys):
    spec = _write_yaml_spec(tmp_path)
    data = tmp_path / "rows.parquet"
    run_cli("generate", spec, "-n", 5, "-o", data)
    with pytest.raises(SystemExit):
        run_cli("drift", spec, data, "--json", "--markdown")


# ---------------------------------------------------------------------------
# --all: a registry of specs, a directory of data files
# ---------------------------------------------------------------------------


def test_generate_all_writes_one_file_per_spec_parents_first(tmp_path, capsys):
    source, _ = _write_orders_specs(tmp_path)
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
    source, _ = _write_orders_specs(tmp_path)
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
    source, _ = _write_orders_specs(tmp_path)
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
    reports = json.loads(capsys.readouterr().out)
    assert set(reports) == {"Customers", "Orders"}
    assert reports["Customers"]["passed"] and not reports["Orders"]["passed"]


def test_validate_all_says_which_specs_have_no_file(tmp_path, capsys):
    source, _ = _write_orders_specs(tmp_path)
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
    source, _ = _write_orders_specs(tmp_path)
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
    reports = json.loads(capsys.readouterr().out)
    assert set(reports) == {"Customers", "Orders"}
    assert reports["Customers"]["unchanged"] and not reports["Orders"]["unchanged"]


def test_drift_all_says_which_specs_have_no_file(tmp_path, capsys):
    source, _ = _write_orders_specs(tmp_path)
    out = tmp_path / "data"
    out.mkdir()
    assert run_cli("drift", "--all", source, out) == 1
    assert "no data file under" in capsys.readouterr().err
    pl.DataFrame({"id": [1, 2]}).write_parquet(out / "Customers.parquet")
    assert run_cli("drift", "--all", source, out) == 0
    assert "(no data file for: Orders)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _write_orders_specs(tmp_path):
    source = tmp_path / "specs.py"
    source.write_text(
        textwrap.dedent(
            """
            import polars as pl
            from polspec import ColSpec, ForeignKey, FrameSpec

            class Customers(FrameSpec):
                id = ColSpec(pl.Int64, unique=True)

            class Orders(FrameSpec):
                order_id = ColSpec(pl.Int64, unique=True)
                customer_id = ColSpec(pl.Int64)
                total = ColSpec(pl.Float64, bounds=(0.0, 100.0))
                __foreign_keys__ = [
                    ForeignKey("customer_id", references=Customers, ref_columns="id")
                ]
            """
        )
    )
    customers = tmp_path / "customers.parquet"
    pl.DataFrame({"id": [1, 2, 3]}).write_parquet(customers)
    return source, customers


def test_validate_passing_data_exits_zero(tmp_path, capsys):
    source, customers = _write_orders_specs(tmp_path)
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
    source, customers = _write_orders_specs(tmp_path)
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
    source, customers = _write_orders_specs(tmp_path)
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
    source, _ = _write_orders_specs(tmp_path)
    data = tmp_path / "orders.parquet"
    pl.DataFrame({"order_id": [1], "customer_id": [1], "total": [5.0]}).write_parquet(
        data
    )
    assert run_cli("validate", source, data, "--class", "Orders") == 1
    assert "no DataFrame for it was supplied" in capsys.readouterr().out


def test_validate_needs_class_when_the_source_defines_several(tmp_path, capsys):
    source, _ = _write_orders_specs(tmp_path)
    data = tmp_path / "orders.parquet"
    pl.DataFrame({"id": [1]}).write_parquet(data)
    assert run_cli("validate", source, data) == 1
    assert "pick one with --class" in capsys.readouterr().err


def test_validate_rejects_malformed_references(tmp_path, capsys):
    source, _ = _write_orders_specs(tmp_path)
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


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli("--version")
    assert exc.value.code == 0
    assert "polspec" in capsys.readouterr().out


def test_no_command_is_an_error():
    with pytest.raises(SystemExit) as exc:
        run_cli()
    assert exc.value.code != 0


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
    from polspec.validation import SWITCHES

    source = _temporal_spec(tmp_path)
    data = tmp_path / "events.csv"
    data.write_text(
        "day,at,zoned,clock,ref\n"
        "2024-06-01,2024-01-01T00:00:00,2024-01-01T00:00:00+0000,12:00:00,x\n",
        encoding="utf-8",
    )
    every = [arg for name in SWITCHES for arg in ("--skip", name)]
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
        "version: 3\n"
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
    assert loaded.col("id").unique and loaded.col("name").dtype == pl.String
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
    assert not before.col("id").unique and after.col("id").unique
    assert isinstance(before.col("name").dtype, pl.Enum)
    assert after.col("name").dtype == pl.String


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
