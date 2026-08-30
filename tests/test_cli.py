"""The CLI is thin argument-parsing over FrameSpec methods that already have
their own tests, so these focus on what the CLI adds: reading data files,
templating, and -- the part with real risk -- generating a test file that
actually passes when run.
"""

import subprocess
import sys
import textwrap

import polars as pl
import pytest
from polspec import FrameSpec
from polspec.cli import main


def run_cli(*args: str) -> int:
    return main([str(a) for a in args])


def run_pytest_on(path) -> subprocess.CompletedProcess:
    """Runs pytest on a generated file in a fresh subprocess.

    A subprocess rather than pytest.main(): the generated file imports
    `polspec` itself, and running it in-process would collect it as part of
    this very test session.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(path), "-q"],
        capture_output=True,
        text=True,
        check=False,
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


def test_schema_infer_sample_limits_rows(sample_data, tmp_path):
    out = tmp_path / "orders.yaml"
    run_cli("schema", "infer", sample_data, "-o", out, "--sample", "10")
    spec_cls = FrameSpec.from_yaml(out)
    # order_id bounds should reflect only the first 10 rows (1..10), not 200.
    assert spec_cls._columns["order_id"].bounds.max == 10


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


def test_generated_test_disables_unique_validation_it_cannot_satisfy(tmp_path):
    """C5 in practice: a unique=True column would otherwise fail its own
    generated test, since generate() does not enforce uniqueness.
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
    assert "validate_unique=False" in content
    assert "not yet generated" in content or "not yet" in content

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
# top level
# ---------------------------------------------------------------------------


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli("--version")
    assert exc.value.code == 0
    assert "polspec" in capsys.readouterr().out


def test_no_command_is_an_error():
    with pytest.raises(SystemExit) as exc:
        run_cli()
    assert exc.value.code != 0
