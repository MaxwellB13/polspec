"""`polspec test`: a test file written from a spec, which has to pass when
it is run -- the part of the command line with real risk.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

from helpers import run_cli


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
    # The key `schema infer` profiled from 200 rows holds 200 values (2.0).
    assert "200 rows, not 500: 'order_id' is unique" in content

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
    assert "unique=False" not in content

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


def test_generated_test_generates_the_parents_a_spec_draws_from(tmp_path):
    """A spec whose foreign keys reach specs loaded with it -- here two levels
    up -- generates them first, so its test runs rather than being skipped."""
    spec_path = tmp_path / "chain.py"
    spec_path.write_text(
        textwrap.dedent(
            """
            import polars as pl
            from polspec import ColSpec, ForeignKey, FrameSpec

            class Customers(FrameSpec):
                id = ColSpec(pl.Int64, bounds=(1, 10_000), unique=True)

            class Orders(FrameSpec):
                order_id = ColSpec(pl.Int64, bounds=(1, 1_000_000), unique=True)
                customer_id = ColSpec(pl.Int64, bounds=(1, 10_000))
                __foreign_keys__ = [
                    ForeignKey("customer_id", references=Customers, ref_columns="id")
                ]

            class Lines(FrameSpec):
                order_id = ColSpec(pl.Int64, bounds=(1, 1_000_000))
                __foreign_keys__ = [
                    ForeignKey("order_id", references=Orders, ref_columns="order_id")
                ]
            """
        ),
        encoding="utf-8",
    )
    test_path = tmp_path / "test_chain.py"
    assert run_cli("test", spec_path, "-o", test_path) == 0
    content = test_path.read_text(encoding="utf-8")
    assert "pytest.mark.skip" not in content
    assert (
        "parents[Orders] = polspec_frames(Orders, 500, references=parents)" in content
    )
    result = run_pytest_on(test_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "failed" not in result.stdout and "skipped" not in result.stdout


def test_generated_test_validates_a_pattern_it_generates(tmp_path):
    """A pattern is generated from since 1.4, so the test checks it; only a
    pattern out of the generator's reach is switched off, saying why."""
    spec_path = tmp_path / "coded.py"
    spec_path.write_text(
        textwrap.dedent(
            r"""
            import polars as pl
            from polspec import ColSpec, FrameSpec

            class Coded(FrameSpec):
                ref = ColSpec(pl.String, pattern=r"^ORD-\d{6}$", unique=True)
                post = ColSpec(pl.String, pattern=r"^[A-Z]{2}\d [0-9][A-Z]{2}$")

            class Bounded(FrameSpec):
                word = ColSpec(pl.String, pattern=r"\bword\b")
            """
        ),
        encoding="utf-8",
    )
    test_path = tmp_path / "test_coded.py"
    assert run_cli("test", spec_path, "-o", test_path) == 0
    content = test_path.read_text(encoding="utf-8")
    coded, bounded = content.split("def test_bounded")
    assert "pattern=False" not in coded
    assert "pattern=False" in bounded
    assert "the pattern of 'word' is not generated from" in bounded
    result = run_pytest_on(test_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "failed" not in result.stdout


def test_generated_test_draws_from_the_fixture_and_reruns_by_its_seed(
    sample_data, tmp_path
):
    """Without --seed the frames are the fixture's: a failure names
    --polspec-seed, and that is what reproduces it."""
    yaml_path = tmp_path / "orders.yaml"
    run_cli("schema", "infer", sample_data, "-o", yaml_path)
    test_path = tmp_path / "test_orders.py"
    run_cli("test", yaml_path, "-o", test_path, "--no-cartesian")
    content = test_path.read_text(encoding="utf-8")
    assert "polspec_frames(" in content
    assert "assert_valid(" in content
    assert "seed=" not in content


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
