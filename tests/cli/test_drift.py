"""`polspec diff` and `polspec drift`."""

import json

import polars as pl
import pytest
from helpers import run_cli, write_yaml_spec
from polspec import ColSpec, FrameSpec

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
    spec = write_yaml_spec(tmp_path)
    data = tmp_path / "rows.parquet"
    run_cli("generate", spec, "-n", 500, "-o", data, "--seed", 1)
    assert run_cli("drift", spec, data) == 0
    assert "No drift" in capsys.readouterr().out


def test_drift_reports_and_gates_on_moved_data(tmp_path, capsys):
    spec = write_yaml_spec(tmp_path)
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
    spec = write_yaml_spec(tmp_path)
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
    spec = write_yaml_spec(tmp_path)
    data = tmp_path / "rows.parquet"
    run_cli("generate", spec, "-n", 5, "-o", data)
    with pytest.raises(SystemExit):
        run_cli("drift", spec, data, "--json", "--markdown")
