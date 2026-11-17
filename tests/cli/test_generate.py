"""`polspec generate`: a data file from a spec."""

import polars as pl
import pytest
from helpers import run_cli, write_orders_specs, write_yaml_spec
from polspec import ColSpec

# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


WRITTEN_SUFFIXES = [
    ".csv",
    ".tsv",
    ".parquet",
    ".pq",
    ".ndjson",
    ".jsonl",
    ".json",
    ".arrow",
    ".feather",
    ".ipc",
]


@pytest.mark.parametrize("suffix", WRITTEN_SUFFIXES)
def test_generate_writes_a_file_the_readers_read_back(tmp_path, suffix, capsys):
    spec = write_yaml_spec(tmp_path)
    out = tmp_path / "out" / f"rows{suffix}"
    assert run_cli("generate", spec, "-n", 40, "-o", out, "--seed", 1) == 0
    assert f"Wrote 40 row(s) of Orders to {out}" in capsys.readouterr().out
    # What generate wrote, the readers read -- through the same suffix map --
    # and the spec accepts: a text format's dates are read back as declared.
    from polspec import from_yaml, read

    back = read(from_yaml(spec), out)
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


def test_every_reader_has_a_writer(tmp_path):
    """Every suffix `read` takes is one `generate` writes, tested above."""
    from polspec import TableSpec, read

    with pytest.raises(ValueError, match="Supported: ") as exc:
        read(TableSpec("T", {"a": ColSpec(pl.Int64)}), tmp_path / "rows.nope")
    supported = str(exc.value).split("Supported: ")[1].split(", ")
    assert sorted(supported) == sorted(WRITTEN_SUFFIXES)


def test_generate_is_reproducible_by_seed(tmp_path):
    spec = write_yaml_spec(tmp_path)
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    run_cli("generate", spec, "-n", 20, "-o", a, "--seed", 7)
    run_cli("generate", spec, "-n", 20, "-o", b, "--seed", 7)
    assert pl.read_parquet(a).equals(pl.read_parquet(b))


def test_generate_threads_references_into_a_foreign_key(tmp_path):
    source, customers = write_orders_specs(tmp_path)
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
    spec = write_yaml_spec(tmp_path)
    out = tmp_path / "c.parquet"
    assert (
        run_cli(
            "generate", spec, "-n", 1, "-o", out, "--method", "cartesian", "--seed", 1
        )
        == 0
    )
    assert set(pl.read_parquet(out)["status"].to_list()) == {"NEW", "PAID"}


def test_generate_refuses_an_unknown_extension_before_generating(tmp_path, capsys):
    spec = write_yaml_spec(tmp_path)
    assert run_cli("generate", spec, "-n", 5, "-o", tmp_path / "x.xlsx") == 1
    assert "don't know how to write '.xlsx'" in capsys.readouterr().err
    assert run_cli("generate", spec, "-n", -1, "-o", tmp_path / "x.csv") == 1
    assert "must be non-negative" in capsys.readouterr().err
