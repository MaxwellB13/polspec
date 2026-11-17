"""Several sources at once: a registry inferred, and a related fake set
written, with the keys between them.
"""

import datetime as dt

import polars as pl
import pytest
from helpers import run_cli
from polspec import Registry

# ---------------------------------------------------------------------------
# Several sources: related tables
# ---------------------------------------------------------------------------


def _shop_files(tmp_path):
    customers, orders = tmp_path / "customers.csv", tmp_path / "orders.parquet"
    pl.DataFrame(
        {"id": range(1, 101), "email": [f"c{i}@example.com" for i in range(100)]}
    ).write_csv(customers)
    pl.DataFrame(
        {
            "id": range(1, 301),
            "customer_id": [i % 37 + 1 for i in range(300)],
            "buyer": [i % 50 + 1 for i in range(300)],
        }
    ).write_parquet(orders)
    return customers, orders


def _keys(registry, name):
    return {
        (fk.columns[0], fk.references, fk.ref_columns[0])
        for fk in registry[name].foreign_keys
    }


@pytest.mark.parametrize("suffix", [".yaml", ".py"])
def test_schema_infer_profiles_several_files_into_a_registry(tmp_path, suffix):

    customers, orders = _shop_files(tmp_path)
    output = tmp_path / f"shop{suffix}"
    code = run_cli(
        "schema",
        "infer",
        customers,
        orders,
        "-o",
        output,
        "--foreign-key",
        "Orders.buyer=Customers.id",
    )
    assert code == 0
    if suffix == ".py":
        assert "registry = Registry(Customers, Orders)" in output.read_text(
            encoding="utf-8"
        )
        data = tmp_path / "data"
        data.mkdir()
        pl.read_csv(customers).write_csv(data / "Customers.csv")
        pl.read_parquet(orders).write_parquet(data / "Orders.parquet")
        assert run_cli("validate", "--all", output, data) == 0
    else:
        registry = Registry.from_yaml(output)
        assert _keys(registry, "Orders") == {
            ("customer_id", "Customers", "id"),
            ("buyer", "Customers", "id"),
        }


def test_all_finds_a_file_by_the_spec_name_its_stem_gives(tmp_path):
    """`schema infer` names `order_lines.csv`'s spec `OrderLines`, and `--all`
    finds it by that name, whatever the file system's case rules."""
    lines, items = tmp_path / "order_lines.csv", tmp_path / "items.csv"
    pl.DataFrame({"line": range(1, 41)}).write_csv(lines)
    pl.DataFrame({"x": range(40)}).write_csv(items)
    spec = tmp_path / "shop.yaml"
    assert run_cli("schema", "infer", lines, items, "-o", spec) == 0
    assert run_cli("validate", "--all", spec, tmp_path) == 0
    assert run_cli("drift", "--all", spec, tmp_path) == 0
    pl.DataFrame({"line": range(1, 41)}).write_parquet(tmp_path / "OrderLines.parquet")
    assert run_cli("validate", "--all", spec, tmp_path) == 1  # two files for one spec


def test_schema_infer_prints_a_registry(tmp_path, capsys):
    customers, orders = _shop_files(tmp_path)
    assert run_cli("schema", "infer", customers, orders, "-o", "-") == 0
    out = capsys.readouterr()
    assert out.out.startswith("version:") and "specs:" in out.out
    assert "1 foreign key(s)" in out.err


def test_synthesize_writes_a_related_fake_set(tmp_path, capsys):
    customers, orders = _shop_files(tmp_path)
    out_dir, spec = tmp_path / "fake", tmp_path / "fake.yaml"
    code = run_cli(
        "synthesize",
        customers,
        orders,
        "-o",
        out_dir,
        "--scale",
        "0.5",
        "--seed",
        "1",
        "--spec",
        spec,
    )
    assert code == 0
    fake_customers = pl.read_csv(out_dir / "customers.csv")
    fake_orders = pl.read_parquet(out_dir / "orders.parquet")
    assert (fake_customers.height, fake_orders.height) == (50, 150)
    assert fake_orders["customer_id"].is_in(fake_customers["id"].implode()).all()
    assert "Kept 1 foreign key(s)" in capsys.readouterr().out
    data = tmp_path / "data"
    data.mkdir()
    fake_customers.write_csv(data / "Customers.csv")
    fake_orders.write_parquet(data / "Orders.parquet")
    assert run_cli("validate", "--all", spec, data) == 0


def test_synthesize_writes_a_set_in_one_format(tmp_path):
    customers, orders = _shop_files(tmp_path)
    out_dir = tmp_path / "fake"
    code = run_cli(
        "synthesize",
        customers,
        orders,
        "-o",
        out_dir,
        "--format",
        "csv",
        "--no-foreign-keys",
    )
    assert code == 0
    assert sorted(p.name for p in out_dir.iterdir()) == ["customers.csv", "orders.csv"]


def test_synthesize_scales_one_file(tmp_path):
    customers, _ = _shop_files(tmp_path)
    output = tmp_path / "fake.csv"
    assert run_cli("synthesize", customers, "-o", output, "--scale", "0.2") == 0
    assert pl.read_csv(output).height == 20


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            ["schema", "infer", "{c}", "-o", "-", "--foreign-key", "A.b=C.d"],
            "two or more sources",
        ),
        (
            ["schema", "infer", "{c}", "{o}", "-o", "-", "--name", "X"],
            "--name names one spec",
        ),
        (
            [
                "schema",
                "infer",
                "{c}",
                "{o}",
                "-o",
                "-",
                "--foreign-key",
                "Orders.buyer",
            ],
            "SPEC.COLUMN=SPEC.COLUMN",
        ),
        (
            [
                "schema",
                "infer",
                "{c}",
                "{o}",
                "-o",
                "-",
                "--foreign-key",
                "A.b=C.d",
                "--no-foreign-keys",
            ],
            "contradict",
        ),
        (
            [
                "schema",
                "infer",
                "{c}",
                "{o}",
                "-o",
                "-",
                "--foreign-key",
                "Nope.b=Customers.id",
            ],
            "not one of the tables",
        ),
        (["synthesize", "{c}", "{o}", "-o", "{t}/x.csv"], "names a directory"),
        (
            ["synthesize", "{c}", "{o}", "-o", "{t}/d", "--format", "xls"],
            "--format 'xls'",
        ),
        (
            ["synthesize", "{c}", "-o", "{t}/x.csv", "--format", "csv"],
            "--format is for two",
        ),
        (
            ["synthesize", "{c}", "-o", "{t}/x.csv", "-n", "5", "--scale", "1"],
            "not both",
        ),
        (["synthesize", "{c}", "-o", "{t}/x.csv", "--scale", "0"], "positive"),
        (["synthesize", "{c}", "{c}", "-o", "{t}/d"], "would both be the spec"),
    ],
)
def test_several_sources_refuse_what_they_cannot_do(tmp_path, capsys, args, message):
    customers, orders = _shop_files(tmp_path)
    filled = [
        a.format(c=customers, o=orders, t=tmp_path) if isinstance(a, str) else a
        for a in args
    ]
    assert run_cli(*filled) == 1
    assert message in capsys.readouterr().err


def test_schema_infer_says_why_a_column_stays_text(tmp_path, capsys):
    source = tmp_path / "bonds.csv"
    pl.DataFrame(
        {
            "NOMINAL": [f"{i},{i:03d}.50" for i in range(1, 60)],
            "maturity": ["30 Sep 2031", "31 Dec 2049", "05 Mar 2027"] * 19
            + ["01 Jan 2030"] * 2,
        }
    ).write_csv(source)
    spec = tmp_path / "bonds.yaml"
    assert run_cli("schema", "infer", source, "-o", spec, "--parse-text") == 0
    err = capsys.readouterr().err
    assert "column 'NOMINAL' stays text: 59 value(s) do not parse as a number" in err
    written = spec.read_text(encoding="utf-8")
    assert "text_format: '%d %b %Y'" in written


def test_synthesize_writes_dates_as_the_source_wrote_them(tmp_path):
    source = tmp_path / "bonds.csv"
    days = [dt.date(2030, 1, 1) + dt.timedelta(days=i) for i in range(200)]
    pl.DataFrame({"maturity": [d.strftime("%d %b %Y") for d in days]}).write_csv(source)
    fake = tmp_path / "fake.csv"
    assert run_cli("synthesize", source, "-o", fake, "--parse-text", "--seed", "1") == 0
    text = pl.read_csv(fake, infer_schema=False)["maturity"]
    assert text.str.to_date("%d %b %Y", strict=True).null_count() == 0
