"""`generate(fixed=)`: columns pinned to values given, with everything else
generated as it would have been."""

import datetime as dt

import polars as pl
import pytest
from polspec import (
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSpec,
    GenerationError,
    Hierarchy,
    Registry,
    RegistryError,
    TableSpec,
    col,
    generate,
    generate_batches,
    scan,
    sink_parquet,
    validate,
)
from polspec.cli import main

SPEC = TableSpec(
    "Orders",
    {
        "id": ColSpec(pl.Int64, unique=True),
        "country": ColSpec(pl.String, choices=["UK", "FR", "DE"]),
        "currency": ColSpec(
            pl.String,
            choices=["GBP", "EUR"],
            rules=[
                ColRule(when=col("country") == "UK", choices=["GBP"]),
                ColRule(when=col("country").is_in(["FR", "DE"]), choices=["EUR"]),
            ],
        ),
        "qty": ColSpec(pl.Int64, bounds=(1, 9), nullable=True),
        "placed": ColSpec(pl.Date),
    },
)


def test_a_scalar_is_every_row():
    frame = generate(SPEC, 20, seed=1, fixed={"country": "FR"})
    assert frame["country"].to_list() == ["FR"] * 20


def test_a_sequence_or_series_is_row_by_row():
    values = ["UK", "FR", "DE"]
    assert (
        generate(SPEC, 3, seed=1, fixed={"country": values})["country"].to_list()
        == values
    )
    series = pl.Series(values)
    assert (
        generate(SPEC, 3, seed=1, fixed={"country": series})["country"].to_list()
        == values
    )


def test_a_rule_reading_a_pinned_column_follows_it():
    frame = generate(SPEC, 50, seed=1, fixed={"country": "UK"})
    assert set(frame["currency"]) == {"GBP"}
    assert validate(SPEC, frame).equals(frame)


def test_no_other_column_moves():
    plain = generate(SPEC, 50, seed=7)
    pinned = generate(SPEC, 50, seed=7, fixed={"qty": 3, "placed": dt.date(2026, 1, 1)})
    for name in ("id", "country", "currency"):
        assert pinned[name].equals(plain[name])
    assert pinned["qty"].to_list() == [3] * 50


def test_without_fixed_nothing_changes():
    assert generate(SPEC, 30, seed=3, fixed={}).equals(generate(SPEC, 30, seed=3))
    assert generate(SPEC, 30, seed=3, fixed=None).equals(generate(SPEC, 30, seed=3))


def test_a_pin_may_hold_nulls_where_the_column_allows_them():
    frame = generate(SPEC, 3, seed=1, fixed={"qty": [1, None, 9]})
    assert frame["qty"].to_list() == [1, None, 9]


def test_a_framespec_takes_it():
    class Tickets(FrameSpec):
        status = ColSpec(pl.Enum(["open", "closed"]))

    frame = Tickets.generate(4, seed=1, fixed={"status": "closed"})
    assert frame["status"].to_list() == ["closed"] * 4
    assert frame.schema["status"] == pl.Enum(["open", "closed"])


def test_pinned_rows_move_with_the_sort():
    ordered = SPEC.with_order_by("qty")
    frame = generate(
        ordered, 3, seed=1, fixed={"qty": [9, 1, 5], "country": ["DE", "UK", "FR"]}
    )
    assert frame["qty"].to_list() == [1, 5, 9]
    assert frame["country"].to_list() == ["UK", "FR", "DE"]


# ---------------------------------------------------------------------------
# Refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fixed", "says"),
    [
        ({"county": "UK"}, "did you mean 'country'"),
        ({"country": ["UK", "FR"]}, "2 value"),
        ({"country": "ES"}, "choices"),
        ({"qty": 10}, "bounds"),
        ({"id": [1, 1, 2]}, "unique"),
        ({"placed": None}, "nullability"),
        ({"qty": 1.5}, "cannot hold"),
        ({"qty": "3"}, "cannot hold"),
        ({"currency": "GBP"}, "the rules on 'currency'"),
    ],
)
def test_a_pin_generate_cannot_keep_is_refused(fixed, says):
    with pytest.raises(GenerationError, match="fixed=") as exc:
        generate(SPEC, 3, seed=1, fixed=fixed)
    assert says in str(exc.value).lower() or says in str(exc.value)


def test_a_column_a_pass_writes_cannot_be_pinned_but_one_it_reads_can():
    parent = TableSpec("Customers", {"cid": ColSpec(pl.Int64, unique=True)})
    child = TableSpec(
        "Orders",
        {"cid": ColSpec(pl.Int64), "n": ColSpec(pl.Int64)},
        foreign_keys=[ForeignKey("cid", references="Customers")],
    )
    with pytest.raises(GenerationError, match="foreign key"):
        generate(
            child, 3, fixed={"cid": 1}, references={"Customers": generate(parent, 3)}
        )
    tree = TableSpec(
        "Tree",
        {"node": ColSpec(pl.String), "parent": ColSpec(pl.String)},
        hierarchy=Hierarchy(child="node", parent="parent", max_depth=2),
    )
    with pytest.raises(GenerationError, match="the hierarchy"):
        generate(tree, 3, fixed={"node": "a"})
    keyed = TableSpec(
        "T",
        {"a": ColSpec(pl.Int64), "b": ColSpec(pl.Int64), "c": ColSpec(pl.Int64)},
        unique_together=[("a", "b")],
    )
    with pytest.raises(GenerationError, match="composite key"):
        generate(keyed, 3, fixed={"a": 1})
    assert generate(keyed, 3, seed=1, fixed={"c": 1})["c"].to_list() == [1, 1, 1]


def test_cartesian_takes_no_pins():
    with pytest.raises(GenerationError, match="cartesian"):
        generate(SPEC, 3, method="cartesian", fixed={"qty": 1})


@pytest.mark.parametrize("dtype", [pl.List(pl.Int64), pl.Struct({"a": pl.Int64})])
def test_a_nested_column_cannot_be_pinned_yet(dtype):
    spec = TableSpec("T", {"x": ColSpec(dtype)})
    with pytest.raises(GenerationError, match="scalar"):
        generate(spec, 2, fixed={"x": [None, None]})


def test_fixed_must_be_a_mapping():
    with pytest.raises(GenerationError, match="map column names"):
        generate(SPEC, 2, fixed=["country"])  # type: ignore[arg-type]


def test_a_pin_is_held_to_a_pattern():
    spec = TableSpec("T", {"code": ColSpec(pl.String, pattern=r"^[A-Z]{3}$")})
    frame = generate(spec, 2, seed=1, fixed={"code": ["ABC", "XYZ"]})
    assert validate(spec, frame).equals(frame)
    with pytest.raises(GenerationError, match="pattern"):
        generate(spec, 1, fixed={"code": "abc"})


def test_a_nan_is_refused_where_the_column_holds_none():
    spec = TableSpec("T", {"x": ColSpec(pl.Float64)})
    with pytest.raises(GenerationError, match="nan"):
        generate(spec, 1, fixed={"x": float("nan")})
    allowed = TableSpec("T", {"x": ColSpec(pl.Float64, nan_probability=0.1)})
    assert generate(allowed, 1, fixed={"x": float("nan")})["x"].is_nan().all()


# ---------------------------------------------------------------------------
# Every way of generating, since 1.4
# ---------------------------------------------------------------------------

BATCHED = TableSpec(
    "Orders",
    {
        "id": ColSpec(pl.Int64, unique=True),
        "country": ColSpec(pl.String, choices=["UK", "FR", "DE"]),
        "qty": ColSpec(pl.Int64, bounds=(1, 9), nullable=True),
    },
)


@pytest.mark.parametrize(
    "fixed",
    [{"country": "FR"}, {"qty": [1, 2, 3, None] * 2_500, "country": "UK"}],
    ids=["scalar", "sequence"],
)
def test_batches_and_scans_are_windows_onto_the_pinned_frame(fixed):
    whole = generate(BATCHED, 10_000, seed=3, fixed=fixed)
    batches = generate_batches(BATCHED, 10_000, batch_size=3_000, seed=3, fixed=fixed)
    assert pl.concat(batches).equals(whole)
    assert (
        scan(BATCHED, 10_000, seed=3, batch_size=3_000, fixed=fixed)
        .collect()
        .equals(whole)
    )
    projected = scan(BATCHED, 10_000, seed=3, fixed=fixed).select("id").collect()
    assert projected.equals(whole.select("id"))


def test_a_scan_checks_its_pins_before_anything_is_collected():
    with pytest.raises(GenerationError, match="does not declare"):
        scan(BATCHED, 10, fixed={"nope": 1})
    with pytest.raises(GenerationError, match="1 value"):
        generate_batches(BATCHED, 10, fixed={"qty": [1]})
    with pytest.raises(GenerationError, match="cartesian"):
        generate_batches(BATCHED, 10, method="cartesian", fixed={"country": "UK"})


def test_a_sink_writes_the_pinned_frame(tmp_path):
    path = tmp_path / "orders.parquet"
    sink_parquet(
        BATCHED, path, 5_000, seed=4, batch_size=2_000, fixed={"country": "DE"}
    )
    assert pl.read_parquet(path).equals(
        generate(BATCHED, 5_000, seed=4, fixed={"country": "DE"})
    )


def test_a_framespec_takes_it_everywhere(tmp_path):
    Orders = FrameSpec.from_spec(BATCHED, name="Orders")
    assert (
        pl.concat(Orders.generate_batches(50, fixed={"qty": 4}))
        .get_column("qty")
        .eq(4)
        .all()
    )
    assert Orders.scan(50, fixed={"qty": 4}).collect()["qty"].eq(4).all()
    Orders.sink_csv(tmp_path / "o.csv", 50, fixed={"qty": 4})
    assert pl.read_csv(tmp_path / "o.csv")["qty"].eq(4).all()


def test_a_registry_pins_spec_by_spec_and_children_follow():
    class Customers(FrameSpec):
        id = ColSpec(pl.Int64, unique=True, bounds=(1, 1_000))
        tier = ColSpec(pl.String, choices=["free", "pro"])

    class Purchases(FrameSpec):
        customer_id = ColSpec(pl.Int64)
        channel = ColSpec(pl.String, choices=["web", "shop"])
        __foreign_keys__ = [
            ForeignKey("customer_id", references=Customers, ref_columns="id")
        ]

    registry = Registry(Customers, Purchases)
    frames = registry.generate_all(
        {Customers: 50, Purchases: 200},
        seed=1,
        fixed={Customers: {"tier": "pro"}, "Purchases": {"channel": "web"}},
    )
    assert frames["Customers"]["tier"].eq("pro").all()
    assert frames["Purchases"]["channel"].eq("web").all()
    registry.validate_all(frames)
    related = registry.generate_related(
        Purchases, 20, seed=1, fixed={Customers: {"tier": "free"}}
    )
    assert related["Customers"]["tier"].eq("free").all()
    lazy = registry.scan_all(20, seed=1, fixed={Purchases: {"channel": "shop"}})
    assert lazy["Purchases"].collect()["channel"].eq("shop").all()
    with pytest.raises(RegistryError, match="Nope"):
        registry.generate_all(10, fixed={"Nope": {"x": 1}})


def test_the_command_line_pins_columns(tmp_path):
    spec_file = tmp_path / "orders.yaml"
    FrameSpec.from_spec(BATCHED, name="Orders").to_yaml(spec_file)
    out = tmp_path / "orders.csv"
    assert (
        main(
            [
                "generate",
                str(spec_file),
                "-n",
                "20",
                "-o",
                str(out),
                "--seed",
                "1",
                "--fixed",
                "country=UK",
                "--fixed",
                "qty=7",
            ]
        )
        == 0
    )
    written = pl.read_csv(out)
    assert written["country"].eq("UK").all() and written["qty"].eq(7).all()
    assert (
        main(
            ["generate", str(spec_file), "-n", "2", "-o", str(out), "--fixed", "qty=x"]
        )
        == 1
    )
