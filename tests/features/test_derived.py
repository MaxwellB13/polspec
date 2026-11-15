"""`derive=`: a column computed from others, since 1.4.

Generation computes it after every column it reads; validation holds the
column to it, exactly or for a float within a relative 1e-9, as a
`derived` finding; spec files hold it; `diff` reports it changing.
"""

from __future__ import annotations

import polars as pl
import pytest
from polspec import (
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSpec,
    GenerationError,
    SpecError,
    TableSpec,
    col,
    diff,
    generate,
    generate_batches,
    inspect,
    validate,
)


class Orders(FrameSpec):
    qty = ColSpec(pl.Int64, bounds=(1, 20))
    price = ColSpec(pl.Float64, bounds=(0.5, 100), nullable=True)
    total = ColSpec(pl.Float64, derive=col("qty") * col("price"), nullable=True)
    large = ColSpec(pl.Boolean, derive=col("total") > 500, nullable=True)


def _expected(frame: pl.DataFrame) -> pl.Series:
    return frame.select(pl.col("qty") * pl.col("price"))["qty"]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_a_derived_column_is_its_expression():
    frame = Orders.generate(2_000, seed=1)
    assert frame["total"].equals(_expected(frame).alias("total"))
    assert frame["large"].equals((frame["total"] > 500).alias("large"))
    assert frame["total"].is_null().equals(frame["price"].is_null())
    Orders.validate(frame)


def test_a_derived_column_reads_fixed_values_and_rules():
    class Shipping(FrameSpec):
        region = ColSpec(pl.Enum(["UK", "US"]))
        rate = ColSpec(
            pl.Int64,
            bounds=(1, 9),
            rules=[ColRule(when=col("region") == "UK", choices=[5])],
        )
        doubled = ColSpec(pl.Int64, derive=col("rate") * 2)

    frame = Shipping.generate(500, seed=2)
    assert frame.filter(pl.col("region") == "UK")["doubled"].eq(10).all()
    pinned = Shipping.generate(50, seed=2, fixed={"region": "UK"})
    assert pinned["doubled"].eq(10).all()
    Shipping.validate(frame)


def test_a_derived_column_reads_a_foreign_key():
    class Products(FrameSpec):
        sku = ColSpec(pl.Int64, bounds=(1, 1_000), unique=True)

    class Lines(FrameSpec):
        sku = ColSpec(pl.Int64)
        sku_plus = ColSpec(pl.Int64, derive=col("sku") + 1)
        __foreign_keys__ = [ForeignKey("sku", references=Products, ref_columns="sku")]

    parents = {Products: Products.generate(100, seed=1)}
    lines = Lines.generate(300, seed=3, references=parents)
    assert (lines["sku_plus"] == lines["sku"] + 1).all()
    Lines.validate(lines, references=parents)


def test_batches_and_cartesian_compute_it_too():
    batches = pl.concat(generate_batches(Orders, 3_000, batch_size=1_000, seed=4))
    assert batches["total"].equals(_expected(batches).alias("total"))

    class Coverage(FrameSpec):
        flag = ColSpec(pl.Boolean)
        inverse = ColSpec(pl.Boolean, derive=~col("flag"))

    covered = Coverage.generate(4, method="cartesian", seed=1)
    assert covered["inverse"].equals((~covered["flag"]).alias("inverse"))


def test_fixed_cannot_pin_a_derived_column():
    with pytest.raises(GenerationError, match="the derivation of 'total' writes it"):
        Orders.generate(10, seed=1, fixed={"total": 1.0})


def test_a_value_its_dtype_cannot_hold_is_refused_at_generation():
    spec = TableSpec(
        "T",
        {
            "big": ColSpec(pl.Int64, bounds=(200, 300)),
            "small": ColSpec(pl.Int8, derive=col("big") * 1),
        },
    )
    with pytest.raises(GenerationError, match="cannot be derived as Int8"):
        generate(spec, 10, seed=1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_a_wrong_value_is_a_derived_finding():
    frame = Orders.generate(1_000, seed=5)
    broken = frame.with_columns(
        pl.when(pl.col("qty") == 1).then(0.0).otherwise(pl.col("total")).alias("total")
    )
    report = inspect(Orders, broken)
    finding = report.by_code("derived")[0]
    assert finding.key == "total__derived"
    assert finding.columns == ("total", "price", "qty")
    # Every row with qty 1 is wrong: 0.0 where its total is a product, or a
    # null one -- a null price makes the expression null.
    wrong = broken.filter(pl.col("qty") == 1).height
    assert finding.count == wrong
    assert "col('qty') * col('price')" in finding.message


def test_a_float_within_a_relative_billionth_passes():
    frame = Orders.generate(1_000, seed=6)
    nudged = frame.with_columns(pl.col("total") * (1 + 1e-12))
    assert inspect(Orders, nudged).passed
    moved = frame.with_columns(pl.col("total") * (1 + 1e-6))
    assert not inspect(Orders, moved).passed


def test_an_integer_must_be_exact_and_a_null_must_be_where_the_expression_is():
    spec = TableSpec(
        "T",
        {
            "a": ColSpec(pl.Int64, nullable=True),
            "b": ColSpec(pl.Int64, derive=col("a") + 1, nullable=True),
        },
    )
    good = pl.DataFrame({"a": [1, None, 3], "b": [2, None, 4]})
    validate(spec, good)
    for b in ([2, 0, 4], [2, None, 5], [None, None, 4]):
        report = inspect(spec, good.with_columns(pl.Series("b", b)))
        assert [f.code for f in report] == ["derived"], b


def test_nan_is_right_where_the_expression_is_nan():
    spec = TableSpec(
        "T",
        {
            "a": ColSpec(pl.Float64, nan_probability=0.1),
            "b": ColSpec(pl.Float64, derive=col("a") * 2),
        },
    )
    nan = float("nan")
    validate(spec, pl.DataFrame({"a": [1.0, nan], "b": [2.0, nan]}))
    assert not inspect(spec, pl.DataFrame({"a": [1.0, nan], "b": [2.0, 0.0]})).passed
    assert not inspect(spec, pl.DataFrame({"a": [1.0, 2.0], "b": [2.0, nan]})).passed


def test_infinity_matches_only_itself():
    spec = TableSpec(
        "T",
        {"a": ColSpec(pl.Float64), "b": ColSpec(pl.Float64, derive=col("a") * 2)},
    )
    inf = float("inf")
    validate(spec, pl.DataFrame({"a": [inf, -inf], "b": [inf, -inf]}))
    assert not inspect(spec, pl.DataFrame({"a": [inf], "b": [5.0]})).passed
    assert not inspect(spec, pl.DataFrame({"a": [5.0], "b": [inf]})).passed
    assert not inspect(spec, pl.DataFrame({"a": [inf], "b": [-inf]})).passed


def test_the_check_can_be_switched_off_and_skips_a_missing_input():
    frame = Orders.generate(100, seed=7).with_columns(pl.lit(-1.0).alias("total"))
    assert not inspect(Orders, frame).passed
    assert [f.code for f in inspect(Orders, frame, derived=False)] == []
    missing = Orders.generate(100, seed=7).drop("qty")
    assert "derived" not in {f.code for f in inspect(Orders, missing)}


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("claims", "named"),
    [
        ({"bounds": (0, 10)}, "bounds"),
        ({"choices": [1, 2]}, "choices"),
        ({"unique": True}, "unique"),
        ({"default": 1}, "default"),
        ({"distribution": "normal"}, "distribution"),
        ({"rules": [ColRule(when=col("a") > 1, choices=[1])]}, "rules"),
    ],
)
def test_a_derived_column_declares_no_values_of_its_own(claims, named):
    with pytest.raises(SpecError, match=f"cannot also declare {named}"):
        ColSpec(pl.Int64, derive=col("a") + 1, **claims)


def test_derive_takes_a_col_expression_reading_a_column():
    with pytest.raises(SpecError, match=r"not a pl.Expr"):
        ColSpec(pl.Int64, derive=pl.col("a") + 1)
    with pytest.raises(SpecError, match="must be a col"):
        ColSpec(pl.Int64, derive="a + 1")
    with pytest.raises(SpecError, match="scalar column"):
        ColSpec(pl.List(pl.Int64), derive=col("a"))


@pytest.mark.parametrize(
    ("columns", "says"),
    [
        ({"b": ColSpec(pl.Int64, derive=col("nope") + 1)}, "unknown column 'nope'"),
        ({"b": ColSpec(pl.Int64, derive=col("b") + 1)}, "derived from itself"),
        (
            {
                "a": ColSpec(pl.Int64, nullable=True),
                "b": ColSpec(pl.Int64, derive=col("a") + 1),
            },
            "declare it nullable=True",
        ),
        (
            {
                "a": ColSpec(pl.Int64, derive=col("b") + 1),
                "b": ColSpec(pl.Int64, derive=col("a") + 1),
            },
            "Cannot order the generation passes",
        ),
    ],
)
def test_the_table_checks_what_a_derived_column_reads(columns, says):
    with pytest.raises(SpecError, match=says):
        TableSpec("T", {"a": ColSpec(pl.Int64), **columns})


def test_a_derived_column_is_written_by_nothing_else():
    with pytest.raises(SpecError, match="composite unique key"):
        TableSpec(
            "T",
            {"a": ColSpec(pl.Int64), "b": ColSpec(pl.Int64, derive=col("a") + 1)},
            unique_together=[("a", "b")],
        )
    with pytest.raises(SpecError, match="derive"):
        ColSpec(
            pl.Struct({"x": pl.Int64}),
            fields={"x": ColSpec(pl.Int64, derive=col("a"))},
        )


# ---------------------------------------------------------------------------
# Specs as values and files
# ---------------------------------------------------------------------------


def test_a_derived_column_round_trips_through_files(tmp_path):
    path = tmp_path / "orders.yaml"
    Orders.to_yaml(path)
    assert "derive:" in path.read_text(encoding="utf-8")
    assert FrameSpec.from_yaml(path).spec == Orders.spec

    namespace: dict = {}
    exec(Orders.to_python(), namespace)
    assert namespace["Orders"].spec == Orders.spec


def test_rename_rewrites_the_expression_and_join_drops_it():
    renamed = Orders.spec.rename({"qty": "quantity"})
    assert repr(renamed.columns["total"].derive) == "col('quantity') * col('price')"
    generate(renamed, 10, seed=1)

    other = TableSpec("Other", {"qty": ColSpec(pl.Int64, bounds=(1, 20))})
    joined = Orders.spec.join(other, on="qty", how="inner")
    assert joined.columns["total"].derive is None


def test_diff_reports_a_derivation_added_as_breaking():
    plain = FrameSpec.from_spec(
        Orders.spec.with_columns(total=ColSpec(pl.Float64, nullable=True)).drop(
            "large"
        ),
        name="Orders",
    )
    derived = FrameSpec.from_spec(Orders.spec.drop("large"), name="Orders")
    added = diff(plain, derived)
    assert [(f.code, f.severity) for f in added] == [("constraint_added", "breaking")]
    removed = diff(derived, plain)
    assert [(f.code, f.severity) for f in removed] == [
        ("constraint_removed", "compatible")
    ]
    assert diff(derived, derived).findings == ()
