"""`TableSpec.join`, `prefix` and `suffix`: the spec of a frame made from
others, carrying what still holds of it."""

import polars as pl
import pytest
from polspec import (
    Check,
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSpec,
    SpecError,
    TableSpec,
    col,
    generate,
    inspect,
)

CUSTOMERS = TableSpec(
    "Customers",
    {
        "cid": ColSpec(pl.Int64, bounds=(1, 10_000), unique=True),
        "tier": ColSpec(pl.Enum(["gold", "silver"]), description="Loyalty tier."),
        "note": ColSpec(pl.String, string_length=(1, 8), validators=col("note") != "x"),
        "joined": ColSpec(pl.Date),
    },
    rows=(1, None),
    checks=[Check(col("cid") > 0, name="positive")],
)

ORDERS = TableSpec(
    "Orders",
    {
        "oid": ColSpec(pl.Int64, unique=True),
        "cid": ColSpec(pl.Int64),
        "region": ColSpec(pl.String, choices=["UK", "FR"]),
        "currency": ColSpec(
            pl.String,
            choices=["GBP", "EUR"],
            rules=[
                ColRule(when=col("region") == "UK", choices=["GBP"]),
                ColRule(when=col("region") == "FR", choices=["EUR"]),
            ],
        ),
        "note": ColSpec(pl.String, nullable=True, pattern="^[a-z]*$"),
        "qty": ColSpec(pl.Int64, bounds=(1, 9), default=1),
    },
    foreign_keys=[ForeignKey("cid", references="Customers")],
    order_by=["oid"],
    unique_together=[("oid", "cid")],
)

HOWS = [
    ("inner", None),
    ("left", None),
    ("left", False),
    ("right", None),
    ("right", False),
    ("full", None),
    ("full", True),
    ("full", False),
    ("semi", None),
    ("anti", None),
]


@pytest.fixture(scope="module")
def frames():
    customers = generate(CUSTOMERS, 40, seed=1)
    # Orders reference half the customers, so every side has unmatched rows;
    # a pattern is not generated, so the note is pinned to one it matches.
    orders = generate(
        ORDERS,
        60,
        seed=2,
        references={"Customers": customers.head(20)},
        fixed={"note": "abc"},
    )
    # And a few orders meet no customer at all.
    orphans = orders.head(5).with_columns(
        oid=pl.Series([-1, -2, -3, -4, -5]), cid=pl.Series([-1, -2, -3, -4, -5])
    )
    return pl.concat([orders, orphans]), customers


# ---------------------------------------------------------------------------
# The contract: a join of frames validates against the join of their specs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("how", "coalesce"), HOWS)
def test_the_join_of_frames_validates_against_the_join_of_specs(frames, how, coalesce):
    orders, customers = frames
    joined = orders.join(customers, on="cid", how=how, coalesce=coalesce)
    spec = ORDERS.join(CUSTOMERS, on="cid", how=how, coalesce=coalesce)
    assert spec.schema() == joined.schema
    # Semi and anti keep the left spec, its foreign key included, which the
    # orphan orders break on purpose.
    report = inspect(spec, joined, foreign_keys=False)
    assert report.passed, str(report)


def test_every_side_has_unmatched_rows(frames):
    orders, customers = frames
    assert orders.join(customers, on="cid", how="anti").height == 5
    assert customers.join(orders, on="cid", how="anti").height > 0


def test_a_cross_join_validates_too(frames):
    orders, customers = frames
    joined = orders.head(5).join(customers.head(4), how="cross")
    spec = ORDERS.join(CUSTOMERS, how="cross")
    assert spec.schema() == joined.schema
    assert inspect(spec, joined).passed


def test_different_key_names_validate_too(frames):
    orders, customers = frames
    renamed = CUSTOMERS.rename({"cid": "customer"})
    frame = customers.rename({"cid": "customer"})
    for how in ("inner", "left", "right", "full"):
        for coalesce in (None, False):
            joined = orders.join(
                frame, left_on="cid", right_on="customer", how=how, coalesce=coalesce
            )
            spec = ORDERS.join(
                renamed, left_on="cid", right_on="customer", how=how, coalesce=coalesce
            )
            assert spec.schema() == joined.schema
            assert inspect(spec, joined).passed, (how, coalesce)


# ---------------------------------------------------------------------------
# What survives
# ---------------------------------------------------------------------------


def test_claims_about_values_survive():
    spec = ORDERS.join(CUSTOMERS, on="cid")
    assert spec["tier"].description == "Loyalty tier."
    assert spec["region"].choices == ("UK", "FR")
    assert spec["qty"].default == 1
    assert spec["note"].pattern == "^[a-z]*$"
    assert spec["note_right"].string_length == CUSTOMERS["note"].string_length
    (validator,) = spec["note_right"].validators
    assert validator.expr.meta.root_names() == ["note_right"]


def test_claims_about_rows_do_not():
    spec = ORDERS.join(CUSTOMERS, on="cid")
    assert spec.checks == ()
    assert spec.foreign_keys == ()
    assert spec.unique_together == ()
    assert spec.order_by == ()
    assert spec.rows is None
    assert spec["currency"].rules == ()


def test_unmatched_sides_become_nullable():
    def nullable(how):
        spec = ORDERS.join(CUSTOMERS, on="cid", how=how)
        return spec["region"].nullable, spec["tier"].nullable

    assert nullable("inner") == (False, False)
    assert nullable("left") == (False, True)
    assert nullable("right") == (True, False)
    assert nullable("full") == (True, True)


def test_unique_survives_only_where_the_other_key_is_unique():
    # Each order meets one customer at most, so no order repeats...
    assert ORDERS.join(CUSTOMERS, on="cid", how="left")["oid"].unique
    # ...but a customer repeats for each of its orders, while an order
    # still meets one customer.
    reverse = CUSTOMERS.join(ORDERS, on="cid", how="left")
    assert not reverse["cid"].unique
    assert reverse["oid"].unique
    assert not ORDERS.join(CUSTOMERS, how="cross")["oid"].unique


def test_a_composite_key_counts_as_unique():
    parent = TableSpec(
        "P",
        {"a": ColSpec(pl.Int64), "b": ColSpec(pl.Int64), "v": ColSpec(pl.Int64)},
        unique_together=[("a", "b")],
    )
    child = TableSpec(
        "C",
        {
            "id": ColSpec(pl.Int64, unique=True),
            "a": ColSpec(pl.Int64),
            "b": ColSpec(pl.Int64),
        },
    )
    assert child.join(parent, on=["a", "b"])["id"].unique
    assert not child.join(parent, on=["a"])["id"].unique


def test_semi_and_anti_keep_the_left_spec_but_its_rows_and_order():
    for how in ("semi", "anti"):
        kept = ORDERS.with_rows((1, None)).join(CUSTOMERS, on="cid", how=how)
        assert kept == ORDERS.with_order_by()
        assert kept.name == "Orders"


def test_the_name_defaults_and_can_be_given():
    assert ORDERS.join(CUSTOMERS, on="cid").name == "Orders_Customers"
    assert ORDERS.join(CUSTOMERS, on="cid", name="Sales").name == "Sales"


def test_a_framespec_joins_and_is_joined():
    class Customers(FrameSpec):
        cid = ColSpec(pl.Int64, unique=True)
        tier = ColSpec(pl.String)

    spec = ORDERS.join(Customers, on="cid")
    assert "tier" in spec


def test_a_suffix_of_ones_own_is_used():
    spec = ORDERS.join(CUSTOMERS, on="cid", suffix="_c")
    assert "note_c" in spec


# ---------------------------------------------------------------------------
# Refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "says"),
    [
        ({"on": "nope"}, "no such column"),
        ({"on": "cid", "how": "outer"}, "how="),
        ({"on": "cid", "left_on": "cid"}, "not both"),
        ({"left_on": "cid"}, "same length"),
        ({"on": "cid", "how": "cross"}, "no keys"),
        ({"left_on": "cid", "right_on": "tier"}, "share a dtype"),
    ],
)
def test_a_join_that_cannot_be_is_refused(kwargs, says):
    with pytest.raises(SpecError, match=says):
        ORDERS.join(CUSTOMERS, **kwargs)


def test_a_raw_expression_validator_a_suffix_renames_is_refused():
    raw = CUSTOMERS.with_columns(
        note=ColSpec(pl.String, validators=pl.col("note") != "")
    )
    with pytest.raises(SpecError, match="raw polars expression"):
        ORDERS.join(raw, on="cid")


# ---------------------------------------------------------------------------
# prefix and suffix
# ---------------------------------------------------------------------------


def test_prefix_and_suffix_rename_every_column_and_claim():
    prefixed = ORDERS.prefix("o_")
    assert list(prefixed.columns) == [f"o_{c}" for c in ORDERS.columns]
    assert prefixed.order_by == ("o_oid",)
    assert prefixed.foreign_keys[0].columns == ("o_cid",)
    assert prefixed["o_currency"].rules[0].when.equals(col("o_region") == "UK")
    assert next(iter(ORDERS.suffix("_old").columns)) == "oid_old"


def test_prefix_refuses_what_rename_refuses():
    raw = TableSpec("T", {"a": ColSpec(pl.Int64, validators=pl.col("a") > 0)})
    with pytest.raises(SpecError, match="raw polars expression"):
        raw.prefix("x_")
