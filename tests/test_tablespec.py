"""`TableSpec`: the spec as a value, and the class body that builds it.

A `FrameSpec` class body is sugar over a `TableSpec`. These tests pin the
data object itself -- construction, validation, the structural operations --
and the metaclass behaviour that keeps columns and methods from colliding.
"""

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
)
from polspec.tablespec import as_spec_name, as_table_spec


class Customers(FrameSpec):
    id = ColSpec(pl.Int64, bounds=(1, 1_000), unique=True)
    name = ColSpec(pl.String, tags="pii")


class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, unique=True)
    customer_id = ColSpec(pl.Int64, bounds=(1, 1_000))
    region = ColSpec(pl.Enum(["UK", "US"]))
    carrier = ColSpec(
        pl.Enum(["RM", "UPS"]),
        rules=[ColRule(when={"column": "region", "equals": "UK"}, choices=["RM"])],
    )
    total = ColSpec(pl.Float64, bounds=(0.0, None))
    __unique_together__ = [["customer_id", "order_id"]]
    __checks__ = [Check(pl.col("total") >= 0, name="non_negative")]
    __foreign_keys__ = [
        ForeignKey("customer_id", references=Customers, ref_columns="id")
    ]


# ---------------------------------------------------------------------------
# The class body builds a TableSpec
# ---------------------------------------------------------------------------


def test_class_body_builds_a_table_spec():
    spec = Orders.spec
    assert isinstance(spec, TableSpec)
    assert spec.name == "Orders"
    assert list(spec) == ["order_id", "customer_id", "region", "carrier", "total"]
    assert spec["total"] == ColSpec(pl.Float64, bounds=(0.0, None))
    assert "region" in spec
    assert len(spec) == 5
    assert spec.unique_together == (("customer_id", "order_id"),)
    assert [c.name for c in spec.checks] == ["non_negative"]
    assert spec.foreign_keys[0].references == "Customers"
    assert spec.foreign_keys[0].target is Customers.spec


def test_columns_are_not_class_attributes_but_still_reachable():
    assert "order_id" not in vars(Orders)
    assert Orders.order_id == ColSpec(pl.Int64, unique=True)
    assert Orders.col("order_id") is Orders.spec["order_id"]
    with pytest.raises(AttributeError):
        Orders.not_a_column  # noqa: B018


def test_table_spec_is_immutable_and_comparable():
    spec = Orders.spec
    with pytest.raises(AttributeError):
        spec.name = "Other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.columns["x"] = ColSpec(pl.Int64)  # type: ignore[index]
    assert spec == TableSpec(
        "Orders",
        dict(spec.columns),
        checks=spec.checks,
        unique_together=spec.unique_together,
        foreign_keys=spec.foreign_keys,
    )
    assert spec != spec.with_name("Other")


def test_from_spec_wraps_a_table_spec_in_a_class():
    spec = TableSpec("Made", {"a": ColSpec(pl.Int64), "b": ColSpec(pl.String)})
    Made = FrameSpec.from_spec(spec)
    assert Made.__name__ == "Made"
    assert Made.spec is spec
    assert Made.generate(3, seed=1).columns == ["a", "b"]

    Renamed = FrameSpec.from_spec(spec, name="Renamed")
    assert Renamed.spec.name == "Renamed"
    assert Renamed.spec.columns == spec.columns


def test_as_table_spec_and_as_spec_name():
    assert as_table_spec(Orders) is Orders.spec
    assert as_table_spec(Orders.spec) is Orders.spec
    assert as_spec_name(Orders) == "Orders"
    assert as_spec_name(Orders.spec) == "Orders"
    assert as_spec_name("Orders") == "Orders"
    with pytest.raises(TypeError, match="Expected a TableSpec"):
        as_table_spec(object())


# ---------------------------------------------------------------------------
# Declaration-time validation lives on the data object
# ---------------------------------------------------------------------------


def test_table_spec_validates_like_a_class_body():
    with pytest.raises(SpecError, match="references unknown column 'zzz'"):
        TableSpec(
            "X",
            {
                "a": ColSpec(
                    pl.Int64,
                    rules=[ColRule(when={"column": "zzz", "equals": 1}, choices=[1])],
                )
            },
        )
    with pytest.raises(SpecError, match="Composite unique key"):
        TableSpec("X", {"a": ColSpec(pl.Int64)}, unique_together=[["a", "b"]])
    with pytest.raises(SpecError, match="Duplicate Check name"):
        TableSpec(
            "X",
            {"a": ColSpec(pl.Int64)},
            checks=[Check(pl.col("a") > 0, name="c"), Check(pl.col("a") < 9, name="c")],
        )
    with pytest.raises(SpecError, match="must be a ColSpec"):
        TableSpec("X", {"a": "not a colspec"})  # type: ignore[dict-item]
    with pytest.raises(SpecError, match="conflicts with its key"):
        TableSpec("X", {"a": ColSpec(pl.Int64, col_name="b")})
    with pytest.raises(SpecError, match="non-empty string"):
        TableSpec("", {"a": ColSpec(pl.Int64)})


def test_foreign_key_to_an_unresolved_name_is_allowed():
    spec = TableSpec(
        "Lines",
        {"order_id": ColSpec(pl.Int64)},
        foreign_keys=[ForeignKey("order_id", references="Orders")],
    )
    fk = spec.foreign_keys[0]
    assert fk.references == "Orders"
    assert fk.target is None
    assert spec.resolve_target(fk) is None
    # Generation leaves the column alone with no parent to sample from.
    assert FrameSpec.from_spec(spec).generate(5, seed=1).height == 5


def test_foreign_key_to_a_bound_target_is_checked():
    with pytest.raises(SpecError, match="unknown column 'nope' on 'Customers'"):
        TableSpec(
            "Lines",
            {"customer_id": ColSpec(pl.Int64)},
            foreign_keys=[
                ForeignKey("customer_id", references=Customers.spec, ref_columns="nope")
            ],
        )


# ---------------------------------------------------------------------------
# Structural operations
# ---------------------------------------------------------------------------


def test_with_columns_adds_and_replaces():
    spec = Orders.spec.with_columns(
        {"placed": ColSpec(pl.Date)}, total=ColSpec(pl.Float32, bounds=(0.0, 10.0))
    )
    assert list(spec) == [
        "order_id",
        "customer_id",
        "region",
        "carrier",
        "total",
        "placed",
    ]
    assert spec["total"].dtype == pl.Float32
    assert Orders.spec["total"].dtype == pl.Float64  # the original is untouched


def test_drop_removes_columns_and_the_keys_that_used_them():
    spec = Orders.spec.drop("customer_id")
    assert "customer_id" not in spec
    assert spec.unique_together == ()
    assert spec.foreign_keys == ()
    assert spec.checks == Orders.spec.checks
    with pytest.raises(SpecError, match="no such column"):
        Orders.spec.drop("nope")


def test_drop_leaves_a_dangling_rule_for_validation_to_reject():
    with pytest.raises(SpecError, match="references unknown column 'region'"):
        Orders.spec.drop("region")


def test_select_keeps_columns_in_the_order_given():
    spec = Orders.spec.select("total", "order_id")
    assert list(spec) == ["total", "order_id"]
    assert spec.unique_together == ()
    assert spec.foreign_keys == ()


def test_rename_rewrites_every_constraint():
    spec = Orders.spec.rename({"customer_id": "cust", "region": "area"})
    assert list(spec) == ["order_id", "cust", "area", "carrier", "total"]
    assert spec["carrier"].rules[0].when == {"column": "area", "equals": "UK"}
    assert spec.unique_together == (("cust", "order_id"),)
    fk = spec.foreign_keys[0]
    assert fk.columns == ("cust",)
    assert fk.ref_columns == ("id",)
    assert fk.references == "Customers"
    assert fk.name == "fk_cust__Customers"  # a defaulted name follows the column
    assert FrameSpec.from_spec(spec).generate(5, seed=1).columns == list(spec)


def test_rename_rejects_collisions_and_validators():
    with pytest.raises(SpecError, match="that name is taken"):
        Orders.spec.rename({"region": "total"})
    with_validator = Orders.spec.with_columns(
        total=ColSpec(pl.Float64, validators=[pl.col("total") >= 0])
    )
    with pytest.raises(SpecError, match="carries validators"):
        with_validator.rename({"total": "amount"})


def test_rename_a_self_referencing_key_rewrites_both_sides():
    class Employees(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        manager_id = ColSpec(pl.Int64, nullable=True)
        __foreign_keys__ = [
            ForeignKey("manager_id", references="self", ref_columns="id")
        ]

    spec = Employees.spec.rename({"id": "emp_id"})
    fk = spec.foreign_keys[0]
    assert fk.ref_columns == ("emp_id",)
    assert fk.columns == ("manager_id",)


def test_with_checks_foreign_keys_and_unique_together_append():
    spec = (
        Customers.spec.with_checks(Check(pl.col("id") > 0, name="positive"))
        .with_unique_together(["id", "name"])
        .with_foreign_keys(ForeignKey("id", references="Accounts"))
    )
    assert [c.name for c in spec.checks] == ["positive"]
    assert spec.unique_together == (("id", "name"),)
    assert spec.foreign_keys[0].references == "Accounts"


def test_structural_ops_round_trip_through_generate_and_validate():
    spec = Orders.spec.drop("customer_id").rename({"total": "amount"})
    Derived = FrameSpec.from_spec(spec, name="Derived")
    df = Derived.generate(50, seed=3)
    assert df.columns == ["order_id", "region", "carrier", "amount"]
    Derived.validate(df, validate_unique=False, validate_checks=False)


# ---------------------------------------------------------------------------
# Inheritance through the metaclass
# ---------------------------------------------------------------------------


def test_subclass_inherits_overrides_and_removes_columns():
    class Base(FrameSpec):
        a = ColSpec(pl.Int64)
        b = ColSpec(pl.String)
        c = ColSpec(pl.Boolean)

    class Child(Base):
        b = ColSpec(pl.Int32)  # override keeps position
        c = None  # removal
        d = ColSpec(pl.Float64)

    assert list(Child.spec) == ["a", "b", "d"]
    assert Child.spec["b"].dtype == pl.Int32
    assert list(Base.spec) == ["a", "b", "c"]  # the parent is untouched


def test_subclass_with_col_name_override_replaces_the_renamed_column():
    class Base(FrameSpec):
        price = ColSpec(pl.Float64, col_name="Unit Price")

    class Child(Base):
        price = ColSpec(pl.Float32, col_name="Price")

    assert list(Child.spec) == ["Price"]


def test_declarations_under_bare_names_do_not_shadow_accessors():
    class Spec(FrameSpec):
        a = ColSpec(pl.Int64)
        checks = [Check(pl.col("a") > 0, name="pos")]
        unique_together = ["a"]

    assert [c.name for c in Spec.checks()] == ["pos"]
    assert Spec.unique_together() == (("a",),)
