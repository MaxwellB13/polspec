"""Every value polspec takes as a declaration or hands back as a result
survives `pickle` and `copy.deepcopy` equal to itself -- so a spec, a
registry or a report can go to another process -- and a report still names
its rows on the far side."""

# ruff: noqa: S301 - unpickling what this file pickled is the property under test

import copy
import pickle

import polars as pl
import polspec
import pytest
from cases import COLUMN_CASES, EVERY_DTYPE
from polspec import (
    Bound,
    CatSpec,
    Check,
    ColSpec,
    ForeignKey,
    FrameSpec,
    Hierarchy,
    Registry,
    TableSpec,
    ValidationOptions,
    col,
)


def _round_trips(value):
    for again in (pickle.loads(pickle.dumps(value)), copy.deepcopy(value)):
        assert again == value
        assert type(again) is type(value)


@pytest.mark.parametrize("name", sorted(COLUMN_CASES))
def test_every_declared_column_pickles(name):
    column = COLUMN_CASES[name]
    _round_trips(column)
    _round_trips(TableSpec("T", {"c": column}))


@pytest.mark.parametrize("dtype", EVERY_DTYPE, ids=str)
def test_a_spec_of_every_dtype_pickles(dtype):
    _round_trips(TableSpec("T", {"c": ColSpec(dtype)}))


def test_a_spec_with_every_table_claim_pickles():
    spec = TableSpec(
        "Nodes",
        {
            "id": ColSpec(pl.Int64, bounds=(1, 1_000), unique=True),
            "parent": ColSpec(pl.Int64, bounds=(1, 1_000), nullable=True),
            "kind": ColSpec(pl.Enum(["a", "b"]), choices={"a": 0.7, "b": 0.3}),
            "x": ColSpec(pl.Float64, validators=[col("x") >= 0]),
            "raw": ColSpec(pl.Int64, validators=[pl.col("raw") > 0]),
        },
        description="A tree.",
        rows=(1, None),
        checks=[Check(col("x") < 100, name="small")],
        unique_together=[["id", "kind"]],
        hierarchy=Hierarchy(child="id", parent="parent", max_depth=3),
        order_by="id",
    )
    _round_trips(spec)


def test_the_other_declarations_pickle():
    for value in (
        Bound(1, None),
        Check(col("a") > 0, name="positive"),
        ForeignKey("a", references="Other"),
        CatSpec(enums={"STATUS": ["NEW", "PAID"]}),
        ValidationOptions(cast=True, parse_text=True),
    ):
        _round_trips(value)


class Customers(FrameSpec):
    customer_id = ColSpec(pl.Int64, bounds=(1, 100), unique=True)


class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, 1_000), unique=True)
    customer_id = ColSpec(pl.Int64, bounds=(1, 200))
    total = ColSpec(pl.Float64, bounds=(0.0, 10.0))
    __foreign_keys__ = [ForeignKey("customer_id", references=Customers)]
    __order_by__ = "order_id"


def test_a_framespec_class_pickles_by_reference():
    assert pickle.loads(pickle.dumps(Orders)) is Orders


def test_a_registry_pickles():
    registry = Registry(Customers, Orders)
    again = pickle.loads(pickle.dumps(registry))
    assert again.names == registry.names
    assert again["Orders"] == registry["Orders"]


def _broken_orders():
    customers = Customers.generate(50, seed=1)
    orders = Orders.generate(200, seed=1, references={Customers: customers})
    return customers, orders.with_columns(
        total=pl.when(pl.col("order_id") % 7 == 0)
        .then(99.0)
        .otherwise(pl.col("total")),
        customer_id=pl.when(pl.col("order_id") % 11 == 0)
        .then(150)
        .otherwise(pl.col("customer_id")),
    )


def test_a_report_pickles_and_still_names_its_rows():
    """Each finding names its rows through a value, not a closure, so the
    rows -- a foreign key's orphans too -- and the settled split come back."""
    customers, orders = _broken_orders()
    report = Orders.inspect(orders, references={Customers: customers})
    assert {f.code for f in report.findings} == {"bounds", "foreign_key"}

    for again in (pickle.loads(pickle.dumps(report)), copy.deepcopy(report)):
        assert again == report
        for before, after in zip(report.findings, again.findings, strict=True):
            assert report.rows(before).collect().equals(again.rows(after).collect())
        assert again.passing_rows().collect().equals(report.passing_rows().collect())
        assert again.failing_rows().collect().equals(report.failing_rows().collect())


def test_a_drift_report_pickles():
    _, orders = _broken_orders()
    report = polspec.drift(Orders, orders)
    again = pickle.loads(pickle.dumps(report))
    assert again.findings == report.findings


def test_a_spec_reads_as_an_arrow_schema():
    """`__arrow_c_schema__`: a spec, or its class, where a schema is taken."""
    pyarrow = pytest.importorskip("pyarrow")
    expected = pl.Schema(
        {"order_id": pl.Int64, "customer_id": pl.Int64, "total": pl.Float64}
    )
    assert pl.Schema(Orders) == expected
    assert pl.Schema(Orders.spec) == expected
    assert pyarrow.schema(Orders).names == ["order_id", "customer_id", "total"]
    assert pl.DataFrame(schema=pl.Schema(Orders)).schema == expected
