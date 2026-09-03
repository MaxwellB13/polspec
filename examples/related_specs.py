from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import polars as pl
from polspec import (
    Bound,
    CatSpec,
    Check,
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSpec,
    col,
)

EXAMPLE_DIR = Path(__file__).resolve().parent

# A registry shared across a Python-declared spec (Orders, below) and a
# YAML-declared one (products.yaml) -- both pull STATUS/CATEGORY/CURRENCY
# from the same file, so a column typed against CURRENCY here and one typed
# against it in products.yaml carry the same physical codes.
categories = CatSpec.from_yaml(EXAMPLE_DIR / "categories.yaml")


class Customers(FrameSpec):
    id = ColSpec(pl.Int64, bounds=(1, 10_000_000), unique=True)
    name = ColSpec(pl.String, string_length=(3, 40))
    email = ColSpec(
        pl.String,
        string_length=(10, 60),
        tags="pii",
        validators=[Check(col("email").str.contains("@"), name="email_has_at")],
    )
    country = ColSpec(pl.Enum(["UK", "US", "DE", "FR", "AU"]))
    signed_up = ColSpec(pl.Date, bounds=(dt.date(2020, 1, 1), dt.date(2026, 1, 1)))
    lifetime_value = ColSpec(
        pl.Float64,
        bounds=(0.0, None),
        distribution="lognormal",
        distribution_params={"mean": 5.0, "std": 1.2},
    )
    is_active = ColSpec(pl.Boolean, weights=[0.2, 0.8])


class Employees(FrameSpec):
    id = ColSpec(pl.Int64, bounds=(1, 1_000_000), unique=True)
    name = ColSpec(pl.String, string_length=(3, 40))
    department = ColSpec(pl.Enum(["SALES", "SUPPORT", "WAREHOUSE", "OPS"]))
    tenure_days = ColSpec(
        pl.Int16, bounds=(0, 10_000), nullable=True, null_probability=0.05
    )
    manager_id = ColSpec(
        pl.Int64, bounds=(1, 1_000_000), nullable=True, null_probability=0.15
    )

    __foreign_keys__ = [
        ForeignKey("manager_id", references="self", ref_columns="id"),
    ]


class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, 100_000_000), unique=True)
    customer_id = ColSpec(pl.Int64, bounds=(1, 10_000_000))
    status = ColSpec(categories.enum.STATUS)
    currency = ColSpec(
        categories.categorical.CURRENCY,
        choices=categories.get_choices("CURRENCY"),
    )
    region = ColSpec(pl.Enum(["UK", "US", "EU"]))
    carrier = ColSpec(
        pl.Enum(["RoyalMail", "UPS", "DHL"]),
        rules=[
            ColRule(when=col("region") == "UK", choices=["RoyalMail"]),
            ColRule(
                when=col("region").is_in(["US", "EU"]),
                choices={"UPS": 3.0, "DHL": 1.0},
            ),
        ],
    )
    subtotal = ColSpec(
        pl.Float64,
        bounds=(0.0, 5_000.0),
        distribution="lognormal",
        distribution_params={"mean": 4.0, "std": 0.6},
    )
    total = ColSpec(pl.Float64, bounds=(0.0, 6_000.0))
    placed_at = ColSpec(
        pl.Datetime("us"),
        bounds=(dt.datetime(2023, 1, 1), dt.datetime(2026, 1, 1)),
    )
    # Bounds given as the physical microsecond count Duration("us") stores,
    # rather than timedelta objects -- PyYAML's safe dumper has no
    # representer for timedelta, so Orders.to_yaml() below would fail on it.
    dispatch_sla = ColSpec(pl.Duration("us"), bounds=(3_600_000_000, 432_000_000_000))

    __checks__ = [
        Check(col("total") >= col("subtotal"), name="total_covers_subtotal"),
    ]
    __foreign_keys__ = [
        ForeignKey("customer_id", references=Customers, ref_columns="id"),
    ]


class OrderLines(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, 100_000_000))
    line_no = ColSpec(pl.Int32, bounds=(1, 1_000_000))
    sku_barcode = ColSpec(pl.Binary, string_length=(12, 12))
    quantity = ColSpec(pl.UInt16, bounds=(1, 500))
    unit_price = ColSpec(pl.Float32, bounds=(0.0, None))
    weight_grams = ColSpec(pl.UInt32, bounds=Bound(1, 20_000))
    pack_station = ColSpec(
        pl.UInt8, bounds=(1, 40), nullable=True, null_probability=0.1
    )
    discount_tier = ColSpec(
        pl.Int8, nullable=True, choices=[0, 1, 2, 3], null_probability=0.3
    )
    packed_at = ColSpec(pl.Time, bounds=(dt.time(6, 0), dt.time(22, 0)))

    __unique_together__ = [["order_id", "line_no"]]
    __foreign_keys__ = [
        ForeignKey("order_id", references=Orders, ref_columns="order_id"),
    ]


# A spec can live entirely in a file instead of a class body -- see
# products.yaml, which declares the same kind of constraints as the classes
# above (bounds, distributions, a self-referencing ForeignKey) and shares
# the CATEGORY/CURRENCY entries from categories.yaml with Orders.
Products = FrameSpec.from_yaml(EXAMPLE_DIR / "products.yaml")


if __name__ == "__main__":
    customers = Customers.generate(200, seed=1)
    employees = Employees.generate(50, seed=2)
    orders = Orders.generate(1_000, seed=3, references={Customers: customers})
    order_lines = OrderLines.generate(3_000, seed=4, references={Orders: orders})
    products = Products.generate(150, seed=5)

    # validators and __checks__ wrap arbitrary predicates that generation
    # never attempts to satisfy (see docs/guide/constraints.md), so they're
    # skipped here and proven separately below against real, compliant data.
    Customers.validate(customers, validate_validators=False)
    Employees.validate(employees)
    Orders.validate(orders, references={Customers: customers}, validate_checks=False)
    OrderLines.validate(order_lines, references={Orders: orders})
    Products.validate(products)

    print("PII columns on Customers:", Customers.tag("pii"))

    coverage_sample = Orders.generate(
        500, method="cartesian", seed=6, references={Customers: customers}
    )
    print("Cartesian coverage rows:", coverage_sample.height)

    # Checks, validators and rules written with col() are written to the
    # file and read back. A cross-spec ForeignKey points at a Python class
    # with no stable name in a standalone file, so to_yaml() warns about it
    # and drops it; from_yaml() recovers by subclassing and re-declaring it.
    # Written to a temporary directory so running this script never dirties
    # the repository.
    scratch = Path(tempfile.mkdtemp(prefix="polspec-example-"))
    Orders.to_yaml(scratch / "orders_generated.yaml")
    LoadedOrders = FrameSpec.from_yaml(scratch / "orders_generated.yaml")

    class OrdersFromYaml(LoadedOrders):
        __foreign_keys__ = [
            ForeignKey("customer_id", references=Customers, ref_columns="id"),
        ]

    OrdersFromYaml.validate(
        OrdersFromYaml.generate(100, seed=7, references={Customers: customers}),
        references={Customers: customers},
        validate_checks=False,
    )

    # Proof that the validator and check skipped above do work -- against
    # real, self-consistent rows rather than independently-drawn random ones.
    first_customer_id = int(customers["id"][0])
    compliant_customer = pl.DataFrame(
        {
            "id": [first_customer_id],
            "name": ["Ada Lovelace"],
            "email": ["ada@example.com"],
            "country": pl.Series(["UK"], dtype=Customers.schema()["country"]),
            "signed_up": [dt.date(2024, 1, 1)],
            "lifetime_value": [120.50],
            "is_active": [True],
        }
    )
    Customers.validate(compliant_customer)

    compliant_order = pl.DataFrame(
        {
            "order_id": [1],
            "customer_id": [first_customer_id],
            "status": pl.Series(["NEW"], dtype=Orders.schema()["status"]),
            "currency": pl.Series(["GBP"], dtype=Orders.schema()["currency"]),
            "region": pl.Series(["UK"], dtype=Orders.schema()["region"]),
            "carrier": pl.Series(["RoyalMail"], dtype=Orders.schema()["carrier"]),
            "subtotal": [80.0],
            "total": [95.0],
            "placed_at": [dt.datetime(2024, 6, 1)],
            "dispatch_sla": [dt.timedelta(days=1)],
        }
    )
    Orders.validate(compliant_order, references={Customers: customers})

    print("All specs generated and validated successfully.")
