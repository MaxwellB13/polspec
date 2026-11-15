"""Validation speed: polspec beside dataframely and pandera.

Each tool validates the same two valid frames -- a million orders and the
hundred thousand customers they belong to -- against the same claims:

- `id` unique; `customer_id` a key into `Customers.customer_id`, itself
  unique;
- `status` nullable, one of three values;
- `amount` nullable, within [0, 1000];
- `ref` matching `^ORD-[0-9]{6}$`;
- `qty` within [1, 20].

Each tool is written the way its documentation writes it: polspec a
`Registry` over two `FrameSpec`s, dataframely a `Collection` whose filter
keeps the orders of known customers, pandera two `DataFrameSchema`s with
the key as a custom check. Eager and lazy where the tool has both: a lazy
run is timed until the validated frames are collected, since dataframely
validates a lazy frame when it is collected. The fastest of several runs,
after one to warm up.

    uv run --group bench python benchmarks/validation.py
    uv run --group bench python benchmarks/validation.py --rows 10000000
"""

# dataframely reads its Collection's annotations at class creation, so this
# module does not use `from __future__ import annotations`.

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from importlib.metadata import version

import dataframely as dy
import numpy as np
import pandera.polars as pa
import polars as pl
from pandera.config import ValidationDepth, config_context
from polspec import ColSpec, ForeignKey, FrameSpec, Registry

STATUSES = ["NEW", "PAID", "SHIPPED"]
PATTERN = r"^ORD-[0-9]{6}$"
SEED = 42


def frames(rows: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """`Customers` and `Orders`, valid: every customer has orders, so a
    filter that drops customers without any keeps every row."""
    customers = max(1, rows // 10)
    rng = np.random.default_rng(SEED)
    present = rng.random(rows) >= 0.05
    ids = rng.permutation(rows) + 1
    orders = pl.DataFrame(
        {
            "id": ids,
            "customer_id": rng.permutation(np.arange(rows) % customers) + 1,
            "status": pl.Series(rng.choice(STATUSES, rows)).set(
                pl.Series(~present), None
            ),
            "amount": pl.Series(np.round(rng.random(rows) * 1000, 2)).set(
                pl.Series(~rng.permutation(present)), None
            ),
            "ref": pl.Series(ids % 1_000_000).cast(pl.String).str.zfill(6),
            "qty": rng.integers(1, 21, rows),
        }
    ).with_columns(ref="ORD-" + pl.col("ref"))
    people = pl.DataFrame(
        {
            "customer_id": np.arange(1, customers + 1),
            "name": [f"customer {i}" for i in range(customers)],
        }
    )
    return people, orders


# --- polspec ---------------------------------------------------------------


class Customers(FrameSpec):
    customer_id = ColSpec(pl.Int64, unique=True)
    name = ColSpec(pl.String)


class Orders(FrameSpec):
    id = ColSpec(pl.Int64, unique=True)
    customer_id = ColSpec(pl.Int64)
    status = ColSpec(pl.String, choices=STATUSES, nullable=True)
    amount = ColSpec(pl.Float64, bounds=(0.0, 1000.0), nullable=True)
    ref = ColSpec(pl.String, pattern=PATTERN)
    qty = ColSpec(pl.Int64, bounds=(1, 20))
    __foreign_keys__ = [
        ForeignKey("customer_id", references="Customers", ref_columns="customer_id")
    ]


SHOP = Registry(Customers, Orders)


def polspec_eager(people: pl.DataFrame, orders: pl.DataFrame) -> None:
    SHOP.validate_all({"Customers": people, "Orders": orders})


def polspec_lazy(people: pl.DataFrame, orders: pl.DataFrame) -> None:
    valid = SHOP.validate_all({"Customers": people.lazy(), "Orders": orders.lazy()})
    pl.collect_all([valid["Customers"], valid["Orders"]])  # ty: ignore[invalid-argument-type]


# --- dataframely -------------------------------------------------------------


class CustomerSchema(dy.Schema):
    customer_id = dy.Int64(primary_key=True)
    name = dy.String()


class OrderSchema(dy.Schema):
    # The collection's members share a primary key: an order's is its
    # customer and its own id, which is unique on its own too.
    customer_id = dy.Int64(primary_key=True)
    id = dy.Int64(primary_key=True, unique=True)
    status = dy.String(nullable=True, check=lambda c: c.is_in(STATUSES))
    amount = dy.Float64(nullable=True, min=0.0, max=1000.0)
    ref = dy.String(regex=PATTERN)
    qty = dy.Int64(min=1, max=20)


class Shop(dy.Collection):
    customers: dy.LazyFrame[CustomerSchema]
    orders: dy.LazyFrame[OrderSchema]

    @dy.filter()
    def known_customer(self) -> pl.LazyFrame:
        return self.customers.join(
            self.orders.select("customer_id").unique(), on="customer_id"
        )


def dataframely_eager(people: pl.DataFrame, orders: pl.DataFrame) -> None:
    CustomerSchema.validate(people)
    OrderSchema.validate(orders)
    Shop.validate({"customers": people, "orders": orders}, skip_member_validation=True)


def dataframely_lazy(people: pl.DataFrame, orders: pl.DataFrame) -> None:
    shop = Shop.validate(
        {"customers": people.lazy(), "orders": orders.lazy()}, lazy=True
    )
    shop.collect_all()


# --- pandera -----------------------------------------------------------------


def _pandera(people: pl.DataFrame) -> tuple[pa.DataFrameSchema, pa.DataFrameSchema]:
    known = people["customer_id"].implode()
    customers = pa.DataFrameSchema(
        {
            "customer_id": pa.Column(pl.Int64, unique=True),
            "name": pa.Column(pl.String),
        }
    )
    orders = pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64, unique=True),
            "customer_id": pa.Column(
                pl.Int64,
                pa.Check(
                    lambda data: data.lazyframe.select(pl.col(data.key).is_in(known)),
                    name="known_customer",
                ),
            ),
            "status": pa.Column(pl.String, pa.Check.isin(STATUSES), nullable=True),
            "amount": pa.Column(
                pl.Float64, pa.Check.in_range(0.0, 1000.0), nullable=True
            ),
            "ref": pa.Column(pl.String, pa.Check.str_matches(PATTERN)),
            "qty": pa.Column(pl.Int64, pa.Check.in_range(1, 20)),
        }
    )
    return customers, orders


def pandera_eager(people: pl.DataFrame, orders: pl.DataFrame) -> None:
    customers, schema = _pandera(people)
    customers.validate(people, lazy=True)
    schema.validate(orders, lazy=True)


def pandera_lazy(people: pl.DataFrame, orders: pl.DataFrame) -> None:
    # A LazyFrame is checked for its schema alone unless asked for its data.
    customers, schema = _pandera(people)
    with config_context(validation_depth=ValidationDepth.SCHEMA_AND_DATA):
        valid = (
            customers.validate(people.lazy(), lazy=True),
            schema.validate(orders.lazy(), lazy=True),
        )
    pl.collect_all(valid)


RUNS: dict[str, Callable[[pl.DataFrame, pl.DataFrame], None]] = {
    "polspec eager": polspec_eager,
    "polspec lazy": polspec_lazy,
    "dataframely eager": dataframely_eager,
    "dataframely lazy": dataframely_lazy,
    "pandera eager": pandera_eager,
    "pandera lazy": pandera_lazy,
}


def best_of(run: Callable[[], None], repeats: int) -> float:
    run()  # warm: imports, compiled patterns, the thread pool
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - start)
    return best


def machine() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "threads": str(pl.thread_pool_size()),
        "cpus": str(os.cpu_count()),
        **{
            name: version(name)
            for name in ("polspec", "polars", "dataframely", "pandera")
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--only", nargs="+", choices=list(RUNS), metavar="RUN")
    parser.add_argument("--json", metavar="PATH", help="Also write the results here")
    args = parser.parse_args(argv)

    people, orders = frames(args.rows)
    results = {}
    for name in args.only or RUNS:
        run = RUNS[name]
        results[name] = best_of(lambda run=run: run(people, orders), args.repeats)
        print(f"{name:<20} {results[name]:>9.4f}s", flush=True)

    fastest = min(results.values())
    print(f"\n{args.rows:,} orders, {people.height:,} customers")
    print("| Tool | Time | vs fastest |\n|:--|--:|--:|")
    for name, seconds in results.items():
        print(f"| {name} | {seconds:.4f}s | {seconds / fastest:.1f}x |")
    about = machine()
    print("\n" + ", ".join(f"{k} {v}" for k, v in about.items()))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump(
                {"rows": args.rows, "results": results, "machine": about}, out, indent=2
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
