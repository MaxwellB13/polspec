# polspec

**One schema for your Polars data. Generate it, validate it, watch it drift.**

[![PyPI](https://img.shields.io/pypi/v/polspec)](https://pypi.org/project/polspec/)
[![Python](https://img.shields.io/pypi/pyversions/polspec)](https://pypi.org/project/polspec/)
[![Tests](https://github.com/MaxwellB13/polspec/actions/workflows/test.yml/badge.svg)](https://github.com/MaxwellB13/polspec/actions/workflows/test.yml)
[![License](https://img.shields.io/pypi/l/polspec)](LICENSE)

Your tests need fixtures, and your pipeline needs a contract. Written
separately, they drift apart. polspec lets you write the schema once, as a
[Polars](https://pola.rs) class, and use it for both: realistic data to
test against, and validation of the data that turns up.

```python
import datetime as dt

import polars as pl
from polspec import ColSpec, FrameSpec

class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, 10**9), unique=True)
    status   = ColSpec(pl.Enum, choices={"NEW": 0.2, "PAID": 0.5, "SHIPPED": 0.3})
    total    = ColSpec(pl.Float64, bounds=(0, 5_000), distribution="lognormal",
                       distribution_params={"mean": 4, "std": 1})
    placed   = ColSpec(pl.Date, bounds=(dt.date(2025, 1, 1), dt.date(2025, 12, 31)),
                       nullable=True, null_probability=0.05)
    email    = ColSpec(pl.String, format="email")

orders = Orders.generate(1_000_000, seed=42)   # a million rows in ~8 ms
Orders.validate(orders)                        # ~26 ms; raises on any breach
```

```text
┌───────────┬────────┬────────────┬────────────┬───────────────────────┐
│ order_id  ┆ status ┆ total      ┆ placed     ┆ email                 │
│ i64       ┆ enum   ┆ f64        ┆ date       ┆ str                   │
╞═══════════╪════════╪════════════╪════════════╪═══════════════════════╡
│ 830916752 ┆ PAID   ┆ 9.187332   ┆ 2025-12-30 ┆ dupebca@xdcbivwq.dev  │
│ 366447256 ┆ PAID   ┆ 19.02604   ┆ 2025-11-28 ┆ hinro@ewzc.com        │
│ 922459406 ┆ PAID   ┆ 336.930617 ┆ 2025-04-11 ┆ jp5uo7@mgayhywoxn.com │
└───────────┴────────┴────────────┴────────────┴───────────────────────┘
```

## What you get

- **Fast.** The generator is a Rust extension that fills columns in
  parallel. `validate()` compiles every check on every column into one
  Polars aggregation, so a wide table costs about what a narrow one does.
  Timings here are from a laptop; more are in
  [Comparison](https://maxwellb13.github.io/polspec/explanation/comparison/).
- **Lazy and streaming.** `scan()` is a `LazyFrame`: filter and project it
  like any other. The sinks write Parquet, IPC, CSV or NDJSON in batches,
  never holding the frame in memory, and a unique column stays unique
  across every batch.
- **Every Polars dtype.** Integers to 128 bits, `Decimal`, time-zoned
  `Datetime`, `Enum`, `Categorical`, and `List`, `Array` and `Struct` nested
  to any depth, and `Map`. Bounds, choices with weights,
  distributions, formats such as `email` and `uuid4`, regex patterns,
  uniqueness, nulls and NaNs are all declared per column, and all of them
  generate.
- **Tables that relate.** Foreign keys within and across specs, conditional
  rules, columns derived from others, composite keys, row order and
  hierarchies are generated to hold, and validated; a row count is validated. Joining two specs gives the spec of
  the joined frame.
  Checks written as Polars expressions are enforced at validation.
- **Validation as data.** `inspect()` returns every finding at once, with the
  offending rows a lazy filter away, and splits a frame into what passed and
  what to quarantine. Declared defaults can fill what is missing first.
- **Learn from real data.** `profile()` turns a frame into a spec, finding
  formats, the patterns codes share, frequencies, distributions and null
  rates — and, with `parse_text=True`, the numbers, flags and dates in a
  file read as text, in the format they are written in. `profile_all()`
  profiles related tables at once and finds the keys between them;
  `synthesize()` and `synthesize_all()` make stand-ins with the same shape,
  a tenth the size or ten times it, every key kept.
- **Drift with statistics behind it.** A move is reported only when it is
  both significant and large, as breaking or compatible.
- **Specs as files and docs.** Save and load specs as YAML or Python. Render
  a Markdown data dictionary or a Mermaid ER diagram. A spec pickles, and
  reads as an Arrow schema: `pl.Schema(Orders)`.
- **Tests that say what broke.** `assert_valid` fails a test with the
  findings, the failing rows and the seed to rerun it with. A pytest fixture
  gives each test frames from a seed of its own, and `strategy` hands a
  spec to Hypothesis.
- **A CLI for your pipeline.** `polspec validate`, `generate`, `drift` and
  `synthesize`, with exit codes for CI.
- **Deterministic.** The same seed gives the same data on every Polars 2,
  on Windows, Linux and macOS, checked case by case in CI.

## A quick tour

### Generate lazily

`scan()` returns a `LazyFrame`, so Polars pushes filters and projections into it:

```python
paid = (
    Orders.scan(10_000_000, seed=1)
    .filter(pl.col("status") == "PAID")
    .select("order_id", "total")
    .collect()                                            # ~0.06 s
)
Orders.sink_parquet("orders.parquet", 1_000_000, seed=1)  # streamed in batches
```

### Validate and quarantine

`inspect()` never raises: it reports every finding, with the rows behind it.

```python
broken = orders.with_columns(
    pl.when(pl.col("order_id") % 1_000 == 0).then(-1.0).otherwise("total").alias("total")
)
report = Orders.inspect(broken)          # 1 finding: 1,053 totals out of bounds
clean = report.passing_rows().collect()  # the other 998,947 rows
```

### Profile real data

```python
from polspec import profile, synthesize

spec = profile(orders.head(100_000))
spec["total"].distribution                         # 'lognormal', fitted
fake = synthesize(orders.head(100_000), seed=1)    # a stand-in with the same shape
```

### Fake a set of related tables

```python
import polspec

customers = pl.DataFrame({"id": range(1, 101), "tier": ["free", "pro"] * 50})
purchases = pl.DataFrame({"id": range(1, 401), "customer_id": [i % 37 + 1 for i in range(400)]})
real = {"Customers": customers, "Purchases": purchases}

registry = polspec.profile_all(real)               # Purchases.customer_id -> Customers.id
fake = polspec.synthesize_all(real, scale=0.1, seed=1)
registry.validate_all(fake)                        # every key held by a fake parent
```

### Catch drift

```python
print(Orders.drift(orders.filter(pl.col("status") != "NEW")))
```

```text
Drift: 0 breaking, 2 compatible, DataFrame against 'Orders'
  - [compatible] Column 'status': 1 of 3 declared value(s) never appear: ['NEW']
  - [compatible] Column 'status': frequencies moved by 20.0% over 800,231 value(s): ...
```

### From the command line

```bash
polspec schema infer orders.csv -o orders_spec.py
polspec schema infer customers.csv orders.csv -o shop.yaml   # a registry, keys found
polspec validate orders_spec.py orders.csv --failing rejected.csv
polspec drift orders_spec.py orders.csv --markdown
polspec generate orders_spec.py -n 1000000 -o fixture.parquet --seed 1
```

## Install

```bash
uv add polspec           # or: pip install polspec
```

polspec works with Polars 2 on Python 3.12 to 3.15. Wheels are published for Linux (x86_64, aarch64), macOS (Intel
and Apple silicon) and Windows (x86_64), so you don't need a Rust toolchain.

> **Stable.** A public name, a spec file's format, a seed's output and the
> command line change only in a major release, after a deprecation.
> [Stability](https://maxwellb13.github.io/polspec/explanation/stability/)
> says what is promised, each promise held by a test, and
> [Known limitations](https://maxwellb13.github.io/polspec/explanation/limitations/)
> what polspec does not do.

## Documentation

**[maxwellb13.github.io/polspec](https://maxwellb13.github.io/polspec/)**: start
with [Getting started](https://maxwellb13.github.io/polspec/tutorial/getting-started/),
find a task among the [how-to guides](https://maxwellb13.github.io/polspec/how-to/columns/),
or look a name up in the [API reference](https://maxwellb13.github.io/polspec/reference/api/).

## Contributing

Setting up a checkout, the checks CI runs and the release process are in
[CONTRIBUTING.md](CONTRIBUTING.md). What changed, and when, is in
[CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE).
