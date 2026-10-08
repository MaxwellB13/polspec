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
    status   = ColSpec(pl.Enum(["NEW", "PAID", "SHIPPED"]), weights=[0.2, 0.5, 0.3])
    total    = ColSpec(pl.Float64, bounds=(0, 5_000), distribution="lognormal",
                       distribution_params={"mean": 4, "std": 1})
    placed   = ColSpec(pl.Date, bounds=(dt.date(2025, 1, 1), dt.date(2025, 12, 31)),
                       nullable=True, null_probability=0.05)
    email    = ColSpec(pl.String, format="email")

orders = Orders.generate(1_000_000, seed=42)   # a million rows in ~12 ms
Orders.validate(orders)                        # ~27 ms; raises on any breach
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
- **Lazy and streaming.** `scan()` is a `LazyFrame`: filter and project it
  like any other. The sinks write Parquet, IPC, CSV or NDJSON in batches,
  never holding the frame in memory, and a unique column stays unique
  across every batch.
- **Every Polars dtype.** Integers to 128 bits, `Decimal`, time-zoned
  `Datetime`, `Enum`, `Categorical`, and `List`, `Array` and `Struct` nested
  to any depth (plus `Map` on Polars 2). Bounds, choices with weights,
  distributions, formats such as `email` and `uuid4`, uniqueness, nulls and
  NaNs are all declared per column.
- **Tables that relate.** Foreign keys within and across specs, conditional
  rules, composite keys and hierarchies are generated to hold, and validated.
  Checks written as Polars expressions are enforced at validation.
- **Validation as data.** `inspect()` returns every finding at once, with the
  offending rows a lazy filter away, and splits a frame into what passed and
  what to quarantine.
- **Learn from real data.** `profile()` turns a frame into a spec, finding
  formats, frequencies, distributions and null rates; `synthesize()` makes a
  stand-in with the same shape.
- **Drift with statistics behind it.** A move is reported only when it is
  both significant and large, as breaking or compatible.
- **Specs as files and docs.** Save and load specs as YAML or Python. Render
  a Markdown data dictionary or a Mermaid ER diagram.
- **A CLI for your pipeline.** `polspec validate`, `generate`, `drift` and
  `synthesize`, with exit codes for CI.
- **Deterministic.** The same seed gives the same data on every Polars from
  1.39 to 2.x, checked case by case in CI.

## A quick tour

**Generate lazily.** Nothing is drawn until you collect, and Polars pushes
your filter and projection into the scan:

```python
paid = (
    Orders.scan(10_000_000, seed=1)       # a LazyFrame
    .filter(pl.col("status") == "PAID")
    .select("order_id", "total")
    .collect()                            # ten million rows drawn in ~0.2 s
)
Orders.sink_parquet("orders.parquet", 1_000_000, seed=1)   # streamed, in batches
```

**Validate, and act on the findings.** `inspect()` never raises. It reports
everything, and hands back the rows each finding is about:

```python
broken = orders.with_columns(
    pl.when(pl.col("order_id") % 1_000 == 0).then(-1.0).otherwise("total").alias("total")
)
report = Orders.inspect(broken)
clean = report.passing_rows().collect()   # 998,947 rows; quarantine the rest
```

```text
Validation failed for DataFrame against 'Orders' (1 error(s) found):
  - Column 'total': found 1053 value(s) out of bounds [0, 5000] (min found: -1.0, ...)
```

**Start from real data.** Profile a frame you already have, and polspec finds
what it can declare:

```python
from polspec import profile, synthesize

spec = profile(orders.head(100_000), name="Orders")
spec["total"].distribution    # 'lognormal', fitted
spec["email"].format          # 'email'
spec["status"].weights        # (0.20209, 0.50006, 0.29785)

fake = synthesize(orders.head(100_000), seed=1)   # a stand-in with the same shape
```

**Watch for drift.** Compare next month's data with the spec:

```python
next_month = orders.with_columns(
    pl.when(pl.col("status") == "NEW").then(pl.lit("PAID")).otherwise("status")
    .cast(orders.schema["status"]).alias("status")
)
print(Orders.drift(next_month))
```

```text
Drift: 0 breaking, 2 compatible, DataFrame against 'Orders'
  - [compatible] Column 'status': 1 of 3 declared value(s) never appear: ['NEW']
  - [compatible] Column 'status': frequencies moved by 20.0% over 1,000,000 value(s): ...
```

**Use it from the command line** in a pipeline or CI job:

```bash
polspec schema infer orders.csv -o orders_spec.py       # a spec from data
polspec validate orders_spec.py orders.csv --failing rejected.csv
polspec drift orders_spec.py orders.csv --markdown      # for a pull-request comment
polspec generate orders_spec.py -n 1000000 -o fixture.parquet --seed 1
```

Timings are from a laptop, warm. More numbers, and comparisons with NumPy
and hand-written fixtures, are in
[Comparison](https://maxwellb13.github.io/polspec/explanation/comparison/).

## Install

```bash
uv add polspec           # or: pip install polspec
```

polspec works with Polars 1.39 and later, Polars 2 included, on Python 3.12
to 3.14. Wheels are published for Linux (x86_64, aarch64), macOS (Intel
and Apple silicon) and Windows (x86_64), so you don't need a Rust toolchain.

> **Pre-1.0.** The API is settling toward a 1.0 with written stability
> promises. Until then, see
> [Roadmap and stability](https://maxwellb13.github.io/polspec/explanation/roadmap/),
> and [Known limitations](https://maxwellb13.github.io/polspec/explanation/limitations/)
> for what polspec does not do. Each limitation there is backed by a test
> that fails the moment it stops being true.

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
