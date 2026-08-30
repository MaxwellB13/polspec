# Testing pipelines with polspec

A spec is a schema and a data source at once, which makes it a natural fit
for the tests around a data pipeline: declare what a stage of the pipeline
expects, generate data that matches, and validate what it produces.

## Why this fits hermetic tests

A hermetic test doesn't reach outside itself — no network call, no shared
fixture file that drifts, no "works on my machine" because someone's local
`sample_data.csv` is newer than the one in CI. `FrameSpec.generate(n, seed=...)`
is a pure function of its arguments: the same seed produces the same frame on
any machine, in any process, with any number of threads. There's no file to
check into the repo, and no file to go stale.

```python
class Customers(FrameSpec):
    customer_id = ColSpec(pl.Int64, bounds=(1, 10_000))
    tier = ColSpec(pl.Enum(["free", "pro", "enterprise"]))
    signed_up = ColSpec(pl.Date, bounds=(date(2020, 1, 1), None))


def test_pipeline_handles_all_tiers():
    df = Customers.generate(500, seed=42)
    assert set(df["tier"].unique()) <= {"free", "pro", "enterprise"}
```

The spec is the fixture. When the pipeline's input schema changes, the type
error is in the `ColSpec` declaration, not in a `.parquet` file nobody
remembers generating.

## Testing a full pipeline

Declare the shape of each stage — including the *output* — and validate the
real function against it. This catches two different kinds of drift: the
pipeline producing the wrong shape, and the test's own expectations going
stale.

```python
class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, None))
    customer_id = ColSpec(pl.Int64, bounds=(1, 10_000))
    amount = ColSpec(pl.Float64, bounds=(0.0, 500.0))
    __foreign_keys__ = [
        ForeignKey("customer_id", references=Customers, ref_columns="customer_id")
    ]


class CustomerSpend(FrameSpec):
    customer_id = ColSpec(pl.Int64, bounds=(1, 10_000))
    tier = ColSpec(pl.Enum(["free", "pro", "enterprise"]))
    total_spend = ColSpec(pl.Float64, bounds=(0.0, None))
    order_count = ColSpec(pl.UInt32)


def summarize_spend(customers: pl.DataFrame, orders: pl.DataFrame) -> pl.DataFrame:
    """The pipeline under test."""
    return (
        orders.group_by("customer_id")
        .agg(
            total_spend=pl.col("amount").sum(),
            order_count=pl.len().cast(pl.UInt32),
        )
        .join(customers.select("customer_id", "tier"), on="customer_id", how="inner")
        .select("customer_id", "tier", "total_spend", "order_count")
    )


def test_summarize_spend_matches_declared_output_shape():
    customers = Customers.generate(200, seed=1)
    orders = Orders.generate(2_000, seed=2, references={Customers: customers})
    Orders.validate(orders, references={Customers: customers})

    result = summarize_spend(customers, orders)
    CustomerSpend.validate(result, extra_cols="allow", missing_cols="allow")
```

`references={Customers: customers}` makes `orders.customer_id` referentially
consistent with the generated `customers` frame, so the join in
`summarize_spend` isn't silently testing against orphaned rows. Validating
the *input* and the *output* against separate specs means a pipeline bug that
drops a column, or a schema change nobody updated the test for, both surface
as a specific, readable `ValidationError` rather than a downstream assertion
failure three functions later.

For a pipeline with more than two stages — raw events into a bronze table,
bronze into a cleaned silver table, silver into an aggregated gold table — the
same pattern repeats at each boundary: a `FrameSpec` per stage, a
`ForeignKey` where one stage's identity flows into the next, `validate()`
between every pair of stages the tests actually exercise.

## Large dataframes and files

Generating a realistic volume of data for a load or performance test doesn't
need a large fixture file checked into version control. `generate_batches`
streams rows without holding all of them in memory:

```python
def test_pipeline_handles_a_million_rows_without_holding_them_all():
    total = 0
    for batch in Customers.generate_batches(1_000_000, batch_size=100_000, seed=1):
        total += process(batch).height
    assert total == 1_000_000
```

For a pipeline stage that specifically reads from a file — a `scan_parquet`
step, an ingestion job watching a directory — `sink_*` writes a large file to
a `tmp_path`, which pytest cleans up automatically:

```python
def test_pipeline_reads_a_large_parquet_file(tmp_path):
    path = tmp_path / "customers.parquet"
    Customers.sink_parquet(path, 2_000_000, batch_size=200_000)

    result = pl.scan_parquet(path).select(pl.len()).collect().item()
    assert result == 2_000_000
```

Nothing here is committed to the repository, nothing needs cleaning up by
hand, and the file is exactly as large as the test needs — a different test
asking for 50,000,000 rows costs nothing to write.

## Edge-case testing

### Guaranteed coverage with `method="cartesian"`

Random generation might never happen to produce a negative amount paired with
a particular payment method in 50 rows. `method="cartesian"` guarantees every
combination of each `Enum`/`Boolean` value with the negative/zero/positive/null
partitions of every bounded numeric column appears at least once:

```python
class Payment(FrameSpec):
    method = ColSpec(pl.Enum(["card", "wire", "cash"]))
    amount = ColSpec(pl.Int64, bounds=(-1000, 1000), nullable=True)


def refund_flag(df: pl.DataFrame) -> pl.DataFrame:
    """The pipeline under test: refunds are negative amounts."""
    return df.with_columns(is_refund=pl.col("amount") < 0)


def test_refund_flag_handles_every_sign_and_method_combination():
    edge_cases = Payment.generate(50, method="cartesian", seed=1)
    result = refund_flag(edge_cases)
    assert result.filter(pl.col("amount") < 0)["is_refund"].all()
    assert not result.filter(pl.col("amount") >= 0)["is_refund"].any()
```

Every method now appears alongside a negative amount, a zero amount, a
positive amount, and a null — the sign/null boundary a naive `amount < 0`
check is actually at risk of getting wrong — without hand-writing sixteen
rows.

### Forcing a specific case with `ColRule`

Cartesian coverage guarantees signs and combinations exist somewhere in the
frame; it doesn't put a specific value on a specific row. When a test needs an
exact scenario — "a wire transfer of exactly zero, paired with this other
column's exact value" — a `ColRule` pins it deterministically instead of
filtering generated rows and hoping one matches:

```python
class PaymentWithForcedCase(FrameSpec):
    method = ColSpec(pl.Enum(["card", "wire", "cash"]))
    amount = ColSpec(
        pl.Int64,
        bounds=(-1000, 1000),
        rules=[ColRule(when={"column": "method", "equals": "wire"}, choices=[0])],
    )


def test_zero_amount_wire_transfer_is_not_a_refund():
    df = PaymentWithForcedCase.generate(20, seed=1)
    result = refund_flag(df)
    assert not result.filter(pl.col("method") == "wire")["is_refund"].any()
```

Every `wire` row is forced to `amount = 0`, while `card` and `cash` still vary
normally — useful for a boundary the pipeline treats specially and cartesian
coverage alone wouldn't reliably isolate.

## Generating the test boilerplate

The [`polspec test`](cli.md) command builds the round-trip skeleton for a
schema automatically:

```bash
polspec test orders.yaml -o test_orders.py
```

Point it at a spec written by hand, or one produced by `polspec schema infer`
against a sample of real production data — a fast way to turn "here's what our
data actually looks like" into a schema you can generate more of.

## A caveat, not a footnote

polspec is early alpha — see [Roadmap and stability](../reference/roadmap.md).
Tests built on it today are exercising real, useful properties (shape,
referential integrity, boundary coverage), but the exact values a given seed
produces are not guaranteed to survive a polspec upgrade. Pin a seed for
*reproducibility within a test run*, not as an assertion baked into a snapshot
that expects byte-identical output after you bump the version.
