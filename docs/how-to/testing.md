# Testing pipelines with polspec

A test of a pipeline stage does three things: make some input, run the
stage, and check what comes out. A spec does the first and the last — it is
a data source and a schema at once — and polspec names each step:

- **`polspec_frames`**, a pytest fixture, makes seeded input;
- **`assert_valid`** checks the output, failing as a test failure that says
  what broke and shows the rows;
- **`check_pipeline`** does all three in one call.

*Added in 1.3.*

The examples on this page test one pipeline:

```python
from datetime import date

import polars as pl
from polspec import ColSpec, ForeignKey, FrameSpec

class Customers(FrameSpec):
    customer_id = ColSpec(pl.Int64, bounds=(1, 10_000), unique=True)
    tier = pl.Enum(["free", "pro", "enterprise"])
    signed_up = ColSpec(pl.Date, bounds=(date(2020, 1, 1), None))


class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, None), unique=True)
    customer_id = ColSpec(pl.Int64, bounds=(1, 10_000))
    amount = ColSpec(pl.Float64, bounds=(0.0, 500.0))
    __foreign_keys__ = [
        ForeignKey("customer_id", references=Customers, ref_columns="customer_id")
    ]


class CustomerSpend(FrameSpec):
    customer_id = ColSpec(pl.Int64, unique=True)
    tier = pl.Enum(["free", "pro", "enterprise"])
    total_spend = ColSpec(pl.Float64, bounds=(0.0, None))
    order_count = pl.UInt32


def summarize_spend(customers: pl.DataFrame, orders: pl.DataFrame) -> pl.DataFrame:
    """The pipeline under test."""
    return (
        orders.group_by("customer_id")
        .agg(total_spend=pl.col("amount").sum(), order_count=pl.len().cast(pl.UInt32))
        .join(customers.select("customer_id", "tier"), on="customer_id")
        .select("customer_id", "tier", "total_spend", "order_count")
    )
```

## Why this fits hermetic tests

A hermetic test doesn't reach outside itself — no network call, no shared
fixture file that drifts, no "works on my machine" because someone's local
`sample_data.csv` is newer than the one in CI. A generated frame is a pure
function of its spec, its row count and its seed: the same on any machine,
in any process, with any number of threads. There is no file to check in,
and none to go stale. When the pipeline's input schema changes, the change
is in the spec, not in a `.parquet` file nobody remembers generating.

## A frame for a test — `polspec_frames`

polspec installs a pytest plugin, and its `polspec_frames` fixture generates
a frame for the test that asks for it:

```python
from polspec import assert_valid


def test_summarize_spend_keeps_its_contract(polspec_frames):
    customers = polspec_frames(Customers, 200)
    orders = polspec_frames(Orders, 2_000, references={Customers: customers})
    assert_valid(CustomerSpend, summarize_spend(customers, orders))
```

`polspec_frames(spec, n, ...)` is `generate()` with a seed of the test's own,
so it takes everything `generate()` does — `references=`, `fixed=`,
`method="cartesian"`. `references={Customers: customers}` draws each order's
customer from the generated customers, so the join in `summarize_spend`
isn't silently testing against orphaned rows.

Each test's seed comes from the session's seed and the test's own name, and
each frame's from that and its spec: so a test gets the same frames on every
run and every machine, two tests never share theirs, and adding, removing or
reordering tests changes no other test's frames. The session seed is 0:

```bash
pytest --polspec-seed=7        # every test's frames from another seed
pytest --polspec-seed=random   # a new seed each run, named in the header
```

The report header names the seed, and a failing test that used polspec
says how to rerun it on the same frames:

```text
---------------------------- polspec ----------------------------
polspec seed 7 -- rerun with --polspec-seed=7
```

Pass `seed=` to pin one frame yourself. `-p no:polspec` turns the plugin off.

## Checking what comes out — `assert_valid`

`assert_valid(spec, df)` is `validate()` as an assertion. It returns the
validated frame, so it chains, and takes `validate()`'s options. A failure
is an `AssertionError` — a failed test in any runner, not an error — whose
message names the spec, each finding, and the first rows that fail:

```text
CustomerSpend: 1 finding(s) in 187 row(s) (polspec seed 0 -- rerun with --polspec-seed=0)
  - total_spend__bounds: Column 'total_spend': found 3 value(s) out of bounds [0.0, None] ...
First failing rows:
┌─────────────┬──────┬─────────────┬─────────────┬─────────────────────┐
│ customer_id ┆ tier ┆ total_spend ┆ order_count ┆ __polspec_finding   │
╞═════════════╪══════╪═════════════╪═════════════╪═════════════════════╡
│ 4127        ┆ pro  ┆ -12.5       ┆ 3           ┆ total_spend__bounds │
...
```

Validating each stage's output against its own spec catches two kinds of
drift: the pipeline producing the wrong shape, and the test's own
expectations going stale. A pipeline bug that drops a column, or a schema
change nobody updated the test for, surfaces as a finding naming the
column, not as an assertion three functions later.

## A whole pipeline in one call — `check_pipeline`

`check_pipeline` generates the inputs, calls the pipeline, and checks the
result:

```python
from polspec import check_pipeline


def test_summarize_spend_in_one_call():
    result, inputs = check_pipeline(
        summarize_spend,
        inputs={"customers": Customers, "orders": Orders},
        output=CustomerSpend,
        n={"customers": 200, "orders": 2_000},
    )
    assert result.height <= inputs["customers"].height
```

Each input names one of the pipeline's arguments, and is a spec to generate
or a frame to pass as it is. The specs are generated together, so the
foreign key from `Orders` to `Customers` lines up without `references=`.
Under pytest the inputs come from the test's seed, as the fixture's do; it
returns the result and the inputs, for anything the spec cannot assert.

For a pipeline of several stages — raw events into a bronze table, bronze
into silver, silver into gold — the same pattern repeats at each boundary:
a spec per stage, a `ForeignKey` where one stage's identity flows into the
next, and a check between every pair of stages the tests exercise.

## A frame shared by tests

A frame is never changed in place, so one generated per session is safe to
share — an ordinary pytest fixture over `generate()`, with a seed of its
own:

```python
import pytest


@pytest.fixture(scope="session")
def customers() -> pl.DataFrame:
    return Customers.generate(200, seed=1)


@pytest.fixture(scope="session")
def orders(customers: pl.DataFrame) -> pl.DataFrame:
    return Orders.generate(2_000, seed=2, references={Customers: customers})
```

## A stage that must reject bad rows

Generated data is valid by construction, so a test of what a stage does with
*bad* data breaks some rows on purpose. `inspect()` then says exactly which
rows are bad and why, so the test asserts against the spec rather than a
hand-kept list:

```python
def clean_orders(raw: pl.DataFrame) -> pl.DataFrame:
    """The stage under test: drops orders whose amount cannot be right."""
    return raw.filter(pl.col("amount").is_between(0.0, 500.0))


def test_clean_orders_drops_exactly_the_rows_the_spec_rejects(customers, orders):
    every_50th = pl.int_range(pl.len()) % 50 == 0
    raw = orders.with_columns(
        amount=pl.when(every_50th).then(-1.0).otherwise(pl.col("amount"))
    )

    report = Orders.inspect(raw, references={Customers: customers})
    assert {finding.code for finding in report.findings} == {"bounds"}

    kept = clean_orders(raw)
    assert kept.equals(report.passing_rows().collect())
    assert raw.height - kept.height == report.failing_rows(once=True).collect().height
```

`passing_rows()` and `failing_rows(once=True)` split a frame exactly, so they
are also what a stage that quarantines bad rows should produce: see
[Findings as data](validating.md#findings-as-data-inspect). The finding codes
are listed in [Errors and findings](../reference/errors.md#finding-codes).

## Many frames — property tests with `strategy`

A property is a claim about every frame a spec allows — "spend is never
negative, whatever the orders". `polspec.strategy(spec)` gives
[Hypothesis](https://hypothesis.readthedocs.io) frames of a spec to try it
on:

```python
from hypothesis import given, settings
from polspec import strategy

CUSTOMERS = Customers.generate(200, seed=1)


@settings(max_examples=50, deadline=None)
@given(orders=strategy(Orders, max_rows=500, references={Customers: CUSTOMERS}))
def test_spend_is_never_negative(orders):
    assert (summarize_spend(CUSTOMERS, orders)["total_spend"] >= 0).all()
```

Each frame is `generate()` on a row count and a seed Hypothesis draws, so it
meets its spec, follows its distributions, and costs what generating it
costs. A failure shrinks to the fewest rows that still fail; the values come
from the seed and do not shrink one by one. Hypothesis is not a dependency
of polspec: install it to use `strategy`. *Added in 1.3.*

## Edge cases

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


def test_refund_flag_handles_every_sign_and_method_combination(polspec_frames):
    result = refund_flag(polspec_frames(Payment, 50, method="cartesian"))
    assert result.filter(pl.col("amount") < 0)["is_refund"].all()
    assert not result.filter(pl.col("amount") >= 0)["is_refund"].any()
```

Every method now appears alongside a negative amount, a zero amount, a
positive amount, and a null — the sign/null boundary a naive `amount < 0`
check is actually at risk of getting wrong — without hand-writing twelve
rows.

### Forcing a specific case with `fixed=`

Cartesian coverage guarantees signs and combinations exist somewhere in the
frame; it doesn't put a specific value on a specific row. When a test needs an
exact scenario — "a wire transfer of exactly zero" — pin those columns with
`fixed=` and let everything else be generated, instead of filtering generated
rows and hoping one matches:

```python
def test_zero_amount_wire_transfer_is_not_a_refund(polspec_frames):
    df = polspec_frames(Payment, 3, fixed={"method": "wire", "amount": [0, -5, 7]})
    assert refund_flag(df)["is_refund"].to_list() == [False, True, False]
```

A pin is a scalar for every row or exactly `n` values, and it must be values
the column accepts; see [Pinning columns](generating.md#pinning-columns-fixed).

When *every* row of some kind should hold a value, rather than the rows of
one test, say so in the spec with a `ColRule`:

```python
from polspec import ColRule, col

class PaymentWithForcedCase(FrameSpec):
    method = ColSpec(pl.Enum(["card", "wire", "cash"]))
    amount = ColSpec(
        pl.Int64,
        bounds=(-1000, 1000),
        rules=[ColRule(when=col("method") == "wire", choices=[0])],
    )
```

Every `wire` row is forced to `amount = 0`, while `card` and `cash` still
vary normally.

## Large frames and files

A load or performance test doesn't need a large fixture file checked into
version control. `generate_batches` streams rows without holding all of
them in memory, and a `sink_*` writes a large file to pytest's `tmp_path`,
which pytest cleans up:

<!-- docs: skip -->
```python
def test_pipeline_handles_a_million_rows_without_holding_them_all():
    total = 0
    for batch in Customers.generate_batches(1_000_000, batch_size=100_000, seed=1):
        total += process(batch).height
    assert total == 1_000_000


def test_pipeline_reads_a_large_parquet_file(tmp_path):
    path = tmp_path / "customers.parquet"
    Customers.sink_parquet(path, 2_000_000, batch_size=200_000)
    assert pl.scan_parquet(path).select(pl.len()).collect().item() == 2_000_000
```

The file is exactly as large as the test needs, and nothing is committed or
left behind.

## Generating the test boilerplate

The [`polspec test`](cli.md#test-a-round-trip-test-from-a-schema) command
writes a round-trip test for a spec — `polspec_frames`, then `assert_valid` —
generating any parent specs declared in the same file first:

```bash
polspec test orders.yaml -o test_orders.py
```

Point it at a spec written by hand, or one produced by `polspec schema infer`
against a sample of real production data — a fast way to turn "here's what our
data actually looks like" into a schema you can generate more of.

## Fixtures that look like production

When a test needs data that looks like the real thing — its skew, its
category mix, its null rates — without carrying any of it,
[`synthesize`](synthesizing.md) makes a stand-in from a sample:

<!-- docs: skip -->
```python
import polspec

fake = polspec.synthesize("sample.parquet", 10_000, seed=1, replace=["email"])
```

Review the profiled spec it generates from before sharing the fake data, as
that page describes.

## A caveat, not a footnote

A seed gives the same frame on every machine, and from 1.0 a minor release
changes it only to fix output that broke its own spec
([Stability](../explanation/stability.md#seeded-output)). Even so, assert
the properties a test cares about — shape, referential integrity, which rows
a stage keeps — rather than a snapshot of the exact values: a test of what
matters survives a fix, and one of everything does not.
