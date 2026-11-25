# Generating data

The examples on this page generate data from these specs:

```python
from datetime import date

import polars as pl
from polspec import ColSpec, ForeignKey, FrameSpec


class Orders(FrameSpec):
    order_id      = ColSpec(pl.Int64, bounds=(1, 1_000_000), unique=True)
    customer_id   = ColSpec(pl.Int64, bounds=(1, 10_000))
    status        = ColSpec(pl.Enum(["NEW", "PAID", "SHIPPED"]))
    total         = ColSpec(pl.Float64, bounds=(0.0, 5_000.0))
    internal_note = ColSpec(pl.String, string_length=(0, 40), nullable=True)


class OrderLines(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, 1_000_000))
    line_no  = ColSpec(pl.Int32, bounds=(1, 50))
    quantity = ColSpec(pl.UInt16, bounds=(1, 500))
    __unique_together__ = [["order_id", "line_no"]]
    __foreign_keys__ = [ForeignKey("order_id", references=Orders)]
```

```python
df = Orders.generate(1_000_000, seed=42)
print(df.head(3))
```

```text
shape: (3, 5)
┌──────────┬─────────────┬────────┬─────────────┬─────────────────────────────────┐
│ order_id ┆ customer_id ┆ status ┆ total       ┆ internal_note                   │
│ ---      ┆ ---         ┆ ---    ┆ ---         ┆ ---                             │
│ i64      ┆ i64         ┆ enum   ┆ f64         ┆ str                             │
╞══════════╪═════════════╪════════╪═════════════╪═════════════════════════════════╡
│ 232729   ┆ 597         ┆ PAID   ┆ 199.269963  ┆ JCiJA1h7VRfOOJeXCdDqNbTeanNRmS… │
│ 193989   ┆ 8167        ┆ PAID   ┆ 744.84394   ┆ 8hUT6kZbrT7jBdPxw61T2FkNk       │
│ 717045   ┆ 4558        ┆ PAID   ┆ 4410.974884 ┆ UNUxTEtawARpr3bkVXXlQIpM17jCu8… │
└──────────┴─────────────┴────────┴─────────────┴─────────────────────────────────┘
```

Columns are generated independently and in parallel by the Rust extension, then
cast to their declared dtypes in one Polars pass. Every value meets its
column's claims — `order_id` unique, `total` within its bounds — but a
column with no further claim, like `internal_note`, is only as realistic as
its declaration: text of the declared lengths. A
[`format`](formats.md) or [`pattern`](formats.md#patterns) says what a value
looks like.

## Reproducibility

A `seed` fixes the result across processes, machines and thread counts. Each
column derives its own seed from the frame seed and its *name*, and each
4,096-row chunk from its index, so the same seed gives the same frame
regardless of how many threads did the work, and adding or reordering columns
never changes the values of the others. A column's nulls are drawn apart from
its values, so changing its `null_probability` moves which rows are null and
leaves the values of the rest where they were. The passes that run after the columns
are filled — rules, the hierarchy, foreign keys, composite uniqueness — are
seeded the same way, from the frame seed and a key naming what the pass is
for, so a column added beside a ruled or foreign-keyed one leaves it alone
too.

```python
Orders.generate(500, seed=7).equals(Orders.generate(500, seed=7))   # True
```

Omit `seed` and generation is seeded from the clock.

### Renaming a column without changing its data

Because a column's seed comes from its name, renaming a column changes the
values it produces — which matters when generated frames are snapshots that
other things are compared against. `seed_name` is the name the seed is
derived from when it is not the column's own, so a renamed column keeps
producing the data it did:

```python
class Before(FrameSpec):
    id     = ColSpec(pl.Int64, unique=True)
    status = ColSpec(pl.String, nullable=True)

class After(FrameSpec):
    id    = ColSpec(pl.Int64, unique=True)
    state = ColSpec(pl.String, nullable=True, seed_name="status")   # renamed

before = Before.generate(100, seed=1)
after = After.generate(100, seed=1)
after["state"].equals(before["status"])   # True
```

A rename is declared, never guessed: `TableSpec.rename()` does not set
`seed_name`, and [`diff(renames=)`](drift.md) is told about the same rename
on the validation side. Two columns of one spec cannot share a seed name, and
one cannot borrow another column's name — they would draw identical values,
which is what a [rule](constraints.md) is for.

`seed_name` covers the column's passes as well as its values: a ruled
column renamed with one keeps its rule's draw. What no seed name holds is
the frame across polspec *versions* — see
[Stability](../explanation/stability.md#seeded-output).

## Pinning columns — `fixed=`

`fixed=` sets columns to values you give, and generates the rest: a scalar
for every row, or a sequence or Series of exactly `n`.

*Added in 1.1.*

```python
df = Orders.generate(3, seed=1, fixed={"status": ["NEW", "PAID", "PAID"]})
df["status"].to_list()   # ['NEW', 'PAID', 'PAID']
```

The pins replace their columns after the draw and before the
[passes](#what-generation-does-not-enforce), so a `ColRule` keyed on a
pinned column follows it: pin `country` to `"UK"` and a currency rule on it
gives `"GBP"`. No other column moves, since each is seeded by its own name.

A pin is held to the promise the rest of `generate()` makes, that its output
validates:

- Its values must be ones the column accepts — dtype, nullability, bounds,
  choices, format, pattern, lengths, `unique` and the column's validators.
  Values are cast strictly, so `1.5` on an `Int64` is refused rather than
  rounded.
- A column a pass writes — a rule's, a foreign key's, a hierarchy's, a
  composite key's, a derived column — cannot be pinned; the error names the
  pass. A column a pass only reads can, and a derived column reading it
  follows the pin.

Each refusal is a `GenerationError` naming the column. With `order_by`,
pinned rows are sorted like any other. `method="cartesian"` takes no pins,
since its row count is not `n`.

### Pins in batches, scans, sinks and registries

`generate_batches`, `scan`, every `sink_*`, and a registry's
`generate_all`, `generate_related` and `scan_all` take `fixed=` too, with
the same rules. *Added in 1.4.*

```python
batches = Orders.generate_batches(30_000, batch_size=10_000, seed=1, fixed={"status": "PAID"})
lazy = Orders.scan(1_000_000, seed=1, fixed={"status": "PAID"})
```

A batch is a window onto the frame `generate()` makes, pins included: a
scalar fills every batch, and a sequence of all `n` values gives each batch
its own stretch of them. A scan checks its pins when it is built, before
anything is collected, and a pin on a column the plan never reads is
checked and then not drawn. A registry takes pins spec by spec —
`{Customers: {"tier": "pro"}}`, keyed by the spec, its class or its name —
and a child's foreign key draws from its pinned parent.

The command line takes `--fixed COLUMN=VALUE`, repeated for several pins,
each value read as its column's dtype: see
[`generate`](cli.md#generate-data-from-a-schema).

## Keeping checks

`__checks__` and column validators hold arbitrary expressions, so a
free draw can break them. Generation keeps them: the frame is read as
always, the rows every check passes are kept, and when fewer than `n`
pass, more of the same frame is read — only the rows past those already
read. *Kept by default since 2.0; `checks=True` added in 1.7.*

```python
from polspec import Check, col


class Invoices(FrameSpec):
    subtotal = ColSpec(pl.Float64, bounds=(0.0, 100.0))
    total    = ColSpec(pl.Float64, bounds=(0.0, 110.0))
    __checks__ = [Check(col("total") >= col("subtotal"), name="total_covers_subtotal")]


invoices = Invoices.generate(1_000, seed=1)
Invoices.inspect(invoices).passed   # True
```

Every row kept is a row of the one frame the seed describes, so:

- A frame whose rows all pass is the frame `checks=False` gives.
- A `unique` column stays unique and a foreign key still points at its
  parent. `generate_batches`, `scan`, the sinks and a registry's
  `generate_all`, `generate_related` and `scan_all` keep them too, and
  each batch reads on from where the last stopped, so uniqueness holds
  across batches.
- A rate a check reads becomes a rate among the rows that pass: keep
  `total >= subtotal` and fewer high subtotals are left.

Some checks cannot be kept: under `method="cartesian"`, whose coverage
set filtering would break; beside a `__hierarchy__` or a key into the
spec's own rows, which link rows to rows that filtering removes; with a
`fixed=` pin of all `n` values (a single-value pin works); and a check
that rejects nearly every row, so that more than fifty times `n` rows
would be needed. Then the frame is the one `checks=False` gives, and a
warning says why. `checks=False` asks for that frame and says nothing;
`checks=True` asks for the checks kept and refuses where they cannot be:
a `SpecError` naming why, or for a check that rejects nearly every row,
a `GenerationError`
naming each check and how many rows it rejected. A spec with no
checks is generated as it always is.

## Lazy output — `scan()`

`scan()` returns a `pl.LazyFrame` that has not been generated. Nothing is
drawn until the plan is collected, and then only the columns and rows the
plan asks for:

<!-- docs: skip -->
```python
lf = Orders.scan(50_000_000, seed=1)

lf.sink_parquet("orders.parquet")               # streams; bounded memory
lf.select("total").head(5).collect()            # five rows of one column
lf.filter(pl.col("status") == "PAID").collect() # every row drawn, matching kept
```

**Projecting cannot change what a column holds.** Every column is seeded by
its name and every pass by what it is for, so dropping a column's
neighbours leaves it alone — `lf.select(cols).collect()` is always
`lf.collect().select(cols)`. Where a column depends on others (a rule reads
the columns its `when` names, a composite key is repaired as a group), those
are generated too and dropped again on the way out. Under
`method="cartesian"` every column is generated whatever the projection, since
the coverage set is the product of all of them.

A predicate filters rows that were drawn; it never narrows the draw. `n`
rows are generated and the matching ones kept, because drawing only matching
rows would quietly change what a `null_probability` or a `unique=True`
column means.

Rows arrive in batches, so a scan carries the terms
[batching](#batching) does: a hierarchy declared `layout="whole"` is
refused, and a composite key that is repaired, not permuted, is distinct
within each batch rather than across the scan; see
[Composite uniqueness](constraints.md#composite-uniqueness-__unique_together__).
Leaving `batch_size` unset lets polars ask for the size it wants; setting it
pins the size.

**Reading `explain()` for a scan.** The generator appears in a query plan as
a Python scan, with the columns and the filter polars handed it underneath:

```
SLICE[offset: 0, len: 5]
  PYTHON SCAN []
  PROJECT 1/18 COLUMNS
  SELECTION: col("int1") > 0
```

`PROJECT 1/18 COLUMNS` says one of the spec's eighteen columns is
generated; `SELECTION` is the predicate applied to the rows drawn; `*` in
place of a count means every column.

On Polars 2 the scan names itself, and says what it was asked for:

```
PYTHON[polspec: Orders] SCAN []
PROJECT 1/18 COLUMNS
INFO: 1,000,000 rows, seed=100, batches chosen by polars
```

`Registry.scan_all()` is the same for a set of specs: parents are generated
eagerly — a foreign key needs the whole parent column to sample from — and
the children are lazy.

!!! note

    `generate()` builds the whole frame before it returns, so `.lazy()` on
    its result is a handle on memory already spent. `scan()` is the frame
    that has not been built. (`generate(lazy=True)` did the former while
    looking like the latter, and was removed in 0.9.0.)

## Coverage — `method="cartesian"`

The default `method="random"` draws each column independently, so a rare enum
value may not appear at all. `method="cartesian"` guarantees it will:

```python
df = Orders.generate(500, method="cartesian", seed=1)
```

It builds the cross-product of every finite domain — each `Enum`'s categories,
both booleans, and the negative / zero / positive / null partitions of every
bounded numeric column — so every combination is present at least once.
Columns with no finite domain (String, bare `Categorical`) are filled in
randomly alongside.

!!! warning "`n` is a minimum here, not a count"

    If the coverage set is smaller than `n` it is padded with random rows. If
    it is **larger**, all of it is kept and `n` is exceeded. Two ten-category
    enums produce 100 rows however small `n` was.

    A safety cap refuses to build more than 50 million coverage rows, naming
    each dimension's cardinality so you can see which one exploded.

## Batching

For volumes that should not be held in memory at once:

<!-- docs: skip -->
```python
for batch in Orders.generate_batches(10_000_000, batch_size=250_000, seed=1):
    process(batch)
```

Each batch is a **window onto the one frame the seed describes**: a column
no pass rewrites holds, batch by batch, exactly the rows
`Orders.generate(n, seed=1)` would, whatever `batch_size` is — so a stream
written at one batch size and re-read at another is the same data, and the
third batch can be checked against `generate(n).slice(...)`. A `List`,
`Array`, `Map` or nested `Struct` column is a window too, elements and all,
and a `unique=True` column, whose values are a permutation of its value
space, unique across every batch, and so is a column rules or a foreign
key into another table rewrite, a composite key whose members have a
finite or bounded space, a hierarchy, and a key into the spec's own rows
whose referenced column is unique with no nulls. What is drawn per batch
instead, deterministic but not row for row the whole frame's, is any other
composite key or key into the spec's own rows. A hierarchy declared
`layout="whole"` is refused. *Nested columns, rules, foreign keys and
composite keys are windows since 2.0; a hierarchy and a key into the
spec's own rows since 2.1.*

A small batch costs little. The columns a window draws are filled many
batches at a time, up to about 64 MiB, in parallel, and each batch is a
slice of that: a million rows in batches of 1,000 take about twice as long
as one `generate()` of them, a `List`, `Array` or `Map` column among
them. *Faster since 1.6; nested columns too since 2.0.*

## Writing straight to a file

Four sinks stream batches to disk without materialising the whole frame:

<!-- docs: skip -->
```python
Orders.sink_parquet("orders.parquet", 50_000_000, compression="zstd")
Orders.sink_csv("orders.csv", 1_000_000)
Orders.sink_ipc("orders.arrow", 1_000_000, compression="zstd")
Orders.sink_ndjson("orders.ndjson", 1_000_000)
```

Each is [`scan()`](#lazy-output-scan) handed to the matching
`LazyFrame.sink_*`, so what a sink writes is what collecting the scan gives.
All four take `batch_size`, `method`, `seed` and `references`, create the
parent directory if needed, and pass extra keyword arguments through to
polars' own sink. Nothing beyond Polars is needed for any of them.

With `n=0`, Parquet, IPC and CSV still write a valid schema-bearing file.
A CSV cannot hold a `Duration`, `List` or `Struct` column, so `sink_csv`
refuses a spec with one; Parquet and IPC keep every dtype. JSON holds
categories, decimals and durations as text, which
[`read()`](validating.md) reads back as declared; it has no bytes, so
`sink_ndjson` refuses a `Binary` column, and no NaN, which it writes as
`null`.

A sink is the shorthand; `Orders.scan(n, seed=1).sink_parquet(path)` is the
same write with the rest of a lazy plan available — a `filter`, a `select`,
a `sort` — before it reaches the file.

## Memory

[Performance](performance.md) gathers what each way of generating costs,
in time and memory.

`generate()` allocates the whole frame before it returns, so it is worth
knowing what that is before asking for it:

```python
Orders.estimated_size(50_000_000) / 1024**3     # gibibytes
```

Read off the declaration — the width of each dtype, the lengths the spec
declares — so it costs nothing and needs no data. Past four gibibytes
`generate()` says so in a warning naming the estimate; `max_bytes=` makes
it a refusal instead, for a CI job that should fail rather than swap, and
`max_bytes=0` silences both.

What each column costs per row:

| Declared | Bytes per row |
|:--|--:|
| `Int8`/`UInt8` … `Int64`/`Float64` | 1 … 8 |
| `Boolean` | ⅛ |
| `Date` | 4; `Time`, `Datetime`, `Duration` | 8 |
| `Decimal` | 16 |
| `Enum` | 1, 2 or 4 — the narrowest that holds the categories |
| `Categorical` | its registry's physical width, 4 by default |
| `String`, `Binary` | **16**, plus the length of any value past 12 bytes, on the rows that are not null |
| `List(inner)` | 8, plus the mean `list_length` × the element's cost, on the rows that are not null |
| `Array(inner, w)` | `w` × the element's cost |
| `Struct` | the sum of its fields, each costed as a column |
| `nullable=True` | + ⅛ |

The sixteen bytes a text value costs before any content is the lever worth
knowing: a low-cardinality string column declared as `pl.Enum([...])` costs
**one** byte per row instead of twenty, and one drawn from `choices` costs
the sixteen but not the content, because the values are gathered from one
shared buffer.

The estimate is the frame, not the process. Generation holds working
buffers on top — most visibly for `Decimal` and `List`, assembled in Polars
rather than filled by the engine — so a peak is higher. For a frame of
scalar columns the two agree within a percent.

[`scan()`](#lazy-output-scan) and [batching](#batching) are the way out of
the question entirely: both hold a batch at a time rather than the frame.

## Foreign keys

`references` maps a parent spec to its data, and makes generated keys
referentially consistent. See [Constraints](constraints.md#referential-integrity-foreignkey).

```python
orders = Orders.generate(10_000, seed=2)
order_lines = OrderLines.generate(30_000, seed=3, references={Orders: orders})
```

## What generation does not enforce

Generation satisfies dtypes, nullability, bounds, string lengths, value
domains, `format`s, `pattern`s, weights, distributions, `unique=True`,
`__unique_together__`, `ColRule`s, derived columns, hierarchies and — when
given parent data — foreign keys.

`ColSpec.validators` and `__checks__` hold arbitrary expressions, which
nothing can draw data to satisfy in general; they are
[kept](#keeping-checks) by keeping the rows that pass them. A `pattern` the generator cannot follow is filled with
plain text, and says so. See [Known limitations](../explanation/limitations.md).
