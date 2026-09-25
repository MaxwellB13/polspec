# Generating data

```python
df = Orders.generate(1_000_000, seed=42)
```

Columns are generated independently and in parallel by the Rust extension, then
cast to their declared dtypes in one Polars pass.

## Reproducibility

A `seed` fixes the result across processes, machines and thread counts. Each
column derives its own seed from the frame seed and its *name*, and each
65,536-row chunk from its index, so the same seed gives the same frame
regardless of how many threads did the work, and adding or reordering columns
never changes the values of the others. The passes that run after the columns
are filled -- rules, the hierarchy, foreign keys, composite uniqueness -- are
seeded the same way, from the frame seed and a key naming what the pass is
for, so a column added beside a ruled or foreign-keyed one leaves it alone
too.

```python
Orders.generate(500, seed=7).equals(Orders.generate(500, seed=7))   # True
```

Omit `seed` and generation is seeded from the clock.

### Renaming a column without changing its data

Because a column's seed comes from its name, renaming a column changes the
values it produces -- which matters when generated frames are snapshots that
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
one cannot borrow another column's name -- they would draw identical values,
which is what a [rule](constraints.md) is for.

`seed_name` covers the column's passes as well as its values: a ruled
column renamed with one keeps its rule's draw. What no seed name holds is
the frame across polspec *versions* -- see
[Roadmap and stability](../explanation/roadmap.md#yaml-format-and-generated-values-may-change).

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
are generated too and dropped again on the way out.

A predicate filters rows that were drawn; it never narrows the draw. `n`
rows are generated and the matching ones kept, because drawing only matching
rows would quietly change what a `null_probability` or a `unique=True`
column means.

Rows arrive in batches, so a scan carries the terms
[batching](#batching) does: a spec declaring a `__hierarchy__` is refused,
and a composite key is distinct within each batch rather than across the
scan; see [Known limitations](../explanation/limitations.md#smaller-sharp-edges).
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

Polars 1 gives a Python source nowhere to put a name, so there the header
stays `PYTHON SCAN` -- polspec asks `register_io_source` what it accepts and
passes the labels only where they are taken.

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
`Orders.generate(n, seed=1)` would, whatever `batch_size` is -- so a stream
written at one batch size and re-read at another is the same data, and the
third batch can be checked against `generate(n).slice(...)`. What is drawn
per batch instead, deterministic but not row for row the whole frame's, is
a column with rules, a foreign key, a composite key, and a `List` column's
elements (its lengths are a window). A `unique=True` column is a window
too: its values are a permutation of its value space, unique across every
batch.

A batch smaller than 65,536 rows -- the engine's chunk -- costs up to one
chunk of extra draws per batch, because a window that starts mid-chunk fills
the chunk from its start and slices the head off. Batches of a chunk or
more cost nothing extra.

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
refuses a spec with one; Parquet and IPC keep every dtype.

A sink is the shorthand; `Orders.scan(n, seed=1).sink_parquet(path)` is the
same write with the rest of a lazy plan available — a `filter`, a `select`,
a `sort` — before it reaches the file.

## Memory

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
orders = Orders.generate(10_000, seed=2, references={Customers: customers})
```

## What generation does not enforce

Generation satisfies dtypes, nullability, bounds, string lengths, value
domains, `format`s, weights, distributions, `unique=True`,
`__unique_together__`, `ColRule`s, hierarchies and — when given parent data —
foreign keys.

It does **not** attempt `ColSpec.validators`, `__checks__` or
`ColSpec.pattern`, by design: the first two hold arbitrary expressions and the
third an arbitrary regex, and nothing can generate data to satisfy either in
general. See [Known limitations](../explanation/limitations.md).
