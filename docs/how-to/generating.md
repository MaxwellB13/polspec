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
and uniqueness holds within a batch. Leaving `batch_size` unset lets polars
ask for the size it wants; setting it pins the size.

`Registry.scan_all()` is the same for a set of specs: parents are generated
eagerly — a foreign key needs the whole parent column to sample from — and
the children are lazy.

!!! note

    `generate(lazy=True)` is a different thing: it builds the whole frame
    and calls `.lazy()` on it, so the memory is already spent. Reach for
    `scan()` for a frame that has not been built.

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
elements (its lengths are a window). Uniqueness holds only *within* a batch.

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

All four take `batch_size`, `method`, `seed` and `references`, create the
parent directory if needed, and pass extra keyword arguments through to the
underlying writer. Parquet and IPC need PyArrow — `pip install "polspec[arrow]"`.

With `n=0`, Parquet, IPC and CSV still write a valid schema-bearing file.

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
