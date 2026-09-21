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
never changes the values of the others.

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

That is all `seed_name` holds. The passes a frame goes through after its
columns are filled -- rules, hierarchy, foreign keys, composite uniqueness --
draw their seeds in declaration order, so inserting a column that carries a
rule ahead of another still changes what the second one draws, and nothing is
promised across polspec versions. See
[Known limitations](../explanation/limitations.md).

## Lazy output

```python
lf = Orders.generate(1_000, seed=1, lazy=True)   # pl.LazyFrame
```

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

Each batch is generated independently, so a `unique=True` foreign key column is
sampled without replacement only *within* a batch.

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
