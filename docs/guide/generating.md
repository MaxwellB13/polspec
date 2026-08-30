# Generating data

```python
df = Orders.generate(1_000_000, seed=42)
```

Columns are generated independently and in parallel by the Rust extension, then
cast to their declared dtypes in one Polars pass.

## Reproducibility

A `seed` fixes the result across processes, machines and thread counts. Each
column derives its own seed from its position, and each 65,536-row chunk from
its index, so the same seed gives the same frame regardless of how many threads
did the work.

```python
Orders.generate(500, seed=7).equals(Orders.generate(500, seed=7))   # True
```

Omit `seed` and generation is seeded from the clock.

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

```python
for batch in Orders.generate_batches(10_000_000, batch_size=250_000, seed=1):
    process(batch)
```

Each batch is generated independently, so a `unique=True` foreign key column is
sampled without replacement only *within* a batch.

## Writing straight to a file

Four sinks stream batches to disk without materialising the whole frame:

```python
Orders.sink_parquet("orders.parquet", 50_000_000, compression="zstd")
Orders.sink_csv("orders.csv", 1_000_000)
Orders.sink_ipc("orders.arrow", 1_000_000, compression="zstd")
Orders.sink_ndjson("orders.ndjson", 1_000_000)
```

All four take `batch_size`, `method`, `seed` and `references`, create the
parent directory if needed, and pass extra keyword arguments through to the
underlying writer. Parquet and IPC need PyArrow — `pip install "polspec[all]"`.

With `n=0`, Parquet, IPC and CSV still write a valid schema-bearing file.

## Foreign keys

`references` maps a parent spec to its data, and makes generated keys
referentially consistent. See [Constraints](constraints.md#referential-integrity-foreignkey).

```python
orders = Orders.generate(10_000, seed=2, references={Customers: customers})
```

## What generation does not enforce

Generation satisfies dtypes, nullability, bounds, string lengths, value
domains, weights, distributions, `ColRule`s and — when given parent data —
foreign keys.

It does **not** attempt `unique=True`, `__unique_together__`,
`ColSpec.validators` or `__checks__`. The last two are impossible in general
(they hold arbitrary expressions); the first two are open work. See
[Known limitations](../reference/limitations.md).
