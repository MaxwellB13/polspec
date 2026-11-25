# Hierarchies and link tables

A link table is an edge list: one column holds a reference, the other holds
the reference it points at, and both draw on the same set of values. A child
pointing at its parent, and that parent pointing at its own parent, are the
same row shape — which is why one table holds both.

`Hierarchy` declares that shape, so `generate()` produces a real tree and
`validate()` can say when data is not one.

## Declaring one

```python
import polars as pl
from polspec import ColSpec, FrameSpec, Hierarchy

class Links(FrameSpec):
    PARENT_REF = ColSpec(pl.String)
    CHILD_REF = ColSpec(pl.String)
    __hierarchy__ = Hierarchy(
        child="CHILD_REF",
        parent="PARENT_REF",
        max_depth=5,
    )

df = Links.generate(1_000_000, seed=7)
```

`child` is the column doing the pointing and `parent` the column being
pointed at. A row is an edge, so `n` is the number of rows. Ultimate parents
have nothing to point at and so have no row of their own.

What that gives you:

- **One parent per reference.** Every value in `CHILD_REF` appears exactly
  once, so walking upward from any row reaches exactly one ultimate parent.
- **A known depth.** No chain is longer than `max_depth` hops, and at least one
  chain is exactly that long — so a test of "resolve to the ultimate parent"
  always exercises the boundary rather than whatever the draw happened to give.
- **No cycles**, unless you ask for them.

`branching` controls the shape — the mean number of children a reference has,
and so how many ultimate parents `n` rows imply. `roots` is the same dial from
the other end: give an exact number of ultimate parents instead, and the
branching follows. They are mutually exclusive, and the default is
`branching=3.0`.

```python
Hierarchy(child="CHILD_REF", parent="PARENT_REF", max_depth=5, roots=40)
```

## Generating data that is deliberately broken

Code that walks a hierarchy has to cope with data that is not one. A loop in
the parent chain is the case that matters, because a resolver written without
a visited set does not fail on it: it runs forever.

```python
Links.generate(1_000_000, seed=7, cycles=10)
Links.generate(1_000_000, seed=7, self_references=5)
```

`cycles=n` closes that many chains into loops, each a few hops long and none
overlapping another. `self_references=n` points that many rows straight at
themselves, which is the degenerate loop your resolver most likely tests for
and the one a `child == parent` guard already catches.

Neither is declared on the spec, because the spec still says the data *should*
be an acyclic forest. That is what makes them useful: `validate()` reports what
was injected, so a test can assert its own resolver and polspec agree.

```python
broken = Links.generate(10_000, seed=7, cycles=3)
report = Links.inspect(broken)

report.passed                                     # False
[f.code for f in report.findings]                 # ['hierarchy_cycle']
report.rows(report.findings[0]).collect().height  # the rows that never terminate
```

## What validation checks

Three findings, on top of everything a column declares for itself:

| code | meaning |
|:--|:--|
| `hierarchy_multi_parent` | a reference appears as a child in more than one row, so it has no single ultimate parent |
| `hierarchy_cycle` | a row's chain of parents never reaches an ultimate parent |
| `hierarchy_depth` | a chain is longer than the declared `max_depth`, without being a loop |

A row inside a loop is reported as a cycle rather than as a depth violation:
it is both, and the loop is the useful half. `hierarchy_cycle` counts every
row that never terminates, which includes rows hanging *below* a loop — they
do not reach an ultimate parent either.

Both checks are bounded. Depth costs `max_depth` steps; cycle detection walks
by pointer doubling, so it covers a chain of a million rows in about twenty.
Validating cyclic data terminates — which matters, because the whole point of
generating it was to have something that breaks a naive walk.

## Resolving the ultimate parent

The reason to generate this data is to test code that collapses it. The same
bounded walk works in Polars:

```python
links = Links.generate(50_000, seed=7)
resolved = links.select(
    node=pl.col("CHILD_REF"), root=pl.col("PARENT_REF")
)
lookup = links.select(
    pl.col("CHILD_REF").alias("root"), pl.col("PARENT_REF").alias("next")
)
for _ in range(5):  # max_depth
    resolved = (
        resolved.join(lookup, on="root", how="left")
        .with_columns(root=pl.coalesce("next", "root"))
        .drop("next")
    )
```

After `max_depth` rounds every `root` is an ultimate parent, because the spec
promised no chain is longer than that. Run the same loop over a frame from
`cycles=10` and it silently returns a value that is not a root — which is
exactly the bug worth having a fixture for.

## Streaming, and the layout

A hierarchy's forest is laid out from each row's position and the seed, so
`scan()`, `generate_batches()` and the `sink_*` functions stream one, and
every batch is the rows `generate()` makes there:

<!-- docs: skip -->
```python
Links.sink_parquet("links.parquet", 50_000_000, seed=1)
for batch in Links.generate_batches(10_000_000, batch_size=500_000, seed=1):
    process(batch)
```

`layout="whole"` builds the forest over the whole frame at once instead, as
2.0 did. It draws the same shape and generates exactly 2.0's frame for a
seed, and the streaming verbs refuse it. Declare it only to keep a frame
2.0 made:

<!-- docs: skip -->
```python
__hierarchy__ = Hierarchy(child="CHILD_REF", parent="PARENT_REF", max_depth=5,
                          layout="whole")
```

`roots=` is exact under the default layout: every root has a child. Under
`layout="whole"` each first-level reference picks its root with
replacement, so a root none picked is not in the data. *Since 2.1, which
made the windowed layout the default and changed a hierarchy's seeded
output once; 2.0 refused to stream one.*

## What it will not do

- **Keep checks.** Generation keeping a spec's checks drops the rows that
  fail, and a kept row's parent may be one that was dropped, so a hierarchy's
  checks are generated unkept, with a warning.
- **The pass owns both columns.** A `null_probability`, `distribution` or
  `weights` on either is not what you get: the references have to come from one
  pool for the two columns to join at all.
- **One parent, not many.** A reference with two parents is a finding, not a
  supported shape.
