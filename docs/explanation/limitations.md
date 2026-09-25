# Known limitations

polspec generates data and validates it from one declaration. Where both
sides read the same definition they cannot drift: what values a column may
hold, and the order the passes that rewrite a generated frame run in, both
live in `polspec.constraints`. What is left below is what generation does not
attempt at all, plus a few edges worth knowing about.

Each is pinned by a test in `tests/contracts/test_roundtrip.py`. A gap meant to close
one day carries `xfail(strict=True)`: the suite stays green while it exists,
and the moment it is fixed pytest turns the XPASS into a failure. A boundary
that is deliberate is pinned by an ordinary passing test instead. Either way
this page cannot quietly go stale — changing what polspec does forces the test
to be updated.

## Generation does not enforce these

### `__checks__`, `ColSpec.validators` and `ColSpec.pattern` are validation-only

This one is by design, not a defect: checks and validators wrap arbitrary
Polars expressions, and a pattern is an arbitrary regex; nothing can generate
data satisfying an arbitrary predicate, and generating from an arbitrary
regex has no answer for `.*`. Validate generated data with
`validate_checks=False` / `validate_validators=False` /
`validate_pattern=False`, or construct the rows those invariants describe
yourself. For a shape polspec *can* generate, declare a
[`format`](../how-to/formats.md) instead of a pattern.

### A self-referencing foreign key is referential, not acyclic

A `ForeignKey(..., references="self")` guarantees exactly what it says: every
non-null value in the child column is a value that exists in the referenced
column of the same frame. It does **not** guarantee the result is a tree.

Parents are sampled from the frame as it stands, which builds a random
functional graph — so a row can be its own parent, and two rows can be each
other's. This is not rare:

```python
class Node(FrameSpec):
    Reference = ColSpec(pl.String, unique=True)
    Parent    = ColSpec(pl.String, nullable=True, null_probability=0.2)
    __foreign_keys__ = [
        ForeignKey("Parent", references="self", ref_columns="Reference")
    ]
```

At 20 rows that typically leaves a handful of rows inside a cycle and one or
two pointing at themselves; at 20,000 it is a fraction of a percent. Rare is
not the same as safe — a cycle is exactly what makes a recursive CTE or a
hierarchy walk fail to terminate, and `validate()` will not report one, because
nothing in a spec can currently say "acyclic".

Where you need a genuine hierarchy, declare one. `Hierarchy` is the same two
columns with the shape written down — one parent per reference, a bounded
depth, no cycles — and generation satisfies it rather than leaving it to the
draw:

```python
class Node(FrameSpec):
    Reference = ColSpec(pl.String)
    Parent    = ColSpec(pl.String)
    __hierarchy__ = Hierarchy(child="Reference", parent="Parent", max_depth=5)
```

See [Hierarchies and link tables](../how-to/hierarchies.md), including how to
ask for the cycles back when they are what you are testing against.

## Cartesian generation

### `n` is a minimum, not a count

Under `method="cartesian"`, if the coverage set is larger than `n` all of it is
kept. `generate_batches` and every `sink_*` inherit this, so asking for 5 rows
from two ten-category enums yields 100.

## A `format` promises syntax, not existence

`format="email"` generates a well-formed address, not a deliverable one, and
validates the shape, not whether anything answers. Nothing is looked up on
either side, so `nobody@example.invalid` passes, `hostname` accepts
`localhost`, and `ipv4` accepts `0.0.0.0`. A column that has to hold *real*
identifiers is a `choices` list or a foreign key into the table that owns
them. See [String formats](../how-to/formats.md).

## Smaller sharp edges

- **A `Decimal`, `Int128` or `UInt128` is drawn through 64 bits.**
  Generation fills a Decimal as its physical integer and a 128-bit integer
  as a 64-bit one, so bounds past what 64 bits hold -- eighteen significant
  digits, for a Decimal -- are refused at `generate()`. Validation checks
  the full range.
- **A `Float16` is drawn as a `Float32` and rounded.** Values are drawn
  between the halves nearest each bound from inside, so rounding never
  carries one past a bound; bounds too close together to hold any half are
  refused where they are written. A unique `Float16` is drawn from the
  finite set of halves between its bounds, since distinct singles can round
  to the same half.
- **`missing_cols="add"` can produce a frame that fails re-validation**, since
  columns are added after validation runs, including for non-nullable columns.
- **A `Hierarchy` cannot be batched or streamed.** `generate_batches` and
  every `sink_*` refuse a spec that declares one, because each batch is
  generated independently and a forest is a property of the whole frame.
- **A `Hierarchy` owns both its columns.** A `null_probability`,
  `distribution` or `weights` declared on either is not what you get: the
  references have to come from one pool for the two columns to join at all.
- **A composite key is distinct within a batch, not across batches.** A
  `unique=True` column is a permutation of its value space, unique across a
  whole frame however it is batched -- including one a foreign key fills,
  which takes a permutation of its parent's keys. A `__unique_together__`
  group, and a unique key that references its own spec, are drawn afresh per
  batch of `generate_batches`, a `scan()` or a `sink_*`, so they collide
  across batches only by chance; generate the frame whole when that
  matters.
- **A unique string favours its longest lengths.** It is drawn uniformly
  from every string its lengths allow, and the longest lengths hold nearly
  all of them.
- **A unique column's space must hold every row, nulls included.** Its
  value at a row comes from that row's place in a permutation, and a null
  row spends its place; so a nullable unique column of a thousand rows
  needs a thousand values, not only as many as are drawn.
- **A `unique` column ignores `weights` and a non-uniform `distribution`** --
  both are refused at declaration rather than silently dropped, since neither
  has anything to say about a draw without replacement.
- **A foreign key still overwrites its column's distribution.** The parent's
  domain has to fit inside the column's own — a contradiction is refused at
  declaration — but within it, values come from the parent, so a declared
  `distribution` or `weights` on a foreign-keyed column is not what you get.
