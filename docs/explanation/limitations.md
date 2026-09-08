# Known limitations

polspec generates data and validates it from one declaration. Where both
sides read the same definition they cannot drift: what values a column may
hold, and the order the passes that rewrite a generated frame run in, both
live in `polspec.constraints`. What is left below is what generation does not
attempt at all, plus a few edges worth knowing about.

Each is pinned by a test in `tests/test_roundtrip.py`. A gap meant to close
one day carries `xfail(strict=True)`: the suite stays green while it exists,
and the moment it is fixed pytest turns the XPASS into a failure. A boundary
that is deliberate is pinned by an ordinary passing test instead. Either way
this page cannot quietly go stale — changing what polspec does forces the test
to be updated.

## Generation does not enforce these

### `__checks__` and `ColSpec.validators` are validation-only

This one is by design, not a defect: both wrap arbitrary Polars expressions,
and nothing can generate data satisfying an arbitrary predicate. Validate
generated data with `validate_checks=False` / `validate_validators=False`, or
construct the rows those invariants describe yourself.

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

### `generate(0, method="cartesian")` is not empty

It emits the whole coverage set, unlike `generate(0)` and the sinks, which all
produce nothing.

## Smaller sharp edges

- **Unsupported dtypes are accepted at declaration.** `ColSpec(pl.List(...))`
  constructs and validates; only `generate()` objects.
- **`missing_cols="add"` can produce a frame that fails re-validation**, since
  columns are added after validation runs, including for non-nullable columns.
- **Rules overwrite nulls**, so a nullable column with a rule ends up with
  fewer nulls than `null_probability` suggests.
- **A `Hierarchy` cannot be batched or streamed.** `generate_batches` and
  every `sink_*` refuse a spec that declares one, because each batch is
  generated independently and a forest is a property of the whole frame.
- **A `Hierarchy` owns both its columns.** A `null_probability`,
  `distribution` or `weights` declared on either is not what you get: the
  references have to come from one pool for the two columns to join at all.
- **Uniqueness holds within a batch, not across one.** `generate_batches` and
  the `sink_*` functions sample each batch independently, so a `unique=True`
  column or a `__unique_together__` group is only distinct inside each batch.
- **A `unique` column ignores `weights` and a non-uniform `distribution`** --
  both are refused at declaration rather than silently dropped, since neither
  has anything to say about a draw without replacement.
- **A foreign key still overwrites its column's distribution.** The parent's
  domain has to fit inside the column's own — a contradiction is refused at
  declaration — but within it, values come from the parent, so a declared
  `distribution` or `weights` on a foreign-keyed column is not what you get.
- **Case-insensitive registry lookup** means a column named `status` binds to a
  registry entry named `STATUS`; entries differing only in case are ambiguous.
- **`to_mermaid` marks every `unique=True` column `PK`**, so several unique
  columns render as several primary keys.
