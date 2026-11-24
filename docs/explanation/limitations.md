# Known limitations

polspec generates data and validates it from one declaration, and the two
directions are not symmetric. A validator only has to *recognise* a
violation, which a Polars expression can do for almost any claim; a
generator has to *avoid* one, which is easy only for a claim with a shape a
sampler can use. `bounds=(1, 100)` has that shape;
`pl.col("email").str.contains("@")` does not — though `format="email"`,
which says the same thing in a sampler's terms, does.

Where both sides read the same definition they cannot drift: what values a
column may hold, and the order in which rules, keys and hierarchies rewrite
a generated frame, are each defined once and read by both. A declaration
whose claims contradict each other — a foreign key whose parent cannot fit
its column, two rules that each read the other — is refused when it is
written, not when it runs. What is left below is what generation does not
attempt at all, plus a few edges worth knowing about.

Each is pinned by a test in polspec's suite. A gap meant to close
one day carries `xfail(strict=True)`: the suite stays green while it exists,
and the moment it is fixed pytest turns the XPASS into a failure. A boundary
that is deliberate is pinned by an ordinary passing test instead. Either way
this page cannot quietly go stale — changing what polspec does forces the test
to be updated.

## Generation does not enforce these

### `__checks__` and `ColSpec.validators` are kept by filtering

Checks and validators wrap arbitrary Polars expressions, and nothing can
draw data satisfying an arbitrary predicate. Generation keeps them by
keeping the rows that pass and reading on for more (*by default since
2.0*). A check rejecting nearly every row defeats that, and so do
`method="cartesian"`, a hierarchy, a key into the spec's own rows, and a
pin of all `n` values: the frame is then generated without them, with a
warning, and `checks=True` refuses instead.
See [Keeping checks](../how-to/generating.md#keeping-checks).

### A pattern the generator cannot follow

A `pattern` is generated from (*since 1.4*), except when its automaton
cannot be walked: it uses a Unicode word boundary (`\b`; the ASCII
`(?-u:\b)` works), its DFA is larger than the generator builds, or it has
no match within the column's `string_length`. That column is filled with
plain random text, as before 1.4, and generation warns, naming the column
and the reason. A pattern that every plain string already passes is
filled with plain text too, on purpose: its seeded output stays what it
was. See [Patterns](../how-to/formats.md#patterns).

### `rows` does not limit `generate()`

This is by design too. A spec's `rows` describes the real table, and generating
a small frame from it is the normal use of `generate()`, so `n` wins: the rows
asked for are made, and `validate()` reports `row_count` if they are outside
`rows`. Validate such a frame with `rows=False`.

### `fixed=` pins scalar columns, and not under `cartesian`

`method="cartesian"` refuses `fixed=`, since its row count is the coverage
it needs rather than `n`. Only a scalar column can be pinned, and a column a
pass writes — a rule's, a key's, a derived column — cannot be.

### A self-referencing foreign key is referential, not acyclic

A `ForeignKey(..., references="self")` guarantees exactly what it says: every
non-null value in the child column is a value that exists in the referenced
column of the same frame. It does **not** guarantee the result is a tree.

Parents are sampled from the frame as it stands, which builds a random
functional graph — so a row can be its own parent, and two rows can be each
other's. This is not rare:

```python
import polars as pl
from polspec import ColSpec, ForeignKey, FrameSpec, Hierarchy

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

## Synthesized data is related only by its keys

`polspec.synthesize` profiles each column on its own and generates each on
its own, so every column of the fake frame looks like the source's and no
relationship between columns survives: a correlation, a conditional mix.
`polspec.synthesize_all` keeps the foreign keys between tables, found from
the values or declared, so every fake child row has a fake parent. Nothing
else between tables survives: how many children a parent has, or how a
child's values follow its parent's. A key is looked for on single columns
only: a composite key, or a hierarchy within a table, is never inferred.
A text column's format is recognised only among the formats
polspec knows, and from at least twenty distinct values; a code — an order
reference, a postcode — is written as the pattern its values share when
they have at most three shapes, and anything else, free text above all,
becomes text of its lengths. See
[Fake data from real data](../how-to/synthesizing.md#how-close-the-fake-data-is).

### A float's infinities are outside its profiled bounds

A bound cannot be infinite, and an infinity is a matter of bounds — an
open end admits one, a closed end refuses it. So `profile()`,
`from_dataframe()`, `polspec schema infer` and `polspec synthesize` take a
float column's bounds from its finite values, and warn once per column that
holds an infinity, saying how many. Validating the source against its own
profile then reports them as `bounds` findings, and a synthesized frame
holds none; open the bound on that side (`None`) where infinities belong.
A column with no finite value at all gets no bounds. This is deliberate, and
pinned by a passing test. A NaN, by contrast, is recorded as the column's
`nan_probability` and carried into the fake data.

## Smaller sharp edges

- **Specs know about each other only through a `Registry`.** A
  `ForeignKey` names the spec it points at, and nothing above one spec
  knows which specs exist unless they are put in a
  [`Registry`](../how-to/registry.md) — two test modules may each define
  an `Orders`. `Registry.discover("specs/")` imports every `.py` file it
  finds, so a declared `Registry(...)` is the safer shape; shared
  categories are checked against a `CatSpec` only when one is given; and a
  single spec's `to_mermaid()` draws one entity, where
  `registry.to_mermaid()` draws them all.
- **A `Float16` is drawn as a `Float32` and rounded.** Values are drawn
  between the halves nearest each bound from inside, so rounding never
  carries one past a bound; bounds too close together to hold any half are
  refused where they are written. A unique `Float16` is drawn from the
  finite set of halves between its bounds, since distinct singles can round
  to the same half.
- **`missing_cols="add"` can produce a frame that fails re-validation**, since
  columns are added after validation runs, including for non-nullable columns.
  A column with a `default` is added holding it; one without is added as
  nulls. `fill_defaults=True` fills before validation instead.
- **A `List`, `Array`, `Map` or `Struct` column takes no `default`.** A
  default is one scalar value; a struct field takes none either.
- **A `Hierarchy` cannot be batched or streamed.** `generate_batches` and
  every `sink_*` refuse a spec that declares one, because each batch is
  generated independently and a forest is a property of the whole frame.
- **An ordered spec cannot be batched or streamed either.** A spec with
  `__order_by__` is refused by `generate_batches`, `scan` and every
  `sink_*`, for the same reason: the rows' order is a property of the whole
  frame. Generate it with `generate()` and write that frame out.
- **A `Hierarchy` owns both its columns.** A `null_probability`,
  `distribution` or `weights` declared on either is not what you get: the
  references have to come from one pool for the two columns to join at all.
  So `drift()` does not hold either column to them.
- **Some composite keys are distinct within a batch, not across
  batches.** A `__unique_together__` group whose members all have a finite
  or bounded space is one permutation of their combinations, distinct across
  the whole frame however it is batched, at the cost of its members' weights
  and distributions. A group with a float or free-text member, or one a
  foreign key fills, is repaired per batch of `generate_batches`, a `scan()`
  or a `sink_*` instead, and so is a unique key that references its own
  spec: they collide across batches only by chance; generate the frame whole
  when that matters.
- **A `unique` column ignores `weights` and a non-uniform `distribution`** —
  both are refused at declaration rather than silently dropped, since neither
  has anything to say about a draw without replacement.
- **A foreign key still overwrites its column's distribution.** The parent's
  domain has to fit inside the column's own — a contradiction is refused at
  declaration — but within it, values come from the parent, so a declared
  `distribution` or `weights` on a foreign-keyed column is not what you get.
