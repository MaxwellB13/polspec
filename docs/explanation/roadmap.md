# Roadmap and stability

!!! warning "Early alpha"

    polspec is early. The sections below are the honest version of "what's
    next" — not a promise of when, just where the rough edges are and which
    direction they're likely to move. Treat everything here, and everything
    the library produces, as breakable between versions until it says
    otherwise.

## Dtype coverage is complete

polspec generates every Polars dtype that holds data — integers to 128
bits, floats from half to double precision,
`Decimal`, booleans, strings, binary, `Date`/`Time`/`Datetime`/`Duration`,
`Enum`, `Categorical`, and `List`, `Array` and `Struct` of any of them,
nested to any depth. The exception is Polars 2's new `Map(key, value)`,
which polspec declares, validates, profiles and writes to a spec file but
does not generate yet -- the next dtype on this list, and the reason the
dtype census in the tests names it rather than covers it. A `Datetime` carrying a `time_zone` is included: the
physical value is an offset from the naive epoch whatever the zone, so the
zone rides along and `generate()` hands back a column of the dtype you
declared. A `Decimal` is the same idea: an integer and a scale, drawn as
the integer and scaled back.

What a nested column *claims* is one declaration per value: a `List`'s
elements take the fields a scalar column takes, and a `Struct`'s take a
`ColSpec` each under
[`fields=`](../how-to/columns.md#fields-what-a-structs-values-are). The
question this answered — whether a struct's fields are a declaration of
their own or a second, smaller thing — is settled the first way: a value
is described the same way wherever it sits, which is why the same change
that generated a struct generated a list of lists.

## Generation is getting more guardrails, not fewer

Two different kinds of "limit" are in scope here, and they're worth telling
apart:

**Safety limits that already exist and will grow.** `method="cartesian"`
refuses to build a coverage set past 50 million rows, naming the dimension
that caused it, rather than silently trying to allocate one, and
`generate()` says how large a frame will be before it allocates it — a
warning by default, a refusal with `max_bytes=`. That's the shape future
guardrails will take elsewhere in generation — an explicit, named refusal
before a runaway allocation, not a mysterious hang. Expect more of these as
generation is asked to handle larger and stranger specs: sanity limits on
distribution parameters, on cartesian dimensionality, on batch sizing.

**Constraints `generate()` doesn't enforce**, which is a different, more
interesting problem. What is left is `__checks__`, `ColSpec.validators` and
`ColSpec.pattern`, and that is by design: the first two wrap arbitrary Polars
expressions, the third is an arbitrary regex, and nothing can generate data
satisfying an arbitrary predicate.
Everything else on this list has been worked through — rule and foreign-key
dependencies by ordering the passes rather than asserting the dependencies
don't exist, and uniqueness by drawing without replacement instead of hoping a
wide domain would do. What remains is narrowing the gap from the other end:
letting a column *describe* its values well enough that a validator becomes
generatable.

**Domains generation can only partly express.** A `String` column can
say what its values look like through a named
[`format`](../how-to/formats.md) — `uuid4`, `email`, `ipv4` and five
others — and is generated to satisfy it, so the validator
`col("email").str.contains("@")` no longer has to fail its own spec. The set
is closed, and [`pattern=`](../how-to/columns.md#string-patterns) is the
honest other half: any regex can be *validated*, and only the curated set
can be generated. Generating from an arbitrary pattern is a much larger
piece of work with no answer for `.*`, and is not planned.

Both directions are active. Neither has a fixed shape yet, so the specific
options `generate()` accepts may well change under you.

## Specs know about each other through a `Registry`, and only there

A `ForeignKey` names the spec it points at; nothing above a single spec knows
which specs exist unless they are put in a
[`Registry`](../how-to/registry.md). That is deliberate — two test modules may
each define an `Orders` — but it leaves edges:

- **`drift` takes one spec.** `polspec generate --all` and `validate --all`
  run over every spec in a directory with the keys between them bound;
  `drift` still takes one spec and its parents as `--references NAME=PATH`.
- **Discovery imports code.** `Registry.discover("specs/")` runs every `.py`
  file it finds. A declared `Registry(...)` in a module of your own is the
  safer shape, and `discover` is a convenience over it.
- **Shared categories are checked only when declared.** `resolve()` compares
  columns against the `CatSpec` a registry was given; without one,
  `catspec()` merges what the specs declare and refuses a disagreement, but
  nothing checks unless asked.
- **A single spec's `to_mermaid()` still draws one entity.** The whole
  picture is `registry.to_mermaid()`.

## YAML format and generated values may change

Two things this project has made no compatibility promise about yet:

- **The YAML spec format.** The keys `to_yaml()` writes and `from_yaml()`
  reads are what today's `ColSpec`/`FrameSpec`/`CatSpec` happen to need. A
  new field, a renamed key, or a different nesting for something like
  distribution parameters could all still happen as the underlying Python API
  settles.
- **The exact values `generate()` produces for a given seed.** Determinism
  *within* a version is a hard guarantee — the same seed on the same version
  always produces the same frame, and that's load-bearing for the round-trip
  tests this project is built around. Determinism *across* versions is not
  guaranteed yet: a bug fix to a distribution, a change to how a chunk's seed
  is derived, or a fix to one of the [known limitations](limitations.md) can
  all legitimately change what a given seed produces. 0.7.0 was such a
  release: every spec with rules, a foreign key, a hierarchy or a composite
  key produces different values for the same seed than 0.6 did, once, so
  that inserting a column never changes its neighbours again. 0.8.0 was
  another, for batched output only: `generate_batches` and the sinks now
  produce windows onto the frame `generate` would, whatever the batch size,
  so every seeded batched or sunk output changed, once. 0.10.0 was the
  third, for unique columns only: a unique column is now a permutation of
  its value space, unique across a whole frame however it is batched, so
  every seeded unique column changed, once.

The first of those is easier to live with than it sounds, because a spec file
now says which format wrote it. Every file `to_yaml()` writes carries
`version: 3`; a file with no `version:` key is read as version 1 and migrated
on load, and one written by a newer polspec than the reader is refused by name
rather than misread. So a format change is a migration to write, not a class of
file that silently stops loading — which is what makes the rest of this section
a smaller promise than it looks.

What is still not promised is that a *given key* survives a minor release. A
renamed key needs a migration, and migrations are written when the rename
happens, not before.

Neither of these is likely to move for the sake of moving — but until this
page says otherwise, don't build something that depends on today's YAML
surviving a version bump byte-for-byte, or on a specific seed producing the
same values after an upgrade.

## Directions, not commitments

Lower confidence than everything above: opportunities noticed rather than gaps
being actively closed. They are here because the machinery each would need
already exists, not because any of them is started.

**Synthetic look-alike data.** `from_dataframe()` profiles real data into a
spec and `generate()` turns a spec back into data, so the trip from a real
table to a statistically similar fake one is already two calls. Making it one —
with `tags` marking which columns should be replaced outright rather than
imitated — would serve the share-realistic-data-without-sharing-real-data case
directly.

**A profiled spec that names a `format`.** `from_dataframe()` reads a
string column as a `String` with a length range; it does not notice that
every value is an email address. Inference is a decision about how sure to be
before naming a format, and a wrong guess is a spec that rejects real data,
so it has not been made. [Drift](../how-to/drift.md) is the reason to want
it: a `format_violated` finding on a column the profiler named would have
been the drift that mattered.

**Test-framework integration.** A pytest fixture or plugin, or a Hypothesis
strategy built from a spec, are the natural adjacent surfaces for a library
whose whole pitch is that fixtures and contracts stay in step. Adjacent,
though — not core.

## Deferred on purpose

**`ColSpec` holding a `Domain` instead of `bounds`/`choices`/`format`.**
`polspec.constraints.Domain` is already the one definition generation,
validation, foreign keys and drift read; `ColSpec` still stores the three
fields it is derived from, and every module re-derives it. Folding the fields
into the domain would remove that repetition and nothing a user can see, at
the cost of touching every attribute read in the library. It has been
considered and set aside at each of the last three releases. The condition
it waited on — nested dtypes forcing a richer domain model — was tested by
0.9.0 and did not hold: a struct's fields are described by `ColSpec`s of
their own, so each field has an ordinary domain and `Domain` needed no
struct case. It stays deferred until something else asks for it.
