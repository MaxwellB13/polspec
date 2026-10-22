# Roadmap and stability

!!! warning "Pre-1.0"

    polspec is settling toward a 1.0 with written stability promises. Until
    then, this page is the honest version of what may change: where the
    rough edges are, and which way they are likely to move. 0.22 writes the
    1.0 promises down, each with a test that holds it.

## Dtype coverage is complete

polspec generates every Polars dtype that holds data — integers to 128
bits, floats from half to double precision,
`Decimal`, booleans, strings, binary, `Date`/`Time`/`Datetime`/`Duration`,
`Enum`, `Categorical`, and `List`, `Array` and `Struct` of any of them,
nested to any depth -- and, on Polars 2, `Map(key, value)`, generated as
the list of `{key, value}` entries it is, with every key present and
distinct within its map. A `Datetime` carrying a `time_zone` is included: the
physical value is an offset from the naive epoch whatever the zone, so the
zone rides along and `generate()` hands back a column of the dtype you
declared. A `Decimal` is the same idea: an integer and a scale, drawn as
the integer and scaled back. A `Null` column, which holds no data at all,
generates as the nulls it is -- declared `nullable=True`, since it can hold
nothing else.

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

- **Discovery imports code.** `Registry.discover("specs/")` runs every `.py`
  file it finds. A declared `Registry(...)` in a module of your own is the
  safer shape, and `discover` is a convenience over it.
- **Shared categories are checked only when declared.** `resolve()` compares
  columns against the `CatSpec` a registry was given; without one,
  `catspec()` merges what the specs declare and refuses a disagreement, but
  nothing checks unless asked.
- **A single spec's `to_mermaid()` still draws one entity.** The whole
  picture is `registry.to_mermaid()`.

## The public API is what `polspec` exports

What you can import from `polspec` -- the names in its `__all__`, which
the [API reference](../reference/api/index.md) documents -- is the public
API, along with three submodules: `polspec.drift` (`drift`, `diff` and
their reports), `polspec.validation` (`FINDING_COLUMN` and the report
types) and `polspec.serialization` (the `CatSpec`, `Registry` and dict
forms of a spec file). Each has an `__all__`, and only those names are
public. Every other module is internal and says so in its docstring:
it can change in any release, without a deprecation. A public name is
never renamed or removed in place: it warns first, for at least one minor
release, and a name deprecated in 0.x goes at 1.0.

## YAML format and generated values may change

Two things this project has made no compatibility promise about yet:

- **The YAML spec format.** The keys `to_yaml()` writes and `from_yaml()`
  reads are what today's `ColSpec`/`FrameSpec`/`CatSpec` happen to need. A
  new field, a renamed key, or a different nesting for something like
  distribution parameters could all still happen as the underlying Python API
  settles.
- **The exact values `generate()` produces for a given seed.** Determinism
  *within* a version is a hard guarantee: the same seed and spec on the same
  version always give the same frame, on every supported operating system
  and Polars. Across versions it is not promised yet. A fix can change what
  a seed produces, and when one does, the changelog says so. So far:

  | Release | What a seed produces differently, once |
  |:--|:--|
  | 0.7.0 | Specs with rules, a foreign key, a hierarchy or a composite key, so that inserting a column never changes its neighbours again |
  | 0.8.0 | Batched and sunk output, now windows onto the frame `generate()` makes, whatever the batch size |
  | 0.10.0 | Unique columns, now a permutation of their value space, unique across a whole frame however it is batched |
  | 0.11.0 | A unique column a foreign key fills, now a permutation of its parent's keys |
  | 0.12.0 | A `List`, `Array` or `Map` column's elements past the first batch (`generate()` did not change) |
  | 0.19.0 | `lognormal` and `beta` columns, where operating systems' maths libraries disagreed: the samplers' `exp`, `ln` and `pow` are polspec's own now, so a seed gives the same values on every OS |

  Every other release since 0.7.0 -- 0.21.0 included -- produces exactly
  what the one before it did, on every Polars it supports.

The first of those is easier to live with than it sounds, because a spec file
now says which format wrote it. Every file `to_yaml()` writes carries
`version: 4`; a file with no `version:` key is read as version 1 and migrated
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

## Fake data from real data exists, one column at a time

`polspec.synthesize` (0.13.0) turns a real table into a fake one: each
column's shape, frequencies, null rate and uniqueness are profiled and
generated again, and `replace=` keeps a column's values out of it. What it
does not do is the next step: columns are profiled independently, so what
they do *together* -- an age that follows a status, a child row that points
at a parent -- is not kept. Capturing that means a spec that can describe a
joint distribution, which `ColRule` does only for conditions written by hand.

Since 0.14.0 a profiled spec also names a text column's `format` when its
values have one of the formats polspec knows, with a near miss's stand-ins
as `extra_values`, and records a float's NaN share -- so `synthesize` makes
email-shaped emails, and `drift` has a `format_violated` to report when a
source starts holding something else.

## Directions, not commitments

Lower confidence than everything above: opportunities noticed rather than gaps
being actively closed. They are here because the machinery each would need
already exists, not because any of them is started.

**Test-framework integration.** A pytest fixture or plugin, or a Hypothesis
strategy built from a spec, are the natural adjacent surfaces for a library
whose whole pitch is that fixtures and contracts stay in step. Adjacent,
though — not core.
