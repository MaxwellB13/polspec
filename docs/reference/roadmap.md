# Roadmap and stability

!!! warning "Early alpha"

    polspec is early. The three sections below are the honest version of
    "what's next" — not a promise of when, just where the rough edges are and
    which direction they're likely to move. Treat everything here, and
    everything the library produces, as breakable between versions until it
    says otherwise.

## Dtype coverage is not complete yet

polspec generates every scalar and temporal Polars dtype — integers, floats,
booleans, strings, binary, `Date`/`Time`/`Datetime`/`Duration`, `Enum` and
`Categorical`. Composite and nested dtypes are not there yet:

- `List`
- `Struct`
- `Array`

A `ColSpec` for one of these constructs without complaint and can be
*validated* against — `FrameSpec.validate()` doesn't need to know how to
generate a dtype to check one. `generate()` is where it stops, with a
`TypeError` naming the dtype. Expanding into nested types is the most
requested kind of gap to close next; if you need one of these today, generate
the column separately and attach it with `with_columns` after `generate()`
returns.

## Generation is getting more guardrails, not fewer

Two different kinds of "limit" are in scope here, and they're worth telling
apart:

**Safety limits that already exist and will grow.** `method="cartesian"`
refuses to build a coverage set past 50 million rows, naming the dimension
that caused it, rather than silently trying to allocate one. That's the shape
future guardrails will take elsewhere in generation — an explicit, named
refusal before a runaway allocation, not a mysterious hang. Expect more of
these as generation is asked to handle larger and stranger specs: sanity
limits on distribution parameters, on cartesian dimensionality, on batch
sizing.

**Constraints `generate()` doesn't enforce yet**, which is a different, more
interesting problem. `unique=True`, `__unique_together__`, chained `ColRule`s,
and a few other interactions are validated but not (yet) satisfied by
generation — see [Known limitations](limitations.md) for the exact list, each
backed by a test that will fail the moment it's fixed. Making generation
smarter about actually satisfying these — sampling without replacement for a
`unique` domain, resolving rule dependencies instead of asserting they don't
exist — is the other half of "improving generation," and the harder half.

Both directions are active. Neither has a fixed shape yet, so the specific
options `generate()` accepts may well change under you.

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
  all legitimately change what a given seed produces.

Neither of these is likely to move for the sake of moving — but until this
page says otherwise, don't build something that depends on today's YAML
surviving a version bump byte-for-byte, or on a specific seed producing the
same values after an upgrade.
