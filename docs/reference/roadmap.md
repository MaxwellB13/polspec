# Roadmap and stability

!!! warning "Early alpha"

    polspec is early. The sections below are the honest version of "what's
    next" — not a promise of when, just where the rough edges are and which
    direction they're likely to move. Treat everything here, and everything
    the library produces, as breakable between versions until it says
    otherwise.

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

`Struct` is the interesting one of the three. A struct column is a set of
named, typed fields, which is what a `FrameSpec` already is — so the question
is less "how is this generated" than whether it reuses that machinery or gets
its own, and getting that wrong would be expensive to undo.

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

**Domains generation cannot currently express.** A `String` column generates
random characters within its `string_length`, and there is no way to say more
than that about its shape. This is why a column carrying a validator as
ordinary as `pl.col("email").str.contains("@")` cannot be generated to satisfy
its own spec. A pattern or named format on `ColSpec` — a regex, or something
like `format="email"` — would let generation produce values its own validators
accept, converting a whole class of the gap above into something that
round-trips rather than something documented. It is also most of what stands
between generated fixtures and fixtures that look like data.

Both directions are active. Neither has a fixed shape yet, so the specific
options `generate()` accepts may well change under you.

## Specs do not know about each other yet

A `ForeignKey` can point at another `FrameSpec`, but nothing above the level of
a single class knows which specs exist in a project. Four separate rough edges
turn out to be the same missing piece.

**Generating a related set is manual.** Parents first, threaded through
`references=` by hand, in an order you work out yourself:

```python
customers = Customers.generate(1_000, seed=1)
orders    = Orders.generate(10_000, seed=2, references={Customers: customers})
lines     = OrderLines.generate(30_000, seed=3, references={Orders: orders})
```

That ordering is already implied by the foreign keys those specs declare.
Walking the graph and returning the whole consistent set from one call is
probably the single most useful thing on this page.

**Cross-spec foreign keys are written by name and resolved by nothing.**
`to_yaml()` writes `references: Customers`; a spec loaded from that file
carries the name and checks nothing about it until a spec called `Customers`
is supplied by hand. A registry of a project's specs is what would resolve
it automatically.

**`to_mermaid()` draws one entity.** The relationships *between* specs — the
reason to draw an ER diagram at all — aren't available to a method that can
only see the class it was called on.

**Shared categories can silently disagree.** Two specs declaring `STATUS` with
different variants, or same-named `pl.Categories` with different physical
dtypes, quietly breaks the join-on-shared-codes property
[Shared categories](../guide/categories.md) describes. Nothing currently
notices.

The likely shape is discovery across a directory — the CLI already finds every
`FrameSpec` subclass defined in a single module — plus a registry the other
three can be built against. Whether that registry is discovered or declared
explicitly is undecided, and it is the decision the rest depends on.

## Validation results are shaped for people, not programs

`validate()` collects every violation across every column before raising, which
is the right behaviour for someone reading a traceback. It is the wrong shape
for anything else: `ValidationError` carries its findings as a list of
formatted strings, so a caller that wants to *act* on them — quarantine the
offending rows, count violations per column, write a report — has to parse
English back into structure.

Two directions here, neither settled:

- **A result object rather than an exception**, so validation can be asked for
  its findings instead of raising them, with failing rows available as a frame
  rather than a handful of samples rendered into a message.
- **A `validate` verb on the command line.**
  [The command line](../guide/cli.md) turns data into a spec and a spec into a
  test, but cannot check data against a spec — the one of the three needing no
  Python at all, and the one that would let a spec gate a pipeline in CI.

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

The first of those is easier to live with than it sounds, and the first step is
small: nothing in a spec file currently records which polspec wrote it. A
version key would let `from_yaml()` say so plainly when a file predates a
format change, instead of failing on an unrecognized key — or, worse, quietly
reading a renamed one as its default.

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

**Drift as a report, not a pass/fail.** "A validation library tells you when
production data drifted" is the claim on the front page, and today the answer
is only *that* it drifted. Diffing a spec against data — new enum variants,
bounds exceeded, cardinality moved — would say how. The same machinery diffs
two specs against each other, which is what reviewing a schema change in a pull
request actually needs.

**Test-framework integration.** A pytest fixture or plugin, or a Hypothesis
strategy built from a spec, are the natural adjacent surfaces for a library
whose whole pitch is that fixtures and contracts stay in step. Adjacent,
though — not core.
