# Changelog

All notable changes to polspec are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Until 1.0, minor
versions may break the Python API, the YAML format, and the values a given
seed produces; see
[Roadmap and stability](https://maxwellb13.github.io/polspec/explanation/roadmap/).

## [Unreleased]

### Added

- `ColSpec(seed_name=...)`: rename a column without changing the data it
  generates. Each column is seeded from the frame seed and its *name*, so a
  rename had always changed a column's values -- which matters when
  generated frames are snapshots other things are compared against. A
  column declared with `seed_name="old"` is seeded as `"old"` and keeps
  producing what it did. Declared, not guessed: `rename()` does not set it,
  and two columns of one spec cannot share one (they would draw identical
  values). It holds across a rename and nothing else: the passes that run
  after the columns are filled draw their seeds in declaration order, so an
  inserted rules column still reshuffles the ones after it, and that
  boundary is pinned and stated in *Known limitations*.

### Documentation

- *Generating data* said each column derives its seed "from its position".
  It is from its name -- the reason `seed_name` is needed at all.

## [0.6.0] - 2026-09-14

The headline is drift: a report of what changed between two specs, or
between a spec and the data it describes, with a severity per finding under
one mechanical rule -- and three CLI verbs so that report can gate a pull
request or a nightly load with no Python. Alongside it, `pattern=` completes
the split `format=` began. Nothing in the engine changes; no seed produces a
different frame, and no spec file needs migrating.

### Added

- **Drift as a report.** `polspec.drift.diff(old, new)` says what changed
  between two declarations; `polspec.drift.drift(spec, df)` says how data
  has moved relative to what its spec declares -- values outside the
  domain, a bound exceeded and by how much, a format no longer matched, a
  null rate that moved, declared values never seen. Both return one
  `DriftReport`, also reachable as `Orders.diff(Other)` and
  `Orders.drift(df)`:

  ```python
  report = OrdersV1.diff(OrdersV2)
  report.breaking          # a narrowed bound, an added column, a new constraint
  report.compatible        # a widened domain, a removed check
  report.to_markdown()     # the shape of a pull-request comment
  ```

  Every finding carries a `severity` under one mechanical rule: *breaking*
  when a frame that satisfied the old side could fail the new one --
  decided, for dtypes, by the same function validation uses. Two tests hold
  the two sides to it: what `generate()` produces never drifts breakingly
  from its own spec, and every breaking finding against data is a column
  `validate()` reports. Sixteen closed finding codes, a `DriftOptions`
  with explicit tolerances, and a comparator per `ColSpec` and `TableSpec`
  field with a parity test, so a field added to a declaration is one entry
  or one red test. `DriftReport`, `DriftFinding` and `DriftOptions` are
  exported from `polspec`.

- `ColSpec(pattern=...)`: a regular expression every value of a `String`
  column must match, **checked by validation only**. Generation does not
  read it -- the column is filled with ordinary random text, and the round
  trip holds only with `validate_pattern=False`, exactly as for
  `validators`. That is the honest half of the split `format=` made: any
  regex can be checked, the curated set can be generated. A `pattern`
  finding code, a `validate_pattern` switch on `ValidationOptions`, one
  `pattern:` key in spec files (the format version stays at 3: an added
  optional key is not a new version, and `migrations.py` now says so).
  Cannot be combined with `format`; compiled by Polars' own engine at
  declaration, so a pattern Polars cannot run is refused with its message.

- **Three CLI verbs.** `polspec generate SPEC -n N -o FILE` writes generated
  rows to any format the CLI reads (`--seed`, `--method`, `--references`
  as for `validate`); `polspec diff OLD NEW` and `polspec drift SPEC DATA`
  print a drift report as text, `--json` or `--markdown`, and gate on it
  with `--fail-on breaking|any|none` -- so a schema change in a pull
  request, or a nightly load, can fail CI with no Python:

  ```bash
  polspec generate orders.yaml -n 1000 -o orders.parquet --seed 1
  polspec drift orders.yaml orders.parquet
  polspec diff main/orders.yaml pr/orders.yaml --markdown --fail-on breaking
  ```

### Documentation

- Five claims the docs had stopped being true about are fixed, and one of
  them is pinned: `unique=True` and `__unique_together__` had been listed as
  work generation does not attempt (generated since 0.2.0); `generate()` on
  an unsupported dtype raises `SpecError`, not `TypeError`; every yaml
  example carried `version: 2` at format version 3, and a test now holds
  each `version:` in the docs to `FORMAT_VERSION`.
- The [roadmap](https://maxwellb13.github.io/polspec/explanation/roadmap/)
  gains a *Deferred on purpose* section, starting with the `ColSpec` ->
  `Domain` restructure that two release plans had set aside without saying
  so on the page, and the condition under which it would be revisited.

## [0.5.0] - 2026-09-11

The headline is `format=`: a `String` column that says what its values look
like, and is generated to satisfy its own validator. Around it, the release
closes five silent edges, settles the validation option surface, spells out
the `FrameSpec` signatures, and puts a type checker in CI. One narrow
breaking change, under *Changed*.

### Added

- `ColSpec(format=...)`: a `String` column that says what its values look
  like, and is generated to satisfy its own validator. Eight formats --
  `uuid4`, `email`, `ipv4`, `ipv6`, `mac`, `hostname`, `iso_country`,
  `iso_currency` -- each with a sampler in the engine and a check in
  validation declared side by side in `polspec.formats`, and pinned by a
  round trip per format:

  ```python
  class Users(FrameSpec):
      user_id = ColSpec(pl.String, format="uuid4", unique=True)
      email   = ColSpec(pl.String, format="email")

  Users.validate(Users.generate(1_000_000, seed=42))   # passes
  ```

  Validation reports a new `format` finding. A format is one `format:` key in
  a spec file and takes part in the domain check a foreign key runs at
  declaration. It cannot be combined with `choices` or `string_length`, and
  only a `String` column can carry one; each is refused with a message
  saying which to drop. What a format promises is syntax: see *Known
  limitations*.

- `ValidationOptions` is exported from `polspec`, and `validate()` and
  `inspect()` take it as `options=`. Every switch as one value, for when the
  same settings go through several calls:

  ```python
  lenient = ValidationOptions(extra_cols="drop", checks=False)
  Orders.validate(df, options=lenient)
  Customers.validate(other_df, options=lenient)
  ```

  `options=` and the individual keywords are alternatives rather than a base
  and an override, so passing both raises instead of quietly picking one.

### Changed

- `FrameSpec.validate`, `inspect` and the four `sink_*` classmethods spell
  their options out instead of forwarding `**kwargs`. Editors complete
  them, a type checker sees a typo, and a mistyped option is a `TypeError`
  from the classmethod's own signature rather than from a function several
  frames away. The validation keywords default to `None`, meaning "the
  `ValidationOptions` default", so the defaults are still defined once. The
  sinks keep their trailing `**kwargs`: that is the documented passthrough
  to the underlying writer, not a gap.

- A misspelled spec name in `references={...}` now warns instead of passing
  silently. A key nothing points at was skipped without a word, so the column
  was generated freely while the caller believed the parent had been used --
  and `validate()` then reported the key as unresolved, which is the round
  trip failing with only its second half audible. The warning names the key
  that went unfilled and suggests the one supplied:

  ```
  Orders: references={...} supplied ['Custmers'] that no foreign key points
  at, while 'Customers' (supplied 'Custmers'?) went unfilled.
  ```

  It takes both halves to warn -- something supplied that went unused *and*
  something unused that went unfilled -- so supplying no parent at all stays
  silent, as documented, and a `Registry` handing every spec the whole set of
  frames says nothing either. `generate_batches` warns once per call rather
  than once per batch.
- `ColSpec(pl.Int64, null_probability=0.9)` now warns. `nullable=False` still
  wins and the rate is still ignored -- turning nullability off should not
  also require deleting the rate beside it -- but the same silence covered
  asking for nulls and forgetting `nullable=True`, where the column generates
  none and nothing says why. Only a rate that cannot be a leftover warns: the
  default and an explicit `0.0` already agree with `nullable=False`.
- `references=` given something that is not a mapping raises `SpecError`
  naming the three key forms it accepts, rather than an `AttributeError` from
  inside `resolve_references`. Every collection argument elsewhere in the API
  is a sequence, so passing one here was an easy mistake with an unhelpful
  answer, and it was the one complaint polspec made that was not a
  `PolspecError`.
- An unknown validation option names the option you meant. It used to surface
  as `ValidationOptions.__init__() got an unexpected keyword argument
  'validate_uniqe'`, naming a private class that is not exported and not the
  option intended; it is now `Unknown validation option(s): 'validate_uniqe'
  (did you mean 'validate_unique'?)`, with the accepted list.

- **Breaking, narrowly: the check switches have one spelling.**
  `inspect(spec, df, unique=False)` used to work alongside
  `validate_unique=False`, while `validate(spec, df, unique=False)` raised --
  a second public spelling reachable through half the API, from a rename map
  that was meant to be internal. Only the `validate_*` form is accepted now,
  by both verbs. The bare names live on as the fields of `ValidationOptions`,
  which is where they were always meant to be.

### Documentation

- `Decimal` joins `List`, `Struct` and `Array` on the list of dtypes that
  declare and validate but cannot be generated. It had been missing from both
  [Dtype coverage](https://maxwellb13.github.io/polspec/explanation/roadmap/)
  and [Known limitations](https://maxwellb13.github.io/polspec/explanation/limitations/),
  and it is the one people miss, being the only one of the four that is not a
  nested type.
- A `Datetime` carrying a `time_zone` generates, which the docs had never said
  either way and readers assumed meant no. Both claims are now pinned by tests
  in `tests/test_roundtrip.py`, so neither page can go stale.

### Internal

- The four copies of "declares no ColSpec columns" are one
  `tablespec.require_columns`.
- **mypy runs in CI.** The package ships `py.typed` and a stub for the Rust
  extension, so every consumer's type checker trusts these signatures, and
  nothing was checking them. The first run found 247 errors, of which 87 trace
  to two declarations: `ColSpec.dtype` and `ForeignKey.references` are both
  annotated with what the constructor accepts rather than what the instance
  ends up holding. Narrowing either is a design decision on a public field, so
  the modules carrying that backlog are listed in `pyproject.toml` and
  everything else is enforced -- the list can only shrink.
- The predicate nodes in `expr.py` no longer each carry their own
  `root_names`, `literals` and `rename`. A node reports its operands through
  `children()` and the three traversals are derived from that on `Pred`, which
  turns thirty-six implementations into twelve. `rename` is the one that
  mattered: the base implementation returned `self`, so a node that forgot to
  override it left a renamed spec pointing at a column that no longer existed,
  with nothing raised. Forgetting `children()` now raises.
- `References` and `Method` were declared identically in four and two modules;
  `_collect`/`_to_lazy` in three. They are one `polspec.frames`.
- The four `sink_*` functions built the same six-argument batch-stream call
  each. `_prepare` now returns the checked call as one value. The public
  signatures stay spelled out, which is what makes a typo in one of them fail
  at the call site.
- `registry.py` imported `serialization` lazily in five methods and `report` in
  a sixth, while importing `generation` and `validation` at module level. There
  was no cycle; all six are hoisted.

## [0.4.1] - 2026-09-09

The hierarchy release. A self-referencing `ForeignKey` says that every parent
value exists somewhere in the frame, and nothing more. This adds the
declaration that says the rest of it -- one parent per reference, a bounded
depth, no cycles -- and makes `generate()` satisfy it rather than leaving it to
the draw.

### Added

- **`Hierarchy`**, declaring that two columns of a spec are a parent/child
  edge list drawn on one pool of references: a child pointing at its parent,
  and that parent pointing at its own parent, are the same row shape.

  ```python
  class Links(FrameSpec):
      PARENT_REF = ColSpec(pl.String)
      CHILD_REF = ColSpec(pl.String)
      __hierarchy__ = Hierarchy(
          child="CHILD_REF", parent="PARENT_REF", max_depth=5
      )
  ```

  `generate()` produces a real forest: one parent per reference, so every row
  resolves to a single ultimate parent, and no chain longer than `max_depth`
  with at least one reaching it exactly -- so a test of a graph walk exercises
  the boundary rather than whatever the draw happened to give. `branching`
  shapes the tree, or `roots` pins the number of ultimate parents.

  `generate(n, cycles=10, self_references=5)` then breaks it on purpose, which
  is the other half of testing a graph walk: a resolver written without a
  visited set does not fail on a loop, it runs forever. The spec still says the
  data should be acyclic, so `validate()` reports what was injected --
  `hierarchy_cycle`, `hierarchy_depth` and `hierarchy_multi_parent` are new
  finding codes -- and a test can assert that its own resolver and polspec
  agree about what is broken.

  Both checks are bounded, so validating deliberately cyclic data terminates:
  depth costs `max_depth` steps and cycle detection walks by pointer doubling,
  covering a million-row chain in about twenty. A validator that walked until
  it reached a root would hang on the fixtures this feature exists to make.

  `generate_batches` and the `sink_*` functions refuse a spec carrying one: a
  forest spans the whole frame, and batches are generated independently. See
  [Hierarchies and link tables](https://maxwellb13.github.io/polspec/how-to/hierarchies/).

### Documentation

- The spec file format is version 3, which adds the `hierarchy:` key. A
  version 2 file loads unchanged.
- [Known limitations](https://maxwellb13.github.io/polspec/explanation/limitations/)
  now points a self-referencing `ForeignKey` at `Hierarchy` for the case that
  wants a real tree, rather than carrying a recipe of its own. Two tests pin
  the two apart.

## [0.4.0] - 2026-09-08

The internals release. 0.2.0 and 0.3.0 settled the vocabulary; this one goes
underneath it, to the Rust generator and the places where the same table was
being maintained in two or three languages.

Nothing about how a spec is written changes. One thing does break, and it is
the same thing the roadmap has always reserved: **the values a given seed
produces are different**. Any test asserting on specific generated values
needs re-baselining; a test asserting on their *properties* -- bounds,
distinctness, null share, distribution shape -- does not. polspec's own suite
needed no changes, which is the shape of test this library is built to support.

Generation got faster, by between a tenth and a third depending on the column.
Measured A/B against v0.3.0 on one machine, same build profile, twenty million
rows: the four-column frame in `benchmarks/bench.py` 1.30x, a
`unique=True` Int64 column 1.21x, a bounded nullable Int64 column 1.17x, an
Enum column 1.25x, a String column unchanged. Treat the ratios rather than the
absolute numbers as the claim.

### Added

- `generate`, `generate_batches`, `inspect`, `validate`, `sink_parquet`,
  `sink_ipc`, `sink_csv` and `sink_ndjson` are exported from `polspec` itself.
  Each takes a `TableSpec` as its first argument and each is what the matching
  `FrameSpec` classmethod already called -- but they lived in
  `polspec.generation` / `polspec.validation`, which the API reference calls
  internal and free to change in a patch release. So the `TableSpec`-first
  half of the library had no stable import path; now it does, and both halves
  appear in [the API reference](https://maxwellb13.github.io/polspec/reference/api/).

### Changed

- **Breaking: the values a given seed produces have changed.** polspec now
  builds on `rand` 0.10 (from 0.8), whose samplers draw differently. Same seed,
  same version, same frame -- as before; across this version boundary, not.
- A generated numeric, boolean, temporal or categorical column arrives as
  **one chunk** rather than one per 65,536 rows. The values buffer and the
  validity bitmap are each allocated once at full length and filled in parallel
  through disjoint slices, instead of being built per chunk and appended
  together. Nothing downstream now pays for a column split into hundreds of
  pieces -- the gather behind a `choices` domain, the cast behind a temporal
  dtype, every `sink_*` write.

  String columns are the exception and stay chunked, because Polars backs them
  with view arrays: merging those copies no string bytes, but it does copy
  sixteen bytes of view per row, which costs more than the split it removes.
  They still gain the other half of the change -- the chunks are collected in
  one go rather than appended one at a time, and an append rescanned both sides
  for their first and last non-null value to maintain a sorted flag that random
  strings will not have set anyway.
- Drawing a `unique=True` column no longer materialises its domain. A domain
  only a little wider than the row count used to be built in full and partially
  shuffled, which allocates in proportion to the domain rather than to the
  output: ten million distinct values from a range of eighty million reserved
  1.4 GB before writing anything. That branch is now Floyd's algorithm, which
  holds only the values it has chosen -- the same case now peaks at 491 MB.
  Roomier domains keep drawing and rejecting, which is faster there and was
  never the memory problem.
- Every character of the generated-string alphabet is now exactly equally
  likely. Six random bits give 64 values for a 62-character alphabet, and the
  two spare ones fell back on `% 62` over a fresh 32-bit draw, which is biased
  by about one part in 70 million -- far too little to see, but free to remove:
  the fallback now rejects properly instead.
- `rand` 0.8 was compiled alongside the `rand` 0.10 that Polars already links,
  so the extension carried two copies of `rand`, `rand_core` and their chacha
  backends. There is now one of each.
- The release profile builds the crate as a single codegen unit. Measured on
  one machine against otherwise identical v0.3.0 code, that alone is worth
  2.1x on the `unique=True` path, for about twenty seconds of build time. Fat
  LTO on top of it was tried and dropped: a further 5% for eight more minutes
  per build.

### Documentation

- A self-referencing `ForeignKey` guarantees that every parent value exists,
  and nothing more -- in particular not that the result is a tree. Parents are
  sampled from the whole frame, so some rows end up in a cycle or pointing at
  themselves, which is what makes a recursive query fail to terminate, and
  `validate()` does not report it because no part of a spec can say "acyclic".
  [Known limitations](https://maxwellb13.github.io/polspec/explanation/limitations/)
  now says so, with a recipe for a genuine hierarchy, and the roadmap carries
  what closing the gap would need. Two tests pin the behaviour.

### Fixed

- `Registry.validate_all` applies each report's structural transformations
  using the same bound spec the report was produced against, rather than the
  unbound copy.

### Internal

- `ColumnPlan::build` takes a `PlanArgs` struct instead of thirteen positional
  arguments, so a call site names what it sets and leaves the rest to
  `Default`. Four `#[allow(clippy::too_many_arguments)]` and a great many
  `None`s went with it.
- `Kind`'s three parallel lists -- the names, the parse, the reverse lookup --
  are generated from one declaration, so a new column kind cannot be added to
  two of them and forgotten in the third.
- The fixed-width integer ranges were written three times: once as polspec's
  default generation range, once as the limits a declared bound may not exceed,
  and once as the Rust samplers' defaults. The first is now read from the
  second.
- The four `sink_*` functions share a typed batch-stream helper rather than
  forwarding `**kwargs`.
- `benchmarks/bench_generate.py` is replaced by `benchmarks/bench.py`, which
  measures the same comparison and adds a regression mode. The old harness
  timed one run per case, in a process shared with the implementations it was
  comparing against, and recorded nothing about the machine -- so its numbers
  varied by around 25% between runs and could not be compared across days. It
  also measured exactly one column shape, which is how a change that made the
  `unique` path half as fast came within an afternoon of being released as a
  speed-up. The new one takes the fastest of several runs, gives every
  measurement its own process, repeats a short case until its floor settles,
  records the CPU, thread count, Polars version and cargo profile, and covers
  each column kind, both branches of the unique draw, the cartesian, rule,
  foreign-key and composite-key passes, and a sink. `record` writes a local
  baseline and `check` exits non-zero when a case regresses past a tolerance;
  repeated measurements now agree to within about 2%.
- The benchmark table in the comparison guide is re-measured. It had been
  recorded on 2026-09-03, which is before both 0.2.0 and 0.3.0, so it had been
  describing an engine two releases old: the four-column frame at twenty
  million rows was published as 0.0827s, measures 0.1409s on v0.3.0, and
  0.1127s here. Some of that gap is still unaccounted for and is worth
  chasing. The NumPy and pure-Python columns re-measure to within 1% of what
  was published, which is what says the difference is polspec's and not the
  machine's.

## [0.3.0] - 2026-09-06

The refactor 0.2.0 started, finished. `CatSpec` was the one declarative
surface left doing everything in one class; it is now a value with a
metaclass facade, like `TableSpec` and `FrameSpec`. Behind it came the fixes
that were waiting for a release allowed to break something.

One thing breaks, and it is worth reading before you upgrade: naming a
`CatSpec` entry now always gives back the dtype, whichever form declared the
registry. `pl.Enum(cats.STATUS)` becomes `cats.STATUS`, and
`cats.CURRENCY.physical()` becomes
`cats.get_categorical("CURRENCY").physical()`. Nothing else in the public API
changed shape.

The documentation gained a test: every Python example in `docs/` is executed
by the suite, which found five broken examples that had been shipping.

### Changed

- **Breaking.** `CatSpec` is a value, and the class body that declares one is
  read by a metaclass rather than left in the namespace -- the same split
  `TableSpec` and `FrameSpec` already had. What follows:

  - **Naming an entry always gives back the dtype.** It used to depend on how
    the registry was built: a class-body entry was a real class attribute and
    returned the dtype, while a dict-built registry's `.STATUS` returned the
    raw category list, and only one of the two could be handed to a `ColSpec`.
    Both now return the dtype, so `ColSpec(cats.STATUS)` and
    `ColSpec(Categories.STATUS)` mean the same thing. `cats["STATUS"]` and
    `cats.get("STATUS")` follow the same rule. Replace `pl.Enum(cats.STATUS)`
    with `cats.STATUS`, and `cats.CURRENCY.physical()` with
    `cats.get_categorical("CURRENCY").physical()`.
  - **An entry may share a name with a method.** Entries are removed from the
    class body before the class exists, so declaring one called `get` no longer
    warns and no longer costs you `CatSpec.get`. The entry is reached through
    the registry (`Categories.spec.get("get")`).
  - **`CatSpec.infer_from_dataframe` and `CatSpec.infer_from_framespec` are
    removed**; `CatSpec.infer(target, ...)` dispatches on what it is given, as
    it already did. `from_dataframe` and `from_framespec` are unchanged --
    those read what is declared rather than inferring what could be.
  - **`Categories.spec`** is the `CatSpec` a class body declares. Anywhere a
    registry is expected -- `with_catspec`, `Registry(categories=...)`,
    `FrameSpec.from_yaml(categories=...)` -- the class and the value are now
    interchangeable.
  - **`CatSpec` has value semantics.** Two registries that say the same thing
    compare equal and hash equal, so one loaded from a file can be checked
    against one a class body declares.
  - **`CatSpec.dtype_of(name)`** is the one lookup everything else is built on:
    the dtype an entry names, or None. `resolve_key` still says which kind of
    entry a name binds to.
  - `enums`, `categoricals` and `choices` are read-only mappings rather than
    fresh dicts. `dict(cats.enums)` if you need a mutable copy.

### Added

- `polspec.MultiValidationError`, raised by `Registry.validate_all` when
  several frames fail at once. It is a `ValidationError`, so an existing
  `except` clause still catches it, and it carries every failing spec's
  `ValidationReport` as `reports`, keyed by spec name -- previously
  `validate_all` raised with a joined string and the reports were lost, so
  `failing_rows()`, `by_code()` and `to_json()` were unreachable from the
  registry path.
- `polspec.CliError` is exported, so `except polspec.CliError` works. It was
  the one exception in the hierarchy reachable only from `polspec.errors`.
- Every Python example in the documentation is executed by
  `tests/test_doc_examples.py`. A block that cannot run standalone says so in
  an HTML comment (`<!-- docs: skip -->`), and one that demonstrates an error
  is checked to still raise (`<!-- docs: raises -->`). This found four broken
  examples, fixed here: a `drop()` of a column the page never declared, a
  `TableSpec` example rebinding the name a later block used, and two blocks
  naming frames (`broken_df`, `existing_df`) that were never built.

### Fixed

- `ColSpec(tags={...})` is reproducible. A `set` was kept in its own iteration
  order, which Python salts per process, so `to_yaml` wrote a different `tags:`
  line on every run and two identically-written specs compared unequal across
  processes. A set is now sorted; a list or tuple keeps the order it was
  written in.
- `Registry.generate_all`, `generate_related` and `inspect_all` bind their
  cross-spec foreign keys before doing anything, so a key whose dtypes do not
  match is a `RegistryError` naming both columns rather than a Polars cast
  error from inside generation. Only `resolve()` used to run that check, and
  nothing said it had to be called first. A key whose target is supplied
  through `references=` rather than held by the registry is still accepted, as
  it was.
- A `ColSpec` carrying the same validator twice keeps it once, so it produces
  one finding rather than two identical ones. `TableSpec` already collapsed
  identical checks and foreign keys.
- `generate_batches` and the `sink_*` functions resolve `references` once per
  call rather than once per batch. A `LazyFrame` parent was collected inside
  every batch, so a scan-backed parent was re-read as many times as there were
  batches.
- `inspect()` no longer raises a raw Polars error for a column whose dtype is
  wrong *and* whose spec declares `choices` or an `Enum`. The domain check was
  built before the dtype check could bail out, and comparing values against
  choices of another type is not something Polars will compile at all, so the
  frame most likely to arrive -- a column read back from CSV or JSON as the
  wrong type -- crashed instead of reporting a `dtype` finding.
- A `ColRule` whose condition is null on a row no longer excuses every later
  rule on that row. Generation folds a null `when` to `False` before testing it
  and before accumulating it into the claimed mask; validation did neither, so
  the null propagated through Kleene logic and left rows that generation *had*
  rewritten unchecked.
- `TableSpec` is hashable, so `references={Orders.spec: df}` works. It is one
  of the three forms `generate()` and `validate()` document, and the only one
  that could not be put in a dict: the dataclass's generated `__hash__` cannot
  hash a mapping of columns, nor a `ColSpec` carrying `distribution_params`.

### Documentation

- The install sections of the README and the documentation home said polspec
  was not published to PyPI, directly below a `pip install polspec` block. Both
  now describe the published wheels, and point at `CONTRIBUTING.md` for
  building from a checkout.
- The documentation workflow runs on changes to `python/**` and
  `scripts/generate_llms_txt.py`. The API reference is `:::` directives filled
  in by mkdocstrings from the live docstrings, so a docstring that breaks
  `--strict` used to pass its own pull request and fail the next one to touch
  `docs/`.
- The roadmap's "YAML format may change" section described the missing format
  version key that 0.2.0 shipped, and said an unsupported dtype raises
  `TypeError` rather than `SpecError`.
- `FINDING_COLUMN` and `ValidationOptions` are documented in
  the validation guide; both are exported and appeared nowhere.
- `how-to/tablespec.md` taught `polspec.generation.generate(spec, ...)` while
  the API reference says anything unlisted may change in a patch. The page now
  says which of the two it is.
- `CONTRIBUTING.md` gives the runnable form of the Windows `cargo test`
  workaround, and names the `STATUS_DLL_NOT_FOUND` failure it fixes.

## [0.2.0] - 2026-09-05

The architecture release. Specs became data, constraints gained one definition
each, and the generator learned to satisfy claims it used to only check.

This release breaks a lot. Every incompatible change below is marked
**Breaking** and says what to do instead. Three are worth knowing before you
upgrade: `Spec._columns` and friends are now `Spec.spec.columns`; the values a
given seed produces have changed, so any test asserting on generated values
needs re-baselining; and several declarations that used to be accepted and
quietly misbehave are now refused at declaration time.

### Added

- `docs/llms.txt` and `docs/llms-full.txt`, published at the documentation
  site root in the [llms.txt](https://llmstxt.org) format: an index of every
  page, and the full text of all of them in one file. Generated by
  `scripts/generate_llms_txt.py` from the nav, the pages, and -- for the API
  reference, whose source is `:::` directives -- the live docstrings, so a
  language model reads signatures rather than an empty page. A test fails if
  either file is stale.
- `polspec.constraints`: the definitions generation and validation both read,
  so they cannot drift. `Domain` is what a column may hold (its `choices`, an
  `Enum`'s categories, its `bounds`); `Pass` and `order` decide which rewrite
  of a generated frame runs first, from the columns each one reads and writes.
- `unique=True` is generated, not just validated. The engine draws the column
  without replacement (`src/unique.rs`): it shuffles a materialised domain
  when the domain is barely larger than the frame, and rejects against a set
  when it is roomy. Every dtype is covered, nulls are exempt (a nullable
  unique column may repeat nulls and nothing else), and a domain too small to
  cover the row count is refused by name instead of quietly producing
  duplicates. `method="cartesian"` holds unique columns out of the coverage
  product and draws them once over the finished frame.
- `__unique_together__` is generated. A pass resamples the rows repeating a
  combination an earlier row already used, so only the repeats move and the
  rest keep the values their own columns' weights and bounds gave them. Rows
  with a null member are exempt, matching validation. A group whose columns
  cannot take enough distinct combinations is refused, naming the group; a
  foreign-keyed member is never resampled, since that would break its key.

- The Rust boundary is typed. Python builds one `ColumnPlan` per column (a
  `#[pyclass]` validated at construction, with errors naming the column)
  instead of a positional tuple. Bounds cross as an `i64`, `u64` or `f64`,
  so `Int64`/`UInt64` bounds beyond 2^53 are exact and the generation clamp
  for an unbounded distribution reaches the dtype's true limits. Columns
  with a finite domain (`choices`, `Enum`) receive indices back and the typed
  values are gathered on the Python side, so a `datetime`, `bytes` or `True`
  choice never passes through a string; choices need only be distinct in the
  column's dtype, not as strings. `python/polspec/_polspec.pyi` is a stub for
  the extension; `src/` is split into `plan.rs`, `dist.rs` and `sample.rs`
  with unit tests under `cargo test`; a test compares the distribution
  parameter tables on both sides.
- `import polspec` works without the Rust extension: validation, spec files,
  the registry and the report renderers need no build. Only generation
  imports it, and raises one actionable `ImportError` when it is missing.
- `Registry`: a declared set of specs. `Registry(Customers, Orders, ...)`
  resolves foreign keys declared against names (`resolve()`, running the
  checks a class-bound key gets at declaration), orders parents before
  children (`order()`), generates the whole set with every key satisfied
  (`generate_all`, with a per-spec seed so adding a table changes no other;
  `generate_related` for one spec and its ancestors), validates it in one
  call (`inspect_all`, `validate_all`), merges or checks shared categories
  (`catspec()`, `categories=`), writes and reads one file for the set
  (`to_yaml`/`from_yaml`, a `specs:` mapping plus `categories:`), collects
  specs from modules and directories (`from_module`, `discover`), and draws
  one entity-relationship diagram (`to_mermaid`). See the new *Multiple
  specs* guide.
- `inspect()`: validation results as data. `FrameSpec.inspect(df)` (and
  `polspec.validation.inspect(spec, df)`) returns a `ValidationReport` of
  `Finding` records -- each with a code, a stable key, the columns involved,
  a count, samples and code-specific details -- and never raises for a bad
  frame. `report.rows(finding)` and `report.failing_rows()` return the
  offending rows lazily; `by_column()`, `by_code()` and `to_json()` slice
  and export them. Checks and validators now carry samples too.
- `ValidationError.report` carries the same `ValidationReport`; `.errors` is
  still the list of messages.
- `polspec validate SPEC DATA [--references NAME=PATH] [--json]` on the
  command line, exiting 1 on findings, so a spec can gate a pipeline in CI.
- A foreign key whose parent was not supplied is a `foreign_key_unresolved`
  finding rather than a `ValueError`, matching how `generate()` already
  treats it; a parent lacking the referenced columns is a `foreign_key`
  finding.
- Spec files carry a `version:` (now 2). Files from version 1 are migrated
  on read; a file from a newer polspec is refused with a clear message. A
  key the reader does not know is an error naming the closest known key;
  `from_yaml(..., strict=False)` downgrades it to a warning.
- Foreign keys to other specs are written to YAML and Python as the target's
  name and read back unresolved, instead of being dropped with a warning.
- `polspec.serialization` is a package driven by one field registry
  (`fields.py`): YAML in both directions, generated Python, and the
  `import datetime` decision all derive from it, and a test asserts every
  dataclass field has an entry. `to_dict`/`from_dict` are public.
- `CatSpec` files keep choices recorded for plain string columns.
- `polspec.col()`, a small predicate language for rules, validators and
  checks: `col("total") >= col("subtotal")`, `col("email").str.contains("@")`,
  `is_in`, `is_between`, `is_null`, `&`/`|`/`~`, arithmetic, and string
  operations. A predicate evaluates like the Polars expression it stands
  for and, unlike one, is written to and read from YAML and generated
  Python. `__checks__` and `ColSpec.validators` written with `col()` now
  round-trip through `to_yaml`/`from_yaml` and `to_python`. Raw `pl.Expr`
  is still accepted and still warns on export.
- `ColRule.when` accepts a predicate, so a rule may depend on several
  columns. The one-column dict form is still accepted and converted.
- `TableSpec`: the spec as an immutable value. A `FrameSpec` class body now
  builds one, reachable as `Spec.spec`, and every verb (`generate`,
  `validate`, `to_yaml`, `to_markdown`, ...) is a function over it in
  `polspec.generation`, `polspec.validation`, `polspec.serialization` and
  `polspec.report`. `TableSpec` offers `with_columns`, `drop`, `select`,
  `rename`, `with_checks`, `with_foreign_keys`, `with_unique_together`,
  `with_name` and `with_catspec`; `FrameSpec.from_spec` wraps one in a class.
  See the new *Specs as values* guide.
- `FrameSpec.col(name)` reaches a column whatever it is called.
- `ForeignKey.references` may be a spec's name, for keys whose target is not
  importable where the key is declared.
- An exception hierarchy under `PolspecError`: `SpecError` for declarations
  that cannot mean anything, `ValidationError` for data that fails its spec,
  `GenerationError` when a spec cannot be turned into data (including every
  error raised inside the Rust engine), `SerializationError` for files that
  cannot be written or read, and `RegistryError`, reserved for the spec
  registry. All are exported from `polspec`; see the new *Errors* reference
  page.

### Changed

- **Breaking.** `ColRule.when` no longer accepts the one-column dict
  (`{"column": "region", "equals": "UK"}`). `col()` is the only spelling:
  write `col("region") == "UK"`. A spec file written by an earlier version
  still loads -- its conditions are converted as the file migrates -- but a
  file declaring the current version must carry the predicate form. The error
  names the column and says what to write.
- **Breaking.** The `le` and `ge` condition keys are gone; they were
  undocumented duplicates of `lte` and `gte`. (`le`/`ge` remain the canonical
  operator names in a predicate's *data* form, which is unrelated.)
- **Breaking.** `polspec.serialization` no longer re-exports the names of the
  pre-package module layout: `_YAML_DTYPES`, `_YAML_NAME_TO_DTYPE`,
  `_dtype_to_yaml`, `_dtype_from_yaml`, `_dtype_to_python`, `_colspec_to_yaml`,
  `_colspec_from_yaml` and `_colspec_to_python`. Use the names in
  `polspec.serialization.fields` and `polspec.serialization.dtypes`.
- **Breaking.** `ColRule.when` is evaluated against the frame as it stands
  when the rule runs, not against the freely generated values. Rules and
  foreign keys are applied in dependency order, so a rule keyed on a column
  that another rule or a foreign key rewrites now reads the rewritten values
  -- the ones `validate()` checks it against. Chained rules, chained foreign
  keys, and a rule keyed on a foreign-keyed column all round-trip; the values
  a given seed produces for such a spec change.
- **Breaking.** Two columns whose rules each read what the other writes have
  no order that satisfies both, and are now refused at declaration with a
  `SpecError` naming them.
- **Breaking.** A `ForeignKey` whose parent's declared domain does not fit
  inside its own column's is refused at declaration (or when a `Registry`
  resolves a key that names its target as a string). A key overwrites its
  column with the parent's values, so `bounds=(1, 50)` on a column
  referencing keys in `100..200` could only ever generate data that fails its
  own validation. A column declaring no `bounds` or `choices` still accepts
  anything.
- **Breaking.** `unique=True` can no longer be combined with `weights`, a
  non-uniform `distribution`, or `rules`. The first two describe how often a
  value recurs, which a draw without replacement has no room for; a rule
  assigns from a fixed set, which is how duplicates would get back in. Each
  is refused at declaration rather than silently ignored.
- **Breaking.** A column carrying `rules` may not also be part of a
  `__unique_together__` group, for the same reason: the repair that separates
  repeated combinations would overwrite what the rule put there.
- **Breaking.** A `ForeignKey` filling a `unique=True` column now refuses when
  the parent holds fewer distinct values than there are rows, instead of
  falling back to sampling with replacement and producing the duplicates the
  column forbids.
- `polspec test` no longer emits `validate_unique=False` in generated tests.
  Uniqueness is generated now, so the generated test asserts it.
- A foreign key spanning textual dtypes -- a `String` column referencing an
  `Enum` key, which declaration has always allowed and generation has always
  handled -- now validates instead of raising `SchemaError` from the
  anti-join. The parent's keys are cast to the local dtype for the join, so
  `ValidationReport.rows()` still returns the frame's own rows unchanged.

- **Breaking.** Each column's generation seed is derived from the frame
  seed and the column's *name*, not its position, so inserting a column no
  longer reshuffles the columns after it. The values a given seed produces
  change from previous versions.
- `ColRule` application samples only as many values as there are matched
  rows and scatters them into place, instead of filling the whole column per
  rule.
- `polspec.validation` is a package (`report.py`, `constraints.py`); foreign
  key anti-joins are collected together with `pl.collect_all` instead of one
  `collect` per key.
- **Breaking.** `ColSpec.distribution` and `distribution_params` are stored
  in canonical form (`"exp"` becomes `"exponential"`, `mu`/`sigma` become
  `mean`/`std`, and so on), so spec files are canonical. Every alias is
  still accepted when declaring.
- **Breaking.** An unrecognised physical dtype in a `CatSpec` entry is now a
  `SerializationError` instead of silently becoming `UInt32`.
- **Breaking.** `ColRule.when` is a predicate after construction rather
  than a dict (`rule.when.root_names()` lists the columns it reads); rules
  in YAML are written in the predicate data form, and the old dict form is
  still read.
- **Breaking.** A column may now share a name with a `FrameSpec` method:
  the metaclass takes `ColSpec` attributes out of the class namespace, so
  `schema`, `tag` and friends no longer shadow anything and no longer warn.
  The private `_columns`, `_checks`, `_unique_together` and `_foreign_keys`
  class attributes are gone; read `Spec.spec.columns` and friends instead.
- **Breaking.** `ForeignKey.references` is the target's *name* after
  construction (the bound spec is available as `ForeignKey.target`), and
  `references={...}` on `generate`/`validate` accepts the class, the
  `TableSpec` or the name as key.
- **Breaking.** Removed: the `FrameSchema` alias; `FrameSpec.generate_catspec`,
  `write_catspec`, `infer_catspec` and `with_inferred_catspec` (use
  `catspec()`, `catspec().to_yaml()`, `CatSpec.infer(...)` and
  `with_catspec(CatSpec.infer(...))`); the `max_unique` and `bounds` alias
  keyword arguments of `from_dataframe` (use `max_unique_enum` and
  `calculate_bounds`).
- `to_yaml` and `to_python` share one set of warnings about what a file
  cannot hold.
- **Breaking, mildly.** Errors that were bare `ValueError` or `TypeError`
  are now the subclass above. Each keeps the built-in type it replaced, so
  `except ValueError` still catches it; only code matching on the exact type
  (`type(exc) is ValueError`) sees a difference. Plain argument misuse
  (`n < 0`, an unknown `method=`) is unchanged.
- The command line prints any `PolspecError` as a one-line `error: ...`
  instead of a `TypeName: message` line.

### Fixed

- Foreign-key sampling during generation drew parent keys from an unordered
  `unique()`, so the same seed could give different child rows between runs.
  The parent's distinct keys now keep their order and generation is
  reproducible.

## [0.1.5] - 2026-09-03

### Added

- `python/polspec/py.typed`, so type checkers use the package's annotations.
- `CONTRIBUTING.md`, this changelog, and a `.python-version` file.
- `examples/related_specs.py`: a worked example of four related specs
  (foreign keys, shared categories, rules, checks, a YAML-declared spec). It
  runs in CI as a smoke test.
- A release-workflow job that refuses a `vX.Y.Z` tag whose version does not
  match `pyproject.toml`.
- CI now runs `ruff check` with a wider rule set, `ruff format --check`, `cargo fmt --check`,
  `cargo clippy -D warnings` and `cargo test`, tests on macOS as well as
  Linux and Windows, and tests against the newest Polars release inside the
  declared bound. The docs build runs strictly on pull requests.
- Release builds now produce wheels for Linux aarch64 and macOS (x86_64 and
  arm64) alongside Linux and Windows x86_64, plus an sdist, and only publish
  when the test workflow is green.
- An optional `.pre-commit-config.yaml` with ruff and cargo fmt hooks.
- `tests/test_colspec.py` (2,000 lines, unsectioned) is split into
  `test_generation.py`, `test_rules.py`, `test_serialization.py`,
  `test_profiler.py`, `test_framespec.py`, `test_report.py` and
  `test_foreign_key.py`, each with a docstring saying what it covers.

### Changed

- The crate version in `Cargo.toml` is a placeholder; `pyproject.toml` is the
  only place the version is set, so `uv version --bump` works.
- The `parquet`, `ipc` and `all` extras (all identical) are replaced by a
  single `arrow` extra. Install with `polspec[arrow]` for the Parquet and
  Arrow IPC sinks.
- `polars` is bounded to `<2`; the Rust extension is coupled to a Polars
  release line.
- The abi3 floor is now Python 3.12, matching `requires-python`.
- `cargo test` links again (`extension-module` is no longer an unconditional
  crate feature; maturin enables it).

### Fixed

- Repository URL in package metadata pointed at the repository's old name.
- README claimed the license was unspecified; it is MIT.
- Documentation: `ColSpec(col_name=...)` is now described in *Declaring
  columns*, `FrameSpec.to_python()` in *YAML specs*, the getting-started
  example imports `date`, and the architecture page lists the `cli` module
  and its tests.

## [0.1.4] - 2026-09-02

### Added

- `polspec schema infer --output spec.py` and `FrameSpec.to_python()`, which
  write a spec as an editable Python module rather than YAML.

## [0.1.3] - 2026-09-01

### Added

- `ColSpec(col_name=...)`, so a column's name in data may differ from the
  attribute name used to declare it.

### Changed

- Roadmap expanded with detailed plans for a spec registry, structured
  validation results, and generation guardrails.

## [0.1.2] - 2026-08-31

Version bump only; no user-facing change.

## [0.1.1] - 2026-08-31

### Added

- Test workflow on GitHub Actions (Linux and Windows, Python 3.12 to 3.14).

### Fixed

- `ColSpec.dtype` accepts a dtype class as well as an instance.

## [0.1.0] - 2026-08-31

First tagged release.

- `ColSpec` and `FrameSpec`: declare a Polars schema with nullability,
  bounds, string lengths, choices and weights, distributions, tags, and
  conditional `ColRule`s.
- `generate()` backed by a parallel Rust extension, `method="cartesian"` for
  coverage sets, batched generation and Parquet/CSV/IPC/NDJSON sinks.
- `validate()` collecting every violation in one Polars aggregation, with
  column validators, multi-column `Check`s, composite uniqueness and
  `ForeignKey`s.
- `CatSpec` registries for shared `Enum`/`Categorical` domains.
- YAML round-trip, `from_dataframe()` profiling, Markdown and Mermaid output.
- CLI: `polspec schema infer`, `polspec schema new`, `polspec test`.
- Documentation site, comparison guide, and release automation.

[Unreleased]: https://github.com/MaxwellB13/polspec/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/MaxwellB13/polspec/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/MaxwellB13/polspec/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/MaxwellB13/polspec/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/MaxwellB13/polspec/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/MaxwellB13/polspec/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/MaxwellB13/polspec/compare/v0.1.5...v0.2.0
[0.1.5]: https://github.com/MaxwellB13/polspec/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/MaxwellB13/polspec/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/MaxwellB13/polspec/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/MaxwellB13/polspec/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/MaxwellB13/polspec/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MaxwellB13/polspec/releases/tag/v0.1.0
