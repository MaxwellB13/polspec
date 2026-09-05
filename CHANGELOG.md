# Changelog

All notable changes to polspec are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Until 1.0, minor
versions may break the Python API, the YAML format, and the values a given
seed produces; see
[Roadmap and stability](https://maxwellb13.github.io/polspec/reference/roadmap/).

## [Unreleased]

### Added

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

### Fixed

- Foreign-key sampling during generation drew parent keys from an unordered
  `unique()`, so the same seed could give different child rows between runs.
  The parent's distinct keys now keep their order and generation is
  reproducible.

### Changed

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

[Unreleased]: https://github.com/MaxwellB13/polspec/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/MaxwellB13/polspec/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/MaxwellB13/polspec/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/MaxwellB13/polspec/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/MaxwellB13/polspec/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/MaxwellB13/polspec/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MaxwellB13/polspec/releases/tag/v0.1.0
