# Stability

What polspec promises: before 1.0, and from it. Each promise names the test
that holds it, so a release cannot break one without a test failing first.

## Where polspec is now

0.22 is the beta. The API is frozen pending 1.0: no public name is renamed
or removed in place, and what is deprecated is listed
[below](#deprecated). Until 1.0, two promises are weaker than they will
be:

- **Seeded output may change in any release.** When one changes what a seed
  produces, its changelog says so, by name. The
  [roadmap](roadmap.md#generated-values-may-change-until-10) lists every
  release that has.
- **A spec file's keys may change**, but only with a migration: a file
  carries its format `version:`, an older one is migrated as it is read,
  and one from a newer polspec is refused by name. So an upgrade never
  leaves a file unreadable.

### Deprecated

Nothing is deprecated. A name deprecated in 1.x is listed here, warns with a
`DeprecationWarning` naming its replacement, and keeps working until the
next major release removes it. To find what your code uses, run its tests
with deprecations as errors: `pytest -W error::DeprecationWarning`.

1.0 removed what 0.x had deprecated: `profile_dataframe`,
`polspec.serialization`'s five `*_FIELDS` tables, and `FrameSpec.checks()`,
`.foreign_keys()` and `.unique_together()`. The
[changelog](../changelog.md) gives the replacement for each.

*Held by* `tests/contracts/test_deprecation.py`, which checks that what is
listed here is exactly what warns.

## What 1.0 promises

### The public API

The public API is `polspec.__all__`: everything public is imported from
`polspec` itself. The [API reference](../reference/api/index.md) documents
each name. Every module is internal, says so in its docstring, and may
change in any release.

A public name is never renamed or removed in place. It is deprecated first:
it keeps working, and each use warns with a `DeprecationWarning` naming what
to use instead. In 1.x a deprecated name is removed only at the next major
release, after at least two minor releases of warning.

*Held by* `tests/contracts/test_boundary.py`, which holds the line between
public and internal, and `tests/contracts/test_contract.py`, which holds the
public names to a golden list: an addition is a deliberate edit, and a
removal fails until it has been through a deprecation.

### Spec files

A spec file any 1.x release writes, every later 1.x release reads, as the
same spec. Each file carries a format `version:`. A new version is added
only with a migration from every earlier one, so an old file is read as
what it declared. A file written by a newer polspec than the reader is
refused by name rather than misread.

*Held by* `tests/serialization/test_migrations.py`, which reads a file of
every earlier version, and `tests/contracts/test_contract.py`, which holds
a golden file of each version to the spec it declares and the current
version's file byte for byte.

### Seeded output

The same call -- spec, seed, row count and method -- gives the same frame
on every supported operating system and Polars, and on every later 1.x
release. A minor release changes what a seed produces
only to fix output that broke its own spec, since data that never met its
declaration is not worth preserving; the changelog names each such fix.
Anything else that changes seeded output waits for the next major release.

*Held by* `tests/contracts/test_seeded.py`, which checks the digest of
every case in the catalogue on every platform CI runs.

### The command line

The verbs, their flags and their exit codes:

| Exit code | Means |
|:--|:--|
| `0` | Success, and the data or spec passed |
| `1` | A reported error -- `error: ...` on stderr -- or a report that failed: `validate`, `diff` and `drift` exit `1` when the report says so |
| `2` | The command line itself was wrong: an unknown verb, flag or value |

Every `--json` document carries a `schema_version`. Its keys, and the types
of their values, are the promise; the values themselves move with the data.

*Held by* `tests/contracts/test_contract.py`, which holds each verb's
`--json` document to a golden copy of its keys and types, and each exit
code to the case that produces it.

### Finding and drift codes

Every finding `inspect()` reports, and every move `drift()` and `diff()`
report, has a code: the full lists are in [Errors and
findings](../reference/errors.md#finding-codes). A code is added only in a
minor release, and is never renamed or removed in 1.x, so code that groups
or filters by one keeps working.

*Held by* `tests/contracts/test_contract.py`, which holds the codes polspec
produces and the tables on that page to each other, in both directions.

## What is not promised

- **The wording of messages.** An error's message, a finding's `message`, a
  warning's text: they are written for a person, and get clearer. Match on a
  finding's `code`, and on an exception's class.
- **`repr`s**, and the layout of rendered documents (`to_markdown`,
  `to_mermaid`), beyond what they contain.
- **Performance.** polspec is benchmarked, and a regression is a bug, but a
  timing is not a promise.
- **What profiling infers.** `profile()`, `from_dataframe()` and
  `polspec schema infer` describe data as well as they can; a better
  description in a minor release is a fix.
- **Anything internal**: a module not named above, and any name starting
  with an underscore.

## Supported versions

- **Python:** every CPython release from 3.12 until its end of life, and
  each new release within one minor polspec release of its final. Today
  that is 3.12 to 3.15.
- **Polars:** 1.39 through 2.x. On Python 3.15, Polars 2 only: Polars 1
  does not run there. The floor moves only in a minor release, announced in
  the changelog of the release before. A new Polars major is supported in a
  minor release once the suite passes on it.
- **Operating systems:** wheels are built, and tested, for Linux (x86-64 and
  aarch64), macOS (Apple silicon, and Intel, built but not run) and
  Windows (x86-64).

Every pull request runs the suite on each supported Python on Linux, macOS
and Windows, and on Linux against both the lowest and the newest Polars
polspec allows. The deep property run is weekly.
