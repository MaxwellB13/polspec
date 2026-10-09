# Contributing

polspec is a Python package over a Rust extension. The Python side owns the
vocabulary (what a column can declare and what it means); the Rust side owns
the inner loop that fills arrays with values. See
[Architecture](https://maxwellb13.github.io/polspec/explanation/architecture/)
for the module map.

## Set up

You need Python 3.12+, [uv](https://docs.astral.sh/uv/), and a Rust toolchain
(`rustup`).

```bash
git clone https://github.com/MaxwellB13/polspec.git
cd polspec
uv sync --group dev          # Python deps, including maturin and pyarrow
uv run maturin develop --release
```

`maturin develop` compiles the extension and installs the package into the
project's virtual environment as editable. Re-run it whenever `src/` changes;
Python-only edits are picked up immediately.

`uv run` syncs the environment first, and a sync that finds the project
changed rebuilds it through uv's own backend -- a *debug* build, two to four
times slower, which the benchmarks will report as a regression in every
case at once. When that happens, re-run `maturin develop --release`, or use
`uv run --no-sync` to leave the installed build alone.

## Generated files

`docs/reference/cli.md` is generated from the `polspec` command's own parser
and committed, and a test fails if it is stale: regenerate it after adding,
removing or rewording a flag.

```bash
uv run python scripts/generate_cli_reference.py
```

`docs/llms.txt` and `docs/llms-full.txt` follow the
[llms.txt convention](https://llmstxt.org): an index of the documentation and
its full text, published at the site root so a language model can read the
library's documentation in one fetch. They are built from the nav in
`zensical.toml`, the pages, and the live docstrings -- by the docs workflow,
before the site, and are not committed. To see them locally, run
`uv run python scripts/generate_llms_txt.py` (it rebuilds the CLI reference
too); `.gitignore` keeps the output out of a commit.

## Check your change

Run everything CI runs before opening a pull request:

```bash
uv run pytest                                # Python test suite
uv run pytest --cov                          # with coverage; fails under the floor
uv run ruff check . && uv run ruff format --check .
uv run ty check                              # type checker; nothing is suppressed
uv run lint-imports                          # the package's layering, [tool.importlinter]
cargo fmt --check
cargo clippy --release -- -D warnings
cargo test --release                         # Rust unit tests
uv run --group docs zensical build --strict  # the docs, as the docs workflow builds them
uv run python examples/related_specs.py      # worked example, doubles as a smoke test
```

A change to the generator is worth a benchmark as well:
`uv run --group bench python benchmarks/bench.py compare` runs it, and
`record` then `check` guard a change against a local baseline.

On Windows, the `cargo test` binary links against the Python DLL, so the
interpreter's directory has to be on `PATH` — without it the tests fail to
start with `STATUS_DLL_NOT_FOUND` (exit code `0xc0000135`) rather than
anything that names the cause:

```bash
PATH="$(uv run python -c 'import sys; print(sys.base_prefix)'):$PATH" cargo test
```

CI runs on Linux, so this only affects local runs.

Optionally, install the pre-commit hooks so ruff and `cargo fmt` run on
every commit:

```bash
uv run --with pre-commit pre-commit install
```

The docs build with [zensical](https://pypi.org/project/zensical/):

```bash
uv run --group docs zensical serve           # live preview
uv run --group docs zensical build --strict  # what the docs workflow runs
```

## Deep runs

CI runs each property in `tests/contracts/test_properties.py` as many times
as it affords. Before a release, or after a change to generation,
validation or rendering, run them deep -- on the locked Polars and on the
newest 2.x:

```bash
POLSPEC_DEEP_EXAMPLES=3000 uv run pytest tests/contracts/test_properties.py
POLSPEC_DEEP_EXAMPLES=3000 uv run --with "polars==2.0.0" pytest -p no:cacheprovider tests/contracts/test_properties.py
```

A failure prints the shrunk spec and a `@reproduce_failure` blob that
replays it. A failure seen once in thousands of draws is still a failure:
every defect the last few reviews found sat that far out.

What polspec writes for people is checked by the parsers people read it
with: drawn specs' ER diagrams with Mermaid 11 and 10, their data
dictionaries' tables with GitHub-flavoured Markdown rules (Node needed):

```bash
cd scripts/deep/render && npm install && cd -
uv run python scripts/deep/render/run.py 400
```

`scripts/deep/polars2_eval_chunks_repro.py` reproduces, without polspec,
the Polars 2.0.0 streaming-engine panic that validation avoids by running on
the in-memory engine; run it on a new Polars to see whether it is fixed.

## Seeded output

`tests/contracts/test_seeded.py` checks that a seed generates the same data
as `tests/contracts/seeded.json` records, case for case, on whichever Polars
runs it -- so CI's floor, lock and newest jobs each hold every supported
Polars to it. A column drawn from a non-uniform distribution is recorded
per operating system (`case@win32`): its sampler uses the platform's math
library, so the rewrite records only the OS it runs on, and the others skip
it (Known limitations). A change that alters seeded output on purpose
rewrites the reference, and says so in the changelog:

```bash
POLSPEC_WRITE_SEEDED=1 uv run pytest tests/contracts/test_seeded.py
POLSPEC_WRITE_SEEDED=1 uv run --with "polars==2.0.0" pytest -p no:cacheprovider tests/contracts/test_seeded.py
```

The second run adds the cases only Polars 2 can generate (`Map`).

## Deprecations

A public name is never renamed or removed in place. It is deprecated first:
it keeps working, unchanged, and each use warns with a `DeprecationWarning`
naming what to use instead and the release it goes in.

- Mark a function with `polspec._deprecation.deprecated(old, use=...)`, or
  call `warn_deprecated` where a decorator does not fit (an argument, an
  attribute).
- A name deprecated in 0.x is removed in 1.0, and after at least one minor
  release of warning. In 1.x a removal waits for the next major, after at
  least two minor releases of warning.
- Say so in the changelog under *Deprecated*, with the replacement; mark it
  "(deprecated)" in the API reference; and test that it warns and still
  works.

## Conventions

- **Generate and validate must agree.** Anything `generate()` produces,
  `validate()` must accept. `tests/contracts/test_roundtrip.py` pins that
  property over the case catalogue in `tests/cases.py`,
  `tests/contracts/test_streaming.py` pins it through `generate_batches()`
  and `scan()`, and `tests/contracts/test_properties.py` pins it over specs
  Hypothesis draws. A new field or dtype belongs in `tests/cases.py`, and in
  the strategy in `test_properties.py`. A known gap is recorded once in
  [`docs/explanation/limitations.md`](docs/explanation/limitations.md) and once as
  an `xfail(strict=True)` test, so fixing it forces the docs to be updated.
- **Error messages name the fix.** Say what was declared, what was expected,
  and what to change. Look at the existing `ValueError`s in
  `python/polspec/spec.py` for the register.
- **Two tables must match.** The distribution parameter aliases live in both
  `python/polspec/distributions.py` and `DistKind::from_spec` in
  `src/lib.rs`. Change them together.
- **A new `ColSpec` field touches four registries**: the `TYPE_CHECKING`
  constructor beside the fields, the serialization `Field` list, the drift
  comparator table, and -- if it is a validation switch -- the `FrameSpec`
  facade signature. `tests/contracts/test_parity.py` holds a test for each,
  and fails naming what is missing.
- **Tests live in a folder by what they cover**: `contracts/` (the
  round trip, whole, streamed and drawn; the parity tests), `declaration/`, `generation/`,
  `validation/`, `drift/`, `serialization/`, `features/` (one file per
  feature across every verb), `cli/`, `docs/`. Every module opens with a
  docstring saying what it covers; `tests/helpers.py` holds what several
  share, and `tests/cases.py` the case catalogue the contracts share.
- **What the constructor accepts is not what the instance holds.** `ColSpec`,
  `Check` and `ForeignKey` annotate their fields with the normalised form
  (`pl.DataType`, `Bound`, tuples) and declare the accepted forms in an
  `__init__` under `if TYPE_CHECKING:`. Read a raw value through `Any` inside
  `__post_init__`; do not widen the field to describe its input.
- **Docs are part of the change.** A new field, option or CLI flag lands with
  its guide page and a `CHANGELOG.md` entry under *Unreleased*.
- Commit messages follow the existing `type: summary` style
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `chore:`).

## Releasing

The version lives in exactly one place: `project.version` in
`pyproject.toml`. maturin prefers it over the crate version in `Cargo.toml`,
which is a placeholder.

1. Bump it and refresh the lock file:

   ```bash
   uv version --bump patch   # or minor / major
   uv lock
   ```

   A feature is a *minor* bump, not a patch one, whatever its size.
2. Move the *Unreleased* section of `CHANGELOG.md` under a
   `## [X.Y.Z] - YYYY-MM-DD` heading, add its compare link at the foot of the
   file, and leave a fresh empty *Unreleased* behind. The release workflow
   refuses a tag whose version has neither, so this is not a step that can be
   skipped and fixed later — v0.4.0 and v0.4.1 both were, and shipped
   claiming a released breaking change had not happened yet.
3. Commit, then tag `vX.Y.Z` and push the tag.
4. The release workflow checks the tag against `pyproject.toml` and
   `CHANGELOG.md`, builds wheels, publishes to PyPI, and *drafts* a GitHub
   release with the wheels attached and notes generated from the merged
   pull requests, grouped by label as `.github/release.yml` says.
5. Read the draft, edit the notes if they need it, and publish it from the
   release page. Nothing on GitHub is public until then; PyPI already is.
