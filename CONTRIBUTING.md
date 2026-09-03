# Contributing

polspec is a Python package over a Rust extension. The Python side owns the
vocabulary (what a column can declare and what it means); the Rust side owns
the inner loop that fills arrays with values. See
[Architecture](https://maxwellb13.github.io/polspec/reference/architecture/)
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

## Check your change

Run everything CI runs before opening a pull request:

```bash
uv run pytest                                # Python test suite
uv run ruff check . && uv run ruff format --check .
cargo test --release                         # Rust unit tests
cargo clippy --release
uv run python examples/related_specs.py      # worked example, doubles as a smoke test
```

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

## Conventions

- **Generate and validate must agree.** Anything `generate()` produces,
  `validate()` must accept. `tests/test_roundtrip.py` pins that property. A
  known gap is recorded once in
  [`docs/reference/limitations.md`](docs/reference/limitations.md) and once as
  an `xfail(strict=True)` test, so fixing it forces the docs to be updated.
- **Error messages name the fix.** Say what was declared, what was expected,
  and what to change. Look at the existing `ValueError`s in
  `python/polspec/spec.py` for the register.
- **Two tables must match.** The distribution parameter aliases live in both
  `python/polspec/distributions.py` and `DistKind::from_spec` in
  `src/lib.rs`. Change them together.
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

   Then move the *Unreleased* section of `CHANGELOG.md` under the new version.
2. Commit, then tag `vX.Y.Z` and push the tag.
3. The release workflow checks the tag matches `pyproject.toml`, builds wheels,
   publishes to PyPI, and attaches the wheels to a GitHub release.
