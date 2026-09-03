# Changelog

All notable changes to polspec are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Until 1.0, minor
versions may break the Python API, the YAML format, and the values a given
seed produces; see
[Roadmap and stability](https://maxwellb13.github.io/polspec/reference/roadmap/).

## [Unreleased]

### Added

- `python/polspec/py.typed`, so type checkers use the package's annotations.
- `CONTRIBUTING.md`, this changelog, and a `.python-version` file.
- `examples/related_specs.py`: a worked example of four related specs
  (foreign keys, shared categories, rules, checks, a YAML-declared spec). It
  runs in CI as a smoke test.
- A release-workflow job that refuses a `vX.Y.Z` tag whose version does not
  match `Cargo.toml`.
- CI now runs `ruff check` with a wider rule set, `cargo fmt --check`,
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

- The package version is read from `Cargo.toml`; `pyproject.toml` declares
  `dynamic = ["version"]`.
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

[Unreleased]: https://github.com/MaxwellB13/polspec/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/MaxwellB13/polspec/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/MaxwellB13/polspec/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/MaxwellB13/polspec/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/MaxwellB13/polspec/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/MaxwellB13/polspec/releases/tag/v0.1.0
