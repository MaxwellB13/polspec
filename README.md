# polspec

Declare a [Polars](https://pola.rs) schema once. Generate data that matches
it, and validate data against it — from the same declaration.

> **Early alpha.** The API, the YAML format, and the exact values a given
> seed produces are all still moving. See
> [Roadmap and stability](https://maxwellb13.github.io/polspec/reference/roadmap/)
> before depending on any of it.

```python
import polars as pl
from polspec import ColSpec, FrameSpec

class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, None))
    status   = ColSpec(pl.Enum(["NEW", "PAID", "SHIPPED"]))
    total    = ColSpec(pl.Float64, bounds=(0.0, None))
    placed   = ColSpec(pl.Date, nullable=True)

df = Orders.generate(1_000_000, seed=42)   # a million rows in well under a second
Orders.validate(df)                        # raises ValidationError on any breach
```

The generator is a Rust extension that fills columns in parallel — ten
million rows across four columns generates in well under 50ms on a release
build. `validate()` compiles every check across every column into a single
Polars aggregation, so validating a wide table costs about the same as a
narrow one.

## Why generate *and* validate from one declaration

A validation library tells you when production data drifted. A fixture
library gives you something to test against. Keeping both behind one
declaration means the fixtures and the contract can't quietly disagree — and
where they still can today, it's written down in
[Known limitations](https://maxwellb13.github.io/polspec/reference/limitations/),
each backed by a test that fails the moment it's fixed.

## Install

Package install (when published to PyPI)

```bash
uv add polspec           # preferred
pip install polspec      # alternative
uv add "polspec[arrow]"  # extra: PyArrow for the Parquet/IPC sinks
```

Not published to PyPI yet — install from a checkout. The generator is a
compiled Rust extension, so this needs a Rust toolchain and
[maturin](https://www.maturin.rs):

```bash
git clone https://github.com/MaxwellB13/polspec.git
cd polspec
uv sync --group dev        # or: pip install -e ".[arrow]" && pip install maturin
maturin develop --release
```

`maturin develop` builds the extension and installs the package into your
active environment, editable. Writing Parquet or Arrow IPC needs PyArrow,
included via `[arrow]`/the `dev` group above.

## What you can declare

- **Types and shape** — every scalar and temporal Polars dtype, nullability,
  bounds (including open-ended, `bounds=(0, None)`), string lengths, and
  value domains with weights.
- **Rules and invariants** — conditional values (`ColRule`), single-column
  validators, multi-column checks, composite uniqueness, and foreign keys
  between specs.
- **Data on demand** — random or coverage-guaranteeing (`method="cartesian"`)
  generation, reproducible seeds, batched streaming straight to Parquet, CSV,
  Arrow IPC or NDJSON.
- **Specs from elsewhere** — infer a spec by profiling an existing DataFrame,
  or load one from YAML.
- **Shared categories** — a `CatSpec` registry so several tables agree on an
  `Enum`/`Categorical` domain instead of each restating it.

```python
class Categories(CatSpec):
    STATUS   = pl.Enum(["NEW", "PAID", "SHIPPED"])
    CURRENCY = pl.Categorical(pl.Categories("CURRENCY", physical=pl.UInt8))

ColSpec(Categories.STATUS)
```

## Command line

```bash
polspec schema infer orders.parquet -o orders.yaml   # profile data into a schema
polspec schema new Payments -o payments.py           # or start from a blank one
polspec test orders.yaml -o test_orders.py           # generate a round-trip pytest file
```

The generated test disables whatever `validate()` flags a constraint
generation can't yet satisfy (`unique=True`, `__checks__`, ...) with a comment
explaining why, and skips a foreign key that needs parent data it wasn't
given — see [Command line](https://maxwellb13.github.io/polspec/guide/cli/).

## Documentation

Full docs: **[maxwellb13.github.io/polspec](https://maxwellb13.github.io/polspec/)**

- [Getting started](https://maxwellb13.github.io/polspec/getting-started/) · [Comparison to other approaches](https://maxwellb13.github.io/polspec/comparison/)
- [Declaring columns](https://maxwellb13.github.io/polspec/guide/columns/) · [Constraints](https://maxwellb13.github.io/polspec/guide/constraints/)
- [Generating data](https://maxwellb13.github.io/polspec/guide/generating/) · [Validating data](https://maxwellb13.github.io/polspec/guide/validating/)
- [Shared categories](https://maxwellb13.github.io/polspec/guide/categories/) · [YAML specs](https://maxwellb13.github.io/polspec/guide/yaml/)
- [Testing pipelines](https://maxwellb13.github.io/polspec/guide/testing/) · [Command line](https://maxwellb13.github.io/polspec/guide/cli/)
- [Architecture](https://maxwellb13.github.io/polspec/reference/architecture/) · [Known limitations](https://maxwellb13.github.io/polspec/reference/limitations/) · [Roadmap and stability](https://maxwellb13.github.io/polspec/reference/roadmap/)

To edit and preview the docs locally instead, see [zensical](https://pypi.org/project/zensical/):

```bash
zensical serve
```

## Benchmarks — polspec vs NumPy vs pure Python

The `benchmarks/bench_generate.py` script compares the Rust-backed generator to:
- a NumPy-based generator that builds equivalent columns, and
- a pure-Python implementation using `random`.

Run locally:

```bash
uv run --group bench python benchmarks/bench_generate.py
```

Example results (illustrative; your hardware will differ):

| n_rows      | polspec (Rust) |   NumPy |  Python |  Polspec v NumPy Speedup |
|-------------|----------------|--------:|--------:|-------------------------:|
| 1,000       | 0.0003s        | 0.0008s | 0.0016s |                     2.7x |
| 10,000      | 0.0006s        | 0.0052s | 0.0145s |                     8.7x |
| 100,000     | 0.0027s        | 0.0480s | 0.1447s |                    17.8x |
| 1,000,000   | 0.0083s        | 0.4867s | 1.4862s |                    58.6x |
| 5,000,000   | 0.0256s        | 2.4326s | skipped |                    95.0x |
| 20,000,000  | 0.0827s        | 9.7523s | skipped |                   117.9x |

*Measured 2026-09-03 on an Intel 13900K with 64GB DDR5; the same run backs the table in
[Comparison](https://maxwellb13.github.io/polspec/comparison/).*

Notes
- All three implementations emit a Polars `DataFrame` with the same schema to keep the comparison fair. The NumPy version uses a fixed-width trick for strings (since vectorized ragged strings are not available), and the pure-Python version builds lists then constructs a `DataFrame`.
- The script warms each implementation once, then times increasing sizes; the pure-Python path is skipped for very large sizes once it exceeds a cutoff.

## Development

After the install steps above:

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
cargo test --release && cargo clippy --release
uv run python examples/related_specs.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and the release
process, and [CHANGELOG.md](CHANGELOG.md) for what changed.

## License

[MIT](LICENSE).
