# polspec

Declare a [Polars](https://pola.rs) schema once. Generate data that matches
it, and validate data against it — from the same declaration.

> **Early alpha.** The API, the YAML format, and the exact values a given
> seed produces are all still moving. See
> [Roadmap and stability](https://maxwellb13.github.io/polspec/explanation/roadmap/)
> before depending on any of it.

```python
import polars as pl
from polspec import ColSpec, FrameSpec

class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, None), unique=True)
    status   = ColSpec(pl.Enum(["NEW", "PAID", "SHIPPED"]))
    total    = ColSpec(pl.Float64, bounds=(0.0, None))
    placed   = ColSpec(pl.Date, nullable=True)

df = Orders.generate(1_000_000, seed=42)   # a million rows in well under a second
Orders.validate(df)                        # raises ValidationError on any breach
```

A validation library tells you when production data drifted. A fixture
library gives you something to test against. Keeping both behind one
declaration means the fixtures and the contract cannot quietly disagree — and
where they still can, it is written down in
[Known limitations](https://maxwellb13.github.io/polspec/explanation/limitations/),
each backed by a test that fails the moment it stops being true.

The generator is a Rust extension that fills columns in parallel; `validate()`
compiles every check across every column into a single Polars aggregation, so
validating a wide table costs about the same as a narrow one. Numbers, and
comparisons to NumPy and hand-written fixtures, are in
[Comparison](https://maxwellb13.github.io/polspec/explanation/comparison/).

## Install

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
uv sync --group dev
maturin develop --release
```

## Documentation

Full docs: **[maxwellb13.github.io/polspec](https://maxwellb13.github.io/polspec/)**

- **Tutorial** — [Getting started](https://maxwellb13.github.io/polspec/tutorial/getting-started/) ·
  [Related tables](https://maxwellb13.github.io/polspec/tutorial/related-tables/)
- **How-to** — [Declare columns](https://maxwellb13.github.io/polspec/how-to/columns/) ·
  [Constraints](https://maxwellb13.github.io/polspec/how-to/constraints/) ·
  [Generate](https://maxwellb13.github.io/polspec/how-to/generating/) ·
  [Validate](https://maxwellb13.github.io/polspec/how-to/validating/) ·
  [Specs as files](https://maxwellb13.github.io/polspec/how-to/files/) ·
  [Command line](https://maxwellb13.github.io/polspec/how-to/cli/)
- **Reference** — [API](https://maxwellb13.github.io/polspec/reference/api/) ·
  [Errors and findings](https://maxwellb13.github.io/polspec/reference/errors/)
- **Explanation** — [Architecture](https://maxwellb13.github.io/polspec/explanation/architecture/) ·
  [Generation and validation](https://maxwellb13.github.io/polspec/explanation/two-sides/) ·
  [Known limitations](https://maxwellb13.github.io/polspec/explanation/limitations/) ·
  [Roadmap](https://maxwellb13.github.io/polspec/explanation/roadmap/)

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
cargo test && cargo clippy --all-targets -- -D warnings
uv run --group docs zensical build --strict
uv run python examples/related_specs.py
```

`uv run --group bench python benchmarks/bench_generate.py` runs the generator
benchmark. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and the
release process, and [CHANGELOG.md](CHANGELOG.md) for what changed.

## License

[MIT](LICENSE).
