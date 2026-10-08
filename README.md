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

A validation library tells you when production data drifted, and
`Orders.drift(df)` says what moved. A fixture library gives you something to
test against. Keeping both behind one declaration means the fixtures and the
contract cannot quietly disagree — and where they still can, it is written
down in
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
```

polspec works with Polars 1.39 and later, Polars 2 included. Wheels are
published for Linux (x86_64, aarch64), macOS (Intel and Apple silicon) and
Windows (x86_64), so using polspec needs no Rust toolchain.
Building from a checkout is covered in [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

Full docs: **[maxwellb13.github.io/polspec](https://maxwellb13.github.io/polspec/)**.
Start with [Getting started](https://maxwellb13.github.io/polspec/tutorial/getting-started/),
find a task among the [how-to guides](https://maxwellb13.github.io/polspec/how-to/columns/),
or look a name up in the [API reference](https://maxwellb13.github.io/polspec/reference/api/).

## Development

Setting up a checkout, the checks CI runs, the conventions and the release
process are in [CONTRIBUTING.md](CONTRIBUTING.md); what changed is in
[CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE).
