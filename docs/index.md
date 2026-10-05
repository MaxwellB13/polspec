# polspec

Declare a Polars schema once. Generate data that matches it, and validate data
against it — from the same declaration.

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

The generator is written in Rust and runs the columns in parallel, so a spec
that describes a realistic table produces millions of rows in the time it takes
to describe one.

## Why two directions from one declaration

Most schema tools do one or the other. A validation library tells you when
production data drifted; a fixture library gives you something to test against.
Keeping both behind one declaration means the fixtures and the contract cannot
disagree — and where they might, polspec has a test suite whose whole job is to
catch it (see [Known limitations](explanation/limitations.md)).

That is the practical payoff: the data in your tests is data your validator
already accepts, so a test that passes locally is not passing on a shape
production will reject.

## What you can declare

<div class="grid cards" markdown>

- **Types and shape**

    Every dtype, generated and validated — integers, floats, `Decimal`,
    booleans, strings, binary, all four temporal types, `Enum`,
    `Categorical`, and a `List`, `Array` or `Struct` of any of them, nested
    to any depth, and on Polars 2 a `Map` — plus nullability, bounds, string
    lengths, value domains, named string formats such as `uuid4` and
    `email`, and regex patterns.

    [Declaring columns](how-to/columns.md) ·
    [String formats](how-to/formats.md)

- **Rules and invariants**

    Conditional values, single-column validators, multi-column checks,
    composite uniqueness, foreign keys between specs, and parent/child
    hierarchies with a bounded depth.

    [Constraints](how-to/constraints.md) ·
    [Hierarchies](how-to/hierarchies.md)

- **Checking data**

    `validate()` raises on any breach; `inspect()` returns every finding as
    data, with the offending rows a filter away. `read()` loads a data file
    in the spec's terms first, parsing dates, decimals and durations that
    arrived as text.

    [Validating data](how-to/validating.md)

- **Data on demand**

    Random or coverage-guaranteeing generation, reproducible seeds, and a
    `LazyFrame` that generates only the columns and rows a plan asks for —
    streaming straight to Parquet, CSV, Arrow IPC or NDJSON.

    [Generating data](how-to/generating.md)

- **Specs from elsewhere**

    Infer a spec by profiling an existing DataFrame, load one from YAML so
    non-Python tooling can read it too, and keep a set of related specs in
    one `Registry` that generates and validates them together.

    [Specs as files](how-to/files.md) · [Multiple specs](how-to/registry.md)

- **Fake data from real data**

    `polspec.synthesize` turns a real dataset -- a frame, a lazy scan or a
    file -- into a fake one with the same schema, shapes, frequencies and
    keys, and none of its rows or sensitive values.

    [Fake data from real data](how-to/synthesizing.md)

- **What changed**

    Diff two versions of a spec, or a spec against data, into a report that
    says what moved and whether a frame that passed before could fail now.

    [Drift](how-to/drift.md)

- **From the command line**

    `polspec generate` writes a data file from a spec, `polspec validate`
    checks one, `polspec diff` and `polspec drift` gate a pull request on
    what changed, and `polspec test` turns a schema into a pytest round-trip.

    [Command line](how-to/cli.md)

</div>

## Install

```bash
uv add polspec           # preferred
pip install polspec      # alternative
```

The generator is a compiled Rust extension, but wheels are published for Linux
(x86_64, aarch64), macOS (Intel and Apple silicon) and Windows (x86_64), so
installing needs no Rust toolchain. At runtime polspec needs Polars and
PyYAML, and on Windows `tzdata` for time-zone data.

Building from a checkout — which does need Rust and
[maturin](https://www.maturin.rs) — is covered in
[CONTRIBUTING.md](https://github.com/MaxwellB13/polspec/blob/main/CONTRIBUTING.md).

## Where to go next

Start with [Getting started](tutorial/getting-started.md) for the full loop — declare,
generate, validate — in about five minutes. If you're weighing polspec
against a hand-rolled fixture, Faker, or a data-quality framework, see
[Comparison to other approaches](explanation/comparison.md) for where each one fits and
the benchmark numbers behind the speed claim.

## For language models

The documentation is published in the [llms.txt](https://llmstxt.org) format:
[`/llms.txt`](https://maxwellb13.github.io/polspec/llms.txt) indexes every
page, and [`/llms-full.txt`](https://maxwellb13.github.io/polspec/llms-full.txt)
carries the full text of all of them -- including the API reference, expanded
to signatures and docstrings -- in one file.
