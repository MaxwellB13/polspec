# polspec

Declare a Polars schema once. Generate data that matches it, and validate data
against it — from the same declaration.

```python
import polars as pl
from polspec import ColSpec, FrameSpec

class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, None), unique=True)
    status   = ColSpec(pl.Enum(["NEW", "PAID", "SHIPPED"]))
    total    = ColSpec(pl.Float64, bounds=(0.0, None))
    placed   = ColSpec(pl.Date, nullable=True)

df = Orders.generate(1_000, seed=42)   # a million rows is just as quick
Orders.validate(df)                    # raises ValidationError on any breach
```

The generator is written in Rust and runs the columns in parallel, so a spec
that describes a realistic table produces millions of rows in the time it takes
to describe one.

## Why two directions from one declaration

Most schema tools do one or the other. A validation library tells you when
production data drifted; a fixture library gives you something to test against.
Keeping both behind one declaration means the fixtures and the contract cannot
disagree — and where they might, polspec has a test suite whose whole job is to
catch it (see [Known limitations](reference/limitations.md)).

That is the practical payoff: the data in your tests is data your validator
already accepts, so a test that passes locally is not passing on a shape
production will reject.

## What you can declare

<div class="grid cards" markdown>

- **Types and shape**

    Every dtype polspec can generate — integers, floats, booleans, strings,
    binary, all four temporal types, `Enum` and `Categorical` — plus
    nullability, bounds, string lengths and value domains.

    [Declaring columns](guide/columns.md)

- **Rules and invariants**

    Conditional values, single-column validators, multi-column checks,
    composite uniqueness and foreign keys between specs.

    [Constraints](guide/constraints.md)

- **Data on demand**

    Random or coverage-guaranteeing generation, reproducible seeds, batched
    streaming straight to Parquet, CSV, Arrow IPC or NDJSON.

    [Generating data](guide/generating.md)

- **Specs from elsewhere**

    Infer a spec by profiling an existing DataFrame, or load one from YAML so
    non-Python tooling can read it too.

    [YAML specs](guide/yaml.md)

</div>

## Install

```bash
pip install polspec
```

Writing Parquet or Arrow IPC needs PyArrow:

```bash
pip install "polspec[all]"
```

Building from a checkout needs a Rust toolchain, since the generator is a
compiled extension:

```bash
maturin develop --release
```

## Where to go next

Start with [Getting started](getting-started.md) for the full loop — declare,
generate, validate — in about five minutes.
