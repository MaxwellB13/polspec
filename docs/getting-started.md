# Getting started

## Declare a spec

A spec is a class. Subclass `FrameSpec` and assign a `ColSpec` per column, in
the order the columns should appear.

```python
import polars as pl
from polspec import ColSpec, FrameSpec

class Customers(FrameSpec):
    customer_id = ColSpec(pl.Int64, bounds=(1, 100_000), unique=True)
    name        = ColSpec(pl.String, string_length=(4, 20))
    tier        = ColSpec(pl.Enum(["free", "pro", "enterprise"]))
    signed_up   = ColSpec(pl.Date, bounds=(date(2020, 1, 1), None))
    churned     = ColSpec(pl.Boolean, nullable=True, null_probability=0.3)
```

Nothing runs at declaration time except validation of the declaration itself.
A contradictory spec fails here, at the line that caused it, rather than
thousands of rows later:

```python
ColSpec(pl.Int8, bounds=(0, 1_000))
# ValueError: ColSpec.bounds max (1000) is outside the range Int8 can represent [-128, 127]
```

## Generate data

```python
df = Customers.generate(10_000, seed=42)
```

`seed` makes the result reproducible across processes and machines. Omit it and
each call differs.

```python
Customers.generate(500, seed=7).equals(Customers.generate(500, seed=7))  # True
```

## Validate data

`validate()` checks a DataFrame or LazyFrame against the same declaration and
returns it, so it drops into a pipeline:

```python
clean = Customers.validate(raw_df, cast=True)
```

Every breach is collected before anything is raised, so one call tells you
everything that is wrong rather than only the first thing:

```python
from polspec import ValidationError

try:
    Customers.validate(broken_df)
except ValidationError as err:
    for problem in err.errors:
        print(problem)
```

```text
Column 'customer_id': unique column contains 3 duplicate value(s). Duplicate samples: [7, 12, 41]
Column 'tier': found 2 invalid value(s) not in allowed choices/categories ['free', 'pro', 'enterprise']. Invalid samples: ['trial']
Column 'name': non-nullable column contains 1 null value(s)
```

!!! tip "One pass, not one per column"

    Every check across every column is compiled into a single Polars
    aggregation and evaluated in one scan. Validating a wide table costs about
    the same as validating a narrow one.

## Handle data that nearly fits

Real input rarely arrives in exactly the declared shape. `validate()` takes
policies for the two structural mismatches:

```python
Customers.validate(
    df,
    extra_cols="drop",     # "raise" (default) | "drop" | "allow"
    missing_cols="raise",  # "raise" (default) | "add" | "allow"
    strict_dtypes=False,   # allow Int32 where Int64 was declared, String for an Enum
    cast=True,             # cast surviving columns to the declared dtype
)
```

By default a String column arriving where an `Enum` was declared is accepted —
that is how data comes back from CSV and JSON. `strict_dtypes=True` demands the
exact dtype.

## Infer a spec instead of writing one

Pointed at an existing DataFrame, polspec writes the spec for you:

```python
Profiled = FrameSpec.from_dataframe(existing_df, weights=True)
print(Profiled.to_markdown())
```

It infers nullability and observed null rates, narrows low-cardinality strings
to `Enum`, and — with `weights=True` — records how often each category actually
occurred, so regenerated data keeps the observed mix rather than a uniform one.

Treat the result as a first draft to edit, not a finished contract: it
describes the sample it saw, which may be narrower than the rule you actually
mean.

## Next

- [Declaring columns](guide/columns.md) — everything a `ColSpec` accepts
- [Constraints](guide/constraints.md) — rules, checks, uniqueness, foreign keys
- [Generating data](guide/generating.md) — coverage, batching, writing to files
