# Comparison to other approaches

polspec sits at the intersection of two things usually solved by separate
tools: generating test data, and validating that data against a schema. This
page is about that intersection — what generating *and* validating from one
declaration buys you that the alternatives don't, where those alternatives
are still the better tool, and the actual numbers behind the speed claim.

## Benchmarks

`benchmarks/bench_generate.py` generates the same four-column frame — a
non-nullable string, a nullable enum, a nullable bounded int, a nullable
bounded float — three ways: polspec's Rust generator, a hand-vectorized NumPy
implementation, and a pure-Python loop using `random`. All three produce an
equivalent `pl.DataFrame`, so the comparison is "how fast can each approach
hand back a usable frame," not raw loop speed in isolation.

```text
      n_rows |         rust |        numpy |       python
---------------------------------------------------------
       1,000 |     0.0004s |     0.0008s |     0.0016s
      10,000 |     0.0005s |     0.0054s |     0.0147s
     100,000 |     0.0025s |     0.0493s |     0.1491s
   1,000,000 |     0.0112s |     0.4932s |     1.4918s
   5,000,000 |     0.0293s |     2.4706s |     skipped
  20,000,000 |     0.0835s |     9.8463s |     skipped
```

Measured on this machine; yours will differ, and the shape matters more than
the absolute numbers. Two things worth reading off it:

- **The gap widens with size, not just the ratio.** At 1,000 rows all three
  are fast enough that the difference doesn't matter to a test suite. At
  20,000,000, pure Python is impractical (skipped past a 5-second cutoff at
  a much smaller size) and NumPy's ~2 million rows/second becomes a real wait
  in a CI loop, while polspec is still under 100ms.
- **NumPy's implementation is the hard-won version.** Its string column uses
  a fixed-width byte-array trick because NumPy has no efficient way to
  vectorize *ragged* per-row lengths — the other two implementations generate
  strings 5–15 characters long; NumPy's are fixed at 15 and decoded back down.
  That's not a knock on NumPy — it's the actual cost of writing this by hand:
  the fast version needs a specific trick per dtype, and someone has to know
  it.

Reproduce it yourself:

```bash
uv run --group bench python benchmarks/bench_generate.py
```

Generation speed is only half the story — [validation](guide/validating.md)
compiles every check across every column into a single Polars aggregation,
so validating a fifty-column table costs about the same as validating a
five-column one. That isn't benchmarked here, since there's no equivalent
"validate this by hand" baseline to compare it against.

## Compared to hand-written fixtures

The common alternative is a Python dict or list literal, copy-pasted between
test files and edited by hand when the shape needs to change:

```python
def make_customer_row(customer_id=1, tier="free"):
    return {"customer_id": customer_id, "tier": tier, "signed_up": "2023-01-01"}
```

This works, and for a handful of fixed cases it's often the right amount of
machinery. It stops working as the schema grows: the dtype lives nowhere —
`tier` being one of three strings is enforced by nobody until something
downstream breaks — and every edge case (a null, a boundary value, a specific
combination of two columns) is a row someone remembered to write by hand.
There's also nothing stopping the fixture and the real schema from drifting
apart; the dict doesn't know the pipeline added a column last month.

A `ColSpec` declaration is both the definition and the generator: the dtype,
the bound, and the domain are enforced the same way whether you're generating
data or checking it, and [`method="cartesian"`](guide/generating.md#coverage-methodcartesian)
covers the boundary/null cases that hand-written fixtures tend to under-cover
because nobody thought to write them.

## Compared to Faker and similar

[Faker](https://faker.readthedocs.io/) and libraries built on it are the
right tool for *semantically realistic* values — names that look like names,
addresses that parse like addresses, emails with plausible domains. polspec
doesn't try to compete there: its strings are bounded-length ASCII, not
locale-aware people or places, because it's solving a different problem —
statistically-shaped data that respects a schema, not human-plausible data
that respects cultural conventions.

The two combine rather than compete. A Faker-generated pool of realistic
values becomes a `ColSpec.choices` list; polspec supplies the bounds,
nullability, cross-column rules, and cross-table referential integrity that
sit around it:

```python
import polars as pl
from faker import Faker
from polspec import ColSpec, FrameSpec

fake = Faker()
first_names = list({fake.first_name() for _ in range(500)})  # choices must be distinct

class Customers(FrameSpec):
    name = ColSpec(pl.String, choices=first_names)
    signup_bonus = ColSpec(pl.Float64, bounds=(0.0, 50.0))
```

What Faker doesn't do on its own is hand back a typed `pl.DataFrame`, enforce
a bound, or keep a foreign key consistent across two generated tables — those
are the parts of the problem polspec is actually for.

## Compared to NumPy or a bespoke script

The benchmark above *is* this comparison: a hand-written NumPy implementation
is faster than pure Python and can be made fast enough for most purposes, but
someone has to write it, and it has to be rewritten — bounds, nullability,
dtype casts — for every new column and every schema change. There's also
nothing left over afterward: the script that generated the data has no
relationship to a validator that checks it, because there was never a shared
declaration for the two to share.

polspec's Rust generator is faster than a hand-written NumPy version because
it doesn't pay Python's per-call overhead and fills columns in parallel — but
the bigger difference for day-to-day use is that the declaration doesn't have
to be rewritten by hand for each column, and the same one both generates and
validates.

## Compared to property-based testing (Hypothesis)

[Hypothesis](https://hypothesis.readthedocs.io/) solves a genuinely different
problem well: given a strategy for producing values, explore the space of
possible inputs, and when one fails, *shrink* it to the smallest failing
case. polspec has no shrinking and makes no attempt at exhaustive space
exploration — `method="cartesian"` is a fixed, deterministic set of
known-important combinations (every enum value, every numeric sign, null),
not an open-ended search.

These are complementary rather than competing: a `FrameSpec.generate(n,
seed=...)` call is a perfectly good data source *inside* a Hypothesis
strategy or a `@given` test, if what you need is Hypothesis's shrinking on
top of polspec's schema-shaped, Polars-native output.

## Compared to data-quality frameworks (Great Expectations, pandera, ...)

These frameworks are built around a different center of gravity: validating,
profiling, and monitoring data that already exists — often production
tables, with drift detection and reporting as first-class concerns. That's a
larger and more operational surface than polspec's `validate()`, which is
schema-shaped correctness checking, not statistical monitoring.

The distinguishing feature runs the other way, too: most validation-first
tools don't generate matching synthetic data for you. `FrameSpec` is meant to
be small enough to declare once and use for both jobs in a test suite, not to
replace a data-quality platform watching a production warehouse.

## What polspec doesn't try to be

Worth being direct about, in the same spirit as the
[known limitations](reference/limitations.md) and
[roadmap](reference/roadmap.md) pages:

- **Not a realistic-fake-data library.** No locales, no plausible names or
  addresses out of the box — pair it with Faker for that.
- **Not a data-quality or monitoring platform.** No drift detection, no
  profiling dashboards, no anomaly scoring.
- **Not a property-based shrinking engine.** No search, no shrinking —
  `method="cartesian"` is a fixed set of known-important cases, not an
  open-ended exploration.
- **Not feature-complete yet.** Nested dtypes (`List`/`Struct`/`Array`) aren't
  generatable, and a handful of constraints are validated but not yet
  generated — see [Known limitations](reference/limitations.md).

## Where it fits

Polars-native pipelines that need fast, schema-shaped synthetic data and
matching validation from one declaration — especially across several related
tables via `ForeignKey`, at volumes where a pure-Python or pandas generator
starts to cost real CI time, in tests that need to stay hermetic. See
[Testing pipelines](guide/testing.md) for that in practice.
