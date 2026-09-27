# Fake data from real data

`polspec.synthesize` turns a real dataset into a fake one that looks like
it: the same schema, the same shape of values in each column, the same
category mix and null rates -- and none of its rows. It is for the data you
cannot share, cannot copy into a test environment, or cannot keep: hand
over a stand-in instead.

```python
import math

import polars as pl
import polspec

# Stand-in for a real table: a key, a skewed amount, a weighted status,
# and a column of names nobody should see outside production.
real = orders.with_columns(
    amount=pl.Series([round(math.exp(3 + (i % 97) / 40), 2) for i in range(500)]),
    name=pl.Series([f"customer {i % 30}" for i in range(500)]),
)

fake = polspec.synthesize(real, seed=1, replace=["name"])

assert fake.schema == real.schema
assert fake.height == real.height
assert fake["order_id"].is_unique().all()        # a key stays a key
assert not set(fake["name"]) & set(real["name"])  # no real name carried over
```

The source can be a `DataFrame`, a `LazyFrame` -- anything Polars can
`scan_*`, a file, a glob, cloud storage -- or the path of a data file:

<!-- docs: skip -->
```python
fake = polspec.synthesize("customers.parquet", 1_000_000, seed=1)
fake = polspec.synthesize(pl.scan_parquet("s3://bucket/customers/*.parquet"), seed=1)
```

`n` defaults to as many rows as the source has; ask for more or fewer.

## The spec in between

`synthesize` is two steps: `polspec.profile` describes the source as a
`TableSpec`, and `generate` makes data from it. Take them one at a time when
the description is worth looking at, editing or keeping -- a reviewed spec
is a safer thing to share than a function call over real data:

```python
spec = polspec.profile(real, replace=["name"])

assert spec["amount"].distribution == "lognormal"   # the shape was fitted
assert spec["order_id"].unique                      # every value distinct: a key

FrameSpec.from_spec(spec).to_yaml("orders_profile.yaml")
fake = polspec.generate(spec, 10_000, seed=1)
```

## What carries over from the source

Everything a spec can say about a column, and nothing else:

| Carried over | Not carried over |
|:--|:--|
| dtypes, and the schema's column order | any row of the source |
| each column's null rate | any value of a column in `replace=` |
| numeric and temporal extremes | any relationship *between* columns |
| the distribution each numeric or temporal column follows | foreign keys between tables |
| a low-cardinality text column's **values** and how often each occurs | |
| string lengths | |
| which columns are keys | |

Two lines of that table decide what to put in `replace=`:

- **A text column with few distinct values keeps them.** A column of at
  most `max_unique_enum` (50) distinct values becomes an `Enum` of them, so
  a status column stays `NEW`/`PAID`/`SHIPPED` -- and a column of thirty
  customer names stays those thirty names. Name every column whose values
  are sensitive in `replace=`: it is then generated from its lengths alone.
- **Extremes are real values.** The smallest and largest amount, the
  earliest and latest date, become the fake column's bounds. If those are
  sensitive, edit them out of the profiled spec before generating.

## How close the fake data is

Each column is described on its own, so each column on its own looks like
the source:

- **Shape.** A numeric or temporal column's values are fitted against every
  distribution polspec can draw -- `normal`, `lognormal`, `gamma`,
  `exponential`, `beta`, `poisson` -- by drawing from each and comparing the
  draws with the data. The closest wins if it beats drawing evenly between
  the extremes; otherwise the column stays even. A log-normal amount comes
  out log-normal, a bell-curved age bell-curved, and a count as counts.
- **Keys.** An integer or text column whose values are all distinct, over
  at least a hundred of them, is declared unique, and the fake one is
  distinct too. Asked for more rows than the key's range holds, the range
  widens upward to make room -- and a nullable key's range has to hold its
  null rows as well, so it can widen by a little even at the source's size.
- **Categories and nulls.** Category frequencies and null rates are
  measured and reproduced.

What columns do *together* is not kept: a fake `age` does not follow a fake
`status`, and a fake child row's key does not point at a fake parent. A
`__checks__` rule or a `ColRule` added to the profiled spec by hand puts a
relationship back where one matters. Nor is a text column's *format*
recognised: an email column becomes random text of email-like length --
declare `format="email"` on the profiled spec to fix that.

## Large sources

`sample=` profiles a random sample of that many rows instead of all of
them, and `seed` makes the sample -- and the fake data -- the same every
run:

<!-- docs: skip -->
```python
fake = polspec.synthesize("events.parquet", 5_000_000, sample=200_000, seed=1)
```

## From the command line

```bash
polspec synthesize customers.parquet -o fake_customers.parquet \
    -n 1000000 --seed 1 --replace name email --spec fake_customers.yaml
```

`--spec` writes the profiled spec beside the data, so the fake data can be
reproduced -- `polspec generate fake_customers.yaml -n 1000000 --seed 1`
-- and the spec reviewed. `polspec schema infer` takes the same profiling
steps one at a time, as `--shape`, `--keys` and `--replace`; see
[Command line](cli.md).
