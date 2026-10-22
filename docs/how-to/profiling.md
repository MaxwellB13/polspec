# Infer a spec from data

When the data already exists, polspec can write the first draft of its spec:
each column's dtype, nulls, extent and domain, read from the data. You then
edit the draft into the contract you mean.

```python
import polars as pl
import polspec
from polspec import FrameSpec, ProfileOptions

n = 2_000
orders = pl.DataFrame(
    {
        "order_id": range(1, n + 1),
        "status": ["NEW", "PAID", "PAID", "SHIPPED"] * (n // 4),
        "total": [round(5 + (i * 37 % 1000) * 0.4871, 2) for i in range(n)],
        "email": [f"user{i % 400}@example.com" for i in range(n)],
        "note": [None if i % 7 else f"note {i}" for i in range(n)],
    }
)
```

## Describe, or imitate

There are two ways in, and they differ in what they record by default.

| | `FrameSpec.from_dataframe(df)` | `polspec.profile(source)` |
|:--|:--|:--|
| Gives | a `FrameSpec` class | a `TableSpec` |
| Reads | a `DataFrame` | a `DataFrame`, a `LazyFrame` or a file path |
| Records | dtypes, nulls and null rates, bounds and lengths, `Enum`s, formats | all of that, plus category frequencies, fitted distributions, unique keys and row order |
| For | a contract: what each column is | a stand-in: data that looks like this data |

Both take the same options, so either can record anything the other does.
`polspec schema infer` on the command line describes, as `from_dataframe`
does, and rounds what it writes (see [Rounding](#rounding)).

### Describe: `from_dataframe`

```python
Described = FrameSpec.from_dataframe(orders, name="Orders")
print(Described.col("status").dtype)
print(Described.col("email").format)
print(Described.col("note").nullable, Described.col("note").null_probability)
```

```text
Enum(categories=['NEW', 'PAID', 'SHIPPED'])
email
True 0.857
```

A text column whose values repeat becomes an `Enum` of them; one of distinct
values stays text. A column whose values are emails, UUIDs, IP addresses,
hostnames or ISO country and currency codes names that
[format](formats.md), so generated values have it too.

### Imitate: `profile`

```python
spec = polspec.profile(orders, name="Orders")
print(spec["status"].weights)
print(spec["order_id"].unique, spec.order_by)
```

```text
(0.25, 0.5, 0.25)
True ('order_id',)
```

`profile` also records how often each category occurs, which column is a
key, the order the rows come in, and the shape a numeric column's values
follow. It is what [`synthesize`](synthesizing.md) generates from, and it
reads a lazy frame or a file as well as a frame.

## Choosing what is inferred

Each option is a keyword on both functions:

| Option | Default (`profile` / `from_dataframe`) | What it does |
|:--|:--|:--|
| `weights` | on / off | Record each category's observed frequency |
| `shape` | on / off | Fit the distribution a numeric or temporal column follows |
| `detect_unique` | on / off | Declare an all-distinct integer or text column `unique` (from 100 values) |
| `detect_order` | on / off | Declare `order_by` when the rows come sorted by a column |
| `calculate_bounds` | on | Record extremes, and text and binary lengths |
| `formats` | on | Name a text column's format |
| `max_unique_enum` | 50 | The most distinct values a text column may have to become an `Enum` |
| `replace` | none | Columns whose values must not reach the spec, such as names or emails |
| `sample` | all rows | Profile a random sample of this many rows |
| `seed` | 0 | Seeds the sample and the fitting, so a profile is the same every run |
| `round_bounds`, `round_rates` | exact | See [Rounding](#rounding) |

```python
Lean = FrameSpec.from_dataframe(orders, name="Orders", weights=True, replace=["email"])
```

To use the same choices in several places, make them a `ProfileOptions`
and pass it as `options=`:

```python
house_style = ProfileOptions(round_bounds="auto", round_rates="auto", replace=["email"])
spec = polspec.profile(orders, options=house_style)
fake = polspec.synthesize(orders, options=house_style)
```

## Rounding

A profile records what it saw: `total` ran from 5.0 to 491.61, and `note`
was null in 85.7% of rows. A person would write `[0, 500]` and `0.86`.
`round_bounds` and `round_rates` write it that way:

```python
Rounded = FrameSpec.from_dataframe(
    orders, name="Orders", round_bounds="auto", round_rates="auto"
)
print(Rounded.col("total").bounds)
print(Rounded.col("note").null_probability)
```

```text
[0.0, 500.0]
0.86
```

`"auto"` keeps two significant figures of a column's span, and rounds a
time to the second. A step rounds to its multiples: `round_bounds=1000`,
`round_rates=0.01`. Bounds always round outward and rates up, within what the
dtype holds, so the data a spec was inferred from still passes it. In Python
the default is exact; `polspec schema infer` rounds `auto` unless told
`--round-bounds exact`.

## From the command line

```bash
polspec schema infer orders.parquet -o orders.yaml
polspec schema infer orders.parquet -o orders.py --weights --keys --order
```

`schema infer` takes the same options as flags: `--weights`, `--shape`,
`--keys`, `--order`, `--no-bounds`, `--no-formats`, `--max-unique-enum`,
`--replace`, `--sample`, `--round-bounds` and `--round-rates`. See
[the command line](cli.md#schema-infer-profile-data-into-a-spec).

## Treat it as a draft

A profile describes the sample it saw, and the sample is narrower than the
rule. Before trusting the draft as a contract:

- **Widen what will grow.** An ID's upper bound, a date range that ends
  today.
- **Add what data cannot show.** [Rules, checks and foreign
  keys](constraints.md) are relationships between columns and tables; a
  profile reads each column on its own.
- **Check it against the data.** The source passes its own profile (any
  infinity aside, which is outside its bounds; see [Known
  limitations](../explanation/limitations.md#a-floats-infinities-are-outside-its-profiled-bounds)):

```python
Described.validate(orders)
```
