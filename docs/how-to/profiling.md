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

There are two ways in, with the same options and the same defaults.

| | `FrameSpec.from_dataframe(df)` | `polspec.profile(source)` |
|:--|:--|:--|
| Gives | a `FrameSpec` class | a `TableSpec` |
| Reads | a `DataFrame`, with rows or without | a `DataFrame`, a `LazyFrame` or a file path |

By default each *imitates* the data, as a stand-in for it needs, and
`describe=True` *describes* it, as a contract a person reads and edits
needs. `polspec schema infer` on the command line does the same, with
`--describe`. *Since 2.0. In 1.x, `from_dataframe` and `schema infer`
described by default, and `schema infer` rounded (see
[Rounding](#rounding)).*

### Imitate: the default

```python
spec = polspec.profile(orders, name="Orders")
print(spec["status"].weights)
print(spec["order_id"].unique, spec.order_by)
```

```text
(0.25, 0.5, 0.25)
True ('order_id',)
```

A profile records each column's dtype, nulls, extremes and values, and
also how often each category occurs, which column is a key, the order the
rows come in, and the shape a numeric column's values follow. It is what
[`synthesize`](synthesizing.md) generates from. A key keeps the range it
was observed in, so a spec profiled from 2,000 rows generates at most
2,000; `synthesize` widens it to the rows it makes.

### Describe: `describe=True`

```python
Described = FrameSpec.from_dataframe(orders, name="Orders", describe=True)
print(Described.column("status").dtype)
print(Described.column("email").format)
print(Described.column("note").nullable, Described.column("note").null_probability)
```

```text
Enum(categories=['NEW', 'PAID', 'SHIPPED'])
email
True 0.857
```

A text column whose values repeat becomes an `Enum` of them; one of distinct
values stays text. A column whose values are emails, UUIDs, IP addresses,
hostnames or ISO country and currency codes names that
[format](formats.md), so generated values have it too. Category
frequencies, fitted distributions, keys and row order are left out unless
asked for one by one: `describe=True, weights=True` describes the data and
records its frequencies.

## Choosing what is inferred

Each option is a keyword on both functions:

| Option | Default (`describe=True`) | What it does |
|:--|:--|:--|
| `weights` | on (off) | Record each category's observed frequency |
| `shape` | on (off) | Fit the distribution a numeric or temporal column follows |
| `detect_unique` | on (off) | Declare an all-distinct integer or text column `unique` (from 100 values) |
| `detect_order` | on (off) | Declare `order_by` when the rows come sorted by a column |
| `calculate_bounds` | on | Record extremes, and text and binary lengths |
| `formats` | on | Name a text column's format |
| `patterns` | on | Write the pattern a code-like text column's values share; see [How close the fake data is](synthesizing.md#how-close-the-fake-data-is) |
| `max_unique_enum` | 50 | The most distinct values a text column may have to become an `Enum` |
| `replace` | none | Columns whose values must not reach the spec, such as names or emails |
| `sample` | all rows | Profile a random sample of this many rows |
| `seed` | 0 | Seeds the sample and the fitting, so a profile is the same every run |
| `round_bounds`, `round_rates` | exact | See [Rounding](#rounding) |
| `parse_text` | off | Read text as the numbers, flags or dates it holds; see [Data read as text](#data-read-as-text) |

```python
Lean = FrameSpec.from_dataframe(
    orders, name="Orders", describe=True, weights=True, replace=["email"]
)
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
print(Rounded.column("total").bounds)
print(Rounded.column("note").null_probability)
```

```text
[0.0, 500.0]
0.86
```

`"auto"` keeps two significant figures of a column's span, and rounds a
time to the second. A step rounds to its multiples: `round_bounds=1000`,
`round_rates=0.01`. Bounds always round outward and rates up, within what the
dtype holds, so the data a spec was inferred from still passes it. The
default is exact, `polspec schema infer` included: `--round-bounds auto
--round-rates auto` rounds there. *Since 2.0. In 1.x, `schema infer`
rounded `auto` by default.*

## Data read as text

A file too unclean to type on reading is often read with its types
switched off, so every column arrives as text. Profiled as it is, a column
of decimals is described as text. `parse_text=True` reads each text
column as the type its values hold:

```python
raw = pl.DataFrame(
    {
        "loan_id": [str(i) for i in range(1, 101)],
        "LGD": [f"0.{i:02d}" if i % 10 else "" for i in range(100)],
        "opened": [f"2024-01-{1 + i % 28:02d}" for i in range(100)],
    }
)  # as `pl.read_csv(path, infer_schema=False)` would hand it back

spec = polspec.profile(raw, parse_text=True)
print(spec.schema())
```

```text
Schema({'loan_id': Int64, 'LGD': Float64, 'opened': Date})
```

*Added in 1.2.*

A column is read as the first of `Boolean` (`true` or `false`, in any
case), `Int64`, `Float64`, `Date` and `Datetime` that every value parses as
exactly. Text is stripped first, and empty text is a null. A column stays
text when any value is written with a leading zero (`007`), or is an
integer too large for an `Int64`: those are codes, and parsing would lose
the zero or round the number. A column in `replace` stays text too.

### Dates written another way

Dates, datetimes and times not in ISO form are read too, each with the
format its values are written in, which the spec keeps as `text_format`:

```python
bonds = pl.DataFrame({"maturity": ["30 Sep 2031", "31 Dec 2049", "15 Mar 2027"]})
print(polspec.profile(bonds, parse_text=True)["maturity"].text_format)
```

```text
%d %b %Y
```

The formats tried are `30 Sep 2031`, `30 September 2031`, `30-Sep-2031`,
`Sep 30, 2031`, `September 30, 2031`, `30/09/2031`, `09/30/2031`,
`2031/09/30` and `30.09.2031`, each with its day and month unpadded too
(`1 Jan 2030`); any of them with a time (`14:05`, `14:05:59`, `2:05 PM`);
and a time alone. A format fits when every value parses with it and
writes back as the text it came from, a month name in any case: so
`30 Feb 2031` fits nothing. Digits alone, such as `20310930`, are a number.
*Added in 1.6.*

Two formats that both fit and read the values differently are no fit:
`01/02/2031` is the first of February one way round and the second of
January the other. The column stays text, and its description says which
formats fit; one day past the twelfth anywhere in the column settles it.
Declare the column with the format you know it to be:

```python
Settled = polspec.TableSpec(
    "Trades", {"settle": ColSpec(pl.Date, text_format="%d/%m/%Y")}
)
trades = polspec.validate(
    Settled, pl.DataFrame({"settle": ["01/02/2031", "03/04/2030"]}), parse_text=True
)
print(trades["settle"].to_list())
```

```text
[datetime.date(2031, 2, 1), datetime.date(2030, 4, 3)]
```

`validate(parse_text=True)` and `read()` read such a column with its
`text_format`, and a value written another way is a `dtype` finding that
names the format. A CSV or NDJSON file polspec writes holds the column in
it too: `synthesize` on the command line, the `sink_*` verbs and
`polspec generate`. In memory, and in Parquet and Arrow IPC, a date is a
date.

### Why a column stays text

With `parse_text=True`, a column that nearly parses as a number or a date,
and stays text, says why in its description, and `schema infer` prints it:

```python
amounts = pl.DataFrame({"NOMINAL": ["1,500,000.00", "250.00", "0.000125"] * 10})
print(polspec.profile(amounts, parse_text=True)["NOMINAL"].description)
```

```text
Stays text: 10 value(s) do not parse as a number, such as ['1,500,000.00'].
```

The reasons are values that do not parse, as here; a value written with a
leading zero; a whole number past an `Int64`; and two date formats that
both fit. polspec reads no thousands separators or decimal commas: only you
know which a comma is. Remove them first, when they are thousands:

```python
amounts = amounts.with_columns(pl.col("NOMINAL").str.replace_all(",", ""))
```

### Left as text

Left off, `parse_text` reads nothing: a text column that would parse stays
text, and its description says what it would parse as, and in which
format — so the spec file tells whoever reads it.
`polspec schema infer` and `polspec synthesize` also say so on the terminal,
and take `--parse-text`.

The frame itself is still text, so validate it with `parse_text=True`
too, which reads the text the same way before checking it — see
[Data read as text](validating.md#data-read-as-text):

```python
typed = polspec.validate(spec, raw, parse_text=True)
print(typed.schema)
```

```text
Schema({'loan_id': Int64, 'LGD': Float64, 'opened': Date})
```

## Several tables

`polspec.profile_all` profiles a set of related tables into a
[`Registry`](registry.md), with the foreign keys between them found from
their values:

```python
customers = pl.DataFrame({"id": range(1, 101)})
purchases = pl.DataFrame({"id": range(1, 301), "customer_id": [1, 2, 3] * 100})
registry = polspec.profile_all({"Customers": customers, "Orders": purchases})
print(registry["Orders"].foreign_keys[0].references)
```

```text
Customers
```

It takes every option `profile` does, applied to each table, and `keys=`
to declare keys rather than find them. [Related
tables](synthesizing.md#related-tables) says how a key is found.
*Added in 1.5.*

## From the command line

```bash
polspec schema infer orders.parquet -o orders.yaml
polspec schema infer orders.parquet -o orders.py --describe --weights
```

`schema infer` takes the same options as flags: `--describe`, `--weights`,
`--shape`, `--keys` and `--order` (each with a `--no-` form), `--no-bounds`, `--no-formats`, `--no-patterns`,
`--max-unique-enum`, `--replace`, `--sample`, `--round-bounds`,
`--round-rates` and `--parse-text`. See
[the command line](cli.md#schema-infer-profile-data-into-a-spec).
Several files make a registry, each spec named after its file:

```bash
polspec schema infer customers.csv orders.csv -o shop.yaml
```

## Treat it as a draft

A profile describes the sample it saw, and the sample is narrower than the
rule. Before trusting the draft as a contract:

- **Widen what will grow.** An ID's upper bound, a date range that ends
  today.
- **Add what data cannot show.** [Rules and checks](constraints.md) are
  relationships between columns, and a profile reads each column on its
  own. `profile_all` finds single-column keys between tables; declare any
  other.
- **Check it against the data.** The source passes its own profile (any
  infinity aside, which is outside its bounds; see [Known
  limitations](../explanation/limitations.md#a-floats-infinities-are-outside-its-profiled-bounds)):

```python
Described.validate(orders)
```
