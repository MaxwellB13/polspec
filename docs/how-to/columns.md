# Declaring columns

A `ColSpec` describes one column. Only `dtype` is required. This page shows
the fields in use; [ColSpec fields](../reference/colspec-fields.md) is the
table of all of them — what each applies to, and what generation and
validation do with it.

Every other field is a keyword, so a declaration reads as a list of claims:

```python
import polars as pl
from polspec import ColSpec

ColSpec(pl.Int64, bounds=(1, None), unique=True, description="The order number.")
```

## Types

polspec generates every dtype below. A dtype passed as a class is instantiated
for you, so `pl.Int64` and `pl.Int64()` mean the same thing. It has to be a
Polars dtype: `ColSpec(int)` or `ColSpec("Int64")` is refused, with the Polars
spelling (`pl.Int64`) in the message.

A column that claims nothing beyond its dtype can be declared as the dtype
alone, in a class body, in `__columns__` or in a `TableSpec`; it is a
`ColSpec` of that dtype. A spec file reads `id: Int64` the same way, and is
still written in the long form, which earlier releases read too. *Added in
1.2.*

```python
class Accounts(FrameSpec):
    account_id = ColSpec(pl.Int64, unique=True)
    holder     = pl.String                          # ColSpec(pl.String)
    opened     = pl.Date
```

Since a dtype in a class body is a column, a class constant meant to hold
a dtype has to live outside the class.

| Family | Types |
|:--|:--|
| Integer | `Int8` `Int16` `Int32` `Int64` `Int128` `UInt8` `UInt16` `UInt32` `UInt64` `UInt128` |
| Float | `Float16` `Float32` `Float64` |
| Decimal | `Decimal(precision, scale)` |
| Boolean | `Boolean` |
| Text | `String` |
| Bytes | `Binary` |
| Temporal | `Date` `Time` `Datetime` `Duration` |
| Categorical | `Enum` `Categorical` |
| Nested | `List(inner)` `Array(inner, width)` `Struct({name: dtype})` — nested to any depth; see [Nested columns](nested.md) |
| Map (Polars 2) | `Map(key, value)` — see [Maps](nested.md#maps-a-list-of-entries) |

That is every dtype that holds data. `Null`, `Object`, `Unknown` and
extension types declare and validate by dtype, but hold nothing polspec
can draw, so `generate()` refuses each by name. A 128-bit integer is drawn
through 64 bits and a `Float16` as a `Float32` rounded to the nearest half,
so bounds past 64 bits are refused at `generate()` and bounds too close to
hold a half at declaration; see
[Known limitations](../explanation/limitations.md#smaller-sharp-edges).

## Nullability

`nullable=False` (the default) means validation rejects any null. When
`nullable=True`, `null_probability` sets how often generation emits one.

```python
from datetime import date

import polars as pl
from polspec import ColSpec, FrameSpec

ColSpec(pl.Int64, nullable=True, null_probability=0.25)   # about a quarter null
```

The rate only matters while `nullable=True`, so you can turn nullability off
without deleting the rate beside it. Setting a rate on a column that is not
nullable warns, though, since it reads as asking for nulls the column will
never hold:

<!-- docs: warns -->
```python
ColSpec(pl.Int64, null_probability=0.25)                   # warns: no nulls
ColSpec(pl.Int64, nullable=True, null_probability=0.25)    # about a quarter null
ColSpec(pl.Int64)                                          # no nulls, no warning
```

The default rate and an explicit `0.0` never warn.

### NaN

A float value can be present and still not a number. `nan_probability` says
how often a *present* value is NaN — nulls are counted first, and NaN is a
share of what is left:

```python
ColSpec(pl.Float64, nullable=True, null_probability=0.1, nan_probability=0.05)
```

The default, `0`, claims a column holds numbers: validation reports a NaN in
it as a `nan` finding. NaN is not a value in any other sense — it sits in no
range and equals nothing — so `bounds` and `choices` never judge one; the
NaN claim does, once. It applies wherever a float value sits: a column, a
list's elements, a struct's field. A `unique=True` column cannot declare it,
since no NaN equals another, and a column of any other dtype cannot hold
one — a missing value there is a null.

## Bounds

`bounds` is an inclusive `[min, max]` for numeric and temporal columns. Pass a
tuple, a list, or a `Bound`:

```python
ColSpec(pl.Int64, bounds=(-100, 100))
ColSpec(pl.Float64, bounds=[0.0, 1.0])
ColSpec(pl.Date, bounds=(date(2020, 1, 1), date(2024, 12, 31)))
```

Temporal bounds accept real `date`, `datetime`, `time` and `timedelta` objects,
or the physical integer the dtype stores. A date, datetime or time may also
be written as ISO text, read where it is declared, as a hand-written spec file
holds it: `bounds=("2020-01-01", None)`. Text that is not one is refused
there, as is text on any other column but a `Decimal`. *Added in 1.2.*

### Decimal bounds

A `Decimal(precision, scale)` column takes its bounds as an `int`, a
`decimal.Decimal`, or a string read exactly — the form a spec file writes.
A float is read through its `repr`, so `0.1` is `0.1`. An endpoint with more
decimal places than the scale keeps is refused rather than rounded, and one
the precision cannot hold is refused like any other out-of-range bound:

```python
from decimal import Decimal

ColSpec(pl.Decimal(10, 2), bounds=(0, "99.99"))
ColSpec(pl.Decimal(10, 2), bounds=(Decimal("0.50"), None))
```

<!-- docs: raises -->
```python
ColSpec(pl.Decimal(10, 2), bounds=(0, "1.005"))
# SpecError: ColSpec.bounds max ('1.005') has more decimal places than Decimal(precision=10, scale=2) keeps (scale 2); ...
```

Generation draws a Decimal as the integer it physically is and scales it
back, so the default range with no bounds is the float default
(`±1,000,000`) or the widest the precision allows, whichever is narrower.
The draw is 64-bit: bounds needing more than eighteen significant digits
are refused at `generate()`, and only there — validation checks the full
precision.

### Open-ended bounds

Either endpoint may be `None`, leaving that side unconstrained:

```python
ColSpec(pl.Int64, bounds=(0, None))    # non-negative
ColSpec(pl.Int64, bounds=(None, 0))    # non-positive
```

!!! warning "An open end means different things to generation and validation"

    `validate()` treats it as genuinely unconstrained. `generate()` cannot
    sample an unbounded range, so it falls back to the same default it would
    use with no bounds at all.

    ```python
    class S(FrameSpec):
        n = ColSpec(pl.Int64, bounds=(0, None))

    S.generate(1000, seed=1)["n"].max()     # ~1_000_000, the Int64 default
    S.validate(pl.DataFrame({"n": [10**15]}))   # accepted — no upper limit
    ```

    A `unique=True` 64- or 128-bit integer with no bounds at all is the
    one exception: when the frame needs more distinct values than the
    default range holds, the range widens upward to fit it, rather than
    refusing an ID column its dtype could hold.

For "always positive", note that bounds are *inclusive*: use `(1, None)` for
integers, and either a small floor like `(1e-9, None)` for floats or an
unsigned dtype, which cannot represent a negative at all.

Bounds outside what the dtype can hold are rejected when you declare them:

<!-- docs: raises -->
```python
ColSpec(pl.Float32, bounds=(-1e40, 1e40))
# SpecError: ColSpec.bounds min (-1e+40) is outside the range Float32 can represent ...
```

## Value domains

`choices` restricts a column to a fixed set:

```python
ColSpec(pl.String, choices=["GBP", "USD", "EUR"])
```

`weights` biases the draw. Supply them positionally, or as a `{choice: weight}`
mapping — never both:

```python
ColSpec(pl.String, choices=["a", "b", "c"], weights=[10.0, 5.0, 1.0])
ColSpec(pl.String, choices={"a": 10.0, "b": 5.0, "c": 1.0})   # same thing
```

Weights need a domain to apply to, so they require `choices`, an `Enum` dtype,
or `Boolean` (where they read `[p_false, p_true]`):

```python
ColSpec(pl.Enum, choices={"x": 0.1, "y": 0.2, "z": 0.7})
ColSpec(pl.Boolean, weights=[0.9, 0.1])   # 10% true
```

A bare `pl.Enum` takes its categories from `choices`, in the order written,
so each category sits beside its share: the first line is
`ColSpec(pl.Enum(["x", "y", "z"]), weights=[0.1, 0.2, 0.7])`, and the two
spellings are the same spec. It is how polspec writes a weighted `Enum` out
as Python; an `Enum` without weights is written with its categories,
`ColSpec(pl.Enum(["x", "y", "z"]))`. *Added in 1.2.*

Choices are held in the column's own dtype, so a `datetime` choice on a
`Datetime` column or a `bytes` choice on a `Binary` column stays what it is.
They must be distinct once cast to that dtype — `1` and `"1"` on a `String`
column are one value:

<!-- docs: raises -->
```python
ColSpec(pl.String, choices=[1, "1"])
# SpecError: ColSpec.choices contains values that are the same once cast to
# String: ['1']
```

## String and binary length

`string_length` is an inclusive `[min, max]` on characters (String) or bytes
(Binary). Unlike `bounds`, both endpoints are required.

```python
ColSpec(pl.String, string_length=(8, 8))    # fixed width
ColSpec(pl.Binary, string_length=(16, 64))
```

## Nested columns

A `List` or `Array` column takes the scalar fields as claims about each
element, plus `list_length`; a `Struct` says what its fields hold with
`fields`; a `Map` is a list of `{key, value}` entries. See
[Nested columns](nested.md).

## String formats

`format=` names the shape a `String` column's values take, and both sides
read it: generation fills the column from that format's sampler and
validation checks every value against it.

```python
ColSpec(pl.String, format="uuid4", unique=True)
ColSpec(pl.String, format="email", nullable=True)
```

The set is `uuid4`, `uuid7`, `ulid`, `email`, `ipv4`, `ipv6`, `mac`,
`hostname`, `url`, `iso_country`, `iso_country3` and `iso_currency`. A format owns the column's domain, so it
cannot sit beside `choices` or `string_length`, and only a `String` column
can carry one. See [String formats](formats.md) for what each generates,
what each accepts, and what none of them promises.

A format can be widened with the values a column holds beside it — ISO
countries plus `UK (ISO)`, IPv4 or `NOT AVAILABLE` — with
[`extra_values`](formats.md#extending-a-format-with-extra-values).

## String patterns

`pattern=` is a regular expression every value must match, and generation
fills the column with full matches of it. See
[Patterns](formats.md#patterns). *Generated from since 1.4.*

## Distributions

Numeric and temporal columns can be drawn from a shape other than uniform:

| Distribution | Parameters (aliases accepted) |
|:--|:--|
| `uniform` | — |
| `normal` | `mean`/`mu`/`loc`, `std`/`sigma`/`scale` |
| `lognormal` | `mean`/`mu`/`meanlog`, `std`/`sigma`/`sdlog` |
| `exponential` (`exp`) | `rate`/`lambda`/`lambda_`, or `scale` |
| `poisson` | `lambda`/`lambda_`/`rate`/`mean` |
| `gamma` | `shape`/`alpha`/`k`, `scale`/`beta`/`theta` |
| `beta` | `alpha`/`a`/`shape1`, `beta`/`b`/`shape2` |

```python
ColSpec(
    pl.Float64,
    bounds=(0.0, 500.0),
    distribution="lognormal",
    distribution_params={"mean": 2.0, "std": 0.6},
)
```

!!! warning "Bounds clamp, they do not resample"

    A draw outside the bounds lands *on* the boundary rather than being drawn
    again. A `normal` centred at 0 squeezed into `(0, 50)` puts roughly half
    the column on the floor as one repeated value.

    When you want a positive-skewed shape, reach for a distribution that is
    already non-negative — `lognormal`, `exponential`, `gamma` — instead of
    clamping a symmetric one.

## Uniqueness

`unique=True` declares that values must be distinct. `generate()` draws the
column as a permutation of everything it could hold — the integers in its
bounds, its choices, a grid over a float range, every string its lengths or
`format` allow — so the data it produces satisfies it, and a value depends
only on its row and the seed. That makes a unique column unique across a
whole frame however it is produced: `generate_batches`, `scan()` and the
sinks give the same values `generate()` does.

Drawn uniformly from every string it could hold, a unique string favours
its longer lengths, since they hold most of the strings: with
`string_length=(5, 15)` nearly every value is fifteen characters. Narrow the
lengths if their spread matters.

Nulls are exempt, as they are for foreign keys: a null means "no value", so a
nullable unique column may repeat nulls and nothing else. A null row still
takes its place in the permutation, so the column's domain has to hold every
row, nulls included.

A domain too small to cover the row count is refused, naming the column:

<!-- docs: raises -->
```python
class Narrow(FrameSpec):
    id = ColSpec(pl.Int8, unique=True)

Narrow.generate(300, seed=1)
# GenerationError: Column 'id' is unique, but its domain holds only 256
# distinct value(s) and 300 are needed. Widen its bounds or choices, or
# generate fewer rows.
```

`unique=True` cannot be combined with `weights`, a non-uniform `distribution`,
or `rules`: the first two describe how often a value recurs, which a draw
without replacement has no room for, and a rule would reintroduce the
duplicates. Each is refused at declaration rather than quietly ignored.

## Describing a column

*Added in 1.1.*

`description=` says what a column means, for a reader. `to_markdown()` shows
it, spec files keep it, and neither generation nor validation reads it:

```python
ColSpec(pl.Int64, unique=True, description="The order's number, as printed on the invoice.")
```

A table's own description is `__description__` on a `FrameSpec`, or
`description=` on a `TableSpec`; see
[Generated documentation](documenting.md#markdown-data-dictionary).

## Defaults and optional columns

Two fields say what to do when a value, or a whole column, is missing.

*Added in 1.1.*

`default=` is the value that stands in for a missing one. It is used only
when you ask: `validate(fill_defaults=True)` fills the column's nulls with
it before checking, and `missing_cols="add"` adds an absent column holding
it. Generation does not read it.

```python
ColSpec(pl.Enum(["open", "closed"]), default="open")
```

A default has to be a value the column itself accepts — its dtype, bounds,
choices, format, pattern, lengths and validators — and one that is not is
refused where it is declared. A `List`, `Array`, `Map` or `Struct` column
takes no default.

`optional=True` means the column may be absent. A frame without it gives no
`missing_columns` finding; a frame with it is checked against every claim
as usual. Generation makes the column like any other.

```python
ColSpec(pl.String, optional=True)
```

[Filling defaults](validating.md#filling-defaults) shows both at work.

## Tags

Tags group columns for later selection. They carry no generation or validation
meaning.

```python
class Events(FrameSpec):
    user_id  = ColSpec(pl.Int64, tags=["pii", "key"])
    email    = ColSpec(pl.String, tags="pii")
    duration = ColSpec(pl.Int64, tags="metric")

Events.tag("pii")                      # ['user_id', 'email']
Events.tag("pii", "key", match="all")  # ['user_id']
```

## Column names that are not identifiers

A column declared as a class attribute takes the attribute's name, and an
attribute name has to be a valid Python identifier. Real data is not so
polite. There are two ways out, for two different situations.

### `col_name`: the data's name has spaces or punctuation

Keep a clean attribute name and tell the `ColSpec` what the column is really
called:

```python
class Sales(FrameSpec):
    unit_price = ColSpec(pl.Float64, col_name="Unit Price", bounds=(0, None))
    region     = ColSpec(pl.Enum(["UK", "US"]), col_name="Sales Region")

Sales.schema()              # Schema({'Unit Price': Float64, 'Sales Region': Enum(...)})
Sales.generate(3).columns   # ['Unit Price', 'Sales Region']
```

`col_name` is the column's name everywhere the spec is used: in the generated
frame, in `validate()`, in a `ColRule` condition built with `col()`, in
`__unique_together__`, in `ForeignKey` columns and in `tag()` results. The
attribute name exists only in the class body. Two attributes that resolve to
the same `col_name` are rejected at declaration, and overriding an attribute
on a subclass removes the column it named, whatever `col_name` it carried.

`to_yaml()` and `to_python()` write the real column name as the key, so a
spec that came from a file never needs `col_name`.

A third name, `seed_name`, is not about what the column is called but about
what it *generates*: a renamed column declared with `seed_name="old"` keeps
producing the data it did under the old name. See
[Renaming a column without changing its data](generating.md#renaming-a-column-without-changing-its-data).

### `__columns__`: the name is an identifier but cannot be an attribute

A leading underscore is skipped by the class-body scan, so a column called
`_id` needs the explicit mapping. A name that matches one of `FrameSpec`'s
methods (`schema`, `tag`, …) is fine either way: the method keeps working and
the column is reachable as `Spec.col("schema")` — see
[Specs as values](tablespec.md#column-names-and-method-names).

```python
class Raw(FrameSpec):
    __columns__ = {
        "_id": ColSpec(pl.Int64),
        "schema": ColSpec(pl.String),
    }
```

`__columns__` is never looked up as an attribute, so both the column and the
method survive. The dict key already is the column name, so a `col_name` that
disagrees with its key is rejected. `from_dataframe`, `from_yaml` and
`to_python` all declare columns this way, since their names come from data
rather than from someone's class body.

