# Declaring columns

A `ColSpec` describes one column. Only `dtype` is required.

<!-- docs: skip -->
```python
ColSpec(
    dtype,
    col_name=None,
    seed_name=None,
    nullable=False,
    bounds=None,
    tags=(),
    unique=False,
    null_probability=0.1,
    string_length=None,
    list_length=None,
    fields=None,
    format=None,
    pattern=None,
    distribution=None,
    distribution_params=None,
    choices=None,
    weights=None,
    rules=(),
    validators=(),
)
```

## Types

polspec generates every dtype below. A dtype passed as a class is instantiated
for you, so `pl.Int64` and `pl.Int64()` mean the same thing.

| Family | Types |
|:--|:--|
| Integer | `Int8` `Int16` `Int32` `Int64` `UInt8` `UInt16` `UInt32` `UInt64` |
| Float | `Float32` `Float64` |
| Decimal | `Decimal(precision, scale)` |
| Boolean | `Boolean` |
| Text | `String` |
| Bytes | `Binary` |
| Temporal | `Date` `Time` `Datetime` `Duration` |
| Categorical | `Enum` `Categorical` |
| Nested | `List(inner)` `Array(inner, width)` `Struct({name: dtype})` — nested to any depth; see [Nested columns](#nested-columns) |

That is every dtype: since 0.9.0 there is none polspec declares and cannot
generate.

## Nullability

`nullable=False` (the default) means validation rejects any null. When
`nullable=True`, `null_probability` sets how often generation emits one.

```python
ColSpec(pl.Int64, nullable=True, null_probability=0.25)   # about a quarter null
```

`null_probability` is ignored when `nullable=False`, so switching nullability
off does not silently leave a stale rate behind.

Writing a rate of your own *without* `nullable=True` warns, though, because
that is the other way round — it reads as asking for nulls, and the column
generates none:

```python
ColSpec(pl.Int64, null_probability=0.25)                   # warns: no nulls
ColSpec(pl.Int64, nullable=True, null_probability=0.25)    # about a quarter null
ColSpec(pl.Int64)                                          # no nulls, no warning
```

Only a rate that cannot be a leftover warns: the default and an explicit
`0.0` both already agree with `nullable=False`.

## Bounds

`bounds` is an inclusive `[min, max]` for numeric and temporal columns. Pass a
tuple, a list, or a `Bound`:

```python
ColSpec(pl.Int64, bounds=(-100, 100))
ColSpec(pl.Float64, bounds=[0.0, 1.0])
ColSpec(pl.Date, bounds=(date(2020, 1, 1), date(2024, 12, 31)))
```

Temporal bounds accept real `date`, `datetime`, `time` and `timedelta` objects,
or the physical integer the dtype stores.

### Decimal bounds

A `Decimal(precision, scale)` column takes its bounds as an `int`, a
`decimal.Decimal`, or a string read exactly -- the form a spec file writes.
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
are refused at `generate()`, and only there -- validation checks the full
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

    This mirrors how `bounds=None` already behaves rather than adding a third
    rule.

For "always positive", note that bounds are *inclusive*: use `(1, None)` for
integers, and either a small floor like `(1e-9, None)` for floats or an
unsigned dtype, which cannot represent a negative at all.

Bounds outside what the dtype can hold are rejected when you declare them:

<!-- docs: raises -->
```python
ColSpec(pl.Float32, bounds=(-1e40, 1e40))
# ValueError: ColSpec.bounds min (-1e+40) is outside the range Float32 can represent
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
ColSpec(pl.Enum(["x", "y", "z"]), weights=[1.0, 2.0, 7.0])
ColSpec(pl.Boolean, weights=[0.9, 0.1])   # 10% true
```

Choices are held in the column's own dtype, so a `datetime` choice on a
`Datetime` column or a `bytes` choice on a `Binary` column stays what it is.
They must be distinct once cast to that dtype -- `1` and `"1"` on a `String`
column are one value:

<!-- docs: raises -->
```python
ColSpec(pl.String, choices=[1, "1"])
# ValueError: ColSpec.choices contains values that are the same once cast to
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

A `List` or `Array` column is described by the same fields as a scalar one,
read as claims about **each element**: `bounds`, `choices`, `weights`,
`format`, `pattern`, `string_length` and `distribution` all apply to the
values inside the list. One field describes the list itself:

```python
ColSpec(pl.List(pl.Int64), bounds=(0, 10), list_length=(1, 5))    # 1 to 5 ints, each 0..10
ColSpec(pl.List(pl.String), format="email")                       # 0 to 5 addresses
ColSpec(pl.List(pl.Enum(["a", "b", "c"])), choices=["a", "b"])    # from a subset of the Enum
ColSpec(pl.Array(pl.Float64, 3), bounds=(0.0, 1.0))               # exactly three, from the dtype
```

`list_length` is the inclusive range of elements a value holds, both ends
required; without it generation makes 0 to 5. An `Array` takes its length
from the dtype and refuses `list_length`.

`nullable` and `null_probability` describe the list: a null cell, never a
null element. Generation never puts a null inside a list, and validation
reports one under `nullability` like a null in a non-nullable column.

Validation runs every element claim inside the list, and a list fails where
*any* element does — the finding's samples and `rows()` are the offending
lists. `list_length` gets its own finding code.

Generation draws the lengths and the elements as two columns of the inner
dtype and wraps one by the other, so an element is made by the same code
that would make a scalar of its dtype, and a `List` column keeps its data
across a rename through `seed_name` like any other.

What a nested column cannot carry: `unique` (a list is not drawn without
replacement) and `rules` (a rule's choices are values, and a list value
would be a list of lists). `validators` and `__checks__` work as on any
column — they are expressions.

### `fields`: what a struct's values are

A `Struct` column's dtype is its schema — every field's name and type comes
from it — and `fields` says what is *claimed* about the values in it:

```python
ColSpec(
    pl.Struct({"lat": pl.Float64, "lon": pl.Float64, "label": pl.String}),
    fields={
        "lat": ColSpec(pl.Float64, bounds=(-90, 90)),
        "lon": ColSpec(pl.Float64, bounds=(-180, 180)),
    },
    nullable=True,
)
```

Each value is a `ColSpec`, so a field is described exactly as a column of
the same dtype would be. `fields` is **partial**: a struct of twenty fields
where one needs bounds spells one field, and the rest are generated from
their dtypes alone. A name the dtype does not declare, or a field spec
whose dtype disagrees with the struct's, is refused where it is written.

A field is a value, not a column, so `unique`, `rules`, `validators`,
`seed_name` and `col_name` are refused on one — each is a claim about a
column among columns. (A validator about a field is written on the struct
column instead: `pl.col("point").struct.field("lat") != 0`.) As with a
list, the column's `nullable` describes the *cell* — a null struct — and a
field's own `nullable` says whether it may be null inside a struct that is
present. It defaults to `False`, like any column's.

A `List` of a `Struct` takes `fields` too, describing its element, and a
field may itself be a struct or a list, so a declaration nests as deeply as
the dtype does:

```python
ColSpec(pl.Struct({"xs": pl.List(pl.Int64)}),
        fields={"xs": ColSpec(pl.List(pl.Int64), bounds=(0, 9), list_length=(2, 2))})
```

Generation makes one column per field and gathers them, so a field is drawn
by the code that draws a column of its dtype — and a struct column is
seeded by name like any other: renaming it with `seed_name` keeps every
field, and adding a field beside one moves nothing.

Validation checks each field's claims in place, and a finding names the
field it is about — its key is `point.lat__bounds`, its message says
`Column 'point.lat'` — while its `columns` stay `("point",)`, so
`report.rows(finding)` returns the rows of the frame that hold the
offending structs. Inside a list the samples are the offending lists, as
for any list. A struct in the data matches a declared one by field name,
not order, each field compatible by the usual rules; a field missing or
added is a `dtype` finding.

## String formats

`format=` names the shape a `String` column's values take, and both sides
read it: generation fills the column from that format's sampler and
validation checks every value against it.

```python
ColSpec(pl.String, format="uuid4", unique=True)
ColSpec(pl.String, format="email", nullable=True)
```

The set is `uuid4`, `email`, `ipv4`, `ipv6`, `mac`, `hostname`,
`iso_country` and `iso_currency`. A format owns the column's domain, so it
cannot sit beside `choices` or `string_length`, and only a `String` column
can carry one. See [String formats](formats.md) for what each generates,
what each accepts, and what none of them promises.

## String patterns

`pattern=` is a regular expression every value must match -- **checked by
validation only**. Generation does not read it: a column with a pattern is
filled with ordinary random text, so the round trip holds only with
`validate_pattern=False`, exactly as for `validators`. That is the honest
half of the split `format=` makes: polspec can check any regex, and can
generate a curated set.

```python
ColSpec(pl.String, pattern=r"^[A-Z]{3}-\d{4}$")     # a SKU shape polspec cannot generate
```

Reach for `format` when the shape is one polspec has; reach for `pattern`
when it is not. The two cannot be combined -- a format already *is* a
pattern with a sampler. A pattern is compiled by Polars' regex engine at
declaration, so one Polars cannot run (look-around, for instance) is refused
with the engine's own message rather than failing on the first validation.

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
column without replacement, so the data it produces satisfies it.

Nulls are exempt, as they are for foreign keys: a null means "no value", so a
nullable unique column may repeat nulls and nothing else.

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
the column is reachable as `Spec.col("schema")` -- see
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

