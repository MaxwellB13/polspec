# Declaring columns

A `ColSpec` describes one column. Only `dtype` is required.

```python
ColSpec(
    dtype,
    nullable=False,
    bounds=None,
    tags=(),
    unique=False,
    null_probability=0.1,
    string_length=None,
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
| Boolean | `Boolean` |
| Text | `String` |
| Bytes | `Binary` |
| Temporal | `Date` `Time` `Datetime` `Duration` |
| Categorical | `Enum` `Categorical` |

Anything else — `List`, `Struct`, `Array` — can be *validated* but not
generated; `generate()` raises `TypeError` naming the dtype.

## Nullability

`nullable=False` (the default) means validation rejects any null. When
`nullable=True`, `null_probability` sets how often generation emits one.

```python
ColSpec(pl.Int64, nullable=True, null_probability=0.25)   # about a quarter null
```

`null_probability` is ignored when `nullable=False`, so switching nullability
off does not silently leave a stale rate behind.

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
frame, in `validate()`, in `ColRule.when={"column": ...}`, in
`__unique_together__`, in `ForeignKey` columns and in `tag()` results. The
attribute name exists only in the class body. Two attributes that resolve to
the same `col_name` are rejected at declaration, and overriding an attribute
on a subclass removes the column it named, whatever `col_name` it carried.

`to_yaml()` and `to_python()` write the real column name as the key, so a
spec that came from a file never needs `col_name`.

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

