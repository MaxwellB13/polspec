# Shared categories

A `CatSpec` is a registry of `Enum` and `Categorical` definitions shared across
specs, so several tables agree on a domain instead of each restating it.

```python
import polars as pl
from polspec import CatSpec

categories = CatSpec(
    enums={"STATUS": ["NEW", "PAID", "SHIPPED"]},
    categoricals={"CURRENCY": pl.Categories("CURRENCY", physical=pl.UInt8)},
)
```

## Using a registry

Three equivalent accessors, whichever reads best where you are:

```python
pl.Enum(categories.STATUS)          # attribute
categories.enum.STATUS              # typed accessor -> pl.Enum
categories.enum["STATUS"]           # item access
categories.categorical.CURRENCY     # -> pl.Categorical
```

```python
class Orders(FrameSpec):
    status   = ColSpec(categories.enum.STATUS)
    currency = ColSpec(categories.categorical.CURRENCY)
```

Lookup falls back to case variants, so a column named `status` finds a registry
entry named `STATUS`. Convenient, but worth knowing about if you have entries
differing only in case.

## Why a shared `Categories` matters

A named `pl.Categories()` registry gives two columns the same physical codes,
so frames can be joined on the code rather than the string. polspec preserves
that identity through generation and through a YAML round-trip.

Choosing a narrow physical dtype is a real memory saving on wide tables:

| Physical | Distinct categories |
|:--|:--|
| `UInt8` | 255 |
| `UInt16` | 65,535 |
| `UInt32` (default) | ~4 billion |

polspec respects the ceiling: a `Categorical` on a `UInt8` registry generates
from a pool sized to the registry rather than overflowing it. Where the
registry is *named*, that pool is derived from the registry's own identity, so
two specs sharing it draw from the same domain.

## Building a registry from what you have

```python
CatSpec.from_dataframe(df)      # existing Enum/Categorical columns
CatSpec.from_framespec(Orders)  # a spec's declared columns
```

## Inferring one

`infer` picks a representation per column by cardinality:

```python
categories = CatSpec.infer(df, max_enum_cardinality=30)
```

- at most `max_enum_cardinality` distinct values → `Enum`
- otherwise, up to `max_categorical_cardinality` and either a low unique ratio
  or under 256 values → `Categorical` with the narrowest physical dtype that fits
- otherwise, left as `String`

Identifier-shaped names are skipped by default, since they are high-cardinality
by nature: `*_id`, `*_uuid`, `*_hash`, `*_url`, `*_key`. Override with
`exclude_patterns`, or force specific columns with `include_columns`.

## Re-typing a spec

`with_catspec` returns a new spec with matching columns re-pointed at the
registry's types:

```python
Optimized = Orders.with_catspec(categories)
Optimized = Orders.with_inferred_catspec(data=df)   # infer, then apply
```

Re-typing changes the dtype and nothing else a column declared — `unique`,
`string_length`, `nullable`, tags, rules and validators all carry over. The two
fields a dtype change can genuinely invalidate are dropped with a warning:
`weights`, which is positional over a domain that just resized, and `choices`,
when the new dtype has no category for them.

## Persisting a registry

```python
categories.to_yaml("categories.yaml")
categories = CatSpec.from_yaml("categories.yaml")
```

```yaml
enums:
  STATUS: [NEW, PAID, SHIPPED]
categoricals:
  CURRENCY:
    name: CURRENCY
    physical: UInt8
    categories: [GBP, USD, EUR]
```

A spec's YAML can point at a registry file, and `FrameSpec.from_yaml` resolves
it automatically — see [YAML specs](yaml.md).

Also available: `to_markdown()` for a documentation table, and `to_mermaid()`
for a class diagram of the registry.
