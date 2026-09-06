# Shared categories

A `CatSpec` is a registry of `Enum` and `Categorical` definitions shared across
specs, so several tables agree on a domain instead of each restating it.

## Declaring one by hand

Subclass `CatSpec`, one line per entry, in the same vocabulary `ColSpec.dtype`
already accepts:

```python
import polars as pl
from polspec import CatSpec, ColSpec, FrameSpec

class Categories(CatSpec):
    STATUS   = pl.Enum(["NEW", "PAID", "SHIPPED"])
    CURRENCY = pl.Categorical(pl.Categories("CURRENCY", physical=pl.UInt8))
```

Naming an entry gives back its dtype, so it plugs straight into a `ColSpec`:

```python
class Orders(FrameSpec):
    status   = ColSpec(Categories.STATUS)
    currency = ColSpec(Categories.CURRENCY)
```

This is deliberately not a dict. `CatSpec(enums={...}, categoricals={...},
choices={...})` puts one name across up to three parallel mappings that all
have to stay in step; a class body puts each entry on its own line, in the
declaration order that also documents it.

The entries are lifted out of the class body before the class exists — the same
thing `FrameSpec` does with `ColSpec` columns — so an entry may be named
anything, including a name `CatSpec` already uses:

```python
class TrickyNames(CatSpec):
    get = pl.Enum(["A", "B"])       # an entry, not a collision

TrickyNames.get                     # still the method
TrickyNames.spec.get("get")         # pl.Enum(["A", "B"])
```

An unnamed `pl.Categorical()` is rejected outright, since a registry entry with
no name can't act as a shared key.

`Categories.spec` is the `CatSpec` value the class body declares. Anywhere a
registry is expected — `with_catspec`, `Registry(categories=...)` — the class
and the value are interchangeable.

## The dict constructor

The form `CatSpec.infer()`, `from_dataframe()` and `from_yaml()` build
programmatically, since their entry names come from data at runtime rather
than from a class body someone writes by hand:

```python
categories = CatSpec(
    enums={"STATUS": ["NEW", "PAID", "SHIPPED"]},
    categoricals={"CURRENCY": pl.Categories("CURRENCY", physical=pl.UInt8)},
)
```

The two forms compose rather than compete: a class-body subclass's entries
become the defaults, and an explicit `enums=`/`categoricals=`/`choices=`
argument at construction time can still add to or override them per key.

```python
extended = Categories(enums={"REASON": ["FRAUD", "DUPLICATE"]})
extended.get_enum("STATUS")   # ["NEW", "PAID", "SHIPPED"] -- inherited
extended.get_enum("REASON")   # ["FRAUD", "DUPLICATE"]     -- added
```

They also compare equal when they say the same thing, so a registry loaded from
a file can be checked against the one a class body declares:

<!-- docs: skip -->
```python
Categories.spec == CatSpec.from_yaml("categories.yaml")
```

## Using a registry

Whichever form built it, the accessors are the same, and naming an entry always
means the same thing: the dtype.

```python
categories.STATUS                   # -> pl.Enum([...])
categories.CURRENCY                 # -> pl.Categorical(...)
categories["STATUS"]                # -> the same dtype
categories.get("STATUS")            # -> the same dtype, or None
```

Ask for the pieces underneath when you want them rather than the dtype:

```python
categories.get_enum("STATUS")           # -> list[str]
categories.get_categorical("CURRENCY")  # -> pl.Categories
categories.get_choices("CURRENCY")      # -> the domain pool, or None
```

And name the kind when you want the lookup to insist on it — these refuse an
entry of the other kind instead of quietly returning it:

```python
categories.enum.STATUS              # -> pl.Enum
categories.enum["STATUS"]           # item access
categories.enum("STATUS")           # callable
categories.categorical.CURRENCY     # -> pl.Categorical
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
Optimized = Orders.with_catspec(CatSpec.infer(df))   # infer, then apply
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
it automatically — see [YAML specs](files.md).

Also available: `to_markdown()` for a documentation table, and `to_mermaid()`
for a class diagram of the registry.
