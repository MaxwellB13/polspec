# YAML specs

A spec can live in a file instead of a class body, so tooling outside Python
can read it and so it can be reviewed as a document.

```python
Orders.to_yaml("orders.yaml")
Loaded = FrameSpec.from_yaml("orders.yaml")

Loaded.generate(1_000, seed=1)
```

The output is plain, readable YAML — defaults are omitted so the file shows
only what you actually declared:

```yaml
name: Orders
columns:
  order_id:
    dtype: Int64
    bounds: [1, 100000]
    unique: true
  status:
    dtype:
      Enum: [NEW, PAID, SHIPPED]
  total:
    dtype: Float64
    bounds: [0.0, null]
  placed:
    dtype: Date
    nullable: true
unique_together:
  - [order_id, status]
```

An open-ended bound writes as `null` and reads back unchanged.

## What survives a round-trip

| | Round-trips |
|:--|:--:|
| dtypes, including parametrized `Enum` / `Datetime` / `Duration` / named `Categorical` | yes |
| `nullable`, `null_probability`, `bounds`, `string_length`, `unique`, `tags` | yes |
| `choices`, `weights`, `distribution`, `distribution_params` | yes |
| `rules` (`ColRule`) | yes |
| `__unique_together__` | yes |
| `__foreign_keys__` with `references="self"` | yes |
| `__foreign_keys__` referencing another spec | **no** |
| `__checks__` | **no** |
| `ColSpec.validators` | **no** |

The three that cannot be written all wrap an arbitrary `polars.Expr` or a
Python class with no stable name in a standalone file. `to_yaml()` warns about
each, naming exactly what will be lost:

```text
UserWarning: Orders declares 1 __checks__ ('total_covers_subtotal') that cannot
be represented in YAML (a Check wraps an arbitrary polars.Expr) and will NOT be
written to orders.yaml. They will be lost on FrameSpec.from_yaml() unless
re-declared on a subclass of the loaded spec.
```

The suggested recovery is to subclass what you loaded:

```python
Loaded = FrameSpec.from_yaml("orders.yaml")

class Orders(Loaded):
    __checks__ = [Check(pl.col("total") >= pl.col("subtotal"), name="total_covers_subtotal")]
```

Columns, rules and unique keys come from the file; the expression-valued parts
are re-declared in Python where they can actually live.

## Sharing categories between files

A spec file can reference a `CatSpec` registry by path, resolved relative to
the spec file:

```yaml
name: Orders
categories: categories.yaml
columns:
  status:
    dtype:
      Enum: STATUS
  currency:
    dtype:
      Categorical: CURRENCY
```

`$categories.STATUS` and `categories.STATUS` are accepted as prefixed forms of
the same reference.

Or pass a registry explicitly, which wins over anything the file names:

```python
FrameSpec.from_yaml("orders.yaml", categories=CatSpec.from_yaml("categories.yaml"))
FrameSpec.from_yaml("orders.yaml", categories="categories.yaml")
```

A spec can also emit the registry its own columns imply:

```python
Orders.write_catspec("categories.yaml")
```

## Column names from data

`from_yaml` declares columns through `__columns__`, so names that could not be
class attributes — a leading underscore, or a collision with a method name like
`schema` — load correctly.
