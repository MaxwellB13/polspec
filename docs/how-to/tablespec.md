# Specs as values

A `FrameSpec` class body is the convenient way to *write* a spec. What it
builds is a `TableSpec`: an immutable value holding the columns, checks,
composite keys and foreign keys, reachable as `.spec` on the class.

```python
import polars as pl
from polspec import ColSpec, ForeignKey, FrameSpec, Registry

class Orders(FrameSpec):
    order_id      = ColSpec(pl.Int64, unique=True)
    total         = ColSpec(pl.Float64, bounds=(0.0, None))
    internal_note = ColSpec(pl.String, nullable=True)

Orders.spec              # TableSpec(name='Orders', columns={...}, ...)
Orders.spec.name         # 'Orders'
list(Orders.spec)        # ['order_id', 'total', 'internal_note']
Orders.spec["total"]     # the ColSpec
Orders.spec.schema()     # the same pl.Schema as Orders.schema()
```

Every verb the library offers is a function over a `TableSpec`; the
classmethods on `FrameSpec` are one-line forwards that pass `cls.spec`. So a
`TableSpec` is the thing being operated on either way:

```python
import polspec

Orders.generate(1_000, seed=1)          # the class
polspec.generate(Orders.spec, 1_000, seed=1)  # the function, over the value
```

Both reach the same code. The functions are exported from `polspec` itself:

```python
from polspec import generate, generate_batches, inspect, validate
from polspec import sink_csv, sink_ipc, sink_ndjson, sink_parquet

df = generate(Orders.spec, 1_000, seed=1)
report = inspect(Orders.spec, df)
validate(Orders.spec, df)
```

Each takes a `TableSpec` as its first argument, and each has a `FrameSpec`
classmethod that forwards to it with `cls.spec`. Use whichever suits the code:
the classmethods read better when a class body declared the spec, the
functions when the spec is a value that was built, loaded or derived.

## Building one directly

A `TableSpec` can be constructed without a class body, which is how
`from_yaml` and `from_dataframe` work internally:

```python
from polspec import TableSpec

spec = TableSpec(
    "Orders",
    {"order_id": ColSpec(pl.Int64, unique=True), "total": ColSpec(pl.Float64)},
    unique_together=[["order_id"]],
)
```

Everything a class body validates at declaration is validated here too. A
`TableSpec` that constructs is one that can be used.

To get the class-shaped API back, wrap it:

```python
Rebuilt = FrameSpec.from_spec(spec)                    # a subclass named Orders
Renamed = FrameSpec.from_spec(spec, name="Orders2026")
```

## Deriving one spec from another

Each operation returns a new `TableSpec`; the original is never changed.

| Operation | Effect |
|:--|:--|
| `with_columns({...}, **cols)` | Add columns, or replace existing ones in place |
| `drop(*names)` | Remove columns, and any composite key, foreign key or check that used them |
| `select(*names)` | Keep only the named columns, in that order, and the constraints that still apply |
| `rename({old: new})` | Rename columns, rewriting rules, validators, checks, composite keys, foreign keys and the hierarchy |
| `prefix(text)`, `suffix(text)` | Rename every column, as `rename` does |
| `join(other, on=..., how=...)` | The spec of a join of two frames; see [below](#joining-two-specs) |
| `with_checks(*checks)`, `with_foreign_keys(*fks)`, `with_unique_together(*groups)` | Append constraints |
| `with_name(name)` | Change the name |
| `with_catspec(registry)` | Re-type columns against a `CatSpec`; see [Shared categories](categories.md) |

```python
staging = Orders.spec.drop("internal_note").rename({"total": "amount"})
Staging = FrameSpec.from_spec(staging, name="StagingOrders")
```

Two deliberate limits. `drop` leaves a rule on a surviving column that points
at a dropped one for validation to reject, since silently dropping a rule
would change what the surviving column generates. `rename` rewrites a
validator or check written with [`col()`](constraints.md#writing-conditions-col)
like everything else, but refuses to rename a column that one written as a
raw `pl.Expr` names: rewriting Polars expressions is not something this
library does, and leaving it would keep a check naming a column the spec no
longer has.

## Joining two specs

`join` takes the arguments `LazyFrame.join` takes -- `on`, or `left_on` and
`right_on`, `how`, `suffix`, `coalesce` -- and returns the spec of the frame
that join makes. Its columns, their order and their names are the ones the
installed Polars gives: polspec runs the join on two empty frames and reads
the result, so a suffixed clash or a coalesced key is named as it will be.

```python
class Customers(FrameSpec):
    customer_id = ColSpec(pl.Int64, unique=True)
    tier        = ColSpec(pl.Enum(["gold", "silver"]))

class Sales(FrameSpec):
    sale_id     = ColSpec(pl.Int64, unique=True)
    customer_id = ColSpec(pl.Int64)

enriched = Sales.spec.join(Customers, on="customer_id", how="left")
enriched.name                  # 'Sales_Customers'
enriched["tier"].nullable      # True: a sale may meet no customer
enriched["sale_id"].unique     # True: each sale meets one customer at most
```

Each column keeps what it claims about its values -- dtype, bounds,
choices, weights, format, pattern, lengths, tags, description, validators
and default -- from the side it came from. A side that can go unmatched
becomes nullable: the right of `left`, the left of `right`, both of `full`.

What a spec says about its *rows* is not carried, since a join changes
which rows there are: checks, rules, foreign keys, composite keys, the
hierarchy, the order and `rows` are dropped; add back any that still holds
with its `with_*`. `unique` stays on a side's columns only when the other
side's join key is declared unique -- a unique column, or a composite key --
because then no row of that side is repeated. `semi` and `anti` keep the
left spec, but for `rows` and the order: Polars does not promise a join
keeps its rows' order. Join keys of different dtypes are refused, as Polars
refuses them.

## Column names and method names

Because the class body's `ColSpec` attributes are taken out of the namespace
before the class exists, a column may share a name with a method. The method
wins on attribute access; the column is reachable by name:

```python
class Raw(FrameSpec):
    schema = ColSpec(pl.String)
    tag    = ColSpec(pl.String)

Raw.schema()            # the method: Schema({'schema': String, 'tag': String})
Raw.col("schema")       # the column
Raw.spec["tag"]         # also the column
```

An ordinary column is still an attribute (`Orders.order_id`), through a
fallback that runs only when normal lookup fails.

## Foreign keys point at names

`ForeignKey.references` is stored as the target spec's *name*. Declaring
`references=Customers` binds the target for declaration-time checks and
stores `"Customers"`; a key can also be declared against a bare name, which
nothing checks until a spec of that name is supplied:

```python
class Customers(FrameSpec):
    id = ColSpec(pl.Int64, bounds=(1, 10_000), unique=True)

ForeignKey("customer_id", references=Customers, ref_columns="id")   # checked now
ForeignKey("customer_id", references="Customers", ref_columns="id") # checked later
```

`generate(references={...})` and `validate(references={...})` accept the
parent frame keyed by the class, the `TableSpec`, or the name. A
[`Registry`](registry.md) holding both specs binds the name and runs the
checks the class form would have run at declaration:

```python
class Shipments(FrameSpec):
    customer_id = ColSpec(pl.Int64, bounds=(1, 10_000))
    __foreign_keys__ = [
        ForeignKey("customer_id", references="Customers", ref_columns="id")
    ]

Shipments.spec.foreign_keys[0].target                     # None -- nothing to check against

bound = Registry(Customers, Shipments).resolve()
bound["Shipments"].foreign_keys[0].target                 # Customers.spec
```
