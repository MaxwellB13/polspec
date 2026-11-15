# Coming from dataframely, pandera or Pointblank

All three validate Polars frames, and two of them generate data too. This
page maps their concepts onto polspec's and translates one small schema
from each. *Added in 1.3.*

What it says about the other tools was checked against their own
documentation on 2026-11-15: dataframely's stable docs, pandera 0.36 and
Pointblank 1.3.0. "Not found" means their documentation did not show it, not
that it cannot be done.

## At a glance

| | polspec | dataframely | pandera | Pointblank |
|:--|:--:|:--:|:--:|:--:|
| Validate Polars frames | yes | yes | yes | yes |
| Generate data from the schema | yes, in Rust | yes, `sample` | not for Polars | yes, `generate_dataset` |
| Rules kept when generating | yes, `ColRule` | yes, by filtering | — | not found |
| Related tables generated together | yes, `ForeignKey` | yes, `Collection` | — | not found |
| Strings from a regex | **no** | yes | — | yes |
| Realistic names and addresses | no | no | — | yes, 100 countries |
| Lazy generation and file sinks | yes | not found | — | not found |
| Bad rows split out | yes | yes, `filter` | `drop_invalid_rows` (pandas docs) | yes, `get_sundered_data` |
| Infer a schema from data | yes | not found | pandas only | yes, `schema_from_tbl` |
| Drift from the spec | yes | not found | not found | profile comparison, `DataScan.compare` |
| Schema in a file | YAML or Python | not found | YAML, JSON | YAML validation plans |
| Typed frames for signatures | **no** | yes | yes | no |
| Reports with thresholds and actions | **no** | no | no | yes |
| pytest fixture | yes, `polspec_frames` | not found | not found | yes, `generate_dataset` |
| Hypothesis strategy | yes, `strategy` | not found | pandas only | not found |

Where polspec trails: strings from a regex (`pattern=` is validated, not
generated; use [`format=`](formats.md) for the shapes it knows), typed
frames, and Pointblank's reports and thresholds. Where it leads: generation
speed, related tables, lazy output, profiling and drift. See
[Comparison](../explanation/comparison.md) for the rest.

## The schema translated

An `Orders` table: a positive key, a country and a carrier with a rule
between them (a UK order ships by Royal Mail), a bounded quantity, and an
optional note. In polspec:

```python
class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, unique=True, bounds=(1, None))
    country  = ColSpec(pl.Enum(["UK", "US"]))
    carrier  = ColSpec(
        pl.Enum(["RoyalMail", "UPS"]),
        rules=[ColRule(when=col("country") == "UK", choices=["RoyalMail"])],
    )
    quantity = ColSpec(pl.Int64, bounds=(1, 100))
    note     = ColSpec(pl.String, nullable=True)

orders = Orders.generate(1_000, seed=1)   # UK rows all ship by Royal Mail
Orders.validate(orders)
```

A column is not nullable unless it says so, in polspec as in dataframely.
The rule is a `ColRule`, which generation follows and validation checks. A
condition no rule can express is a [`Check`](constraints.md#multi-column-invariants-__checks__),
which validation checks and generation does not attempt.

### From dataframely

<!-- docs: skip -->
```python
import dataframely as dy

class Orders(dy.Schema):
    order_id = dy.Int64(primary_key=True, min=1)
    country  = dy.Enum(["UK", "US"])
    carrier  = dy.Enum(["RoyalMail", "UPS"])
    quantity = dy.Int64(min=1, max=100)
    note     = dy.String(nullable=True)

    @dy.rule()
    def uk_ships_royal_mail(cls) -> pl.Expr:
        return (pl.col("country") != "UK") | (pl.col("carrier") == "RoyalMail")

orders = Orders.sample(1_000)
good, failures = Orders.filter(orders)
```

| dataframely | polspec |
|:--|:--|
| `class Orders(dy.Schema)` | `class Orders(FrameSpec)` |
| `dy.Int64(min=1, max=100)` | `ColSpec(pl.Int64, bounds=(1, 100))` |
| `min_exclusive=0` | `bounds=(1, None)` on an integer; bounds are inclusive, so a float takes `validators=[col("x") > 0]` |
| `dy.Enum([...])` | `ColSpec(pl.Enum([...]))`, or `choices={...}` with weights |
| `primary_key=True` on one column | `unique=True` |
| `primary_key=True` on several | `__unique_together__` |
| `regex=` | `pattern=` (validated only) or `format=` |
| `@dy.rule()` | `ColRule` when it picks values; `Check` otherwise |
| `Orders.validate(df, cast=True)` | `Orders.validate(df, cast=True)` |
| `Orders.filter(df)` | `report = Orders.inspect(df)`, then `report.passing_rows()` and `report.failing_rows(once=True)` |
| `failures.counts()` | `{f.key: f.count for f in report}` |
| `Orders.sample(n, overrides=..., generator=...)` | `Orders.generate(n, fixed=..., seed=...)` |
| `dy.Collection` | `ForeignKey` between specs, and a [`Registry`](registry.md) |
| `Orders.create_empty()` | `Orders.generate(0)` |
| `dy.DataFrame[Orders]` | none; annotate `pl.DataFrame` |
| `pyarrow.schema(Orders)` | `Orders.spec.schema()`, a `pl.Schema` |

### From pandera

<!-- docs: skip -->
```python
import pandera.polars as pa
from pandera.polars import PolarsData

class Orders(pa.DataFrameModel):
    order_id: int = pa.Field(ge=1, unique=True)
    country: str = pa.Field(isin=["UK", "US"])
    carrier: str = pa.Field(isin=["RoyalMail", "UPS"])
    quantity: int = pa.Field(ge=1, le=100)
    note: str = pa.Field(nullable=True)

    @pa.dataframe_check
    def uk_ships_royal_mail(cls, data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select(
            (pl.col("country") != "UK") | (pl.col("carrier") == "RoyalMail")
        )

Orders.validate(df, lazy=True)
```

| pandera | polspec |
|:--|:--|
| `class Orders(pa.DataFrameModel)` | `class Orders(FrameSpec)` |
| `quantity: int` | `ColSpec(pl.Int64)`: the Polars dtype itself |
| `Field(ge=1, le=100)` | `bounds=(1, 100)` |
| `Field(gt=0)` | `bounds=(1, None)` on an integer; `validators=[col("x") > 0]` on a float |
| `Field(isin=[...])` | `pl.Enum([...])`, or `choices=[...]` on a `String` |
| `Field(unique=True)` | `unique=True` |
| `@pa.check("col")` | `ColSpec(validators=[...])` |
| `@pa.dataframe_check` | `Check` in `__checks__`, or a `ColRule` |
| `validate(df, lazy=True)` | `validate(df)`: every problem is collected, always |
| `SchemaErrors` | `ValidationError`, with `.errors` and `.report` |
| `coerce=True` | `cast=True` (after checking), or `parse_text=True` for text |
| a `LazyFrame` checked for its schema only | a `LazyFrame` checked in full, in one query |
| `@pa.check_types`, `DataFrame[Orders]` | none; call `validate` at the boundary |
| `.example()` (pandas only) | `Orders.generate(n)`, or [`strategy`](testing.md#many-frames-property-tests-with-strategy) |

### From Pointblank

Pointblank declares validation as a chain of steps over one table, and
generation as a separate `pb.Schema`. polspec holds both in one spec, and
every claim on a column is a step.

<!-- docs: skip -->
```python
import pointblank as pb

validation = (
    pb.Validate(data=df, tbl_name="orders")
    .col_vals_ge(columns="order_id", value=1)
    .rows_distinct(columns_subset=["order_id"])
    .col_vals_in_set(columns="country", set=["UK", "US"])
    .col_vals_in_set(columns="carrier", set=["RoyalMail", "UPS"])
    .col_vals_between(columns="quantity", left=1, right=100)
    .col_vals_not_null(columns=["order_id", "country", "carrier", "quantity"])
    .interrogate()
)
validation.all_passed()

schema = pb.Schema(
    order_id=pb.int_field(min_val=1, unique=True),
    country=pb.string_field(allowed=["UK", "US"]),
    carrier=pb.string_field(allowed=["RoyalMail", "UPS"]),
    quantity=pb.int_field(min_val=1, max_val=100),
    note=pb.string_field(nullable=True),
)
fake = pb.generate_dataset(schema, n=1_000, seed=1)
```

| Pointblank | polspec |
|:--|:--|
| `col_vals_between`, `col_vals_ge`, … | `bounds=` |
| `col_vals_in_set` | `pl.Enum([...])` or `choices=` |
| `col_vals_not_null` | the default; `nullable=True` allows nulls |
| `col_vals_regex` | `pattern=` |
| `rows_distinct` | `unique=True`, or `__unique_together__` |
| `col_schema_match` | always checked: each column's dtype |
| `row_count_match` | `__rows__` |
| `col_vals_expr` | `Check` in `__checks__` |
| `interrogate()` then `all_passed()` | `Orders.inspect(df).passed` |
| `get_sundered_data(type="pass")` | `report.passing_rows()` |
| thresholds and actions | none: a finding has a `count`, so gate on it yourself |
| `yaml_interrogate("plan.yaml")` | `polspec validate orders.yaml data.parquet` |
| `schema_from_tbl(df)` | `polspec.profile(df)` |
| `DataScan(...).compare(...)` | `polspec.drift(Orders, df)` |
| `pb.Schema(...)` and `generate_dataset` | the same `Orders`, and `Orders.generate(n, seed=...)` |
| `string_field(preset="name", country=...)` | none: pass [Faker's values as `choices`](../explanation/comparison.md#compared-to-faker-and-similar) |

The one translation polspec cannot make is a regex Pointblank generates
from, such as `string_field(pattern=r"SKU-[A-Z]{2}-[0-9]{5}")`: polspec
validates `pattern=` but fills the column with ordinary text. See
[Known limitations](../explanation/limitations.md).
