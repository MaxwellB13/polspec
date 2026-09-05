# Constraints

Beyond the shape of a single value, a spec can assert relationships. They fall
into two groups worth keeping straight:

| | Generated | Validated |
|:--|:--:|:--:|
| `ColRule` | yes | yes |
| `ForeignKey` | yes, when given parent data | yes |
| `unique=True`, `__unique_together__` | no | yes |
| `ColSpec.validators`, `__checks__` | no | yes |

The last two rows are validation-only by design: they wrap arbitrary Polars
expressions, and nothing can produce data satisfying an arbitrary predicate.
Generation makes no attempt, and that boundary is pinned down by tests.

## Writing conditions — `col()`

Rules, validators and checks all take a condition. Write it with `col()`,
which builds a small predicate tree rather than a Polars expression:

```python
from polspec import col

col("total") >= col("subtotal")
col("email").str.contains("@")
col("status").is_in(["NEW", "PAID"]) & (col("qty") > 0)
col("shipped").is_null() | (col("shipped") >= col("placed"))
```

Supported: comparisons (`== != < <= > >=`), arithmetic (`+ - * /`),
`&`, `|`, `~`, `is_in`, `is_null`, `is_not_null`, `is_between`, and the
string operations `str.contains` (a literal substring), `str.starts_with`,
`str.ends_with`, `str.matches` (a regular expression) and `str.len_chars`.
Scalars, dates and datetimes are fine as operands.

A predicate evaluates exactly as the Polars expression it stands for, and
unlike one it can be written to a spec file and read back, so rules, checks
and validators written this way survive `to_yaml` and `to_python`. A raw
`pl.Expr` is still accepted everywhere a predicate is, for anything the
predicate language cannot say; it just cannot be persisted.

Comparison operators build predicates, as they do on `pl.Expr`, so a
predicate has no truth value. Compare two structurally with `Pred.equals`.

## Conditional values — `ColRule`

A rule overwrites a column on the rows where its condition matches.

```python
from polspec import ColRule

class Shipments(FrameSpec):
    region = ColSpec(pl.Enum(["UK", "US", "EU"]))
    carrier = ColSpec(
        pl.Enum(["RoyalMail", "UPS", "DHL"]),
        rules=[
            ColRule(when={"column": "region", "equals": "UK"}, choices=["RoyalMail"]),
            ColRule(when={"column": "region", "in": ["US", "EU"]}, choices=["UPS", "DHL"]),
        ],
    )
```

Multiple rules on one column are tried in declaration order, first match wins,
like a SQL `CASE`. `choices` may be weighted exactly as on a `ColSpec`:

```python
ColRule(when={"column": "region", "equals": "US"}, choices={"UPS": 3.0, "DHL": 1.0})
```

`when` is a small dict rather than an expression so that rules survive a YAML
round-trip. The supported forms:

```python
{"column": "c", "equals": "A"}          {"column": "c", "not_equals": "A"}
{"column": "c", "in": ["A", "B"]}       {"column": "c", "not_in": ["A", "B"]}
{"column": "c", "lt": 10}               {"column": "c", "lte": 10}
{"column": "c", "gt": 10}               {"column": "c", "gte": 10}
{"column": "c", "between": [1, 10]}
{"column": "c", "is_null": True}        {"column": "c", "is_not_null": True}
```

!!! note "Rules see the pre-rule frame"

    Every `when` is evaluated against the freely-generated values, never
    another rule's output, so rules on different columns need no dependency
    ordering. The cost is that a rule keyed on a *rule-targeted* column does
    not round-trip — see [Known limitations](../reference/limitations.md).

A rule also overwrites nulls on matching rows, so a nullable column with a rule
ends up with fewer nulls than `null_probability` suggests.

## Single-column predicates — `validators`

A validator is a Polars expression that each row must satisfy, referencing only
its own column:

```python
class Accounts(FrameSpec):
    email = ColSpec(pl.String, validators=[pl.col("email").str.contains("@")])
```

Wrap one in a `Check` to name it, describe it, or change null handling:

```python
from polspec import Check

ColSpec(
    pl.Float64,
    validators=[
        Check(
            pl.col("score") <= 100,
            name="score_ceiling",
            description="Scores are a percentage",
        )
    ],
)
```

Referencing another column is rejected at declaration time — use `__checks__`
for that.

## Multi-column invariants — `__checks__`

```python
class Invoices(FrameSpec):
    subtotal = ColSpec(pl.Float64, bounds=(0.0, 1000.0))
    total    = ColSpec(pl.Float64, bounds=(0.0, 2000.0))

    __checks__ = [
        Check(pl.col("total") >= pl.col("subtotal"), name="total_covers_subtotal"),
    ]
```

By default a row whose check evaluates to null passes, matching SQL `CHECK`
semantics. `Check(..., ignore_nulls=False)` treats null as a failure.

Checks are inherited: a subclass collects its bases' checks as well as its own,
de-duplicated. Two *different* checks sharing a name is an error, since the name
is what an error message points at.

## Composite uniqueness — `__unique_together__`

```python
class Memberships(FrameSpec):
    user_id = ColSpec(pl.Int64)
    team_id = ColSpec(pl.Int64)

    __unique_together__ = [["user_id", "team_id"]]
```

Rows where any constituent column is null are exempt. A flat list of strings is
read as one composite key; a list of lists declares several.

## Referential integrity — `ForeignKey`

```python
from polspec import ForeignKey

class Customers(FrameSpec):
    id = ColSpec(pl.Int64, bounds=(1, 10_000))

class Orders(FrameSpec):
    customer_id = ColSpec(pl.Int64, bounds=(1, 10_000))

    __foreign_keys__ = [
        ForeignKey("customer_id", references=Customers, ref_columns="id"),
    ]
```

Rows where any key column is null are exempt — a null foreign key means "no
reference", not an invalid one.

### Generating consistent data

Pass the parent frame and the child's key values are sampled from it, so the
result is referentially consistent by construction:

```python
customers = Customers.generate(1_000, seed=1)
orders    = Orders.generate(10_000, seed=2, references={Customers: customers})

Orders.validate(orders, references={Customers: customers})
```

Without `references`, generation leaves the column freely generated — but
`validate()` then *raises*, because it has nothing to check against. Supply the
parent to both calls, or disable the check with
`validate(..., validate_foreign_keys=False)`.

Composite keys are sampled as one joint pick per row, so multi-column keys stay
internally consistent.

With several related specs, a [`Registry`](registry.md) does the walk:
`Registry(Customers, Orders, OrderLines).generate_all(1_000, seed=1)` generates
parents first and threads each into its children, and `validate_all` checks
the whole set.

### Self-references

```python
class Employees(FrameSpec):
    id         = ColSpec(pl.Int64, bounds=(1, 500))
    manager_id = ColSpec(pl.Int64, bounds=(1, 500), nullable=True)

    __foreign_keys__ = [
        ForeignKey("manager_id", references="self", ref_columns="id"),
    ]
```

`"self"` resolves to whichever spec the key ends up declared on, and needs no
`references` entry in either call.

!!! warning "Declare the same domain on both sides"

    A foreign key overwrites its column with values from the parent, ignoring
    that column's own `bounds` or `choices`. Declaring `bounds=(1, 50)` on a
    column that references keys in `100..200` produces data that fails
    validation. Keep the two declarations compatible.
