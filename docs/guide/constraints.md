# Constraints

Beyond the shape of a single value, a spec can assert relationships. They fall
into two groups worth keeping straight:

| | Generated | Validated |
|:--|:--:|:--:|
| `ColRule` | yes | yes |
| `ForeignKey` | yes, when given parent data | yes |
| `unique=True`, `__unique_together__` | yes | yes |
| `ColSpec.validators`, `__checks__` | no | yes |

The last row is validation-only by design: both wrap arbitrary Polars
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

!!! note "Rules see the frame as it stands"

    Every `when` is evaluated against the values the column actually holds
    when the rule runs, and the passes run in dependency order: a rule keyed
    on a column that another rule or a foreign key rewrites reads the
    rewritten values — the same ones `validate()` will check the rule
    against. So rules chain:

    ```python
    class Orders(FrameSpec):
        region  = ColSpec(pl.Enum(["UK", "US"]))
        carrier = ColSpec(
            pl.Enum(["RoyalMail", "UPS"]),
            rules=[ColRule(when=col("region") == "UK", choices=["RoyalMail"])],
        )
        tracked = ColSpec(  # keyed on a column that carries rules of its own
            pl.Enum(["yes", "no"]),
            rules=[ColRule(when=col("carrier") == "RoyalMail", choices=["yes"])],
        )
    ```

    Two columns whose rules each read what the other writes have no such
    order, and are refused at declaration with `SpecError`.

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

A composite key declares that a *combination* of columns is distinct, even
where each column on its own repeats.

```python
class Assignments(FrameSpec):
    employee_id = ColSpec(pl.Int64, bounds=(1, 500))
    project_id  = ColSpec(pl.Int64, bounds=(1, 200))

    __unique_together__ = [["employee_id", "project_id"]]
```

`generate()` satisfies it by resampling the rows that repeat a combination an
earlier row already used. Only the repeats move, so on a roomy domain almost
every row keeps the value it was generated with, along with whatever weights
or bounds shaped it. Rows where any member is null are exempt, matching how
the key is validated.

A group whose columns cannot take enough distinct combinations between them is
refused, naming the group:

```python
class TooTight(FrameSpec):
    a = ColSpec(pl.Enum(["x", "y"]))
    b = ColSpec(pl.Enum(["p", "q"]))
    __unique_together__ = [["a", "b"]]

TooTight.generate(300, seed=1)
# GenerationError: Composite unique key ['a', 'b'] cannot be satisfied: the
# columns take 4 distinct combination(s) between them and 300 row(s) need one.
```

A member column may not also carry `rules`: a rule assigns from a fixed set,
which is how two rows come to share a combination, and the repair would
overwrite what the rule put there. Declare one or the other.

A foreign-keyed member is never resampled — that would break the key — so the
repair works with the other members. If *every* member is foreign-keyed there
is nothing it can move, and generation says so rather than returning data that
fails its own validation.

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

!!! warning "The parent's domain has to fit inside the column's own"

    A foreign key overwrites its column with values from the parent, so the
    parent's `bounds` or `choices` have to be ones the column itself declares
    it can hold. Declaring `bounds=(1, 50)` on a column referencing keys in
    `100..200` would produce data that fails its own validation, so it is
    refused when you declare it:

    ```python
    class Orders(FrameSpec):
        customer_id = ColSpec(pl.Int64, bounds=(1, 50))
        __foreign_keys__ = [
            ForeignKey("customer_id", references=Customers, ref_columns="id")
        ]
    # SpecError: ... column 'customer_id' is declared bounds [1, 50], but the
    # key fills it with values from 'id' on 'Customers', where bounds
    # [1, 10000] do not fit inside [1, 50].
    ```

    A column that declares no `bounds` or `choices` accepts anything, so the
    check only fires on a genuine contradiction. Widen or drop the child's
    declaration, or narrow the parent's.

    The check needs both specs, so a key naming its target as a string is
    checked when a [`Registry`](registry.md) resolves it, not before.
