# Recipes

Short answers to "how do I…", each linking to the page that covers it in
full. *Added in 1.3.*

The examples share these two specs:

```python
class Customers(FrameSpec):
    customer_id = ColSpec(pl.Int64, unique=True, bounds=(1, 10_000))
    tier        = ColSpec(pl.Enum, choices={"free": 0.8, "pro": 0.2})

class Orders(FrameSpec):
    order_id    = ColSpec(pl.Int64, unique=True, bounds=(1, 1_000_000))
    customer_id = ColSpec(pl.Int64)
    total       = ColSpec(pl.Float64, bounds=(0, 500))
    __foreign_keys__ = [
        ForeignKey("customer_id", references=Customers, ref_columns="customer_id")
    ]

customers = Customers.generate(200, seed=1)
orders = Orders.generate(1_000, seed=2, references={Customers: customers})
```

## Declaring

### A primary key

`unique=True` on one column; `__unique_together__` for a key of several.
Generation draws distinct values, and validation rejects a repeat.

```python
class OrderLines(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, 1_000))
    line_no  = ColSpec(pl.Int64, bounds=(1, 20))
    __unique_together__ = [("order_id", "line_no")]
```

A key is not nullable unless you say so. See
[Composite uniqueness](constraints.md#composite-uniqueness-__unique_together__).

### IDs `1, 2, …, n`

A unique column bounded `(1, n)` in a frame of `n` rows holds every value
from 1 to `n`; order the frame by it for the sequence.

```python
class Invoices(FrameSpec):
    __order_by__ = "invoice_id"
    invoice_id = ColSpec(pl.Int64, unique=True, bounds=(1, 500))

assert Invoices.generate(500, seed=0)["invoice_id"].to_list() == list(range(1, 501))
```

See [A sequence](constraints.md#a-sequence-1-2-n).

### Sorted timestamps

`__order_by__` names the column, and generation sorts by it. Validation
checks the frame comes sorted.

```python
class Events(FrameSpec):
    __order_by__ = "at"
    at = ColSpec(pl.Datetime("us"), bounds=(datetime(2026, 1, 1), datetime(2026, 12, 31)))

assert Events.generate(1_000, seed=0)["at"].is_sorted()
```

`["user_id", "at"]` sorts within each user. `scan()` cannot produce an
ordered frame; see [Row order](constraints.md#row-order-__order_by__).

### An order reference, a postcode, a SKU

A `pattern` is generated from, so a code with a shape needs nothing else,
and `unique=True` draws each value once:

```python
class Parcels(FrameSpec):
    ref      = ColSpec(pl.String, pattern=r"^PCL-\d{8}$", unique=True)
    postcode = ColSpec(pl.String, pattern=r"^[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}$")

Parcels.validate(Parcels.generate(1_000, seed=1))
```

*Added in 1.4.* See [Patterns](formats.md#patterns).

### A total computed from other columns

`derive=` declares the expression; generation computes it, and
validation holds the column to it:

```python
class Lines(FrameSpec):
    qty   = ColSpec(pl.Int64, bounds=(1, 20))
    price = ColSpec(pl.Float64, bounds=(0.5, 100))
    total = ColSpec(pl.Float64, derive=col("qty") * col("price"))

Lines.validate(Lines.generate(1_000, seed=1))
```

*Added in 1.4.* See [Derived columns](constraints.md#derived-columns-derive).

### A nullable foreign key

Declare the column nullable. A null references nothing, so it is neither
checked against the parent nor drawn from it.

```python
class Tickets(FrameSpec):
    ticket_id   = ColSpec(pl.Int64, unique=True, bounds=(1, 100_000))
    customer_id = ColSpec(pl.Int64, nullable=True, null_probability=0.3)
    __foreign_keys__ = [
        ForeignKey("customer_id", references=Customers, ref_columns="customer_id")
    ]

tickets = Tickets.generate(500, seed=3, references={Customers: customers})
Tickets.validate(tickets, references={Customers: customers})
```

See [Referential integrity](constraints.md#referential-integrity-foreignkey).

### One spec derived from another

Every structural operation on a `TableSpec` returns a new one: `drop`,
`select`, `rename`, `with_columns`, `join`. `FrameSpec.from_spec` makes a
class of it again.

```python
Staging = FrameSpec.from_spec(
    Orders.spec.drop("total").rename({"order_id": "id"}), name="StagingOrders"
)
```

A `FrameSpec` subclass inherits its parent's columns too. See
[Deriving one spec from another](tablespec.md#deriving-one-spec-from-another).

## Generating

### An empty frame with the spec's schema

```python
empty = Orders.generate(0)
assert empty.height == 0 and empty.schema == Orders.spec.schema()
```

### A frame for one test, or one for the session

The `polspec_frames` fixture gives each test its own seed; a
session-scoped fixture over `generate()` gives every test the same frame.

<!-- docs: skip -->
```python
def test_spend(polspec_frames):
    orders = polspec_frames(Orders, 500, references={Customers: customers})
    ...

@pytest.fixture(scope="session")
def orders(customers):
    return Orders.generate(2_000, seed=2, references={Customers: customers})
```

See [Test pipelines](testing.md#a-frame-shared-by-tests).

### A fake copy of related tables

`synthesize_all` profiles the tables, finds the keys between them, and
generates a fake set in which every key holds:

```python
real = {"Customers": customers, "Orders": orders}
fake = polspec.synthesize_all(real, scale=0.1, seed=1)   # a tenth of each
assert fake["Orders"]["customer_id"].is_in(fake["Customers"]["customer_id"].implode()).all()
```

*Added in 1.5.* From files, `polspec synthesize customers.csv orders.csv -o
fake/`. See [Related tables](synthesizing.md#related-tables).

### A key the profile missed

A key is found only where the names link the tables, or the column is text
of many values. Declare any other with `keys=`, and it is checked against
the data:

```python
renamed = {"Customers": customers, "Orders": orders.rename({"customer_id": "buyer"})}
keys = {"Orders": [ForeignKey("buyer", references="Customers", ref_columns="customer_id")]}
registry = polspec.profile_all(renamed, keys=keys)
```

*Added in 1.5.* On the command line, `--foreign-key
Orders.buyer=Customers.customer_id`. See
[Related tables](synthesizing.md#related-tables).

## Validating

### Validate in a method chain

`validate` hands the frame back, so it sits in `pipe`:

```python
big_orders = (
    orders.pipe(Orders.validate, references={Customers: customers})
    .filter(pl.col("total") > 400)
)
```

### Check a lazy pipeline's shape without running it

```python
pipeline = pl.LazyFrame(schema=Orders.spec.schema()).with_columns(pl.col("total") * 2)
Orders.validate(pipeline, schema_only=True)   # columns and dtypes; no rows read
```

*Added in 1.4.* See [The schema alone](validating.md#the-schema-alone).

### Quarantine the bad rows

`inspect()` never raises. Its report splits the frame in two: the rows that
pass, and the rows that fail, each naming the claims it broke.

```python
suspect = orders.with_columns(
    pl.when(pl.col("order_id") % 10 == 0).then(-1.0).otherwise(pl.col("total")).alias("total")
)
report = Orders.inspect(suspect, references={Customers: customers})

kept = report.passing_rows().collect()
quarantined = report.failing_rows(once=True).collect()
assert kept.height + quarantined.height == suspect.height
```

`polspec validate --failing bad.parquet` writes the second half from the
shell. See [Findings as data](validating.md#findings-as-data-inspect).

### A CSV read as text

A file too unclean to type on reading arrives as text. `parse_text=True`
reads each column as the type its spec declares before checking it, and
`profile(parse_text=True)` infers those types to begin with.

```python
raw = pl.DataFrame({"customer_id": ["1", "2"], "tier": ["free", "pro"]})
# as pl.read_csv(path, infer_schema=False) hands it back

Customers.validate(raw, parse_text=True, cast=True)
```

See [Data read as text](validating.md#data-read-as-text) and
[profiling it](profiling.md#data-read-as-text).

### Dates written as `30 Sep 2031`

Declare the format the file writes them in; `parse_text=True` and `read()`
read the column with it, and a CSV polspec writes holds it the same way:

```python
class Bonds(FrameSpec):
    maturity = ColSpec(pl.Date, text_format="%d %b %Y")

Bonds.validate(pl.DataFrame({"maturity": ["30 Sep 2031"]}), parse_text=True)
```

*Added in 1.6.* `profile(parse_text=True)` finds the format itself; see
[Dates written another way](profiling.md#dates-written-another-way).

## In CI

### Gate on drift

`polspec drift` exits 1 when last night's data would fail the spec, and
`polspec diff` when a pull request's change to a spec would fail data that
passed before. `--markdown` writes a pull-request comment.

```bash
polspec drift specs/orders.yaml nightly/orders.parquet --fail-on breaking
polspec diff main/orders.yaml pr/orders.yaml --markdown --fail-on breaking > drift.md
```

See [`diff` and `drift`](cli.md#diff-and-drift-what-moved) and
[Schema and data drift](drift.md).

### Print a spec from data

```bash
polspec schema infer orders.parquet -o -
```

In Python, `print(polspec.profile(df).to_yaml())`. See
[Infer a spec from data](profiling.md).
