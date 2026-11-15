# A spec for data you already have

The other tutorials start from a spec you write. This one starts from files
you already have: three tables from a small shop. You profile them into
specs, with the keys between them found from the values; edit what the
profile got too exact; validate the files against the specs; watch a later
month for drift; and make a fake tenth of the data for your tests.

Download the three files into a folder called `shop`, and run everything
from the folder that holds it:

- [`customers.csv`](../data/shop/customers.csv): 150 customers, with an
  `id`, an `email`, a `country` and the date they `signed_up`.
- [`orders.csv`](../data/shop/orders.csv): 600 orders, each with a
  `customer_id`, a `ref` such as `ORD-100037`, the date it was `placed_on`
  and a `status`.
- [`lines.csv`](../data/shop/lines.csv): 1,500 order lines, each with an
  `order_id`, a `sku` such as `SKU-9881`, a `qty` and a `price`.

## Profile the files

One command profiles all three:

<!-- docs: run -->
```bash
polspec schema infer shop/customers.csv shop/orders.csv shop/lines.csv -o shop.yaml --weights
```

It prints a line saying it inferred 3 specs, with 2 foreign keys between them.
`shop.yaml` is a registry: one spec per file, named after it
(`Customers`, `Orders`, `Lines`). Each column has the claims its values
support:

- **Its dtype and extremes:** `qty` is an `Int64` from 1 to 5, and
  `placed_on` a `Date` within 2025.
- **Its values, where there are few:** `status` is an `Enum` of four values,
  and `--weights` records how often each occurs.
- **Its shape, where it has one:** `email` names the `email` format, and
  `ref` and `sku` get the pattern their values share, `^ORD-[0-9]{6}$` and
  `^SKU-[0-9]{4}$`.
- **Its key, where it is one:** every `Orders.customer_id` is a
  `Customers.id`, and its name says so, so it becomes a
  [foreign key](../how-to/registry.md). The same goes for `Lines.order_id`
  into `Orders.id`.

The same thing in Python is `profile_all`, over frames you have already read:

```python
shop = {
    name: pl.read_csv(f"shop/{name.lower()}.csv", try_parse_dates=True)
    for name in ("Customers", "Orders", "Lines")
}
inferred = polspec.profile_all(shop)
for spec in inferred.specs:
    for key in spec.foreign_keys:
        print(f"{spec.name}.{key.columns[0]} -> {key.references}.{key.ref_columns[0]}")
```

```text
Orders.customer_id -> Customers.id
Lines.order_id -> Orders.id
```

A key is inferred only where the evidence is strong. See
[how keys are found](../how-to/synthesizing.md#related-tables), and how to
declare one that was missed.

## Read the registry

`shop.yaml` is meant to be read, and committed beside the pipeline that
writes these files. Load it to look at a column:

```python
registry = Registry.from_yaml("shop.yaml")
print(registry["Orders"]["status"])
print(registry["Orders"]["ref"])
```

```text
ColSpec(Enum, choices={'NEW': 0.085, 'PAID': 0.27, 'RETURNED': 0.055, 'SHIPPED': 0.59})
ColSpec(String, string_length=(10, 10), pattern='^ORD-[0-9]{6}$')
```

## Edit what is too exact

A profile says what this data held, not what the data may hold. The
command line rounds bounds outward to two significant figures, but a few
claims still need a person to decide. Here, an order line may hold up to
20 of an item, and a price has no upper limit:

```python
lines = registry["Lines"]
lines = lines.with_columns({
    "qty": lines["qty"].replace(bounds=(1, 20)),
    "price": lines["price"].replace(bounds=(0.0, None)),
})
registry = Registry(*(lines if spec.name == "Lines" else spec for spec in registry.specs))
registry.to_yaml("shop.yaml")
```

Editing the YAML by hand does the same. `-o shop.py` writes the registry as
Python instead: a `FrameSpec` class per file, to edit and import like any
other.

## Validate the files

Validate each file against its spec, and the keys between them:

<!-- docs: run -->
```bash
polspec validate --all shop.yaml shop
```

It exits 1 when a file breaks its spec, so a pipeline can run it as a gate
on its own output. In Python, `registry.validate_all(shop)` does the same.

## Watch for drift

Validation asks whether the data breaks a claim. Drift asks whether it has
moved from what the spec recorded, while still passing. Say next month's
orders come back with a quarter of them returned:

```python
Path("next").mkdir()
for name in ("customers", "lines"):
    shop[name.title()].write_csv(f"next/{name}.csv")
returned = pl.when(pl.col("id") % 4 == 0).then(pl.lit("RETURNED")).otherwise("status")
shop["Orders"].with_columns(status=returned).write_csv("next/orders.csv")
```

<!-- docs: run -->
```bash
polspec drift --all shop.yaml next
polspec drift --all shop.yaml next --fail-on any   # exits 1
```

The first command reports that the frequencies of `status` moved, with
`RETURNED` at 29% where the spec recorded 5.5%, and exits 0: the change is
*compatible*, since every value is still one the spec allows. A *breaking*
change, which validation would reject, fails the run on its own.
`--fail-on any` fails on any change, for a CI job that should stop and ask.
`--markdown` writes the report as a pull-request comment.

## Make a tenth for your tests

Tests need data shaped like the real thing, without the real rows. Make a
fake set a tenth of the size, with every key held by a fake parent row:

<!-- docs: run -->
```bash
polspec synthesize shop/customers.csv shop/orders.csv shop/lines.csv -o fake --scale 0.1 --seed 1
polspec validate --all shop.yaml fake
```

`fake/` holds 15 customers, 60 orders and 150 lines, and validates against
the same registry as the real files. No source row is in it, and no email,
`ref` or `sku` either: those are drawn from their format or pattern. Values
of a column that became an `Enum`, such as `country`, are carried. Name a
column in `--replace` to keep its values out.

In Python, `synthesize_all` returns the frames:

```python
fake = polspec.synthesize_all(shop, scale=0.1, seed=1)
print({name: frame.height for name, frame in fake.items()})
registry.validate_all(fake)
```

```text
{'Customers': 15, 'Orders': 60, 'Lines': 150}
```

Commit `fake/` as test fixtures, or generate frames from `shop.yaml` in the
tests themselves, as [Testing pipelines](../how-to/testing.md) shows.

## Where next

- [Infer a spec from data](../how-to/profiling.md): every profiling option,
  and data read as text.
- [Fake data from real data](../how-to/synthesizing.md): how close the fake
  data is, and related tables.
- [Schema and data drift](../how-to/drift.md): what drift measures, and its
  thresholds.
- [Multiple specs](../how-to/registry.md): a registry, and the keys between
  them.
