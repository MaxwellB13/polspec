# Schema and data drift

`validate()` says whether data meets its spec. Drift says *what moved* --
between two versions of a declaration, or between a declaration and the
data it is supposed to describe -- and whether each move would break
validation.

<!-- docs: skip -->
```python
from polspec.drift import diff, drift

report = diff(OldOrders, NewOrders)   # two declarations
report = drift(Orders, df)            # a declaration and a frame
```

Both are also methods on a spec: `OldOrders.diff(NewOrders)` and
`Orders.drift(df)`. Either way the result is a `DriftReport`.

## One rule for severity

Every finding is `breaking` or `compatible`, and the line between them is
mechanical: a finding is **breaking when a frame that satisfied the old side
could fail the new one**. For `drift()`, that means this frame fails this
spec on that column. Nothing else is a judgement call:

| Change | Severity | Because |
|:--|:--|:--|
| a bound or domain narrowed | breaking | values that validated before now fail |
| a bound or domain widened | compatible | nothing that passed now fails |
| a column added | breaking | a frame without it fails `missing_cols="raise"` |
| a column removed | breaking | a frame carrying it fails `extra_cols="raise"` |
| a constraint added (`unique`, a validator, a check, a key) | breaking | rows that passed may fail |
| a constraint removed | compatible | |
| `nullable` turned off | breaking | nulls that validated before now fail |
| a null rate that moved within a nullable column | compatible | the data still validates; the declaration describes it less well |

A column added is breaking under this rule, which surprises people in a pull
request. It is the honest answer -- validation of the old data would fail --
and the CLI's `--fail-on` is how a team decides what to gate on.

## Two declarations

```python
import datetime as dt

class OrdersV1(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, None), unique=True)
    status   = ColSpec(pl.Enum(["NEW", "PAID"]))
    total    = ColSpec(pl.Float64, bounds=(0.0, 1_000.0))
    placed   = ColSpec(pl.Date, bounds=(dt.date(2024, 1, 1), dt.date(2025, 1, 1)))

class OrdersV2(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, None), unique=True)
    status   = ColSpec(pl.Enum(["NEW", "PAID", "SHIPPED"]))   # widened
    total    = ColSpec(pl.Float64, bounds=(0.0, 500.0))         # narrowed
    placed   = ColSpec(pl.Date)                                 # widened
    channel  = ColSpec(pl.Enum(["web", "store"]))               # added

report = OrdersV1.diff(OrdersV2)
print(report)
```

```
Drift: 2 breaking, 3 compatible, 'OrdersV1' and 'OrdersV2'
  - [breaking] Column 'channel' added; a frame without it fails missing_cols='raise'
  - [breaking] Column 'total': domain narrowed from bounds [0.0, 1000.0] to bounds [0.0, 500.0]; values that validated before now fail
  - [compatible] Column 'status': dtype changed from Enum(categories=['NEW', 'PAID']) to Enum(categories=['NEW', 'PAID', 'SHIPPED'])
  - [compatible] Column 'status': domain widened from one of ['NEW', 'PAID'] to one of ['NEW', 'PAID', 'SHIPPED']; values that failed before are now accepted
  - [compatible] Column 'placed': domain widened from bounds [2024-01-01, 2025-01-01] to any Date; values that failed before are now accepted
```

Widened and narrowed are decided by the same `Domain` comparison a foreign
key uses at declaration, run in both directions. A change that is neither --
`choices=["A", "B"]` to `["B", "C"]`, or `format="email"` to `"uuid4"` -- is
`domain_changed`, and breaking.

A rename is not guessed from similar names. Say it, and it is reported as
one `column_renamed` finding instead of a column removed and another added:

```python
class Renamed(FrameSpec):
    order_ref = ColSpec(pl.Int64, bounds=(1, None), unique=True)
    status    = ColSpec(pl.Enum(["NEW", "PAID"]))
    total     = ColSpec(pl.Float64, bounds=(0.0, 1_000.0))
    placed    = ColSpec(pl.Date, bounds=(dt.date(2024, 1, 1), dt.date(2025, 1, 1)))

assert [f.code for f in OrdersV1.diff(Renamed, renames={"order_id": "order_ref"})] == [
    "column_renamed"
]
```

## A declaration and data

`drift()` asks the frame the questions the spec makes answerable: is
anything outside the declared bounds, and by how much; which values are
outside the declared `choices`, `Enum` or `format`; which declared values
never appear; has the null rate moved.

```python
good = OrdersV1.generate(2_000, seed=1)
assert OrdersV1.drift(good).unchanged

moved = good.with_columns(
    total=pl.col("total") * 3,
    placed=pl.col("placed") + pl.duration(days=200),
    status=pl.lit("PAID").cast(pl.String),
)
print(OrdersV1.drift(moved))
```

```
Drift: 2 breaking, 2 compatible, DataFrame against 'OrdersV1'
  - [breaking] Column 'total': values escape bounds [0.0, 1000.0]: max found 2999.36861541796 by 1999.36861541796 above. Widen the bounds, or fix the source
  - [breaking] Column 'placed': values escape bounds [2024-01-01, 2025-01-01]: max found 2025-07-20 by 200 days above. Widen the bounds, or fix the source
  - [compatible] Column 'status': holds String, declared Enum(categories=['NEW', 'PAID'])
  - [compatible] Column 'status': 1 of 2 declared value(s) never appear: ['NEW']
```

The generated frame drifts by nothing, and that is pinned by a test: what
`generate()` produces never drifts breakingly from the spec that produced
it, the same round trip validation is held to. The other direction is pinned
too -- every breaking finding is a column `validate()` would report -- so
`breaking` never means more than "validation fails here".

A struct column is compared field by field, in both directions: each
field its `fields` describes -- or its dtype alone, where `fields` says
nothing -- goes through every comparison a column does, and a finding names
it by path. Describing `point.lat` with bounds is `domain_narrowed` on
`point.lat`, breaking for the same reason it is on a column; data whose
`lat` escapes those bounds is `bounds_exceeded` on `point.lat`. A field's
null rate is measured inside the structs that are present, which is what
its `null_probability` claims. The finding's `columns` stay `("point",)`.

What `drift()` does **not** measure is what validation already does:
uniqueness, composite keys, foreign keys and checks are pass/fail claims
about rows, not summaries that move. `validate()` is still the verdict.

### Options

```python
from polspec import DriftOptions

lenient = DriftOptions(null_rate_tolerance=0.2, unseen_values=False)
OrdersV1.drift(good, options=lenient)
OrdersV1.drift(good, null_rate_tolerance=0.2)   # or one keyword at a time, not both
```

| Option | Default | Meaning |
|:--|:--|:--|
| `null_rate_tolerance` | `0.05` | how far the observed null rate may sit from `null_probability` before `null_rate_moved` is reported; absolute, not relative |
| `unseen_values` | `True` | report declared values the data never holds (`cardinality_moved`) |
| `strict_dtypes` | `False` | the same switch as validation's, decided by the same function: whether a `dtype_changed` is breaking |
| `max_samples` | `10` | how many offending values a finding's `details` carry |

## The report

A `DriftReport` is data first. `report.breaking` and `report.compatible`
are the two halves; `by_column()` and `by_code()` slice it; `to_dict()` and
`to_json()` serialise it; `bool(report)` is `report.unchanged`, the way
`bool(ValidationReport)` is `passed`.

`to_markdown()` renders the shape of a pull-request comment, breaking
findings first:

```python
text = OrdersV1.diff(OrdersV2).to_markdown()
assert text.index("## Breaking") < text.index("## Compatible")
```

The finding codes are listed with the validation codes in
[Errors and findings](../reference/errors.md#drift-codes).
