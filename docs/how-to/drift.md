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
never appear; have the null rate, the NaN share, the weights or the
distribution moved.

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
Drift: 2 breaking, 1 compatible, DataFrame against 'OrdersV1'
  - [breaking] Column 'total': values escape bounds [0.0, 1000.0]: max found 2999.36861541796 by 1999.36861541796 above. Widen the bounds, or fix the source
  - [breaking] Column 'placed': values escape bounds [2024-01-01, 2025-01-01]: max found 2025-07-20 by 200 days above. Widen the bounds, or fix the source
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

`status` arrives as a `String` where an `Enum` is declared -- the way a CSV
hands one back -- and that is no finding: the column validates as it is,
and data that validates has not drifted on dtype. `strict_dtypes=True`
reports it, as validation's own switch would.

### Significant, and large enough to matter

A null rate, a NaN share, a weighted column's frequencies and a declared
distribution are claims about how often, and a sample always wanders from
them a little. So each is reported as moved only when the move is both:

- **significant** -- unlikely, at `significance` (0.001), to be what sampling
  alone would do over the rows measured: an exact binomial test for a rate,
  Pearson's chi-square for frequencies, a two-sample Kolmogorov-Smirnov
  test for a distribution;
- **large** -- at least its tolerance: `null_rate_tolerance` for a rate,
  `frequency_tolerance` for frequencies (the total variation distance, half
  the summed gap between observed and declared shares), and
  `distribution_tolerance` for a distribution (the KS distance).

Fifty rows of a column declared 30% null hold 22% nulls often enough --
that used to be a finding, and is not now: a move that size over fifty rows
happens by chance more than one run in four. Ten million rows prove every
difference significant, and the tolerance is what keeps a rounding error
from being reported. Both directions are pinned: a spec's own seeded output
does not drift from it at fifty, five thousand or five million rows, and a
real move over enough rows always does.

A distribution is compared by drawing: `drift()` generates a sample from the
declaration itself -- with a fixed seed, so a report is the same every run
-- and measures the data's distance from it. So clamping at the bounds and
the rounding of an integer column are exactly what generation does, the
same way the profiler fits shapes. A declared value that never appears is
weighed the same way: `cardinality_moved` names it only when, at the share
it is generated at, missing it from the rows seen is itself unlikely -- a
category of weight 1% is absent from fifty rows six times in ten.

A column a pass rewrites -- one a rule narrows, a foreign key fills, a
hierarchy links, or a composite key repairs -- is not drawn from its own
weights or distribution, so neither is compared on it, and a declared value
it never holds is not reported: a rule may leave one out of every row.

Each column is tested at `significance` on its own. A frame of a hundred
columns, each tested once, expects a false alarm about one run in ten;
lower `significance` for a wide frame watched often.

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
| `significance` | `0.001` | the false-alarm rate of each statistical test: how unlikely a move must be, were the declaration right, before it is reported |
| `null_rate_tolerance` | `0.05` | the smallest move of a null rate, an element null rate or a NaN share worth reporting (`null_rate_moved`, `nan_rate_moved`); absolute, not relative |
| `frequency_tolerance` | `0.05` | the smallest move of a weighted column's frequencies worth reporting (`frequencies_moved`), as the total variation distance |
| `distribution_tolerance` | `0.05` | the smallest move of a declared distribution worth reporting (`distribution_moved`), as the Kolmogorov-Smirnov distance |
| `unseen_values` | `True` | report declared values the data never holds and, at their declared share, would not miss by chance (`cardinality_moved`) |
| `strict_dtypes` | `False` | the same switch as validation's, decided by the same function: a dtype that validates is no finding, and held strictly it is a breaking `dtype_changed` |
| `max_samples` | `10` | how many offending values a finding's `details` carry |

## The report

A `DriftReport` is data first. `report.breaking` and `report.compatible`
are the two halves; `by_column()` and `by_code()` slice it; `to_dict()` and
`to_json()` serialise it. Like a `ValidationReport` it is a collection of
findings, so `len(report)` counts them and an empty report is falsy:
`if report:` means something changed, and `report.unchanged` says the
opposite in words.

`to_markdown()` renders the shape of a pull-request comment, breaking
findings first:

```python
text = OrdersV1.diff(OrdersV2).to_markdown()
assert text.index("## Breaking") < text.index("## Compatible")
```

The finding codes are listed with the validation codes in
[Errors and findings](../reference/errors.md#drift-codes).
