# Known limitations

polspec implements each constraint twice — once to generate data and once to
validate it — and there is no structural guarantee the two agree. The gaps
below are the places they currently do not.

Each is pinned by a test in `tests/test_roundtrip.py` carrying
`xfail(strict=True)`. While the gap exists the test reports XFAIL and the suite
stays green; the moment it is fixed pytest turns the XPASS into a failure. So
this page cannot quietly go stale — closing a gap forces the test to be updated.

## Generation does not enforce these

### `unique=True` is validated but not generated

A `unique` column whose domain is smaller than `n` emits duplicates that its own
spec rejects.

```python
class A(FrameSpec):
    id = ColSpec(pl.Int8, unique=True)

A.validate(A.generate(200, seed=1))
# ValidationError: Column 'id': unique column contains 110 duplicate value(s)
```

**Work around it** by giving the column a domain comfortably larger than `n`, or
by validating with `validate_unique=False`.

### `__unique_together__` is validated but not generated

The composite sibling of the above, with the same workaround
(`validate_unique=False`).

### `__checks__` and `ColSpec.validators` are validation-only

This one is by design, not a defect: both wrap arbitrary Polars expressions,
and nothing can generate data satisfying an arbitrary predicate. Validate
generated data with `validate_checks=False` / `validate_validators=False`, or
construct the rows those invariants describe yourself.

## Interactions that do not round-trip

### Chained `ColRule`s

`_apply_rules` evaluates every `when` against the *pre-rule* frame, so rules on
different columns need no dependency ordering. Validation evaluates the same
`when` against the *final* frame. A rule keyed on a column that is itself
rule-targeted therefore fails its own validation.

```python
b = ColSpec(pl.Enum(["x", "y"]), rules=[ColRule(when={"column": "a", "equals": "x"}, choices=["y"])])
c = ColSpec(pl.Enum(["x", "y"]), rules=[ColRule(when={"column": "b", "equals": "y"}, choices=["x"])])
# rows where b == "y" but c != "x"
```

**Avoid** keying a rule on a column that carries rules of its own.

### Several foreign keys touching the same column

Every key's replacement is computed against the original frame and applied in
one pass, so a key referencing a column that another key rewrites samples
values that no longer exist.

**Avoid** chaining a self-referencing key onto a column that is itself
foreign-keyed.

### A foreign key against its column's own bounds

A key overwrites its column with parent values regardless of that column's
declared `bounds` or `choices`. Declaring `bounds=(1, 50)` on a column
referencing keys in `100..200` produces data that fails validation, and nothing
flags the contradiction when you declare it.

**Keep** the two declarations compatible.

### Textual foreign keys across `Enum` and `String`

A `String` key may reference an `Enum` key — the declaration-time check allows
it deliberately, and generation handles it. Validation anti-joins the columns
without casting, and Polars raises `SchemaError` rather than a
`ValidationError`.

**Work around it** by declaring both sides with the same dtype.

## Numeric precision

### Integer bounds beyond 2^53

Bounds cross into the Rust engine as `f64`, so `Int64` and `UInt64` bounds
above 2^53 round. Generation can overshoot a declared maximum by one, and
`UInt64` bounds near the dtype maximum collapse to a single repeated value.

This is also why the generation clamp for an unbounded distribution is capped
at 2^53: a limit that rounds outward would enforce nothing.

**Work around it** by keeping explicit bounds within ±2^53
(±9,007,199,254,740,992).

## Cartesian generation

### `n` is a minimum, not a count

Under `method="cartesian"`, if the coverage set is larger than `n` all of it is
kept. `generate_batches` and every `sink_*` inherit this, so asking for 5 rows
from two ten-category enums yields 100.

### `generate(0, method="cartesian")` is not empty

It emits the whole coverage set, unlike `generate(0)` and the sinks, which all
produce nothing.

## Smaller sharp edges

- **Unsupported dtypes are accepted at declaration.** `ColSpec(pl.List(...))`
  constructs and validates; only `generate()` objects.
- **`missing_cols="add"` can produce a frame that fails re-validation**, since
  columns are added after validation runs, including for non-nullable columns.
- **Rules overwrite nulls**, so a nullable column with a rule ends up with
  fewer nulls than `null_probability` suggests.
- **`generate()` silently ignores a foreign key it has no parent data for**,
  while `validate()` raises for the same key.
- **`CatSpec.to_dict()` drops choices not attached to a categorical**, so
  choices recorded for plain string columns do not survive a YAML round-trip.
- **Case-insensitive registry lookup** means a column named `status` binds to a
  registry entry named `STATUS`; entries differing only in case are ambiguous.
- **`to_mermaid` marks every `unique=True` column `PK`**, so several unique
  columns render as several primary keys.
