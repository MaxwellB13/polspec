# Validating data

```python
clean = Orders.validate(df)
```

`validate()` accepts a `DataFrame` or a `LazyFrame` and returns the same kind,
so it drops into a pipeline. On success the returned frame has its declared
columns first, in declaration order.

## Collecting every problem at once

All checks across all columns are compiled into one Polars aggregation and
evaluated in a single scan. Every breach is gathered before anything is raised:

```python
from polspec import ValidationError

try:
    Orders.validate(df)
except ValidationError as err:
    print(len(err.errors), "problems")
    for problem in err.errors:
        print(problem)
```

`ValidationError` subclasses `ValueError`. `err.errors` is the list of
individual messages; `str(err)` is the same list formatted as a report.

## Options

```python
Orders.validate(
    df,
    extra_cols="raise",        # "raise" | "drop" | "allow"
    missing_cols="raise",      # "raise" | "add" | "allow"
    strict_dtypes=False,
    cast=False,
    streaming=False,
    references=None,
    validate_rules=True,
    validate_validators=True,
    validate_unique=True,
    validate_checks=True,
    validate_foreign_keys=True,
)
```

### Structural mismatches

`extra_cols` decides what happens to columns the spec does not declare —
refuse, drop them from the result, or keep them (appended after the declared
ones).

`missing_cols` decides what happens to declared columns the frame lacks —
refuse, add them as all-null, or ignore them.

!!! warning "`missing_cols="add"` can produce a frame that fails re-validation"

    Columns are added *after* validation has run, including for columns
    declared `nullable=False`. Feed the result straight back into `validate()`
    and it will object to the nulls it just inserted.

### Dtype strictness

By default polspec accepts what a real pipeline delivers: any integer width for
a declared integer, an integer or float for a declared float, any temporal for
a temporal, and `String`/`Categorical` for a declared `Enum`. `strict_dtypes=True`
requires the exact dtype, treating only `String` and `Utf8` as interchangeable.

### Casting

`cast=True` casts each column to its declared dtype *after* validation passes,
so a String column that holds only valid enum members comes back as the `Enum`.

### Streaming

`streaming=True` evaluates with the Polars streaming engine, for frames larger
than memory.

### Turning checks off

The five `validate_*` flags disable whole categories. Useful when generation
cannot satisfy something yet:

```python
Orders.validate(Orders.generate(1_000, seed=1), validate_checks=False)
```

## What gets checked

| Check | From |
|:--|:--|
| Column present / not extra | the spec's column set |
| Dtype compatible | `ColSpec.dtype` |
| No unexpected nulls | `nullable` |
| Value in domain | `choices`, `Enum` categories |
| Value within range | `bounds` |
| Length within range | `string_length` |
| Conditional values hold | `rules` |
| Single-column predicates | `validators` |
| Values distinct | `unique` |
| Composite key distinct | `__unique_together__` |
| Multi-column invariants | `__checks__` |
| Referential integrity | `__foreign_keys__` |

Bounds, lengths, rules, validators and uniqueness are skipped for a column
whose dtype is already wrong — comparing values of the wrong type would bury
the dtype error under noise.

## Foreign keys need their parent

A key referencing another spec needs that spec's data:

```python
Orders.validate(orders, references={Customers: customers})
```

Without it, `validate()` raises `ValueError` naming the key and the missing
spec. Self-referencing keys are checked against the frame itself and need
nothing.

Each foreign key is an anti-join against the parent, so these run separately
from the single-pass aggregation above.
