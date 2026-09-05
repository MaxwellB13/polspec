# Errors

Everything polspec raises on its own behalf derives from one base class, so
a caller can separate "polspec objected" from "something else went wrong"
with a single clause:

```python
from polspec import PolspecError

try:
    Orders.validate(df)
except PolspecError as exc:
    log.warning("rejected: %s", exc)
```

| Exception | Raised when | Also a |
|:--|:--|:--|
| `PolspecError` | Base class; never raised directly | `Exception` |
| `SpecError` | A declaration cannot mean anything: bounds a dtype cannot hold, a rule naming a column that does not exist, two attributes resolving to one column name | `ValueError`, `TypeError` |
| `ValidationError` | Data does not meet its spec. `err.report` is the `ValidationReport`; `err.errors` lists its messages | `ValueError` |
| `GenerationError` | A spec that declares fine cannot be turned into data as asked: no column for `method="cartesian"` to cover, a coverage set past the size cap, a foreign key with an empty parent. Errors from the Rust engine surface as this | `ValueError` |
| `SerializationError` | A spec file cannot be written or read: a dtype with no file representation, an unrecognised dtype name, a category reference the registry does not hold | `ValueError` |
| `RegistryError` | A collection of specs is inconsistent (reserved for the spec registry) | `LookupError` |

Each subclass keeps the built-in type it replaced, so `except ValueError`
written against an earlier version still catches it.

Ordinary argument misuse is not a `PolspecError`. A negative row count, an
unknown `method=`, a `batch_size` of zero, or the wrong object passed where a
DataFrame was expected raise the plain `ValueError` or `TypeError` any Python
API would.

The command line prints a `PolspecError` as a one-line `error: ...` and exits
with status 1; anything else is a bug and keeps its traceback.

## Finding codes

Every violation `inspect()` reports, and `validate()` raises, is a `Finding`
with one of these codes. Row-level findings can return the offending rows
through `report.rows(finding)`; structural ones describe the frame's shape.

| Code | Kind | Raised when |
|:--|:--|:--|
| `extra_columns` | structural | the frame has columns the spec does not declare (`extra_cols="raise"`) |
| `missing_columns` | structural | the frame lacks declared columns (`missing_cols="raise"`) |
| `dtype` | structural | a column's dtype is not compatible with its declaration |
| `foreign_key_unresolved` | structural | a key references another spec and `references=` had no entry for it |
| `nullability` | row-level | a non-nullable column holds nulls |
| `choices` | row-level | a value is outside `choices` or the `Enum` categories |
| `bounds` | row-level | a value is outside `bounds`; `details` carry the extremes found |
| `string_length` | row-level | a string or binary value's length is outside `string_length` |
| `rule` | row-level | a row matched a `ColRule` but holds a value outside its choices |
| `validator` | row-level | a `ColSpec.validators` predicate is false |
| `unique` | row-level | a `unique=True` column holds duplicates |
| `unique_together` | row-level | a composite key holds duplicate combinations |
| `check` | row-level | a `__checks__` predicate is false |
| `foreign_key` | row-level | a key value has no matching parent row (also structural when the parent lacks the referenced columns) |
