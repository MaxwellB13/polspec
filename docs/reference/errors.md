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
| `ValidationError` | Data does not meet its spec. `err.errors` lists every violation found | `ValueError` |
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
