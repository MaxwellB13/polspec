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

`ValidationError` is a `PolspecError` (and still a `ValueError`); see
[Errors](../reference/errors.md). `err.errors` is the list of
individual messages; `str(err)` is the same list formatted as a report, and
`err.report` is the `ValidationReport` behind both.

## Findings as data — `inspect()`

An exception is the right shape for someone reading a traceback. For code
that wants to *act* on what was found — quarantine the offending rows, count
problems per column, write a report — use `inspect()`, which returns the same
findings as a `ValidationReport` and never raises for a bad frame:

```python
suspect = df.with_columns(pl.col("total") * -1)   # every total now negative
report = Orders.inspect(suspect)

report.passed                 # False
for finding in report:
    finding.code              # "bounds", "check", "foreign_key", ...
    finding.key               # "total__bounds", "check:total_covers_subtotal",
                              # "point.lat__bounds" for a struct's field
    finding.columns           # ("total",)
    finding.count             # rows violating it (None for structural findings)
    finding.samples           # up to five offending values
    finding.details           # {"bounds": [0.0, None], "min_found": -3.0, ...}
    finding.message           # the same text validate() would have raised

report.by_column()["total"]   # every finding involving one column
report.by_code("foreign_key") # every finding of one kind
report.to_json()              # everything above, JSON-safe
```

The offending rows are reachable lazily, so nothing is materialised until you
ask:

```python
bad = report.by_code("bounds")[0]
report.rows(bad).collect()        # just the rows violating that one claim
report.failing_rows().collect()   # every violating row, with a `__polspec_finding`
                                  # column naming the claim (a row violating two
                                  # claims appears twice)
```

The column `failing_rows()` adds is named by `polspec.validation.FINDING_COLUMN`
rather than spelled out, so grouping by it does not hard-code the name:

```python
from polspec.validation import FINDING_COLUMN

quarantined = report.failing_rows().collect()
quarantined.group_by(FINDING_COLUMN).len()   # how many rows each claim caught
```

Structural findings (`extra_columns`, `missing_columns`, `dtype`,
`foreign_key_unresolved`) describe the frame's shape rather than its rows and
have no rows to return. `inspect()` takes exactly the options `validate()`
does; `validate()` is `inspect()` followed by `report.raise_if_failed()` and
the structural transformations below. The full list of codes is in
[Errors](../reference/errors.md#finding-codes), and `polspec validate` on the
[command line](cli.md#validate-check-data-against-a-schema) prints the same
report.

## Checking a file you were given

The usual way in: someone hands you a file, and there is a spec it should
meet. Read it loosely, let `inspect()` say everything that is wrong, decide
whether the file or the spec is at fault, and only then ask for the typed
frame.

```python
Path("customers.csv").write_text(          # the file you were handed
    "id,name,country,signed_up\n"
    "1,Ada Lovelace,UK,2021-03-04\n"
    "2,Grace Hopper,US,2019-12-31\n"
    "2,Al,FR,2022-01-01\n"
)
```

**1. Read it without forcing the spec's types.**

```python
given = Customers.read("customers.csv")
```

`read()` picks the reader from the extension (`.csv`, `.tsv`, `.parquet`,
`.ndjson`, `.json`, `.arrow` -- the set `polspec validate` reads, with any
Polars reader option passed through, such as `separator=";"`) and does one
thing in the spec's name. A CSV has no date type, so `signed_up` arrives as
text; `read()` parses each column the spec declares as a date or time, when
every value in it parses. Without that, the report would say only *expected
Date, got String*, and none of the column's own checks would run. The other
gaps between a CSV and a spec take care of themselves -- a `String` column
is accepted where an `Enum` or `Categorical` is declared, an integer where a
float or `Decimal` is, and the values are checked either way.

In plain Polars, `pl.read_csv("customers.csv", try_parse_dates=True)` comes
close; the difference is that Polars parses every column that looks like a
date, where `read()` parses only the ones the spec says are dates.

Reading with the spec's schema is the tempting alternative, and the wrong
first step: Polars stops at the first value that does not fit, so one
error replaces the whole report.

<!-- docs: raises -->
```python
pl.read_csv("customers.csv", schema_overrides=Customers.schema())
# ComputeError: could not parse `FR` as dtype `enum` at column 'country'
```

**2. Ask what is wrong -- all of it.**

```python
report = Customers.inspect(given)
print(report)
```

```
Validation failed for DataFrame against 'Customers' (4 error(s) found):
  - Column 'id': unique column contains 2 duplicate value(s). Duplicate samples: [2]
  - Column 'name': found 1 value(s) with string length outside [3, 20]. Invalid samples: ['Al']
  - Column 'country': found 1 invalid value(s) not in allowed choices/categories ['UK', 'US', 'DE']. Invalid samples: ['FR']
  - Column 'signed_up': found 1 value(s) out of bounds [2020-01-01, 2026-01-01] (min found: 2019-12-31, max found: 2022-01-01). Out of bounds samples: [datetime.date(2019, 12, 31)]
```

`report.rows(finding)` is the offending rows, and `report.to_json()` is
something to send back to whoever sent the file:

```python
report.rows(report.by_code("choices")[0]).collect()   # the row with country "FR"
```

**3. Decide which is wrong: the file, or the spec.** A bad row is fixed or
filtered at the source. A spec that has fallen behind -- `FR` is a real
country now -- is changed, and [`diff`](drift.md#two-declarations) says
whether the change is breaking for anything already validated against it:

```python
Widened = Customers.spec.with_columns(country=ColSpec(pl.Enum(["UK", "US", "DE", "FR"])))
Customers.diff(Widened).breaking      # () -- widening a domain breaks nothing
```

While you are still finding out what the ranges really are,
`validate_bounds=False` checks everything but the bounds, and
[`from_dataframe`](../tutorial/getting-started.md#infer-a-spec-instead-of-writing-one)
describes what the file actually holds, to compare against what the spec
says it should.

**4. Then take the typed frame.** Once it passes, `cast=True` returns each
column as its declared dtype -- the `Enum` an `Enum`, not the `String` the
CSV held -- so the code downstream reads the types the spec promises:

```python
corrected = given.filter(pl.col("name") == "Ada Lovelace")   # the file, fixed
clean = Customers.validate(corrected, cast=True)
clean.schema["country"]       # Enum(categories=['UK', 'US', 'DE'])
```

From the shell, `polspec validate customers.py customers.csv` does steps 1
and 2 in one go, parsing the columns the spec declares as dates and times;
see [the CLI](cli.md#validate-check-data-against-a-schema).

!!! note "What a CSV cannot hold"

    A CSV has no way to write a `Duration`, a `List` or a `Struct`, so a
    spec with one of those cannot be met by a CSV however it is read.
    Parquet or Arrow IPC keeps every dtype.

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
    validate_hierarchy=True,
    validate_pattern=True,
    validate_bounds=True,
)
```

Every one of these is a field of `ValidationOptions`, which is what a report
carries as `report.options` — so a report says what it was asked to check, not
only what it found:

```python
report = Orders.inspect(df, validate_checks=False)
report.options.checks        # False
report.options.extra_cols    # "raise"
```

The `validate_*` switches are named for what they switch, so
`validate_checks` is `options.checks`. An option name polspec does not accept
is a `TypeError` naming the closest one it does.

You can also pass the whole set as one value, which is the shape to reach for
when the same settings go through several calls:

```python
from polspec import ValidationOptions

lenient = ValidationOptions(extra_cols="drop", checks=False)

for frame in (df, df.head(10)):
    Orders.validate(frame, options=lenient)
```

`options=` and the individual keywords are alternatives, not a base and an
override — passing both raises rather than quietly picking one.

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

Each `validate_*` switch disables a whole category of check. Most exist for
what generation cannot satisfy yet -- `__checks__`, validators and `pattern`
are validated but not generated:

```python
Orders.validate(Orders.generate(1_000, seed=1), validate_checks=False)
```

`validate_bounds=False` is the other way round: generation always stays in
bounds, so it is for real data -- a file whose ranges you want to look at
before holding it to them, while every other claim is still checked. It
covers every `bounds`, including a `List`'s elements and a struct's fields;
`string_length` and `list_length` have codes of their own and stay on.

```python
Orders.validate(df, validate_bounds=False)
```

To loosen one column rather than all of them, validate against a spec with
that column's bounds removed:

```python
import dataclasses
from polspec import validate

loose = Orders.spec.with_columns(
    total=dataclasses.replace(Orders.col("total"), bounds=None)
)
validate(loose, df)
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
| List has the declared number of elements | `list_length` |
| Value has the declared shape | `format` |
| Value matches the regex | `pattern` |
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

Without it, the key is reported as a `foreign_key_unresolved` finding naming
the spec it needed, so `validate()` raises and `inspect()` lists it alongside
everything else. `references` may be keyed by the class, its `TableSpec`, or
the spec's name. Self-referencing keys are checked against the frame itself
and need nothing.

Each foreign key is an anti-join against the parent, so these run separately
from the single-pass aggregation above.
