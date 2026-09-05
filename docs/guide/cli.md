# Command line

`polspec` has two things to do with a schema: create one, and turn one into a
test. Both are thin wrappers over `FrameSpec` methods that already exist —
`from_dataframe`, `to_yaml`, `generate`, `validate` — so the CLI is argument
parsing and templating, not new behaviour.

```bash
polspec schema infer orders.parquet -o orders.yaml
polspec schema new Orders -o orders.py
polspec test orders.yaml -o test_orders.py
```

## `schema infer` — profile data into a spec

```bash
polspec schema infer SOURCE -o OUTPUT.yaml [options]
polspec schema infer SOURCE -o OUTPUT.py [options]
```

`SOURCE` is a `.csv`, `.tsv`, `.parquet`, `.ndjson`/`.jsonl`, `.json`, or Arrow
IPC (`.arrow`/`.ipc`/`.feather`) file. It is read with the matching Polars
reader and profiled with `FrameSpec.from_dataframe`, the same function behind
[Getting started](../getting-started.md#infer-a-spec-instead-of-writing-one).

`OUTPUT`'s extension picks the format: `.yaml`/`.yml` writes a YAML spec via
`FrameSpec.to_yaml`; `.py` writes a `FrameSpec` subclass via
`FrameSpec.to_python` — a starting-point module you can edit like any other
source file, rather than a data file `from_yaml` re-parses.

```console
$ polspec schema infer orders.parquet -o orders.yaml --weights
Inferred 3 column(s) from 12,483 row(s) of orders.parquet -> orders.yaml

$ polspec schema infer orders.parquet -o orders.py --weights
Inferred 3 column(s) from 12,483 row(s) of orders.parquet -> orders.py
```

```python
"""Declares the Orders schema."""

import polars as pl
from polspec import ColSpec, FrameSpec


class Orders(FrameSpec):
    __columns__ = {
        "order_id": ColSpec(pl.Int64, bounds=(1, 12483)),
        "status": ColSpec(pl.Enum(["NEW", "PAID", "SHIPPED"]), weights=[0.4, 0.3, 0.3]),
        "total": ColSpec(pl.Float64, bounds=(10.0, 500.0)),
    }
```

Columns are declared through `__columns__` rather than as class attributes,
same as `from_yaml` — see [Column names that are not
identifiers](columns.md#column-names-that-are-not-identifiers). The `.py`
output is passed through `ruff format` when it's on `PATH`, same as `schema
new`.

| Option | Effect |
|:--|:--|
| `--name NAME` | Class name (default: derived from the file name) |
| `--weights` | Record each category's observed frequency |
| `--max-unique-enum N` | Max distinct values for a string column to become an `Enum` (default 50) |
| `--no-bounds` | Skip computing numeric/temporal bounds and string lengths |
| `--sample N` | Profile only the first N rows |

Treat the output as a draft. It describes the sample it saw — edit bounds,
add rules, tighten a domain — before trusting it as a contract.

## `schema new` — start from nothing

```bash
polspec schema new NAME -o OUTPUT.py
```

Writes a blank `FrameSpec` with the two imports it will need and a few
commented `ColSpec` examples, for the case where there's no data yet to
profile.

## `test` — a round-trip test from a schema

```bash
polspec test SOURCE -o OUTPUT_test.py [options]
```

`SOURCE` is a `.yaml`/`.yml` spec (from `schema infer`, or written by
`FrameSpec.to_yaml`) or a `.py` file defining one or more `FrameSpec`
subclasses (from `schema new`, filled in). The generated file asserts the
property this project's own test suite is built around:

```python
def test_orders_roundtrip():
    df = Orders.generate(500, seed=42)
    Orders.validate(df)


def test_orders_cartesian_coverage():
    df = Orders.generate(500, method="cartesian", seed=42)
    Orders.validate(df)
```

| Option | Effect |
|:--|:--|
| `--rows N` | Rows to generate (default 500) |
| `--seed N` | Generation seed (default 42) |
| `--no-cartesian` | Skip the coverage-guaranteeing test |
| `--class NAME` | Generate a test for only this class, when the source defines several |

### It will not hand you a test that fails on the spot

`generate()` does not attempt everything `validate()` checks — see
[Known limitations](../reference/limitations.md). A spec using `unique=True`,
`__unique_together__`, `__checks__` or `ColSpec.validators` would otherwise
generate a test that fails the moment it runs. The generator checks for each
and disables the corresponding `validate()` flag, with a comment explaining
why:

```python
def test_narrow_roundtrip():
    # unique=True / __unique_together__ is validated but not yet generated
    # (see docs/reference/limitations.md)
    df = Narrow.generate(500, seed=42)
    Narrow.validate(df, validate_unique=False)
```

A spec with a foreign key referencing *another* spec needs that spec's data
via `references=`, which the CLI cannot supply on its own — that test is
marked `@pytest.mark.skip` with a reason, rather than guessed at:

```python
@pytest.mark.skip(
    reason=(
        "Child has foreign key(s) 'fk_parent_id__Parent' referencing another "
        "FrameSpec. generate()/validate() need a parent DataFrame via "
        "references={OtherSpec: parent_df} -- see "
        "docs/guide/constraints.md#referential-integrity-foreignkey."
    )
)
def test_child_roundtrip():
    pass
```

Similarly, the cartesian test is only emitted when the spec actually has
something for `method="cartesian"` to build coverage from — an `Enum`,
`Boolean`, or bounded numeric column. A spec of only unbounded strings gets a
comment instead of a test that would raise `ValueError` on the first run.

### Regenerating

The generated file names the command that made it:

```python
"""Generated by `polspec test orders.yaml`.

Regenerate with:

    polspec test orders.yaml -o test_orders.py

This file is only overwritten by running that command again -- edit freely.
"""
```

It is a plain file, not managed state — add assertions, rename the functions,
delete the parts you don't want. Nothing re-reads it.

## `validate` — check data against a schema

```bash
polspec validate orders.yaml orders.parquet
polspec validate specs.py orders.parquet --class Orders --references Customers=customers.parquet
polspec validate orders.yaml orders.csv --json > report.json
```

Reads a data file (CSV, Parquet, NDJSON or Arrow IPC), runs
[`inspect()`](validating.md#findings-as-data-inspect) against the spec, and
prints the report: the same text `validate()` would raise, or the full
structured report with `--json`. The exit status is 0 when the data passes
and 1 when it does not, so a spec can gate a pipeline step in CI with no
Python at all.

`--references NAME=PATH` supplies parent data for a foreign key to another
spec, by that spec's name; repeat it for several. `--allow-extra` and
`--allow-missing` relax the structural checks; `--strict-dtypes` tightens the
dtype check.

## Exit codes and errors

Every subcommand returns `0` on success and `1` on a reported error, printed
as `error: ...` on stderr rather than a traceback — a missing file, an
unreadable format, an invalid class name.

```console
$ polspec schema infer nope.csv -o out.yaml
error: no such file: nope.csv
```
