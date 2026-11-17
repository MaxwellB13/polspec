# Command line

Every `polspec` command and argument, generated from the command's own
parser — `polspec <command> --help` prints the same. [Command line](../how-to/cli.md)
shows the workflows, and what the exit codes mean.

Generate a schema from data, a test from a schema, or check data against a schema.

```
polspec [-h] [--version]
               {schema,test,validate,generate,synthesize,diff,drift} ...
```

| Argument | Meaning |
|:--|:--|
| `--version` | show program's version number and exit |

## `polspec schema`

Create or infer a FrameSpec.

## `polspec schema infer`

Profile a data file into a YAML schema.

```
polspec schema infer [-h] -o OUTPUT [--name NAME] [--weights]
                            [--max-unique-enum N] [--no-formats] [--no-patterns]
                            [--parse-text] [--no-bounds] [--sample N] [--shape]
                            [--keys] [--order] [--round-bounds STEP]
                            [--round-rates STEP] [--replace COL [COL ...]]
                            [--foreign-key SPEC.COLUMN=SPEC.COLUMN]
                            [--no-foreign-keys]
                            source [source ...]
```

| Argument | Meaning |
|:--|:--|
| `source` | Path to a CSV, Parquet, NDJSON or IPC file; several are related tables, profiled into a registry with the foreign keys between them |
| `-o`, `--output` `OUTPUT` | **Required.** File to write: .yaml/.yml, or .py for a FrameSpec subclass (one per source, and a Registry, with several); '-' prints the YAML instead |
| `--name` `NAME` | Spec class name (default: derived from source) |
| `--weights` | Record each category's observed frequency |
| `--max-unique-enum` `N` | Max distinct values for a text column to keep its values, as an Enum, when they repeat (default: 50) |
| `--no-formats` | Do not name the format a text column's values have (email, uuid4, ...) |
| `--no-patterns` | Do not write the pattern a code-like text column's values share (ORD-[0-9]{6}, ...) |
| `--parse-text` | Read a text column as the number, flag or date every value holds (Boolean, Int64, Float64, Date, Datetime) |
| `--no-bounds` | Skip computing numeric/temporal bounds and string lengths |
| `--sample` `N` | Profile a random sample of N rows (seeded, so the same every run) |
| `--shape` | Fit the distribution each numeric or temporal column follows |
| `--keys` | Declare an all-distinct integer or text column unique |
| `--order` | Declare order_by when the rows come sorted by a column |
| `--round-bounds` `STEP` | Round each numeric column's bounds outward to STEP (1000, 1k, 10k, 1M, 0.01), within its dtype, so the data stays inside; 'auto' keeps two significant figures of the span and whole seconds; 'exact' writes them as observed (default: auto) |
| `--round-rates` `STEP` | Round each null and NaN rate up to STEP (0.01), and each --weights frequency to its nearest STEP; 'auto' keeps two significant figures; 'exact' writes them as observed. A rate of 0 stays 0 (default: auto) |
| `--replace` `COL` | Columns whose values must not be carried over: text in them is described by its lengths, never by the values the source holds |
| `--foreign-key` `SPEC.COLUMN=SPEC.COLUMN` | With several sources, declare a foreign key, child then parent, each under its file's spec name: Orders.buyer=Customers.id |
| `--no-foreign-keys` | With several sources, find no foreign keys between them |

## `polspec schema new`

Write a blank FrameSpec to edit by hand.

```
polspec schema new [-h] -o OUTPUT name
```

| Argument | Meaning |
|:--|:--|
| `name` | Spec class name, e.g. Orders |
| `-o`, `--output` `OUTPUT` | **Required.** Python file to write |

## `polspec test`

Generate a pytest round-trip test from a schema.

```
polspec test [-h] -o OUTPUT [--rows N] [--seed SEED] [--no-cartesian]
                    [--class NAME]
                    source
```

| Argument | Meaning |
|:--|:--|
| `source` | A .yaml/.yml spec, or a .py file defining one |
| `-o`, `--output` `OUTPUT` | **Required.** Test file to write |
| `--rows` `N` | Rows to generate (default: 500) |
| `--seed` `SEED` | Pin every frame to this seed (default: none -- each test's own, from the polspec_frames fixture and --polspec-seed) |
| `--no-cartesian` | Skip the coverage-guaranteeing cartesian test |
| `--class` `NAME` | Generate a test for only this class (a .py source may define several) |

## `polspec validate`

Check a data file against a schema.

```
polspec validate [-h] [--all] [--references NAME=PATH] [--class NAME]
                        [--allow-extra] [--allow-missing] [--strict-dtypes]
                        [--fill-defaults] [--skip CHECK] [--json] [--output PATH]
                        [--failing PATH]
                        spec data
```

| Argument | Meaning |
|:--|:--|
| `spec` | A .yaml/.yml spec, or a .py file defining one; with --all, a directory |
| `data` | Path to a CSV, Parquet, NDJSON or IPC file; with --all, a directory of files named after the specs |
| `--all` | Every spec found under SPEC against DATA/<name>.<suffix>, each seeing the others as parents |
| `--references` `NAME=PATH` | Parent data for a foreign key to another spec, as the spec's name and a data file; repeat for several |
| `--class` `NAME` | Validate against only this class (a .py source may define several) |
| `--allow-extra` | Do not report columns the spec does not declare |
| `--allow-missing` | Do not report declared columns the data lacks |
| `--strict-dtypes` | Require exact dtypes rather than compatible ones |
| `--fill-defaults` | Fill each defaulted column's nulls with its default, and add it when absent, before checking; --output writes the filled rows |
| `--skip` `CHECK` | Turn off one kind of check, as CHECK=False does in Python; repeat for several. One of `rules`, `derived`, `validators`, `unique`, `checks`, `foreign_keys`, `hierarchy`, `order`, `rows`, `pattern`, `bounds`. |
| `--json` | Print the report as JSON instead of text |
| `--output` `PATH` | Write the rows that passed, typed as the spec declares, to PATH (the format by its extension) |
| `--failing` `PATH` | Write the rows that failed, with the finding each broke, to PATH |

## `polspec generate`

Generate rows from a schema into a data file.

```
polspec generate [-h] [--all] [--format EXT] -n N -o OUTPUT [--seed SEED]
                        [--method {random,cartesian}] [--references NAME=PATH]
                        [--fixed COLUMN=VALUE] [--class NAME]
                        spec
```

| Argument | Meaning |
|:--|:--|
| `spec` | A .yaml/.yml spec, or a .py file defining one; with --all, a directory |
| `--all` | Every spec found under SPEC, parents first, to -o/--output as a directory of <name> files in --format |
| `--format` `EXT` | With --all, the file format to write (default: parquet) |
| `-n`, `--rows` `N` | **Required.** Rows to generate |
| `-o`, `--output` `OUTPUT` | **Required.** File to write; the extension picks the format (.parquet, .csv, ...) |
| `--seed` `SEED` | Generation seed (default: random) |
| `--method` `METHOD` | Sampling method (default: random) One of `random`, `cartesian`. |
| `--references` `NAME=PATH` | Parent data for a foreign key to another spec; repeat for several |
| `--fixed` `COLUMN=VALUE` | Pin a column to one value in every row, read as its dtype; repeat for several (with --all, SPEC.COLUMN=VALUE) |
| `--class` `NAME` | Generate from only this class (a .py source may define several) |

## `polspec synthesize`

Write a fake data file that looks like a real one.

```
polspec synthesize [-h] -o OUTPUT [-n N] [--scale X] [--seed SEED]
                          [--format EXT] [--foreign-key SPEC.COLUMN=SPEC.COLUMN]
                          [--no-foreign-keys] [--max-unique-enum N] [--no-formats]
                          [--no-patterns] [--parse-text] [--sample N]
                          [--replace COL [COL ...]] [--spec PATH] [--name NAME]
                          source [source ...]
```

| Argument | Meaning |
|:--|:--|
| `source` | Path to a CSV, TSV, Parquet, NDJSON, JSON or IPC file; several are related tables, synthesized with the foreign keys between them |
| `-o`, `--output` `OUTPUT` | **Required.** File to write, whose extension picks the format (.parquet, .csv, ...); with several sources, a directory to write one file each in |
| `-n`, `--rows` `N` | Rows to generate (default: as many as the source has) |
| `--scale` `X` | Rows as a multiple of each source's: 0.1 for a tenth, keeping the ratios between related tables |
| `--seed` `SEED` | Generation seed (default: random) |
| `--format` `EXT` | With several sources, write each as EXT (parquet, csv, ...) (default: its own) |
| `--foreign-key` `SPEC.COLUMN=SPEC.COLUMN` | With several sources, declare a foreign key, child then parent, each under its file's spec name: Orders.buyer=Customers.id |
| `--no-foreign-keys` | With several sources, find no foreign keys between them |
| `--max-unique-enum` `N` | Max distinct values for a text column to keep its values, as an Enum, when they repeat (default: 50) |
| `--no-formats` | Do not name the format a text column's values have (email, uuid4, ...) |
| `--no-patterns` | Do not write the pattern a code-like text column's values share (ORD-[0-9]{6}, ...) |
| `--parse-text` | Read a text column as the number, flag or date every value holds (Boolean, Int64, Float64, Date, Datetime) |
| `--sample` `N` | Profile a random sample of N rows (seeded, so the same every run) |
| `--replace` `COL` | Columns whose values must not be carried over: text in them is described by its lengths, never by the values the source holds |
| `--spec` `PATH` | Also write the spec the data was generated from (.yaml, or .py); with several sources, the registry of them |
| `--name` `NAME` | The spec's class name (default: derived from source) |

## `polspec diff`

What changed between two schemas, and whether it breaks.

```
polspec diff [-h] [--rename OLD=NEW] [--json | --markdown]
                    [--fail-on {breaking,any,none}] [--strict-dtypes] [--class NAME]
                    old new
```

| Argument | Meaning |
|:--|:--|
| `old` | The earlier spec: .yaml/.yml, or a .py file |
| `new` | The later spec |
| `--rename` `OLD=NEW` | A column renamed between the two; repeat for several |
| `--json` | Print the report as JSON |
| `--markdown` | Print the report as Markdown, for a pull-request comment |
| `--fail-on` `FAIL_ON` | Which findings make the exit status 1: breaking (default), any, or none. One of `breaking`, `any`, `none`. |
| `--strict-dtypes` | A dtype change is breaking unless the dtypes are identical |
| `--class` `NAME` | Use only this class (a .py source may define several) |

## `polspec drift`

How a data file has moved relative to its schema.

```
polspec drift [-h] [--all] [--sample N] [--significance P]
                     [--null-rate-tolerance F] [--frequency-tolerance F]
                     [--distribution-tolerance F] [--no-unseen] [--max-samples N]
                     [--json | --markdown] [--fail-on {breaking,any,none}]
                     [--strict-dtypes] [--class NAME]
                     spec data
```

| Argument | Meaning |
|:--|:--|
| `spec` | A .yaml/.yml spec, or a .py file defining one; with --all, a directory |
| `data` | Path to a CSV, Parquet, NDJSON or IPC file; with --all, a directory of files named after the specs |
| `--all` | Every spec found under SPEC against DATA/<name>.<suffix> |
| `--sample` `N` | Measure a random sample of N rows (seeded, so the same every run) |
| `--significance` `P` | How unlikely, were the spec right, a moved rate, frequency or distribution must be to be reported (default: 0.001) |
| `--null-rate-tolerance` `F` | The smallest move of a null rate or NaN share worth reporting (default: 0.05) |
| `--frequency-tolerance` `F` | The smallest move of a weighted column's frequencies worth reporting, as total variation (default: 0.05) |
| `--distribution-tolerance` `F` | The smallest move of a declared distribution worth reporting, as the KS distance (default: 0.05) |
| `--no-unseen` | Do not report declared values the data never holds |
| `--max-samples` `N` | Offending values to carry per finding (default: 10) |
| `--json` | Print the report as JSON |
| `--markdown` | Print the report as Markdown, for a pull-request comment |
| `--fail-on` `FAIL_ON` | Which findings make the exit status 1: breaking (default), any, or none. One of `breaking`, `any`, `none`. |
| `--strict-dtypes` | A dtype change is breaking unless the dtypes are identical |
| `--class` `NAME` | Use only this class (a .py source may define several) |
