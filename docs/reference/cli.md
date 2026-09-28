# Command line

Every `polspec` command and argument, generated from the command's own
parser -- `polspec <command> --help` prints the same. [Command line](../how-to/cli.md)
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
                            [--max-unique-enum N] [--no-bounds] [--sample N] [--shape]
                            [--keys] [--replace COL [COL ...]]
                            source
```

| Argument | Meaning |
|:--|:--|
| `source` | Path to a CSV, Parquet, NDJSON or IPC file |
| `-o`, `--output` `OUTPUT` | **Required.** File to write: .yaml/.yml, or .py for a FrameSpec subclass |
| `--name` `NAME` | Spec class name (default: derived from source) |
| `--weights` | Record each category's observed frequency |
| `--max-unique-enum` `N` | Max distinct values for a string column to become an Enum, when they repeat (default: 50) |
| `--no-bounds` | Skip computing numeric/temporal bounds and string lengths |
| `--sample` `N` | Profile only the first N rows |
| `--shape` | Fit the distribution each numeric or temporal column follows |
| `--keys` | Declare an all-distinct integer or text column unique |
| `--replace` `COL` | Columns whose values must not be carried into the spec |

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
| `--seed` `SEED` | Generation seed (default: 42) |
| `--no-cartesian` | Skip the coverage-guaranteeing cartesian test |
| `--class` `NAME` | Generate a test for only this class (a .py source may define several) |

## `polspec validate`

Check a data file against a schema.

```
polspec validate [-h] [--all] [--references NAME=PATH] [--class NAME]
                        [--allow-extra] [--allow-missing] [--strict-dtypes]
                        [--skip CHECK] [--json] [--output PATH] [--failing PATH]
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
| `--skip` `CHECK` | Turn off one kind of check, as validate_CHECK=False does; repeat for several. One of: rules, validators, unique, checks, foreign_keys, hierarchy, pattern, bounds One of `rules`, `validators`, `unique`, `checks`, `foreign_keys`, `hierarchy`, `pattern`, `bounds`. |
| `--json` | Print the report as JSON instead of text |
| `--output` `PATH` | Write the rows that passed, typed as the spec declares, to PATH (the format by its extension) |
| `--failing` `PATH` | Write the rows that failed, with the finding each broke, to PATH |

## `polspec generate`

Generate rows from a schema into a data file.

```
polspec generate [-h] [--all] [--format EXT] -n N -o OUTPUT [--seed SEED]
                        [--method {random,cartesian}] [--references NAME=PATH]
                        [--class NAME]
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
| `--class` `NAME` | Generate from only this class (a .py source may define several) |

## `polspec synthesize`

Write a fake data file that looks like a real one.

```
polspec synthesize [-h] -o OUTPUT [-n N] [--seed SEED]
                          [--replace COL [COL ...]] [--sample N] [--max-unique-enum N]
                          [--spec PATH] [--name NAME]
                          source
```

| Argument | Meaning |
|:--|:--|
| `source` | Path to a CSV, TSV, Parquet, NDJSON, JSON or IPC file |
| `-o`, `--output` `OUTPUT` | **Required.** File to write; the extension picks the format (.parquet, .csv, ...) |
| `-n`, `--rows` `N` | Rows to generate (default: as many as the source has) |
| `--seed` `SEED` | Generation seed (default: random) |
| `--replace` `COL` | Columns whose values must not be carried over: text in them is generated from its lengths, never from the values the source holds |
| `--sample` `N` | Profile a random sample of N rows rather than every row |
| `--max-unique-enum` `N` | Max distinct values for a text column to keep its values, when they repeat (default: 50) |
| `--spec` `PATH` | Also write the spec the data was generated from (.yaml, or .py) |
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
| `--fail-on` `FAIL_ON` | Which findings make the exit status 1: breaking (default), any, or none One of `breaking`, `any`, `none`. |
| `--strict-dtypes` | A dtype change is breaking unless the dtypes are identical |
| `--class` `NAME` | Use only this class (a .py source may define several) |

## `polspec drift`

How a data file has moved relative to its schema.

```
polspec drift [-h] [--all] [--sample N] [--null-rate-tolerance F] [--no-unseen]
                     [--max-samples N] [--json | --markdown]
                     [--fail-on {breaking,any,none}] [--strict-dtypes] [--class NAME]
                     spec data
```

| Argument | Meaning |
|:--|:--|
| `spec` | A .yaml/.yml spec, or a .py file defining one; with --all, a directory |
| `data` | Path to a CSV, Parquet, NDJSON or IPC file; with --all, a directory of files named after the specs |
| `--all` | Every spec found under SPEC against DATA/<name>.<suffix> |
| `--sample` `N` | Measure only the first N rows |
| `--null-rate-tolerance` `F` | How far the null rate may sit from null_probability (default: 0.05) |
| `--no-unseen` | Do not report declared values the data never holds |
| `--max-samples` `N` | Offending values to carry per finding (default: 10) |
| `--json` | Print the report as JSON |
| `--markdown` | Print the report as Markdown, for a pull-request comment |
| `--fail-on` `FAIL_ON` | Which findings make the exit status 1: breaking (default), any, or none One of `breaking`, `any`, `none`. |
| `--strict-dtypes` | A dtype change is breaking unless the dtypes are identical |
| `--class` `NAME` | Use only this class (a .py source may define several) |
