# Spec files

The YAML a spec, a category registry or a registry of specs is written as,
key by key. [Specs as files](../how-to/files.md) shows the workflow; this is
the format. A test holds every table here to the reader's own field
registry, so a key the reader accepts cannot be missing from this page.

Every file starts with `version: 3`, the format version that wrote it. A
file from an older version is migrated as it is read; one from a newer
polspec is refused, naming the version it needs. A key the reader does not
know is an error naming the closest known key; `strict=False` downgrades it
to a warning. A key left out takes its default, and a writer leaves out
every key still at its default.

## A spec file

`Orders.to_yaml(path)` and `serialization.to_yaml(spec, path)` write one;
`FrameSpec.from_yaml(path)` reads one.

| Key | Holds |
|:--|:--|
| `version` | The format version, `3` |
| `name` | The spec's name; foreign keys refer to a spec by it |
| `categories` | A category registry: a path to one (relative to the file), or one written inline |
| `columns` | Column name to column, in order |
| `unique_together` | A list of composite keys, each a list of column names |
| `foreign_keys` | A list of foreign keys |
| `checks` | A list of checks |
| `hierarchy` | A hierarchy |

### A column

Each key is the `ColSpec` field of the same name; see
[ColSpec fields](colspec-fields.md) for what each means.

| Key | Written as |
|:--|:--|
| `dtype` | A dtype -- see [Dtypes](#dtypes) |
| `col_name` | A string |
| `nullable` | `true` or `false` |
| `bounds` | `[min, max]`; either end may be `null`. A `Decimal`'s fractional ends are strings, such as `'1.50'`, so they read back exactly; a time or a duration is [tagged](#times-and-durations) |
| `tags` | A string, or a list of them |
| `unique` | `true` or `false` |
| `null_probability` | A number from 0 to 1 |
| `element_null_probability` | A number from 0 to 1 |
| `string_length` | `[min, max]` |
| `list_length` | `[min, max]` |
| `fields` | Field name to column, for a `Struct` -- or `key` and `value`, for a `Map` |
| `format` | A format's name, such as `email` |
| `pattern` | A regular expression |
| `seed_name` | A string |
| `distribution` | A distribution's name, such as `normal` |
| `distribution_params` | Parameter name to number |
| `choices` | A list of values; a time or a duration is [tagged](#times-and-durations) |
| `weights` | A list of numbers, one per choice |
| `rules` | A list of rules |
| `validators` | A list of checks, each over this column |

### A rule

| Key | Written as |
|:--|:--|
| `when` | A predicate -- see [Predicates](#predicates) |
| `choices` | A list of values |
| `weights` | A list of numbers, one per choice |

### A check or validator

| Key | Written as |
|:--|:--|
| `expr` | A predicate |
| `name` | A string; defaults to the predicate written out |
| `description` | A string |
| `ignore_nulls` | `true` (the default: a null result passes) or `false` |

Only a check written with `col()` can be written to a file. One over a raw
`pl.Expr` is left out, with a warning naming it.

### A foreign key

| Key | Written as |
|:--|:--|
| `columns` | A list of this spec's columns |
| `references` | The referenced spec's name, or `self` |
| `ref_columns` | A list of the referenced spec's columns, in the same order |
| `name` | A string; defaults to one derived from the columns and the target |

### A hierarchy

| Key | Written as |
|:--|:--|
| `child` | The column holding a reference |
| `parent` | The column holding the reference it points at |
| `max_depth` | The deepest chain, in hops |
| `branching` | The mean children per reference |
| `roots` | How many ultimate parents |

## A category registry

`CatSpec.to_yaml(path)` writes one; `CatSpec.from_yaml(path)` reads one,
and a spec file's `categories:` may name one.

| Key | Holds |
|:--|:--|
| `version` | The format version |
| `enums` | Name to a list of categories |
| `categoricals` | Name to a categorical: `name`, and optionally `namespace`, `physical` (`UInt8`, `UInt16` or `UInt32`) and `categories` |
| `choices` | Name to a list of values, for a name that is neither |

## A registry of specs

`Registry.to_yaml(path)` writes one; `Registry.from_yaml(path)` reads one.

| Key | Holds |
|:--|:--|
| `version` | The format version |
| `categories` | A category registry, inline or as a path |
| `specs` | Spec name to a spec, written as a spec file is but without `version` or `name` |

## Dtypes

A dtype with no parameters is its name: `Int8`, `Int16`, `Int32`, `Int64`,
`Int128`, `UInt8`, `UInt16`, `UInt32`, `UInt64`, `UInt128`, `Float16`,
`Float32`, `Float64`, `Boolean`, `String`, `Binary`, `Date`, `Time`,
`Datetime`, `Duration` and `Categorical`. One with parameters is a mapping
of one key:

| Dtype | Written as |
|:--|:--|
| `Enum` | `{Enum: [a, b, c]}`, or `{Enum: NAME}` for a registry's |
| `Categorical` | `{Categorical: {name: ..., namespace: ..., physical: ...}}`, or `{Categorical: NAME}` for a registry's |
| `Datetime` | `{Datetime: {time_unit: us, time_zone: UTC}}` |
| `Duration` | `{Duration: {time_unit: ms}}` |
| `Decimal` | `{Decimal: {precision: 10, scale: 2}}` |
| `List` | `{List: <dtype>}` |
| `Array` | `{Array: {inner: <dtype>, width: 3}}` |
| `Struct` | `{Struct: {field: <dtype>, ...}}` |
| `Map` | `{Map: {key: <dtype>, value: <dtype>}}` -- Polars 2 only |

A bare name a category registry declares -- `STATUS`, or
`$categories.STATUS` -- is that registry's `Enum` or `Categorical`.

## Predicates

A predicate is the data form of one written with `col()`: a mapping of one
operation to its operands, or a scalar for a literal. `col("total") >=
col("subtotal")` is written `{ge: [{col: total}, {col: subtotal}]}`.

| Operation | Operands |
|:--|:--|
| `col` | A column name |
| `lit` | A value; needed only for a list or mapping |
| `eq`, `ne`, `lt`, `le`, `gt`, `ge` | `[left, right]` |
| `add`, `sub`, `mul`, `div` | `[left, right]` |
| `and`, `or` | A list of two or more predicates |
| `not` | A predicate |
| `is_in` | `[predicate, [values...]]` |
| `is_null` | A predicate |
| `between` | `[predicate, lower, upper]` |
| `str_contains`, `str_starts_with`, `str_ends_with`, `str_matches` | `[predicate, text]`; `str_matches` is a regular expression, `str_contains` a literal substring |
| `str_len` | A predicate |
| `time`, `duration` | A literal time or duration, in its [tagged form](#times-and-durations) |

## Times and durations

YAML has forms for numbers, text, dates, datetimes and bytes, and none for a
time of day or a duration. Wherever a spec file holds one -- a bound, a
choice, a rule's choice, a predicate's literal -- it is a mapping of one key
saying which it is:

| Value | Written as |
|:--|:--|
| `datetime.time(12, 30)` | `{time: "12:30:00"}` -- the ISO form, microseconds included when there are any |
| `datetime.timedelta(days=1, hours=2)` | `{duration: {days: 1, seconds: 7200, microseconds: 0}}` -- `timedelta`'s own fields; a field left out is `0` |

```yaml
columns:
  opens:
    dtype: Time
    bounds: [{time: "06:00:00"}, {time: "10:00:00"}]
checks:
- expr: {ne: [{col: opens}, {time: "08:00:00"}]}
  name: not_eight
```

A file holding neither reads as before, so these are not a new format
version: a file could not hold a time or a duration at all until 0.13.0.

## Versions

| Version | Changed |
|:--|:--|
| 1 | The original format, with no `version` key: tags could be spelt `category` or `categories`, a rule's `when` was a one-column mapping such as `{column: region, equals: UK}`, and distribution parameters took any alias |
| 2 | `version` recorded; tags spelt `tags`; conditions and checks as predicates; distribution parameters canonical; foreign keys written with their target's name; `checks` and `validators`; registry files |
| 3 | `hierarchy` |

A key added without changing another's meaning -- `format`, `pattern`,
`seed_name` -- is not a new version: a file without it reads as before.
