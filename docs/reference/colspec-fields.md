# ColSpec fields

Every field a `ColSpec` takes, what it applies to, and what each side of the
spec does with it. The how-to pages
([Declaring columns](../how-to/columns.md),
[Constraints](../how-to/constraints.md)) show them in use; the
[API reference](api/columns.md#colspec) has the full parameter
descriptions. A test holds this table to `ColSpec` itself, so a field added
without a row here fails the suite.

On a `List`, `Array` or `Map` column, a field that describes a *value*
describes each element (for a `Map`, each `{key, value}` entry); `nullable`
and `null_probability` describe the cell itself.

| Field | File key | Applies to | Generation | Validation |
|:--|:--|:--|:--|:--|
| `dtype` | `dtype` | every column | draws values of this dtype | the column's dtype must be compatible (exactly, with `strict_dtypes`) |
| `col_name` | `col_name` | a `FrameSpec` attribute | names the column instead of the attribute | the column is looked up by this name |
| `seed_name` | `seed_name` | every column | seeds the column, and its rules, from this name instead of its own | -- |
| `nullable` | `nullable` | every column | allows nulls, at `null_probability` | a null is a `nullability` finding unless this is set |
| `bounds` | `bounds` | numeric and temporal values | draws within them; an open end falls back to the default range | a value outside a declared end is a `bounds` finding |
| `tags` | `tags` | every column | -- | -- (selects columns: `spec.tag(...)`) |
| `unique` | `unique` | scalar columns | draws without replacement, unique across a whole frame however it is batched | a repeated value is a `unique` finding |
| `null_probability` | `null_probability` | a `nullable` column | the rate nulls are drawn at | -- (drift compares it to the observed rate) |
| `nan_probability` | `nan_probability` | float values -- a column's, a list's elements, a struct's fields | the share of present values drawn as NaN | a NaN is a `nan` finding while this is `0`; `bounds` and `choices` never judge a NaN |
| `string_length` | `string_length` | `String` and `Binary` values | draws lengths within it | a length outside it is a `string_length` finding |
| `list_length` | `list_length` | `List` and `Map` columns | draws each cell's length within it | a length outside it is a `list_length` finding |
| `element_null_probability` | `element_null_probability` | `List` and `Array` columns | the rate an element inside a present list is null | a null element is a `nullability` finding while this is `0` |
| `fields` | `fields` | `Struct` values, and a `Map`'s `key`/`value` | draws each field from its own `ColSpec` | checks each field as a value, named `column.field` |
| `format` | `format` | `String` values | fills each value from the format's template | a value not in the format is a `format` finding |
| `extra_values` | `extra_values` | a `format` column | draws each extra on its share of the present rows -- 1% each by default -- and the format's own values on the rest | a value that is neither the format nor an extra is a `format` finding |
| `pattern` | `pattern` | `String` values | -- (not read; values are ordinary random text) | a value not matching is a `pattern` finding |
| `distribution` | `distribution` | numeric and temporal values | shapes the draw | -- |
| `distribution_params` | `distribution_params` | a `distribution`, or a `Boolean`'s `p` | the distribution's parameters | -- |
| `choices` | `choices` | any scalar value | draws from them, as a finite domain | a value outside them is a `choices` finding |
| `weights` | `weights` | `choices`, an `Enum`, or a `Boolean` | biases the draw | -- (drift compares them to observed frequencies) |
| `rules` | `rules` | scalar columns | rewrites the rows a rule's condition matches | a matched row outside the rule's choices is a `rule` finding |
| `validators` | `validators` | every column | -- (not read) | a row failing one is a `validator` finding |

A `--` in the generation column is a claim generation does not act on; the
round trip still holds, except for `pattern` and `validators`, which are
validation-only by design -- see
[Known limitations](../explanation/limitations.md).
