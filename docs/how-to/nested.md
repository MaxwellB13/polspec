# Nested columns

`List`, `Array`, `Struct` and Polars 2's `Map` columns, nested to any
depth. Each is declared with the fields a scalar column takes, read as
claims about the values inside it. The scalar fields themselves are on
[Declaring columns](columns.md).

## Lists and arrays

A `List` or `Array` column is described by the same fields as a scalar one,
read as claims about **each element**: `bounds`, `choices`, `weights`,
`format`, `pattern`, `string_length` and `distribution` all apply to the
values inside the list. One field describes the list itself:

```python
ColSpec(pl.List(pl.Int64), bounds=(0, 10), list_length=(1, 5))    # 1 to 5 ints, each 0..10
ColSpec(pl.List(pl.String), format="email")                       # 0 to 5 addresses
ColSpec(pl.List(pl.Enum(["a", "b", "c"])), choices=["a", "b"])    # from a subset of the Enum
ColSpec(pl.Array(pl.Float64, 3), bounds=(0.0, 1.0))               # exactly three, from the dtype
```

`list_length` is the inclusive range of elements a value holds, both ends
required; without it generation makes 0 to 5. An `Array` takes its length
from the dtype and refuses `list_length`.

`nullable` and `null_probability` describe the list: a null *cell*.
`element_null_probability` describes what is inside it — how often an
element is null:

```python
ColSpec(pl.List(pl.Int64), element_null_probability=0.1)   # about one element in ten is null
```

It defaults to 0, which says elements are never null: generation puts none
inside a list, and validation reports one under `nullability` like a null in
a non-nullable column. Any other rate accepts them, and every other element
claim still applies to the elements that are present. The rate applies to a
`List` and an `Array` alike, and to an element that is itself a struct or a
list, which is then null as a whole.

Validation runs every element claim inside the list, and a list fails where
*any* element does — the finding's samples and `rows()` are the offending
lists. `list_length` gets its own finding code.

Generation draws the lengths and the elements as two columns of the inner
dtype and wraps one by the other, so an element is made by the same code
that would make a scalar of its dtype, and a `List` column keeps its data
across a rename through `seed_name` like any other.

What a nested column cannot carry: `unique` (a list is not drawn without
replacement) and `rules` (a rule's choices are values, and a list value
would be a list of lists). `validators` and `__checks__` work as on any
column — they are expressions.

## Structs — `fields`

A `Struct` column's dtype is its schema — every field's name and type comes
from it — and `fields` says what is *claimed* about the values in it:

```python
ColSpec(
    pl.Struct({"lat": pl.Float64, "lon": pl.Float64, "label": pl.String}),
    fields={
        "lat": ColSpec(pl.Float64, bounds=(-90, 90)),
        "lon": ColSpec(pl.Float64, bounds=(-180, 180)),
    },
    nullable=True,
)
```

Each value is a `ColSpec`, so a field is described exactly as a column of
the same dtype would be. `fields` is **partial**: a struct of twenty fields
where one needs bounds spells one field, and the rest are generated from
their dtypes alone. A name the dtype does not declare, or a field spec
whose dtype disagrees with the struct's, is refused where it is written.

A field is a value, not a column, so `unique`, `rules`, `validators`,
`seed_name` and `col_name` are refused on one — each is a claim about a
column among columns. (A validator about a field is written on the struct
column instead: `pl.col("point").struct.field("lat") != 0`.) As with a
list, the column's `nullable` describes the *cell* — a null struct — and a
field's own `nullable` says whether it may be null inside a struct that is
present. It defaults to `False`, like any column's.

A `List` of a `Struct` takes `fields` too, describing its element, and a
field may itself be a struct or a list, so a declaration nests as deeply as
the dtype does:

```python
ColSpec(pl.Struct({"xs": pl.List(pl.Int64)}),
        fields={"xs": ColSpec(pl.List(pl.Int64), bounds=(0, 9), list_length=(2, 2))})
```

Generation makes one column per field and gathers them, so a field is drawn
by the code that draws a column of its dtype — and a struct column is
seeded by name like any other: renaming it with `seed_name` keeps every
field, and adding a field beside one moves nothing.

Validation checks each field's claims in place, and a finding names the
field it is about — its key is `point.lat__bounds`, its message says
`Column 'point.lat'` — while its `columns` stay `("point",)`, so
`report.rows(finding)` returns the rows of the frame that hold the
offending structs. Inside a list the samples are the offending lists, as
for any list. A struct in the data matches a declared one by field name,
not order, each field compatible by the usual rules; a field missing or
added is a `dtype` finding.

## Maps — a list of entries

Polars 2's `Map(key, value)` is physically a list of `{key, value}`
structs, and Polars casts one to the other and back. polspec describes a
map as that list, so it takes what a `List(Struct)` takes: `list_length`
is how many entries a map holds, and `fields` says what its `key` and its
`value` are.

```python
ColSpec(
    pl.Map(pl.String, pl.Int64),
    list_length=(1, 4),                          # entries per map
    fields={
        "key": ColSpec(pl.String, choices=["cpu", "mem", "disk", "net"]),
        "value": ColSpec(pl.Int64, bounds=(0, 100), nullable=True),
    },
    nullable=True,                               # a null map, not an empty one
)
```

A map adds two rules of its own: **a key is never null**, and **no key
repeats within a map**. So a nullable key is refused, and so is a
`list_length` longer than the keys can fill — a map of up to five entries
from three possible keys is refused where it is written. With no
`list_length`, a map is as long as a list (0 to 5 entries) or as its keys
allow, if fewer: a map of `Boolean` keys holds at most two.

Generation draws the entries as a list of structs, redraws any key that
repeats within its map — only the repeats, so the key's declared shape
survives — and casts the result to the `Map`. Validation casts the map to
its entries and checks them as a list of structs: a finding about a key or
a value is named `m.key` or `m.value`, the length `m__list_len`, and a map
holding one key twice — which Polars folds away when it casts, but not
when a map arrives from Arrow — is `m.key__unique`. A map in the data
stands in for a declared one when its key and value are each compatible by
the usual rules; a `List(Struct)` does not, since it is not a map. Drift,
`from_dataframe` and `estimated_size` read a map as its entries too.

`choices`, `weights` and `element_null_probability` have no meaning on the
map itself (an entry is never null) and are refused; say what a key or
a value may be in `fields`.

