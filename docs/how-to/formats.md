# String formats

A `String` column can say how long its values are. `format=` lets it say what
they look like, and the two sides of the spec read that one word: generation
fills the column from the format's own sampler, and validation checks every
value against the format's own test.

```python
class Users(FrameSpec):
    user_id = ColSpec(pl.String, format="uuid4", unique=True)
    email   = ColSpec(pl.String, format="email")
    host    = ColSpec(pl.String, format="ipv4", nullable=True)

df = Users.generate(10_000, seed=42)
Users.validate(df)   # passes -- which is the point
```

Without a format, a column carrying a validator as ordinary as
`col("email").str.contains("@")` could not be generated to satisfy its own
spec. With one it can, because the sampler and the validator are one
declaration in `polspec.formats`, reviewed in one diff and pinned by one
round-trip test per format.

## The formats

| `format` | Generates | Validates |
|:--|:--|:--|
| `uuid4` | 32 hex digits in 8-4-4-4-12 groups, version nibble `4`, variant `8`–`b` | the same shape, either case |
| `email` | a local part, `@`, a domain and a TLD from a small pool | one `@`, non-empty both sides, a dot in the domain |
| `ipv4` | four dotted octets | four dotted decimals, each 0–255 |
| `ipv6` | eight colon-separated four-digit hextets | eight colon-separated hextets of 1–4 hex digits, uncompressed |
| `mac` | six colon-separated hex pairs | the same shape, either case |
| `hostname` | two or three labels ending in a TLD | dot-separated RFC 1123 labels, at most 253 characters |
| `iso_country` | one of the 249 ISO 3166-1 alpha-2 codes | membership in that list |
| `iso_currency` | one of the ISO 4217 alpha-3 codes of circulating currencies | membership in that list |

A validator is deliberately wider than its sampler where real data is: an
uppercase UUID from another system validates, even though polspec generates
lowercase ones. It is never wider than the standard it names.

The set is closed. A named format is a twenty-line sampler with an
unambiguous test; generating from an arbitrary regex would need a
regex-to-sampler compiler and has no answer for `.*`. Validation of an
arbitrary regex is what [`pattern=`](columns.md#string-patterns) is for --
checked like a format, generated like nothing at all.

## What a format promises

Syntax. `format="email"` generates a well-formed address and validates a
well-formed address; nothing is looked up, so `nobody@example.invalid` is
accepted and no generated address is deliverable. `hostname` accepts
`localhost`; `ipv4` accepts `0.0.0.0`. See
[Known limitations](../explanation/limitations.md).

## What a format combines with

A format owns the column's whole domain, so it refuses anything that says
what the values are a second time:

<!-- docs: raises -->
```python
ColSpec(pl.String, format="email", choices=["a@b.co"])
# SpecError: ColSpec cannot carry both format='email' and choices: each is a
# complete description of the column's domain, and they cannot both hold.

ColSpec(pl.String, format="uuid4", string_length=(36, 36))
# SpecError: ... the format already fixes how long a value is.

ColSpec(pl.Int64, format="uuid4")
# SpecError: ColSpec.format is only supported for pl.String, got Int64.
```

Everything that says how the values are *distributed* still applies:

- `nullable` and `null_probability` work as on any column.
- `unique=True` draws without replacement. `uuid4` has room for any frame;
  `iso_country` runs out at 249 rows and says so by name, the way a small
  `choices` list does.
- `validators` run alongside the format check. A column can carry
  `format="email"` and a validator on the domain it should come from.
- `tags` and `col_name` are untouched.

## Foreign keys

A format is part of the column's domain, so a key into a formatted parent is
checked at declaration like every other domain narrowing. A child column with
`format="uuid4"` accepts a parent with the same format and refuses one with
another format or none; a plain `String` child accepts any parent. A parent
whose `choices` are listed is checked value by value.

```python
class Sessions(FrameSpec):
    user_id = ColSpec(pl.String, format="uuid4")
    __foreign_keys__ = [ForeignKey("user_id", references=Users)]

users = Users.generate(100, seed=1)
sessions = Sessions.generate(1_000, seed=2, references={Users: users})
Sessions.validate(sessions, references={Users: users})
```

## Reports and files

The `format` finding reports values that do not match, with the format named
and samples listed:

```
Column 'host': found 3 value(s) that are not ipv4 (four dotted decimal
octets, each 0-255). Invalid samples: ['300.1.1.1', 'abc']
```

A format is one scalar key in a spec file -- `format: email` -- and one
keyword in generated Python. Neither changes the file format version.
`from_dataframe()` does not infer formats: guessing `email` from a sample is a
guess, and the profiler's contract is to describe what is there.
