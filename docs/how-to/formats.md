# String formats

A `String` column can say how long its values are. `format=` lets it say what
they look like, and the two sides of the spec read that one word: generation
fills the column from the format's own sampler, and validation checks every
value against the format's own test.

```python
import polars as pl
from polspec import ColSpec, ForeignKey, FrameSpec

class Users(FrameSpec):
    user_id = ColSpec(pl.String, format="uuid4", unique=True)
    email   = ColSpec(pl.String, format="email")
    host    = ColSpec(pl.String, format="ipv4", nullable=True)

df = Users.generate(10_000, seed=42)
Users.validate(df)   # passes -- which is the point
```

Without a format, a column carrying a validator as ordinary as
`col("email").str.contains("@")` could not be generated to satisfy its own
spec: generation cannot read a validator. A format is a sampler and a test
defined together, so the data it generates always passes its own check.

## The formats

| `format` | Generates | Validates |
|:--|:--|:--|
| `uuid4` | 32 hex digits in 8-4-4-4-12 groups, version nibble `4`, variant `8`–`b` | the same shape, either case |
| `uuid7` | the same, version nibble `7`; *added in 1.4* | the same shape, either case; the timestamp is not read |
| `ulid` | 26 characters of Crockford base 32, the first `0`–`7`; *added in 1.4* | the same, either case; the timestamp is not read |
| `email` | a local part, `@`, a domain and a TLD from a small pool | one `@`, non-empty both sides, a dot in the domain |
| `ipv4` | four dotted octets | four dotted decimals, each 0–255 |
| `ipv6` | eight colon-separated four-digit hextets | eight colon-separated hextets of 1–4 hex digits, uncompressed |
| `mac` | six colon-separated hex pairs | the same shape, either case |
| `hostname` | two or three labels ending in a TLD | dot-separated RFC 1123 labels, at most 253 characters |
| `url` | `https://`, a hostname, and a path or query from a small pool; *added in 1.4* | `http` or `https`, a hostname, an optional port, then anything without whitespace; at most 2,048 characters |
| `iso_country` | one of the 249 ISO 3166-1 alpha-2 codes | membership in that list |
| `iso_country3` | one of the same 249 countries' alpha-3 codes, such as `GBR`; *added in 1.4* | membership in that list |
| `iso_currency` | one of the ISO 4217 alpha-3 codes of circulating currencies | membership in that list |

A validator is deliberately wider than its sampler where real data is: an
uppercase UUID from another system validates, even though polspec generates
lowercase ones. It is never wider than the standard it names.

For any other shape, declare a [`pattern=`](#patterns):
it is generated from and checked as a format is. A format names the shape
and keeps its test exactly as wide as the standard; a pattern says the
shape itself.

## Patterns

`pattern=` is a regular expression every value must match. Generation
fills the column with full matches of it, so generated data passes:

```python
class Orders(FrameSpec):
    order_ref = ColSpec(pl.String, pattern=r"^ORD-\d{6}$", unique=True)
    postcode  = ColSpec(pl.String, pattern=r"^[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}$")

orders = Orders.generate(1_000, seed=1)
Orders.validate(orders)
```

*Generated from since 1.4.*

The values are drawn through the automaton Polars' own regex engine builds
for the pattern, so generation and validation read it alike:

- **Characters:** ASCII wherever the pattern allows it, so `\w` draws
  `[A-Za-z0-9_]` and `.` printable ASCII; a pattern that needs `é` draws it.
- **Lengths:** within `string_length` when declared, counted in characters
  as validation counts them. An unbounded repeat (`+`, `*`, `{n,}`) runs to
  16 characters past the pattern's shortest match.
- **Unique:** `unique=True` draws each value of the pattern once, as for any
  other unique column; a pattern with fewer values than rows is refused.

Two kinds of pattern are filled with plain random text instead, as before
1.4:

- **One every plain string already passes**, such as `^[A-Za-z0-9]+$`. Its
  output stays exactly what it was, as seeded output does across 1.x.
- **One the generator cannot follow:** a Unicode word boundary (`\b`; the
  ASCII `(?-u:\b)` works), an automaton too large to build, or no match
  within the declared `string_length`. Generation warns, naming the
  column and the reason; validate with `pattern=False`, or pin the column
  with [`generate(fixed=)`](generating.md#pinning-columns-fixed).

Reach for `format` when the shape is one polspec names (`email`, `uuid4`,
...); reach for `pattern` for any other. The two cannot be combined. A pattern is compiled by Polars' regex engine at
declaration, so one Polars cannot run (look-around, for instance) is refused
with the engine's own message rather than failing on the first validation.

## What a format promises

Syntax. `format="email"` generates a well-formed address and validates a
well-formed address; nothing is looked up, so `nobody@example.invalid` is
accepted and no generated address is deliverable. `hostname` accepts
`localhost`; `ipv4` accepts `0.0.0.0`. See
[Known limitations](../explanation/limitations.md).

## Extending a format with extra values

Real columns rarely hold a clean format. A country column is ISO 3166 *plus*
a few codes of the organisation's own; an address column is IPv4 *or* a
sentinel for a value nobody captured. `extra_values` names them:

```python
country = ColSpec(pl.String, format="iso_country", extra_values=["UK (ISO)", "UK (ISLANDS)"])
ip = ColSpec(pl.String, format="ipv4", extra_values={"NOT AVAILABLE": 0.05, "INVALID": 0.01})

class Sessions(FrameSpec):
    __columns__ = {"country": country, "ip": ip}

sessions = Sessions.generate(100_000, seed=1)
Sessions.validate(sessions)
assert (sessions["ip"] == "NOT AVAILABLE").mean() > 0.04  # about 5% of the rows
```

Validation accepts a value that has the format *or* is one of the extras;
anything else — `uk (iso)`, `NA` — is still a `format` finding, and the
finding's details name the extras beside the format. Generation draws each
extra on its share of the present rows: a list gives every extra 1%, a
mapping sets each share. The format's own values fill the rest, so test data
reaches the code that handles `NOT AVAILABLE` as well as the code that
parses an address.

In a spec file:

```yaml
country:
  dtype: String
  format: iso_country
  extra_values: [UK (ISO), UK (ISLANDS)]
ip:
  dtype: String
  format: ipv4
  extra_values: {NOT AVAILABLE: 0.05, INVALID: 0.01}
```

It is refused where it could not mean anything, each with a message saying
why: without a `format` (a plain `String` column lists its values as
`choices`), for a value the format already accepts (`"GB"` on
`iso_country` would be counted twice), for a repeated or empty extra, for
shares summing past 1, and beside `unique=True`.

## What a format combines with

A format owns the column's whole domain, so it refuses anything that says
what the values are a second time:

<!-- docs: raises -->
```python
ColSpec(pl.String, format="email", choices=["a@b.co"])
# SpecError: ColSpec cannot carry both format='email' and choices: each is a
# complete description of the column's domain, and they cannot both hold.
# ... or, for values the column holds beside the format's own, declare them
# as extra_values.

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

A format is one scalar key in a spec file — `format: email` — and one
keyword in generated Python. Neither changes the file format version.
`from_dataframe()`, `profile()` and `polspec schema infer` name a format
when at least twenty distinct values have it and every value does — see
[Fake data from real data](synthesizing.md#how-close-the-fake-data-is) for
the near misses kept as `extra_values`, and for when a country or currency
column stays an `Enum` of its codes instead. `formats=False` turns it off.
