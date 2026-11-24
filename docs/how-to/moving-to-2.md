# Moving to 2.0

polspec 2.0 is the one release that breaks: a faster engine whose seeds
generate different data, defaults changed where 1.x had two of them, and
names made consistent. Nothing was deprecated in 1.x first. Instead every
old spelling fails at once with an error that names what to write — "since
2.0" is in each — so running your tests on 2.0 finds what to change.

```bash
pip install "polspec>=2,<3"
```

2.0 needs Polars 2 and reads spec files of format 4, which every polspec
since 0.14.0 writes. The rest of this page is each change, with what to
write instead.

## Before you upgrade

- **Keep any generated frame that must not change as a file.** The same
  spec and seed generate different data on 2.0 (see
  [Seeds](#seeds-generate-new-data)). A test that compares generated values
  to numbers written into it will need those numbers again.
- **Rewrite spec files older than format 4.** A file whose `version:` is 1,
  2 or 3, written by polspec 0.13 or earlier, is refused on 2.0. polspec
  1.x reads it and writes it back as version 4: with `polspec<2`
  installed, `polspec.to_yaml(polspec.from_yaml(path), path)`.
- **Move to Polars 2.** polspec 2.0 needs `polars>=2,<3`.

## Seeds generate new data

The engine was rebuilt for 2.0, and the same spec and seed give a
different frame. Within 2.x a seed's output is promised again, on every
platform, Polars 2 release and thread count, as it was within 1.x
([Stability](../explanation/stability.md#seeded-output)).

What changed, and what each buys:

- Columns are seeded every 4,096 rows, so a small frame fills on every
  core, and nulls come from a stream of their own, so a column's values no
  longer move when its `null_probability` does.
- Strings are drawn in bulk, about twice as fast.
- Unique columns draw through a cheaper permutation. A unique string
  spreads over its lengths rather than nearly always taking the longest,
  and a null in a unique column spends no value.
- Weighted categories are drawn by the alias method, at the same cost
  whatever their number.
- Lists, rules and foreign keys are windows onto the whole frame, so a
  batch, a scan or a sink holds, row for row, what `generate()` holds
  there.
- A composite key whose members all have a finite or bounded space is
  distinct across batches, not only within one.
- `Decimal`, `Int128` and `UInt128` generate at full width: bounds past 64
  bits are no longer refused. A range 64 bits held draws what it did.

## Generated data keeps its checks by default

`checks=` now defaults to `None`: a spec's `__checks__` and column
validators are kept, as `checks=True` keeps them, so fake data passes its
own spec unasked. Where they cannot be kept — `method="cartesian"`, a
hierarchy, a key into the spec's own rows, a pin of all `n` values, a
check that rejects nearly every row — the frame is the one `checks=False`
gives, with a warning saying why.

| 1.x | 2.0 |
|:--|:--|
| `generate(spec, n)`, checks not kept | `generate(spec, n)` keeps them where it can; `checks=False` for 1.x's frame, unfiltered and silent |
| `generate(spec, n, checks=True)` | unchanged: keeps them, or refuses |

## One set of profiling defaults

`FrameSpec.from_dataframe` and `polspec schema infer` record what
`profile()` records — category frequencies, fitted distributions, keys and
row order — and write bounds and rates exactly. `describe=True`, or
`--describe`, is 1.x's lean form.

| 1.x | 2.0 |
|:--|:--|
| `FrameSpec.from_dataframe(df)` | `FrameSpec.from_dataframe(df, describe=True)` |
| `FrameSpec.from_dataframe(df, weights=True)` | `FrameSpec.from_dataframe(df, describe=True, weights=True)`, or `from_dataframe(df)` for everything |
| `polspec schema infer data.csv -o spec.yaml` | `polspec schema infer data.csv -o spec.yaml --describe --round-bounds auto --round-rates auto` |
| `schema infer ... --weights --keys --order` | `schema infer ...`: they are on; `--no-weights`, `--no-keys`, `--no-order` turn each off |

A key keeps the range it was seen in, so a spec inferred from 200 rows
generates at most 200 rows of it; `synthesize` widens it for the rows it
makes, and `polspec test` asks for no more rows than a unique column holds.

## Names and signatures

Each 1.x spelling raises an error naming the 2.0 one.

| 1.x | 2.0 |
|:--|:--|
| `assert_valid(df, spec)` | `assert_valid(spec, df)`, the spec first as in every other verb |
| `Orders.col("id")` | `Orders.column("id")`; `polspec.col("id")` is still the predicate |
| `Orders.tag("pii")`, `spec.tag("pii")` | `Orders.tagged("pii")`, `spec.tagged("pii")` |
| `CatSpec.from_framespec(spec)` | `CatSpec.from_spec(spec)` |
| `CatSpec.infer(spec, max_enum_cardinality=20)` | `CatSpec.infer(spec, max_unique_enum=20)` |
| `from_yaml(path, categories=cats)`, and on `from_dict`, `FrameSpec.from_yaml`, `Registry`, `Registry.discover`, `Registry.from_module` | `catspec=cats` |
| `registry.categories` | `registry.catspec()` |
| `spec.to_yaml(path)` returned `None` | it returns the text, and writes the file too; so does every writer |

## Weights inside `choices`

`weights=` is gone from `ColSpec` and `ColRule`: weights are written beside
their values, as the `repr` and `to_python` already wrote them. A column
still reads its weights back as `.weights`, and spec files keep their
`weights:` key, so no file changes.

| 1.x | 2.0 |
|:--|:--|
| `ColSpec(pl.String, choices=["a", "b"], weights=[3, 1])` | `ColSpec(pl.String, choices={"a": 3, "b": 1})` |
| `ColSpec(pl.Enum(["a", "b"]), weights=[3, 1])` | `ColSpec(pl.Enum, choices={"a": 3, "b": 1})` |
| `ColSpec(pl.Boolean, weights=[0.9, 0.1])` | `ColSpec(pl.Boolean, choices={False: 0.9, True: 0.1})` |
| `ColRule(when=..., choices=["x", "y"], weights=[1, 2])` | `ColRule(when=..., choices={"x": 1, "y": 2})` |
| `spec.replace(weights=[...])` | `spec.replace(choices={...})` |

## Errors are polspec's own

`SpecError`, `ValidationError`, `GenerationError`, `SerializationError`
and `RegistryError` are each a `PolspecError` and nothing else: they no
longer also subclass `ValueError`, `TypeError` or `LookupError`.

| 1.x | 2.0 |
|:--|:--|
| `except ValueError:` around a polspec call | `except PolspecError:`, or the class itself |
| `pytest.raises(ValueError)` for a bad spec | `pytest.raises(SpecError)` |

A wrong argument — a string where a list was asked for — is still a
`TypeError` or a `ValueError`, as Python's own are.

## Polars 2, and format 4

| 1.x | 2.0 |
|:--|:--|
| `polars>=1.39,<3` | `polars>=2,<3` |
| A spec file of format 1, 2 or 3 is migrated as it is read | refused, naming its version and saying 1.x reads and rewrites it |
| A file or mapping with no `version:` key is format 1 | read as format 4; an unknown key in one says the file may be format 1 |

See [Specs as files](files.md) for the format, and the
[changelog](https://github.com/MaxwellB13/polspec/blob/main/CHANGELOG.md)
for everything 2.0 changed.
