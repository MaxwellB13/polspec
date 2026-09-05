# Generation and validation

polspec does two things with one declaration: it makes data that matches a
spec, and it checks whether data matches a spec. That sounds like one job read
in two directions, but the two directions are not symmetric, and most of the
library's design follows from where they differ.

## Why one declaration is harder than two

A validator only has to *recognise* a violation. A generator has to *avoid*
one. Recognising is easy for almost any claim you can write down: a Polars
expression over the frame gives you the answer. Avoiding is only easy for
claims with a shape a sampler can exploit.

`bounds=(1, 100)` has that shape — draw uniformly from the range and no value
can be out of bounds. `pl.col("email").str.contains("@")` does not. Both are
perfectly good validators; only one is a usable generator.

So the honest position is that the two sides cover different amounts of
ground, and the interesting engineering is in narrowing the gap without
pretending it isn't there.

## The failure mode: two implementations

The dangerous version of this is implementing each claim twice — once in the
sampler, once in the checker — and hoping they agree. They drift. A bound is
inclusive on one side and exclusive on the other; a choice is compared as a
string here and as a typed value there; a rule is evaluated against the
freely-generated frame while validation reads the final one.

Every one of those was a real bug in polspec, and each produced the same
symptom: `Spec.validate(Spec.generate(n))` raising. Data the library made,
rejected by the library that made it.

That property has a name here — the *round-trip* — and it is asserted
directly, in `tests/test_roundtrip.py`:

```python
SpecCls.validate(SpecCls.generate(n, seed=...))   # must not raise
```

## What is shared, and what is only tested

The structural answer is to give both sides one definition to read. That is
what `polspec.constraints` holds:

- **`Domain`** — the values a column may hold: its `choices`, an `Enum`'s
  categories, its `bounds`. Generation samples from it, validation checks
  against it, and a foreign key asks whether a parent's domain fits inside a
  child's. One definition, three readers.
- **`Pass` and `order`** — which rewrite of a generated frame runs first,
  derived from the columns each pass reads and writes. This is what lets a
  rule keyed on a foreign-keyed column see the parent's values, which is the
  same thing validation will check the rule against.

What is not shared is held in step by the round-trip test instead. That is a
weaker guarantee than a shared definition, and the difference is deliberate:
sharing costs an abstraction, and it is only worth paying where the two sides
genuinely say the same thing.

## Where the gap remains

Three claims are validated and not generated, and one of them is permanent:

`__checks__` and `ColSpec.validators` wrap arbitrary Polars expressions.
Nothing can generate data satisfying an arbitrary predicate — that is a
statement about predicates, not about polspec — so generation makes no
attempt, and the boundary is pinned by its own tests rather than papered over.

The way to close *that* gap is not a cleverer generator. It is a richer
vocabulary for describing values: a `pattern=` or a `format="email"` on
`ColSpec` says the same thing as the validator, in a shape a sampler can use.
See the [roadmap](roadmap.md).

Everything else on the list has been closed rather than documented away:
uniqueness by drawing without replacement, rule and foreign-key dependencies
by ordering the passes. The current list is in
[Known limitations](limitations.md), and each entry there is backed by a test
that fails the moment the entry stops being true.

## What this means when you declare a spec

Two practical consequences.

**A contradiction is refused when you write it, not when you run it.** A
foreign key whose parent domain cannot fit inside its own column's, two
columns whose rules each depend on the other, `unique=True` alongside
`weights` — none of these have a coherent reading, so they raise `SpecError`
at declaration rather than producing data that fails its own spec.

**A validation-only claim is still worth declaring.** A validator that
generation cannot satisfy is not wasted: it still guards real data on the way
in. Generate with `validate_validators=False` when you need synthetic rows,
and keep the claim for the data that matters.
