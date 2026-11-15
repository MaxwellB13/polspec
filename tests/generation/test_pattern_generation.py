"""Strings from a `pattern`, since 1.4.

Each value is a full match of the pattern, walked through the automaton the
validator's engine builds, so every value generated passes validation. A
pattern that every plain string already passes keeps the plain strings, as
a seed's output changes only where it broke its own spec; one out of the
generator's reach is a plain string, as before.
"""

from __future__ import annotations

import polars as pl
import pytest
from polspec import ColSpec, TableSpec, generate, generate_batches, scan, validate

PATTERNS = [
    r"^ORD-[0-9]{6}$",
    r"^[A-Z]{1,2}[0-9][A-Z0-9]? [0-9][A-Z]{2}$",  # a UK postcode
    r"SKU-[A-Z]{2}-\d{5}",  # unanchored: a full match passes `contains`
    r"^(cat|dog|bird)s?$",
    r"^\w+@\w+\.(com|org|net)$",
    r"^[^aeiou\s]{3,5}$",
    r"^a+b*c?$",
    r"(?i)^hello$",
    r"^\d{3}(-\d{4})?$",
    r"^[\p{Lu}]{2}\d$",
    r"^é+ü?$",
]


def _spec(**column) -> TableSpec:
    return TableSpec("P", {"c": ColSpec(pl.String, **column)})


@pytest.mark.parametrize("pattern", PATTERNS)
def test_every_value_matches_its_pattern(pattern):
    spec = _spec(pattern=pattern)
    frame = generate(spec, 2_000, seed=3)
    assert frame["c"].str.contains(pattern).all(), frame.filter(
        ~pl.col("c").str.contains(pattern)
    )
    validate(spec, frame)


def test_string_length_is_held_in_characters():
    spec = _spec(pattern=r"^[a-z]+$", string_length=(3, 6))
    lengths = generate(spec, 2_000, seed=1)["c"].str.len_chars()
    assert lengths.min() == 3 and lengths.max() == 6
    accented = generate(_spec(pattern=r"^é+$", string_length=(2, 2)), 10, seed=1)
    assert accented["c"].to_list() == ["éé"] * 10


def test_an_unbounded_repeat_runs_sixteen_past_its_shortest():
    lengths = generate(_spec(pattern=r"^ab+$"), 5_000, seed=1)["c"].str.len_chars()
    assert (lengths.min(), lengths.max()) == (2, 18)


def test_values_are_ascii_wherever_the_pattern_allows_it():
    values = generate(_spec(pattern=r"^\w{6}$"), 2_000, seed=1)["c"]
    assert values.str.contains(r"^[A-Za-z0-9_]{6}$").all()


def test_nulls_are_drawn_as_declared():
    spec = _spec(pattern=r"^X\d$", nullable=True, null_probability=0.5)
    frame = generate(spec, 4_000, seed=2)
    assert 0.45 < frame["c"].null_count() / 4_000 < 0.55
    validate(spec, frame)


def test_a_list_of_patterned_strings():
    spec = TableSpec(
        "L",
        {
            "tags": ColSpec(
                pl.List(pl.String), pattern=r"^#[a-z]{3,6}$", list_length=(1, 3)
            )
        },
    )
    frame = generate(spec, 500, seed=4)
    matches = frame["tags"].list.eval(pl.element().str.contains(r"^#[a-z]{3,6}$"))
    assert matches.list.all().all()
    validate(spec, frame)


def test_a_seed_gives_the_same_values_and_batches_are_windows_onto_them():
    spec = _spec(pattern=r"^[A-F]{2}\d{3}$")
    whole = generate(spec, 200_000, seed=9)
    assert whole.equals(generate(spec, 200_000, seed=9))
    batches = pl.concat(generate_batches(spec, 200_000, batch_size=70_000, seed=9))
    assert batches.equals(whole)
    assert scan(spec, 200_000, seed=9).collect().equals(whole)


def test_a_pattern_plain_strings_always_pass_keeps_the_plain_strings():
    plain = generate(_spec(), 1_000, seed=5)
    for pattern in (r"^[A-Za-z0-9]+$", r"^\w{5,15}$", r"[A-Za-z0-9]"):
        assert generate(_spec(pattern=pattern), 1_000, seed=5).equals(plain), pattern


def test_a_pattern_plain_strings_can_fail_is_generated_from():
    plain = generate(_spec(), 1_000, seed=5)
    patterned = generate(_spec(pattern=r"^[A-Za-z0-9]{5,14}$"), 1_000, seed=5)
    assert not patterned.equals(plain)
    assert patterned["c"].str.len_chars().max() <= 14


@pytest.mark.parametrize(
    "column",
    [
        {"pattern": r"\bword\b"},  # a Unicode word boundary
        {"pattern": r"^[0-9]{6}$", "string_length": (2, 3)},  # no match that long
    ],
)
def test_a_pattern_out_of_reach_is_a_plain_string(column):
    plain = generate(_spec(string_length=column.get("string_length")), 300, seed=6)
    assert generate(_spec(**column), 300, seed=6).equals(plain)


def test_a_unique_pattern_column_is_a_plain_string_for_now():
    spec = _spec(pattern=r"^ID\d{8}$", unique=True)
    plain = generate(_spec(unique=True), 300, seed=6)
    assert generate(spec, 300, seed=6).equals(plain)
