"""Strings from a `pattern`, since 1.4.

Each value is a full match of the pattern, walked through the automaton the
validator's engine builds, so every value generated passes validation. A
pattern that every plain string already passes keeps the plain strings, as
a seed's output changes only where it broke its own spec; one out of the
generator's reach is a plain string, as before.
"""

from __future__ import annotations

import itertools
import warnings

import polars as pl
import pytest
from polspec import (
    ColSpec,
    GenerationError,
    TableSpec,
    generate,
    generate_batches,
    scan,
    validate,
)

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
    ("column", "why"),
    [
        ({"pattern": r"\bword\b"}, "Unicode word boundary"),
        ({"pattern": r"^[0-9]{6}$", "string_length": (2, 3)}, "2 to 3 characters"),
    ],
)
def test_a_pattern_out_of_reach_is_a_plain_string_and_says_why(column, why):
    plain = generate(_spec(string_length=column.get("string_length")), 300, seed=6)
    with pytest.warns(UserWarning, match=why) as warned:
        frame = generate(_spec(**column), 300, seed=6)
    assert frame.equals(plain)
    assert "pattern=False" in str(warned[0].message)


def test_a_pattern_that_generates_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        generate(_spec(pattern=r"^ORD-\d{6}$"), 100, seed=1)


@pytest.mark.parametrize(
    ("pattern", "rows"),
    [(r"^ID\d{8}$", 5_000), (r"^[A-C]{2}-(x|y|z)\d$", 270), (r"^é{1,3}[ab]?$", 9)],
)
def test_a_unique_pattern_column_draws_each_value_once(pattern, rows):
    spec = _spec(pattern=pattern, unique=True)
    frame = generate(spec, rows, seed=7)
    assert frame["c"].is_unique().all()
    validate(spec, frame)


def test_a_unique_pattern_fills_its_whole_space_and_no_more():
    spec = _spec(pattern=r"^[ab]{1,3}$", unique=True)
    every = generate(spec, 14, seed=1)["c"].to_list()
    expected = [
        "".join(p) for k in (1, 2, 3) for p in itertools.product("ab", repeat=k)
    ]
    assert sorted(every) == sorted(expected)
    with pytest.raises(GenerationError, match="14"):
        generate(spec, 15, seed=1)


def test_a_unique_pattern_column_is_a_window_in_batches():
    spec = _spec(pattern=r"^K\d{7}$", unique=True)
    whole = generate(spec, 150_000, seed=2)
    batches = pl.concat(generate_batches(spec, 150_000, batch_size=40_000, seed=2))
    assert batches.equals(whole)
    assert whole["c"].is_unique().all()
