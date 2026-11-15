"""A code-like text column profiled as the pattern its values share."""

from __future__ import annotations

import polars as pl
import polspec
import pytest


def _pattern(values: list[str], **options) -> str | None:
    frame = pl.DataFrame({"code": values})
    return polspec.profile(frame, **options)["code"].pattern


ORDERS = [f"ORD-{i:06d}" for i in range(0, 900_000, 997)]
NAMES = [
    "Ada", "Grace", "Alan", "Barbara", "Edsger", "Donald", "Ken", "Dennis",
    "Linus", "Guido", "Brian", "Bjarne", "James", "Niklaus", "John", "Leslie",
    "Tony", "Robin", "Frances", "Margaret",
]  # fmt: skip


@pytest.mark.parametrize(
    ("values", "pattern"),
    [
        (ORDERS, "^ORD-[0-9]{6}$"),
        ([*ORDERS[:30], "N/A"], "^(?:ORD-[0-9]{6}|N/A)$"),
        (NAMES, "^[A-Z][a-z]{2,7}$"),
        ([f"a.b(c){i % 30}" for i in range(40)], r"^a\.b\(c\)[0-9]{1,2}$"),
        # Fewer than ten values sharing a part is chance: it is a class.
        ([*ORDERS[:20], "AB1", "AB2", "AB3"], "^(?:ORD-[0-9]{6}|[A-Z]{2}[0-9])$"),
    ],
)
def test_a_shape_becomes_a_pattern(values, pattern):
    assert _pattern(values) == pattern


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(
            [f"the {'very ' * (i % 5)}quick fox {i}" for i in range(40)], id="prose"
        ),
        pytest.param([f"{i * 7919:x}" for i in range(100)], id="hex, many shapes"),
        pytest.param([f"a-----------b{i}" for i in range(40)], id="a long literal"),
        pytest.param([f"c{i}" for i in range(19)] * 3, id="19 distinct"),
        pytest.param([f"AB{i:02d}" for i in range(100)], id="a key with no room"),
        pytest.param([f"a\x01{i}" for i in range(40)], id="a shape mark"),
        pytest.param([str(i % 30) for i in range(40)], id="numbers as text"),
        pytest.param(
            [f"A{i:05d}" for i in range(20_000)] * 2 + ["b1", "c-1", "d_e"],
            id="more shapes past the first values",
        ),
        pytest.param(
            [f"A{i:04d}" for i in range(1_000)] * 2 + ["b1", "c-1", "d_e"],
            id="more shapes past the first thousand",
        ),
        pytest.param(
            [f"A{i:05d}" for i in range(20_000)] * 2 + ["A\x01"],
            id="a shape mark past the first values",
        ),
    ],
)
def test_no_pattern(values):
    assert _pattern(values) is None


def test_every_source_value_matches_and_the_fake_codes_have_the_shape():
    areas, units = ["SW", "M", "B", "EC"], ["AA", "AE", "TH"]
    postcodes = [
        f"{areas[i % 4]}{i % 9 + 1} {i % 7 + 1}{units[i % 3]}" for i in range(903)
    ]
    frame = pl.DataFrame({"ref": ORDERS, "postcode": postcodes})
    spec = polspec.profile(frame)
    assert spec["postcode"].pattern == "^[A-Z]{1,2}[0-9] [0-9][A-Z]{2}$"
    polspec.validate(spec, frame)
    fake = polspec.synthesize(frame, n=2_000, seed=1)
    assert fake["ref"].str.contains(r"^ORD-\d{6}$").all()
    polspec.validate(spec, fake, bounds=False)


def test_a_key_has_room_for_a_larger_stand_in():
    frame = pl.DataFrame({"ref": [f"R{i:04d}" for i in range(500)]})
    assert polspec.profile(frame)["ref"].pattern == "^R[0-9]{4}$"
    fake = polspec.synthesize(frame, scale=10, seed=1)
    assert fake["ref"].is_unique().all()


def test_lengths_past_the_first_values_are_read_from_every_value():
    assert _pattern([f"R{i * i}" for i in range(50_000)]) == "^R[0-9]{1,10}$"


def test_switched_off_and_under_replace():
    assert _pattern(ORDERS, patterns=False) is None
    assert _pattern(ORDERS, replace=["code"]) is None


def test_a_format_wins():
    emails = [f"user{i}@example.com" for i in range(40)]
    spec = polspec.profile(pl.DataFrame({"code": emails}))["code"]
    assert spec.format == "email"
    assert spec.pattern is None
