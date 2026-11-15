"""`synthesize_all`: fake related tables, with the keys between them kept."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import polspec
import pytest
from polspec import ForeignKey, GenerationError

DOCS_DATA = Path(__file__).resolve().parents[2] / "docs" / "data" / "shop"


@pytest.fixture
def shop() -> dict[str, pl.DataFrame]:
    customers = pl.DataFrame(
        {"id": range(1, 101), "email": [f"c{i}@example.com" for i in range(100)]}
    )
    orders = pl.DataFrame(
        {
            "id": range(1, 301),
            "customer_id": pl.Series([i % 37 + 1 for i in range(300)], dtype=pl.Int32),
            "status": ["NEW", "PAID", "SHIPPED"] * 100,
        }
    )
    lines = pl.DataFrame(
        {"order_id": [i % 300 + 1 for i in range(600)], "line": [1, 2] * 300}
    )
    return {"Customers": customers, "Orders": orders, "OrderLines": lines}


def _held(fake: dict[str, pl.DataFrame]) -> None:
    """Every child row points at a fake parent row."""
    for child, column, parent, ref in (
        ("Orders", "customer_id", "Customers", "id"),
        ("OrderLines", "order_id", "Orders", "id"),
    ):
        values = fake[child][column].drop_nulls()
        assert values.is_in(fake[parent][ref].cast(values.dtype).implode()).all()


@pytest.mark.parametrize(
    ("kwargs", "heights"),
    [
        ({}, (100, 300, 600)),
        ({"scale": 0.1}, (10, 30, 60)),
        ({"scale": 3}, (300, 900, 1800)),
        ({"n": 40}, (40, 40, 40)),
        ({"n": {"Orders": 50}}, (100, 50, 600)),
    ],
)
def test_sizes_and_keys(shop, kwargs, heights):
    fake = polspec.synthesize_all(shop, seed=1, **kwargs)
    assert list(fake) == ["Customers", "Orders", "OrderLines"]
    assert tuple(frame.height for frame in fake.values()) == heights
    assert all(fake[name].schema == shop[name].schema for name in shop)
    _held(fake)
    registry = polspec.profile_all(shop)
    registry.validate_all(fake, bounds=False)


def test_a_larger_set_widens_parent_and_child_together(shop):
    fake = polspec.synthesize_all(shop, scale=3, seed=2)
    assert fake["Customers"]["id"].max() > 100
    assert fake["Orders"]["customer_id"].max() > 100
    _held(fake)


def test_a_seed_repeats_the_frames(shop):
    first = polspec.synthesize_all(shop, scale=0.5, seed=7)
    again = polspec.synthesize_all(shop, scale=0.5, seed=7)
    other = polspec.synthesize_all(shop, scale=0.5, seed=8)
    assert all(first[name].equals(again[name]) for name in shop)
    assert not all(first[name].equals(other[name]) for name in shop)


def test_no_real_value_is_carried_where_replaced(shop):
    fake = polspec.synthesize_all(shop, seed=1, replace=["email"])
    assert not set(fake["Customers"]["email"]) & set(shop["Customers"]["email"])


def test_declared_keys_are_kept(shop):
    shop = {**shop, "Orders": shop["Orders"].rename({"customer_id": "buyer"})}
    keys = {"Orders": [ForeignKey("buyer", references="Customers", ref_columns="id")]}
    fake = polspec.synthesize_all(shop, scale=2, seed=1, keys=keys)
    assert fake["Orders"]["buyer"].is_in(fake["Customers"]["id"].implode()).all()


def test_keys_false_keeps_none(shop):
    fake = polspec.synthesize_all(shop, seed=1, keys=False)
    assert fake["Orders"]["customer_id"].max() <= 37


def test_a_key_too_wide_for_its_child_is_refused():
    frames = {
        "Customers": pl.DataFrame({"id": pl.Series(range(1, 101), dtype=pl.Int64)}),
        "Orders": pl.DataFrame(
            {"customer_id": pl.Series([1, 2, 3] * 10, dtype=pl.Int8)}
        ),
    }
    with pytest.raises(GenerationError, match=r"'Orders\.customer_id' is a Int8 key"):
        polspec.synthesize_all(frames, n={"Customers": 500})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n": 5, "scale": 0.5}, "not both"),
        ({"scale": 0}, "positive"),
        ({"scale": float("nan")}, "positive"),
        ({"scale": True}, "positive"),
        ({"n": -1}, ">= 0"),
        ({"n": {"Nope": 5}}, "not among the tables"),
    ],
)
def test_bad_sizes(shop, kwargs, match):
    with pytest.raises(ValueError, match=match):
        polspec.synthesize_all(shop, **kwargs)


@pytest.mark.parametrize("scale", [1, 0.1, 3])
def test_the_tutorials_shop_synthesizes_to_frames_its_registry_accepts(scale):
    """Every check on, foreign keys included; a larger set's keys outrun the
    profiled bounds on purpose, to have room, so those are left out there."""
    shop = {
        name: pl.read_csv(DOCS_DATA / f"{name.lower()}.csv", try_parse_dates=True)
        for name in ("Customers", "Orders", "Lines")
    }
    registry = polspec.profile_all(shop)
    fake = polspec.synthesize_all(shop, scale=scale, seed=1)
    registry.validate_all(fake, bounds=scale <= 1)
