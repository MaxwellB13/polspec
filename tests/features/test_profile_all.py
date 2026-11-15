"""`profile_all`: related tables profiled together, keys found by containment."""

from __future__ import annotations

import datetime as dt
import warnings

import polars as pl
import polspec
import pytest
from polspec import ForeignKey, Registry, SpecError


def _codes(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(n)]


def _keys(registry: Registry, table: str) -> set[tuple[str, str, str]]:
    return {
        (fk.columns[0], fk.references, fk.ref_columns[0])
        for fk in registry[table].foreign_keys
    }


def _quiet(frames, **kwargs) -> Registry:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return polspec.profile_all(frames, **kwargs)


def _passes(registry: Registry, frames) -> None:
    """The registry holds its own sources, and what it generates."""
    registry.validate_all(frames)
    sizes = {name: frame.height for name, frame in frames.items()}
    registry.validate_all(registry.generate_all(sizes, seed=3))


@pytest.fixture
def shop() -> dict[str, pl.DataFrame]:
    customers = pl.DataFrame(
        {"id": range(1, 101), "email": [f"c{i}@example.com" for i in range(100)]}
    )
    orders = pl.DataFrame(
        {
            "id": range(1, 301),
            "customer_id": [i % 37 + 1 for i in range(300)],
            "qty": [i % 5 + 1 for i in range(300)],
        }
    )
    lines = pl.DataFrame(
        {"order_id": [i % 300 + 1 for i in range(600)], "line": [1, 2] * 300}
    )
    return {"Customers": customers, "Orders": orders, "OrderLines": lines}


def test_keys_follow_containment_and_names(shop):
    registry = _quiet(shop)
    assert _keys(registry, "Customers") == set()
    assert _keys(registry, "Orders") == {("customer_id", "Customers", "id")}
    assert _keys(registry, "OrderLines") == {("order_id", "Orders", "id")}
    assert registry.names == ("Customers", "Orders", "OrderLines")
    _passes(registry, shop)


def test_child_takes_the_parents_claims(shop):
    registry = _quiet(shop)
    child = registry["Orders"]["customer_id"]
    # Profiled alone it would be (1, 37); the key fills it from all 100.
    assert child.bounds.min == 1 and child.bounds.max == 100
    assert child.unique is False
    assert child.description == "A key into Customers.id, inferred from its values."


def test_an_unrelated_integer_is_not_a_key(shop):
    # Orders.id and qty are contained in Customers.id, and named for nothing.
    registry = _quiet(shop)
    assert {fk.columns for fk in registry["Orders"].foreign_keys} == {("customer_id",)}
    assert registry["Orders"]["qty"].bounds.max == 5


@pytest.mark.parametrize(
    ("table", "column", "parent_column"),
    [
        ("Customers", "customer", "id"),
        ("Customers", "customer_id", "customer_id"),
        ("Categories", "category_id", "id"),
        ("Addresses", "address_id", "id"),
        ("OrderLines", "order_line_id", "id"),
        ("Person", "person_id", "id"),
        ("Statuses", "status_code", "code"),
    ],
)
def test_name_links(table, column, parent_column):
    frames = {
        table: pl.DataFrame({parent_column: range(1, 51)}),
        "Child": pl.DataFrame({column: [i % 20 + 1 for i in range(200)]}),
    }
    assert _keys(_quiet(frames), "Child") == {(column, table, parent_column)}


def test_a_shared_generic_name_is_no_link():
    frames = {
        "Countries": pl.DataFrame({"code": range(1, 51)}),
        "Shops": pl.DataFrame({"code": [1, 2, 3, 4] * 10}),
    }
    assert _keys(_quiet(frames), "Shops") == set()


def test_text_is_a_key_from_twenty_values():
    countries = pl.DataFrame({"iso": _codes("C", 40)})
    many = pl.DataFrame({"ship_to": _codes("C", 20) * 3})
    few = pl.DataFrame({"ship_to": _codes("C", 19) * 3})
    assert _keys(_quiet({"Countries": countries, "Shipments": many}), "Shipments") == {
        ("ship_to", "Countries", "iso")
    }
    assert (
        _keys(_quiet({"Countries": countries, "Shipments": few}), "Shipments") == set()
    )


def test_a_low_cardinality_text_key_is_text_not_an_enum():
    frames = {
        "Countries": pl.DataFrame({"country": _codes("C", 40)}),
        "Shops": pl.DataFrame({"country": _codes("C", 3) * 20}),
    }
    registry = _quiet(frames)
    assert _keys(registry, "Shops") == {("country", "Countries", "country")}
    assert registry["Shops"]["country"].dtype == pl.String
    _passes(registry, frames)


def test_an_orphan_means_no_key():
    frames = {
        "Customers": pl.DataFrame({"id": range(1, 51)}),
        "Orders": pl.DataFrame({"customer_id": [1, 2, 3, 99]}),
    }
    assert _keys(_quiet(frames), "Orders") == set()


def test_a_parent_with_a_repeat_or_a_null_is_no_parent():
    child = pl.DataFrame({"customer_id": [1, 2, 3, 2]})
    repeated = pl.DataFrame({"id": [1, 2, 3, 3]})
    nulled = pl.DataFrame({"id": [1, 2, 3, None]})
    for parent in (repeated, nulled):
        assert _keys(_quiet({"Customers": parent, "Orders": child}), "Orders") == set()


def test_a_nullable_key_keeps_its_nulls():
    frames = {
        "Customers": pl.DataFrame({"id": range(1, 51)}),
        "Orders": pl.DataFrame({"customer_id": [1, None, 2, None, 3, 4] * 10}),
    }
    registry = _quiet(frames)
    child = registry["Orders"]["customer_id"]
    assert child.nullable and child.null_probability == pytest.approx(1 / 3)
    _passes(registry, frames)


def test_a_one_to_one_key_stays_unique():
    frames = {
        "Customers": pl.DataFrame({"id": range(1, 501)}),
        "Profiles": pl.DataFrame({"customer_id": range(1, 301)}),
    }
    registry = _quiet(frames)
    assert _keys(registry, "Profiles") == {("customer_id", "Customers", "id")}
    assert registry["Profiles"]["customer_id"].unique
    _passes(registry, frames)


def test_two_unique_text_columns_with_the_same_values_are_no_key():
    frames = {
        "Left": pl.DataFrame({"a": _codes("X", 30)}),
        "Right": pl.DataFrame({"b": _codes("X", 30)}),
    }
    registry = _quiet(frames)
    assert _keys(registry, "Left") == _keys(registry, "Right") == set()


def test_a_self_reference():
    frames = {
        "Employees": pl.DataFrame(
            {"id": range(1, 41), "parent_id": [None] + [i // 2 + 1 for i in range(39)]}
        )
    }
    registry = _quiet(frames)
    assert _keys(registry, "Employees") == {("parent_id", "self", "id")}
    registry.validate_all(frames)


def test_a_child_with_two_linked_parents_warns_and_gets_none():
    frames = {
        "Customers": pl.DataFrame({"id": range(1, 51), "customer_id": range(1, 51)}),
        "Orders": pl.DataFrame({"customer_id": [1, 2, 3] * 5}),
    }
    with pytest.warns(UserWarning, match=r"Orders\.customer_id.*Customers\.id"):
        registry = polspec.profile_all(frames)
    assert _keys(registry, "Orders") == set()


def test_a_name_link_beats_containment_alone():
    frames = {
        "Countries": pl.DataFrame({"code": _codes("C", 40)}),
        "Regions": pl.DataFrame({"code": _codes("C", 40)[::-1], "n": range(40)}),
        "Shops": pl.DataFrame({"country_code": _codes("C", 25) * 2}),
    }
    registry = _quiet(frames)
    assert _keys(registry, "Shops") == {("country_code", "Countries", "code")}


def test_a_key_closing_a_cycle_warns_and_is_dropped():
    a, b = _codes("A", 30), _codes("B", 30)
    frames = {
        "First": pl.DataFrame({"code": a, "other": b[:25] + b[:5]}),
        "Second": pl.DataFrame({"code": b, "other": a[:25] + a[:5]}),
    }
    with pytest.warns(UserWarning, match="already depends on"):
        registry = polspec.profile_all(frames)
    assert _keys(registry, "First") == {("other", "Second", "code")}
    assert _keys(registry, "Second") == set()
    _passes(registry, frames)


def test_declared_keys_replace_inferred_ones():
    frames = {
        "Customers": pl.DataFrame({"id": range(1, 51), "alt": range(101, 151)}),
        "Orders": pl.DataFrame(
            {"buyer": [1, 2, 3] * 5, "customer_id": [101, 102, 103] * 5}
        ),
    }
    registry = _quiet(
        frames,
        keys={
            "Orders": [
                ForeignKey("buyer", references="Customers", ref_columns="id"),
                ForeignKey("customer_id", references="Customers", ref_columns="alt"),
            ]
        },
    )
    assert _keys(registry, "Orders") == {
        ("buyer", "Customers", "id"),
        ("customer_id", "Customers", "alt"),
    }
    assert registry["Orders"]["buyer"].description is None
    assert registry["Orders"]["buyer"].bounds.max == 50
    _passes(registry, frames)


def test_a_declared_composite_key():
    frames = {
        "Slots": pl.DataFrame({"day": [1, 1, 2, 2], "hour": [9, 10, 9, 10]}),
        "Bookings": pl.DataFrame({"day": [1, 2, 2], "hour": [9, 9, 10]}),
    }
    registry = _quiet(
        frames,
        keys={"Bookings": [ForeignKey(["day", "hour"], references="Slots")]},
    )
    (fk,) = registry["Bookings"].foreign_keys
    assert fk.columns == ("day", "hour")
    registry.validate_all(frames)


def test_a_declared_key_the_data_breaks_is_refused():
    frames = {
        "Customers": pl.DataFrame({"id": range(1, 11)}),
        "Orders": pl.DataFrame({"buyer": [1, 2, 99, 98]}),
    }
    key = ForeignKey("buyer", references="Customers", ref_columns="id")
    with pytest.raises(SpecError, match=r"2 value\(s\).*99"):
        polspec.profile_all(frames, keys={"Orders": [key]})


@pytest.mark.parametrize(
    ("keys", "match"),
    [
        ({"Nope": []}, "not one of the tables"),
        ({"Orders": [ForeignKey("buyer", references="Nope")]}, "not one of the tables"),
        (
            {"Orders": [ForeignKey("nope", references="Customers", ref_columns="id")]},
            "does not have",
        ),
    ],
)
def test_declared_keys_are_checked(keys, match):
    frames = {
        "Customers": pl.DataFrame({"id": [1, 2]}),
        "Orders": pl.DataFrame({"buyer": [1, 2]}),
    }
    with pytest.raises(SpecError, match=match):
        polspec.profile_all(frames, keys=keys)


def test_keys_false_infers_none(shop):
    registry = _quiet(shop, keys=False)
    assert all(not registry[name].foreign_keys for name in registry)


def test_keys_come_from_all_the_rows_claims_from_the_sample():
    customers = pl.DataFrame({"id": range(1, 2001)})
    orders = pl.DataFrame({"customer_id": [i % 1500 + 1 for i in range(3000)]})
    registry = _quiet({"Customers": customers, "Orders": orders}, sample=200, seed=1)
    assert _keys(registry, "Orders") == {("customer_id", "Customers", "id")}


def test_dates_and_narrow_integers():
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(60)]
    frames = {
        "Days": pl.DataFrame({"day": days}),
        "Customers": pl.DataFrame({"id": pl.Series(range(1, 61), dtype=pl.Int64)}),
        "Visits": pl.DataFrame(
            {
                "day": days[:30] * 2,
                "customer_id": pl.Series([1, 2, 3] * 20, dtype=pl.Int16),
            }
        ),
    }
    registry = _quiet(frames)
    assert _keys(registry, "Visits") == {
        ("day", "Days", "day"),
        ("customer_id", "Customers", "id"),
    }
    assert registry["Visits"]["customer_id"].dtype == pl.Int16
    _passes(registry, frames)


def test_parsed_text_keys():
    frames = {
        "Customers": pl.DataFrame({"id": [str(i) for i in range(1, 51)]}),
        "Orders": pl.DataFrame({"customer_id": ["1", "2", "", "3"] * 5}),
    }
    registry = _quiet(frames, parse_text=True)
    assert _keys(registry, "Orders") == {("customer_id", "Customers", "id")}
    assert registry["Orders"]["customer_id"].dtype == pl.Int64


def test_sources_of_every_kind(tmp_path, shop):
    path = tmp_path / "customers.parquet"
    shop["Customers"].write_parquet(path)
    frames = {"Customers": path, "Orders": shop["Orders"].lazy()}
    assert _keys(_quiet(frames), "Orders") == {("customer_id", "Customers", "id")}


def test_replace_need_only_name_one_table(shop):
    registry = _quiet(shop, replace=["email"])
    assert registry["Customers"]["email"].dtype == pl.String
    with pytest.raises(ValueError, match="none of the tables"):
        polspec.profile_all(shop, replace=["nope"])


@pytest.mark.parametrize("frames", [{}, [pl.DataFrame({"a": [1]})]])
def test_frames_must_be_a_mapping(frames):
    with pytest.raises(TypeError, match="non-empty mapping"):
        polspec.profile_all(frames)


def test_bad_arguments():
    with pytest.raises(TypeError, match="keys takes a mapping"):
        polspec.profile_all({"T": pl.DataFrame({"a": [1]})}, keys=[])
    with pytest.raises(ValueError, match="'T' has no rows"):
        polspec.profile_all({"T": pl.DataFrame({"a": []})})


def test_a_registry_file_round_trips(shop):
    registry = _quiet(shop)
    again = Registry.from_dict(registry.to_dict())
    assert _keys(again, "Orders") == _keys(registry, "Orders")
    _passes(again.resolve(), shop)
