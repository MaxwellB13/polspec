"""Every pass is seeded from the frame seed and a key naming what it is
for, so the columns declared around a pass never change what it draws.

The column seeds always worked that way (`sample.rs::seed_for_column`); the
Python-side passes drew theirs in declaration order until 0.7.0. The port
that keys them is pinned bit for bit against the engine's golden value, so
"keyed by name" means one thing on both sides of the boundary.
"""

from __future__ import annotations

import polars as pl
from polspec import ColRule, ColSpec, ForeignKey, FrameSpec, Hierarchy, col
from polspec.generation.seeds import pass_seed

ROWS, SEED = 300, 11
RULE = [ColRule(when=col("flag"), choices=tuple(range(50, 100)))]


def test_the_port_matches_the_engine_bit_for_bit():
    # The golden value `src/sample.rs` pins for seed_for_column(42, "order_id").
    assert pass_seed(42, "order_id") == 26_122_605_474_442_453
    assert pass_seed(1, "a") == pass_seed(1, "a")
    assert pass_seed(1, "a") != pass_seed(1, "b")
    assert pass_seed(1, "a") != pass_seed(2, "a")
    assert 0 <= pass_seed(2**63 - 1, "x") < 2**64


def _column(spec_cls: type[FrameSpec], name: str) -> pl.Series:
    return spec_cls.generate(ROWS, seed=SEED)[name]


def test_an_inserted_rules_column_leaves_its_neighbour_alone():
    class Original(FrameSpec):
        flag = ColSpec(pl.Boolean)
        ruled = ColSpec(pl.Int64, bounds=(0, 100), rules=RULE)

    class Inserted(FrameSpec):
        flag = ColSpec(pl.Boolean)
        inserted = ColSpec(pl.Int64, bounds=(0, 100), rules=RULE)
        ruled = ColSpec(pl.Int64, bounds=(0, 100), rules=RULE)

    assert _column(Inserted, "ruled").equals(_column(Original, "ruled"))


def test_an_added_foreign_key_leaves_the_others_alone():
    class Parent(FrameSpec):
        id = ColSpec(pl.Int64, unique=True, bounds=(1, 50))

    parent = Parent.generate(50, seed=1)

    class One(FrameSpec):
        a = ColSpec(pl.Int64)
        __foreign_keys__ = [ForeignKey("a", references=Parent, ref_columns="id")]

    class Two(FrameSpec):
        z = ColSpec(pl.Int64)
        a = ColSpec(pl.Int64)
        __foreign_keys__ = [
            ForeignKey("z", references=Parent, ref_columns="id"),
            ForeignKey("a", references=Parent, ref_columns="id"),
        ]

    one = One.generate(ROWS, seed=SEED, references={Parent: parent})["a"]
    two = Two.generate(ROWS, seed=SEED, references={Parent: parent})["a"]
    assert two.equals(one)


def test_a_composite_key_is_keyed_by_its_members_not_its_position():
    class One(FrameSpec):
        a = ColSpec(pl.Int64, bounds=(0, 5))
        b = ColSpec(pl.Int64, bounds=(0, 1_000_000))
        __unique_together__ = [("a", "b")]

    class Two(FrameSpec):
        a = ColSpec(pl.Int64, bounds=(0, 5))
        b = ColSpec(pl.Int64, bounds=(0, 1_000_000))
        c = ColSpec(pl.Int64, bounds=(0, 1_000_000))
        d = ColSpec(pl.Int64, bounds=(0, 1_000_000))
        __unique_together__ = [("c", "d"), ("b", "a")]

    assert (
        Two.generate(ROWS, seed=SEED)
        .select("a", "b")
        .equals(One.generate(ROWS, seed=SEED).select("a", "b"))
    )


def test_a_hierarchy_is_unmoved_by_a_column_added_beside_it():
    class One(FrameSpec):
        ref = ColSpec(pl.Int64)
        parent = ColSpec(pl.Int64)
        __hierarchy__ = Hierarchy(child="ref", parent="parent", max_depth=4)

    class Two(FrameSpec):
        flag = ColSpec(pl.Boolean)
        extra = ColSpec(pl.Int64, bounds=(0, 100), rules=RULE)
        ref = ColSpec(pl.Int64)
        parent = ColSpec(pl.Int64)
        __hierarchy__ = Hierarchy(child="ref", parent="parent", max_depth=4)

    assert (
        Two.generate(ROWS, seed=SEED)
        .select("ref", "parent")
        .equals(One.generate(ROWS, seed=SEED).select("ref", "parent"))
    )


def test_a_bounded_categorical_pool_is_keyed_by_the_column():
    class One(FrameSpec):
        c = ColSpec(pl.Categorical(pl.Categories("cats", physical=pl.UInt8)))

    class Two(FrameSpec):
        z = ColSpec(pl.Categorical(pl.Categories("other", physical=pl.UInt8)))
        c = ColSpec(pl.Categorical(pl.Categories("cats", physical=pl.UInt8)))

    assert _column(Two, "c").cast(pl.String).equals(_column(One, "c").cast(pl.String))


def test_cartesian_representatives_are_keyed_by_the_column():
    class One(FrameSpec):
        n = ColSpec(pl.Int64, bounds=(-100, 100))

    class Two(FrameSpec):
        m = ColSpec(pl.Float64, bounds=(-1.0, 1.0))
        n = ColSpec(pl.Int64, bounds=(-100, 100))

    one = One.generate(1, seed=SEED, method="cartesian")["n"].unique().sort()
    two = Two.generate(1, seed=SEED, method="cartesian")["n"].unique().sort()
    assert two.equals(one)


def test_seed_name_carries_the_rule_pass_too():
    class Original(FrameSpec):
        flag = ColSpec(pl.Boolean)
        ruled = ColSpec(pl.Int64, bounds=(0, 100), rules=RULE)

    class Renamed(FrameSpec):
        flag = ColSpec(pl.Boolean)
        other = ColSpec(pl.Int64, bounds=(0, 100), rules=RULE, seed_name="ruled")

    assert _column(Renamed, "other").equals(_column(Original, "ruled"))


def test_a_plain_column_is_what_it_was():
    """The frame seed is still the first draw of the caller's seed, so a
    column with no pass over it produces exactly the 0.6 values."""

    class S(FrameSpec):
        n = ColSpec(pl.Int64, bounds=(0, 1_000_000))

    # The first five values at this seed, as 0.6.0 produced them.
    assert S.generate(5, seed=42)["n"].to_list() == [
        162650,
        591325,
        734681,
        365482,
        294775,
    ]
