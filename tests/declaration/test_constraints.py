"""The definitions generation and validation share.

What a column may hold is read from one declaration by both sides, so a
foreign key whose parent holds values its column refuses is refused as it
is declared. The rewrites of a generated frame run writers before readers,
so the data a spec produces satisfies the same claims validation checks it
against. These tests pin both, and the round-trips they make possible.
"""

import polars as pl
import pytest
from polspec import (
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSpec,
    GenerationError,
    SpecError,
    TableSpec,
    col,
)

# ---------------------------------------------------------------------------
# What a column is declared to hold, and whether a parent's values fit it
# ---------------------------------------------------------------------------


def _keyed(child: ColSpec, parent: ColSpec) -> SpecError | None:
    """The refusal of a key filling `child` from `parent`, if it is refused."""
    parent_spec = TableSpec("Parent", {"k": parent})
    try:
        TableSpec(
            "Child",
            {"k": child},
            foreign_keys=[ForeignKey("k", references=parent_spec, ref_columns="k")],
        )
    except SpecError as refused:
        return refused
    return None


@pytest.mark.parametrize(
    ("child", "parent", "fits"),
    [
        (ColSpec(pl.Int64, bounds=(1, 100)), ColSpec(pl.Int64, bounds=(10, 20)), True),
        (ColSpec(pl.Int64, bounds=(1, 100)), ColSpec(pl.Int64, bounds=(1, 100)), True),
        (
            ColSpec(pl.Int64, bounds=(1, 100)),
            ColSpec(pl.Int64, bounds=(50, 500)),
            False,
        ),
        (ColSpec(pl.Int64, bounds=(1, 100)), ColSpec(pl.Int64), False),
        # an open end on either side is unbounded in that direction
        (
            ColSpec(pl.Int64, bounds=(1, None)),
            ColSpec(pl.Int64, bounds=(5, None)),
            True,
        ),
        (
            ColSpec(pl.Int64, bounds=(1, None)),
            ColSpec(pl.Int64, bounds=(None, 9)),
            False,
        ),
        # a column that declares nothing accepts anything
        (ColSpec(pl.Int64), ColSpec(pl.Int64, bounds=(1, 5)), True),
        (ColSpec(pl.Int64), ColSpec(pl.Int64), True),
        # finite domains
        (
            ColSpec(pl.String, choices=["a", "b"]),
            ColSpec(pl.String, choices=["a"]),
            True,
        ),
        (
            ColSpec(pl.String, choices=["a", "b"]),
            ColSpec(pl.String, choices=["a", "z"]),
            False,
        ),
        (ColSpec(pl.String, choices=["a"]), ColSpec(pl.String), False),
        # a finite parent has to sit inside a bounded child
        (ColSpec(pl.Int64, bounds=(1, 10)), ColSpec(pl.Int64, choices=[2, 4]), True),
        (ColSpec(pl.Int64, bounds=(1, 10)), ColSpec(pl.Int64, choices=[2, 40]), False),
    ],
)
def test_a_key_is_declared_only_where_the_parents_values_fit(child, parent, fits):
    assert (_keyed(child, parent) is None) is fits


@pytest.mark.parametrize(
    ("child", "parent", "says"),
    [
        (ColSpec(pl.Int64, bounds=(1, 10)), ColSpec(pl.Int64), "an unbounded Int64"),
        (ColSpec(pl.Int64, bounds=(1, 10)), ColSpec(pl.Int64), "bounds [1, 10]"),
        (
            ColSpec(pl.String, choices=["a"]),
            ColSpec(pl.String, choices=["a", "z"]),
            "one of ['a']",
        ),
    ],
)
def test_a_refused_key_names_both_sides(child, parent, says):
    refused = _keyed(child, parent)
    assert refused is not None and says in str(refused)


def test_an_enum_key_and_a_string_column_compare_by_the_text_they_hold():
    enum_side = ColSpec(pl.Enum(["A", "B"]))
    string_side = ColSpec(pl.String, choices=["A", "B"])
    assert _keyed(enum_side, string_side) is None
    assert _keyed(string_side, enum_side) is None
    # choices narrow an Enum rather than replacing it
    assert _keyed(ColSpec(pl.Enum(["A", "B"]), choices=["B"]), enum_side) is not None
    assert _keyed(enum_side, ColSpec(pl.String, choices=["A", "C"])) is not None


# ---------------------------------------------------------------------------
# The rewrites run writers before readers
# ---------------------------------------------------------------------------


def test_rules_that_read_each_other_cannot_be_ordered():
    with pytest.raises(SpecError, match="Cannot order the generation passes on S"):
        TableSpec(
            "S",
            {
                "a": ColSpec(
                    pl.Enum(["x", "y"]),
                    rules=[ColRule(when=col("b") == "x", choices=["y"])],
                ),
                "b": ColSpec(
                    pl.Enum(["x", "y"]),
                    rules=[ColRule(when=col("a") == "x", choices=["y"])],
                ),
            },
        )


def test_a_self_key_draws_from_the_column_another_key_fills():
    """`b` takes its values from `a`, which a key fills from the parent; so
    `b` is drawn after `a` is filled, and holds only the parent's keys."""

    class Parent(FrameSpec):
        k = ColSpec(pl.Int64, bounds=(40, 50))

    class Child(FrameSpec):
        a = ColSpec(pl.Int64, bounds=(1, 50))
        b = ColSpec(pl.Int64, bounds=(1, 50), nullable=True)
        __foreign_keys__ = [
            ForeignKey("a", references=Parent, ref_columns="k", name="fk_a"),
            ForeignKey("b", references="self", ref_columns="a", name="fk_b"),
        ]

    parent = Parent.generate(11, seed=1)
    df = Child.generate(300, seed=2, references={Parent: parent})
    assert set(df["b"].drop_nulls().to_list()) <= set(parent["k"].to_list())
    Child.validate(df, references={Parent: parent})


# ---------------------------------------------------------------------------
# What the ordering buys: passes see each other's work
# ---------------------------------------------------------------------------


class FkParent(FrameSpec):
    k = ColSpec(pl.Int64, bounds=(160, 200))


def test_a_rule_keyed_on_a_foreign_keyed_column_sees_the_parents_values():
    """The interaction the ordering exists for.

    `tier` is decided by `amount`, which the key overwrites. Reading the
    freely generated `amount` would set `tier` from values the frame no
    longer holds, and validation -- which reads the final frame -- would
    flag every row the key moved across the threshold.
    """

    class Child(FrameSpec):
        amount = ColSpec(pl.Int64, bounds=(1, 1_000))
        tier = ColSpec(
            pl.Enum(["low", "high"]),
            rules=[ColRule(when=col("amount") > 150, choices=["high"])],
        )
        __foreign_keys__ = [ForeignKey("amount", references=FkParent, ref_columns="k")]

    parent = FkParent.generate(50, seed=1)
    df = Child.generate(500, seed=3, references={FkParent: parent})
    # every amount now comes from the parent, so every tier is "high"
    assert df["amount"].min() >= 160
    assert set(df["tier"].to_list()) == {"high"}
    Child.validate(df, references={FkParent: parent})


def test_supplying_references_does_not_reshuffle_the_other_columns():
    """Seeds are drawn per pass in declaration order, so a key the caller
    gave no data for still costs its draw and leaves its neighbours alone.
    """

    class Child(FrameSpec):
        amount = ColSpec(pl.Int64, bounds=(1, 1_000))
        note = ColSpec(
            pl.Enum(["x", "y"]),
            rules=[ColRule(when=col("amount") > 500, choices=["y"])],
        )
        __foreign_keys__ = [ForeignKey("amount", references=FkParent, ref_columns="k")]

    without = Child.generate(200, seed=7)
    with_parent = Child.generate(
        200, seed=7, references={FkParent: FkParent.generate(50, seed=1)}
    )
    # `amount` is replaced, but the rule on `note` still drew the same seed
    assert not without["amount"].equals(with_parent["amount"])
    assert without["note"].dtype == with_parent["note"].dtype


# ---------------------------------------------------------------------------
# Uniqueness: drawn without replacement, repaired where it is composite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column",
    [
        ColSpec(pl.Int64, unique=True),
        ColSpec(pl.Int16, bounds=(1, 400), unique=True),
        ColSpec(pl.UInt16, unique=True),
        ColSpec(pl.Float64, unique=True),
        ColSpec(pl.String, unique=True),
        ColSpec(pl.String, choices=[f"c{i}" for i in range(500)], unique=True),
        ColSpec(pl.Enum([f"e{i}" for i in range(500)]), unique=True),
        ColSpec(pl.Datetime("us"), unique=True),
    ],
)
def test_a_unique_column_repeats_nothing(column):
    spec = type("U", (FrameSpec,), {"__columns__": {"c": column}})
    values = spec.generate(300, seed=4)["c"]
    assert values.n_unique() == 300


def test_nulls_do_not_count_as_repeated_values():
    """A null means "no value", so two of them are not two rows sharing one."""

    class Spec(FrameSpec):
        c = ColSpec(
            pl.Int32, bounds=(1, 400), unique=True, nullable=True, null_probability=0.4
        )

    values = Spec.generate(400, seed=6)["c"]
    assert values.null_count() > 0
    present = values.drop_nulls()
    assert present.n_unique() == len(present)
    Spec.validate(Spec.generate(400, seed=6))


def test_unique_refuses_what_it_cannot_be_combined_with():
    """Nothing that describes how often a value recurs survives a draw
    without replacement, and a rule would put the repeats back.
    """
    with pytest.raises(SpecError, match="carry weights"):
        ColSpec(
            pl.String, choices=["a", "b", "c"], weights=[1.0, 2.0, 3.0], unique=True
        )
    with pytest.raises(SpecError, match="carry distribution"):
        ColSpec(pl.Int64, distribution="normal", unique=True)
    with pytest.raises(SpecError, match="carry rules"):
        ColSpec(
            pl.String,
            unique=True,
            rules=[ColRule(when=col("g") == "x", choices=["a"])],
        )


def test_a_composite_key_orders_after_what_writes_its_columns():
    class Parent(FrameSpec):
        k = ColSpec(pl.Int64, bounds=(1, 5_000))

    class Child(FrameSpec):
        a = ColSpec(pl.Int64, bounds=(1, 5_000))
        b = ColSpec(pl.Int64, bounds=(1, 5_000))
        __foreign_keys__ = [ForeignKey("a", references=Parent, ref_columns="k")]
        __unique_together__ = [["a", "b"]]

    # The repair reads both members -- so it runs after the key that fills
    # `a` -- but rewrites only `b`, since rewriting `a` would break that key.

    parent = Parent.generate(200, seed=1)
    df = Child.generate(300, seed=2, references={Parent: parent})
    assert df.select(["a", "b"]).n_unique() == 300
    assert set(df["a"].to_list()) <= set(parent["k"].to_list())
    Child.validate(df, references={Parent: parent})


def test_a_composite_key_of_only_foreign_keyed_columns_says_so():
    class Parent(FrameSpec):
        k1 = ColSpec(pl.Int64, bounds=(1, 3))
        k2 = ColSpec(pl.Int64, bounds=(1, 3))

    class Child(FrameSpec):
        a = ColSpec(pl.Int64, bounds=(1, 3))
        b = ColSpec(pl.Int64, bounds=(1, 3))
        __foreign_keys__ = [
            ForeignKey(["a", "b"], references=Parent, ref_columns=["k1", "k2"])
        ]
        __unique_together__ = [["a", "b"]]

    with pytest.raises(GenerationError, match="every column in it is foreign-keyed"):
        Child.generate(300, seed=1, references={Parent: Parent.generate(50, seed=1)})


def test_a_key_filling_a_unique_column_needs_a_parent_that_can():
    """A unique column's values all come from the parent, so the parent has
    to hold at least as many distinct ones as there are rows.
    """

    class Parent(FrameSpec):
        k = ColSpec(pl.Int64, bounds=(1, 50_000), unique=True)

    class Child(FrameSpec):
        fk = ColSpec(pl.Int64, bounds=(1, 50_000), unique=True)
        __foreign_keys__ = [ForeignKey("fk", references=Parent, ref_columns="k")]

    with pytest.raises(GenerationError, match="offers only 20 distinct value"):
        Child.generate(100, seed=2, references={Parent: Parent.generate(20, seed=1)})

    parent = Parent.generate(500, seed=1)
    df = Child.generate(100, seed=2, references={Parent: parent})
    assert df["fk"].n_unique() == 100
    assert set(df["fk"].to_list()) <= set(parent["k"].to_list())
    Child.validate(df, references={Parent: parent})
