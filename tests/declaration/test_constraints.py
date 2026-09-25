"""The definitions generation and validation share.

`Domain` is what a column may hold, read from one declaration by both sides.
`Pass` and `order` decide which rewrite of a generated frame runs first, so
the data a spec produces satisfies the same claims validation checks it
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
    col,
)
from polspec.constraints import Domain, Pass, order, ordered_passes, passes_of

# ---------------------------------------------------------------------------
# Domain: what a column is declared to hold
# ---------------------------------------------------------------------------


def test_domain_reads_choices_bounds_and_categories():
    assert Domain.of(ColSpec(pl.Int64)).is_open
    assert Domain.of(ColSpec(pl.Int64, bounds=(None, None))).is_open

    numeric = Domain.of(ColSpec(pl.Int64, bounds=(1, 10)))
    assert numeric.values is None
    assert (numeric.bounds.min, numeric.bounds.max) == (1, 10)

    assert Domain.of(ColSpec(pl.String, choices=["a", "b"])).values == ("a", "b")
    assert Domain.of(ColSpec(pl.Enum(["A", "B"]))).values == ("A", "B")
    # choices narrow an Enum rather than replacing it
    assert Domain.of(ColSpec(pl.Enum(["A", "B"]), choices=["B"])).values == ("B",)


def test_domain_describes_itself_for_an_error_message():
    assert str(Domain.of(ColSpec(pl.Int64, bounds=(1, 10)))) == "bounds [1, 10]"
    assert str(Domain.of(ColSpec(pl.String, choices=["a"]))) == "one of ['a']"
    assert str(Domain.of(ColSpec(pl.Int64))) == "any Int64"


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
def test_domain_knows_whether_a_parents_values_fit(child, parent, fits):
    assert (Domain.of(child).rejects(Domain.of(parent)) is None) is fits


def test_textual_domains_compare_by_the_string_they_hold():
    enum_side = Domain.of(ColSpec(pl.Enum(["A", "B"])))
    string_side = Domain.of(ColSpec(pl.String, choices=["A", "B"]))
    assert enum_side.rejects(string_side) is None
    assert string_side.rejects(enum_side) is None

    narrower = Domain.of(ColSpec(pl.String, choices=["A", "C"]))
    assert enum_side.rejects(narrower) is not None


def test_declarations_that_cannot_be_compared_prove_nothing():
    # Nothing is guessed from values the two sides cannot be ranked by.
    child = Domain.of(ColSpec(pl.Int64, bounds=(1, 10)))
    assert child.rejects(Domain(dtype=pl.Int64, values=("x",))) is None


# ---------------------------------------------------------------------------
# Passes: what each rewrite reads and writes
# ---------------------------------------------------------------------------


class Ordered(FrameSpec):
    region = ColSpec(pl.Enum(["UK", "US"]))
    carrier = ColSpec(
        pl.Enum(["RM", "UPS"]),
        rules=[ColRule(when=col("region") == "UK", choices=["RM"])],
    )
    plain = ColSpec(pl.Int64, bounds=(0, 10))


def test_a_pass_reading_what_it_writes_does_not_depend_on_itself():
    # A rule keyed on its own column reads the values it is about to replace.
    # That is not a dependency on any other pass -- but if something *else*
    # writes that column, this pass has to run after it.
    self_reading = Pass(
        key="rules:n", label="the rules on 'n'", writes={"n"}, reads={"n", "g"}
    )
    assert self_reading.reads == frozenset({"n", "g"})
    assert order([self_reading], "S") == [self_reading]

    earlier = Pass(key="fk:n", label="the foreign key on 'n'", writes={"n"})
    assert [p.key for p in order([self_reading, earlier], "S")] == [
        "fk:n",
        "rules:n",
    ]


def test_passes_come_from_rules_and_foreign_keys():
    passes = {p.key: p for p in passes_of(Ordered.spec)}
    assert set(passes) == {"rules:carrier"}
    assert passes["rules:carrier"].writes == frozenset({"carrier"})
    assert passes["rules:carrier"].reads == frozenset({"region"})


def test_a_cross_spec_key_reads_nothing_local_and_a_self_key_does():
    class Parent(FrameSpec):
        k = ColSpec(pl.Int64, bounds=(1, 50))

    class Child(FrameSpec):
        a = ColSpec(pl.Int64, bounds=(1, 50))
        b = ColSpec(pl.Int64, bounds=(1, 50), nullable=True)
        __foreign_keys__ = [
            ForeignKey("a", references=Parent, ref_columns="k", name="fk_a"),
            ForeignKey("b", references="self", ref_columns="a", name="fk_b"),
        ]

    passes = {p.key: p for p in passes_of(Child.spec)}
    assert passes["fk:fk_a"].reads == frozenset()
    assert passes["fk:fk_b"].reads == frozenset({"a"})
    # so the key drawing from `a` runs after the one that fills it
    assert [p.key for p in ordered_passes(Child.spec)] == ["fk:fk_a", "fk:fk_b"]


def test_order_puts_writers_before_readers_and_keeps_declaration_order():
    declared = [
        Pass(key="third", label="third", writes={"c"}, reads={"b"}),
        Pass(key="first", label="first", writes={"a"}),
        Pass(key="second", label="second", writes={"b"}, reads={"a"}),
        Pass(key="independent", label="independent", writes={"z"}),
    ]
    # Among the passes ready to run, the earliest declared goes first, so
    # the sequence is the declaration order a dependency does not disturb.
    assert [p.key for p in order(declared, "S")] == [
        "first",
        "second",
        "third",
        "independent",
    ]


def test_order_refuses_a_cycle():
    cycle = [
        Pass(key="a", label="the rules on 'a'", writes={"a"}, reads={"b"}),
        Pass(key="b", label="the rules on 'b'", writes={"b"}, reads={"a"}),
    ]
    with pytest.raises(SpecError, match="Cannot order the generation passes on S"):
        order(cycle, "S")


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

    assert [p.key for p in ordered_passes(Child.spec)] == [
        "fk:fk_amount__FkParent",
        "rules:tier",
    ]
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

    passes = {p.key: p for p in passes_of(Child.spec)}
    repair = passes["unique_together:0"]
    # It reads both members -- so it runs after the key that fills `a` -- but
    # writes only `b`, since rewriting `a` would break that key.
    assert repair.reads == frozenset({"a", "b"})
    assert repair.writes == frozenset({"b"})
    assert [p.key for p in ordered_passes(Child.spec)] == [
        "fk:fk_a__Parent",
        "unique_together:0",
    ]

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

    assert passes_of(Child.spec)[-1].writes == frozenset()
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
