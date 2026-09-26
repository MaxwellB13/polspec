"""The round-trip property: anything ``generate()`` produces, ``validate()`` accepts.

``FrameSpec`` declares each constraint once and acts on it twice -- once in the
generation path (``engine``/``rules``/``foreign_key``/``generation``) and once
in the validation path (``validation``). What both sides read from
``constraints`` cannot drift; the rest is held in step only by this module,
which asserts the invariant that ties them together::

    SpecCls.validate(SpecCls.generate(n, seed=...))   # must not raise

Cases that fail today carry ``@pytest.mark.xfail(strict=True)`` with the
finding they belong to. Strict matters: while a bug is present the case
reports XFAIL and the suite stays green, and the moment it is fixed the case
reports XPASS, which pytest turns into a *failure*. That is the signal to
delete the marker -- so this file tells you when each bug is genuinely fixed
rather than silently going quiet.

Three things sit deliberately outside the property, because generation makes
no claim to satisfy them; see the "boundaries" section at the bottom for the
tests that pin that down.
"""

import datetime as dt

import polars as pl
import pytest
from cases import COLUMN_CASES, EVERY_DTYPE
from polspec import (
    CatSpec,
    Check,
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSpec,
    GenerationError,
    SpecError,
    TableSpec,
    col,
    generate,
    validate,
)
from polspec.formats import FORMATS

ROWS = 300
SEED = 11


def _spec_for(name: str, column: ColSpec) -> type[FrameSpec]:
    """A single-column FrameSpec named after the case, with the column as `c`."""
    return type(f"RoundTrip_{name}", (FrameSpec,), {"c": column})


def assert_roundtrip(
    spec_cls: type[FrameSpec],
    *,
    n: int = ROWS,
    seed: int = SEED,
    method: str = "random",
    references=None,
    validate_checks: bool = True,
) -> pl.DataFrame:
    """Generates from `spec_cls` and asserts its own validate() accepts the result.

    Also materializes every value. `validate()` only inspects what the spec
    declares, so a column with no declared bounds can carry values that are
    not representable in its own dtype and still validate clean -- the
    conversion only fails when someone reads the frame.
    """
    df = spec_cls.generate(n, method=method, seed=seed, references=references)
    assert df.schema == spec_cls.schema()
    df.to_dicts()
    validated = spec_cls.validate(
        df, references=references, validate_checks=validate_checks
    )
    assert validated.equals(df)
    return df


# ---------------------------------------------------------------------------
# Column matrix -- one declared column per case (`cases.COLUMN_CASES`),
# round-tripped in isolation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(COLUMN_CASES))
def test_column_roundtrips(case):
    assert_roundtrip(_spec_for(case, COLUMN_CASES[case]))


def test_a_unique_column_with_no_room_refuses_by_name():
    """The one thing generation cannot do is invent a 301st Int8.

    A domain smaller than the frame is a contradiction in the declaration,
    not a gap in generation, so it is reported rather than quietly producing
    the duplicates the spec itself rejects.
    """
    spec_cls = _spec_for("narrow", ColSpec(pl.Int8, unique=True))
    with pytest.raises(GenerationError, match="only 256 distinct value"):
        spec_cls.generate(300, seed=SEED)
    # 256 rows is the whole domain exactly, and still round-trips.
    assert_roundtrip(spec_cls, n=256)


def test_a_unique_finite_format_covers_its_list_exactly_once():
    """A finite format is a domain like `choices`: `unique=True` can use all
    of it and not one row more, and the refusal counts the list.
    """
    currencies = len(FORMATS["iso_currency"].values)
    spec_cls = _spec_for(
        "currency", ColSpec(pl.String, format="iso_currency", unique=True)
    )
    with pytest.raises(GenerationError, match=f"only {currencies} distinct value"):
        spec_cls.generate(currencies + 1, seed=SEED)
    df = assert_roundtrip(spec_cls, n=currencies)
    assert df["c"].n_unique() == currencies


# ---------------------------------------------------------------------------
# Generation must fail cleanly, never panic across the FFI boundary.
# A Rust panic! surfaces as pyo3_runtime.PanicException, which callers cannot
# reasonably handle; an invalid spec should raise a normal Python error.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "bounds"),
    [
        (pl.Float32, (-1e40, 1e40)),
        (pl.Float32, (0.0, 1e39)),
        (pl.Int8, (0, 1_000)),
        (pl.UInt8, (-5, 10)),
        (pl.Int32, (0, 2**40)),
        (pl.Date, (0, 10**12)),
    ],
)
def test_bounds_wider_than_dtype_raise_cleanly(dtype, bounds):
    """C1: an unrepresentable bound is a declaration error, not a process abort."""
    with pytest.raises(ValueError, match="outside the range"):
        ColSpec(dtype, bounds=bounds)


@pytest.mark.parametrize("bounds", [(float("nan"), 1.0), (0.0, float("inf"))])
def test_non_finite_bounds_raise_cleanly(bounds):
    with pytest.raises(ValueError, match="finite"):
        ColSpec(pl.Float64, bounds=bounds)


def test_bounds_at_the_dtype_limit_are_accepted():
    """The check rejects what falls outside the domain, not what sits on its edge."""
    assert_roundtrip(_spec_for("int8_full", ColSpec(pl.Int8, bounds=(-128, 127))))
    assert_roundtrip(_spec_for("uint8_full", ColSpec(pl.UInt8, bounds=(0, 255))))
    assert_roundtrip(
        _spec_for("date_full", ColSpec(pl.Date, bounds=(dt.date.min, dt.date.max)))
    )


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Date,
        pl.Time,
        pl.Datetime("ms"),
        pl.Datetime("us"),
        pl.Datetime("ns"),
        pl.Duration("ms"),
        pl.Duration("us"),
        pl.Duration("ns"),
    ],
    ids=str,
)
@pytest.mark.parametrize("std", [1e3, 1e9, 1e18])
def test_distribution_on_temporal_column_stays_in_dtype_range(dtype, std):
    """C2: the distribution governs the shape; the dtype still governs the domain.

    Unbounded temporal columns reach the engine as a bare int32/int64, so a
    distribution wide enough to leave the dtype's range used to yield values
    that could not be read back -- an out-of-range date panicked outright.
    """
    assert_roundtrip(
        _spec_for(
            f"temporal_normal_{std:.0e}",
            ColSpec(
                dtype,
                distribution="normal",
                distribution_params={"mean": 0.0, "std": std},
            ),
        )
    )


def test_uint64_bounds_near_maximum_keep_precision():
    spec_cls = _spec_for(
        "uint64_huge",
        ColSpec(
            pl.UInt64,
            bounds=(18_446_744_073_709_551_600, 18_446_744_073_709_551_615),
        ),
    )
    df = spec_cls.generate(ROWS, seed=SEED)
    assert df["c"].n_unique() > 1


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


class SingleRuleSpec(FrameSpec):
    g = ColSpec(pl.Enum(["x", "y"]))
    v = ColSpec(
        pl.Int64,
        bounds=(0, 100),
        rules=[ColRule(when=col("g") == "x", choices=[7])],
    )


class MultiRuleSpec(FrameSpec):
    g = ColSpec(pl.Enum(["x", "y", "z"]))
    v = ColSpec(
        pl.String,
        choices=["a", "b", "c"],
        rules=[
            ColRule(when=col("g") == "x", choices=["a"]),
            ColRule(when=col("g").is_in(["y", "z"]), choices=["b"]),
        ],
    )


class ChainedRuleSpec(FrameSpec):
    a = ColSpec(pl.Enum(["x", "y"]))
    b = ColSpec(
        pl.Enum(["x", "y"]),
        rules=[ColRule(when=col("a") == "x", choices=["y"])],
    )
    c = ColSpec(
        pl.Enum(["x", "y"]),
        rules=[ColRule(when=col("b") == "y", choices=["x"])],
    )


def test_single_rule_roundtrips():
    assert_roundtrip(SingleRuleSpec)


def test_multiple_rules_on_one_column_roundtrip():
    assert_roundtrip(MultiRuleSpec)


def test_rule_roundtrips_under_cartesian():
    assert_roundtrip(SingleRuleSpec, n=50, method="cartesian")


def test_chained_rules_roundtrip():
    assert_roundtrip(ChainedRuleSpec)


def test_rules_that_each_read_the_other_are_refused():
    """No order runs both against their own inputs, so neither is allowed."""
    with pytest.raises(SpecError, match="Cannot order the generation passes"):

        class Circular(FrameSpec):
            a = ColSpec(
                pl.Enum(["x", "y"]),
                rules=[ColRule(when=col("b") == "x", choices=["y"])],
            )
            b = ColSpec(
                pl.Enum(["x", "y"]),
                rules=[ColRule(when=col("a") == "x", choices=["y"])],
            )


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------


class UniqueTogetherSpec(FrameSpec):
    a = ColSpec(pl.Enum(["x", "y"]))
    b = ColSpec(pl.Int64, bounds=(1, 10_000))
    __unique_together__ = [["a", "b"]]


class ThreeWayUniqueSpec(FrameSpec):
    a = ColSpec(pl.Enum(["x", "y"]))
    b = ColSpec(pl.Boolean)
    c = ColSpec(pl.String, choices=[f"s{i}" for i in range(500)])
    __unique_together__ = [["a", "b", "c"]]


class CrowdedUniqueTogetherSpec(FrameSpec):
    """Two by two: four combinations, however many rows are asked for."""

    a = ColSpec(pl.Enum(["x", "y"]))
    b = ColSpec(pl.Enum(["p", "q"]))
    __unique_together__ = [["a", "b"]]


def test_unique_together_roundtrips():
    assert_roundtrip(UniqueTogetherSpec)
    assert_roundtrip(ThreeWayUniqueSpec)


def test_unique_together_only_moves_the_rows_that_repeat():
    """A roomy domain keeps almost everything it generated."""
    df = UniqueTogetherSpec.generate(300, seed=SEED)
    assert df.select(["a", "b"]).n_unique() == 300


def test_unique_together_with_no_room_refuses_by_name():
    with pytest.raises(GenerationError, match="cannot be satisfied"):
        CrowdedUniqueTogetherSpec.generate(300, seed=SEED)
    # All four combinations fit in four rows.
    assert_roundtrip(CrowdedUniqueTogetherSpec, n=4)


def test_a_unique_member_already_makes_the_combination_distinct():
    class Spec(FrameSpec):
        a = ColSpec(pl.Enum(["x", "y"]))
        b = ColSpec(pl.Int64, bounds=(1, 100_000), unique=True)
        __unique_together__ = [["a", "b"]]

    assert_roundtrip(Spec)


# ---------------------------------------------------------------------------
# Foreign keys
# ---------------------------------------------------------------------------


class SelfRefSpec(FrameSpec):
    id = ColSpec(pl.Int64, bounds=(1, 50))
    mgr = ColSpec(pl.Int64, bounds=(1, 50), nullable=True)
    __foreign_keys__ = [ForeignKey("mgr", references="self", ref_columns="id")]


class ParentSpec(FrameSpec):
    k = ColSpec(pl.Int64, bounds=(100, 200))


class CompositeParentSpec(FrameSpec):
    k1 = ColSpec(pl.Int64, bounds=(1, 20))
    k2 = ColSpec(pl.String, choices=["a", "b"])


class CompositeChildSpec(FrameSpec):
    f1 = ColSpec(pl.Int64, bounds=(1, 20))
    f2 = ColSpec(pl.String, choices=["a", "b"])
    __foreign_keys__ = [
        ForeignKey(
            ["f1", "f2"], references=CompositeParentSpec, ref_columns=["k1", "k2"]
        )
    ]


class ChainedFkSpec(FrameSpec):
    a = ColSpec(pl.Int64, bounds=(100, 200))
    b = ColSpec(pl.Int64, bounds=(100, 200))
    __foreign_keys__ = [
        ForeignKey("a", references=ParentSpec, ref_columns="k", name="fk_a"),
        ForeignKey("b", references="self", ref_columns="a", name="fk_b"),
    ]


class TextualParentSpec(FrameSpec):
    code = ColSpec(pl.String, choices=["a", "b"])


class TextualChildSpec(FrameSpec):
    code = ColSpec(pl.Enum(["a", "b"]))
    __foreign_keys__ = [
        ForeignKey("code", references=TextualParentSpec, ref_columns="code")
    ]


def test_self_referencing_fk_roundtrips():
    assert_roundtrip(SelfRefSpec)


def test_composite_fk_roundtrips():
    parent = CompositeParentSpec.generate(200, seed=1)
    assert_roundtrip(CompositeChildSpec, references={CompositeParentSpec: parent})


def test_fk_whose_parent_does_not_fit_the_column_is_refused():
    """A key overwrites its column, so the parent's domain has to fit inside
    the column's own. Data that could not round-trip is refused as it is
    declared, rather than generated and then failed by its own spec.
    """
    with pytest.raises(SpecError, match="do not fit inside"):

        class WideningChild(FrameSpec):
            fk = ColSpec(pl.Int64, bounds=(1, 50))  # parent keys are 100..200
            __foreign_keys__ = [
                ForeignKey("fk", references=ParentSpec, ref_columns="k")
            ]

    with pytest.raises(SpecError, match="is not one of"):

        class NarrowChoices(FrameSpec):
            code = ColSpec(pl.String, choices=["a"])
            __foreign_keys__ = [
                ForeignKey("code", references=TextualParentSpec, ref_columns="code")
            ]


def test_fk_whose_parent_fits_is_accepted():
    class FittingChild(FrameSpec):
        fk = ColSpec(pl.Int64, bounds=(50, 500))
        __foreign_keys__ = [ForeignKey("fk", references=ParentSpec, ref_columns="k")]

    parent = ParentSpec.generate(200, seed=1)
    assert_roundtrip(FittingChild, references={ParentSpec: parent})


def test_chained_fk_roundtrips():
    parent = ParentSpec.generate(200, seed=1)
    assert_roundtrip(ChainedFkSpec, references={ParentSpec: parent})


def test_textual_fk_across_enum_and_string_roundtrips():
    parent = TextualParentSpec.generate(100, seed=1)
    assert_roundtrip(TextualChildSpec, references={TextualParentSpec: parent})


# ---------------------------------------------------------------------------
# Generation modes -- the property must hold however the rows were produced.
# ---------------------------------------------------------------------------


class MixedSpec(FrameSpec):
    e = ColSpec(pl.Enum(["x", "y"]))
    b = ColSpec(pl.Boolean)
    n = ColSpec(pl.Int64, bounds=(-10, 10), nullable=True)
    s = ColSpec(pl.String, string_length=(2, 4))


def test_random_roundtrips():
    assert_roundtrip(MixedSpec)


def test_cartesian_roundtrips():
    assert_roundtrip(MixedSpec, n=200, method="cartesian")


def test_a_lazy_frame_roundtrips():
    """`validate` takes a LazyFrame and hands one back -- from `scan()`, or
    from `.lazy()` on a generated frame."""
    assert MixedSpec.validate(MixedSpec.scan(ROWS, seed=SEED)).collect().height == ROWS
    eager = MixedSpec.generate(ROWS, seed=SEED)
    assert MixedSpec.validate(eager.lazy()).collect().height == ROWS


def test_every_batch_roundtrips():
    total = 0
    for batch in MixedSpec.generate_batches(500, batch_size=128, seed=SEED):
        MixedSpec.validate(batch)
        total += batch.height
    assert total == 500


def test_zero_rows_roundtrips():
    assert_roundtrip(MixedSpec, n=0)


def test_zero_rows_cartesian_is_empty():
    """C12, closed in 0.7.0: nothing asked for is nothing covered, the same
    empty typed frame `generate(0)` and the sinks produce."""
    df = MixedSpec.generate(0, method="cartesian", seed=SEED)
    assert df.height == 0 and df.schema == MixedSpec.schema()


# ---------------------------------------------------------------------------
# Spec transformations must not weaken the spec they transform.
# ---------------------------------------------------------------------------


class RetypedSpec(FrameSpec):
    code = ColSpec(
        pl.String,
        unique=True,
        string_length=(1, 1),
        choices=["a", "b", "c"],
    )
    amount = ColSpec(pl.Int64, bounds=(1, 10))


def test_with_catspec_preserves_column_constraints():
    """C10: re-typing changes the dtype and nothing else the column declared."""
    original = RetypedSpec.spec.columns["code"]
    retyped = RetypedSpec.with_catspec(CatSpec(enums={"code": ["a", "b", "c"]}))
    retyped = retyped.spec.columns["code"]

    assert isinstance(retyped.dtype, pl.Enum)
    for field in ("unique", "string_length", "nullable", "null_probability", "tags"):
        assert getattr(retyped, field) == getattr(original, field), field


def test_with_catspec_leaves_untouched_columns_identical():
    retyped = RetypedSpec.with_catspec(CatSpec(enums={"code": ["a", "b", "c"]}))
    assert retyped.spec.columns["amount"] is RetypedSpec.spec.columns["amount"]


def test_with_catspec_drops_weights_it_cannot_carry():
    """Weights are positional over a domain the re-type just resized."""

    class Weighted(FrameSpec):
        code = ColSpec(pl.String, choices=["a", "b", "c"], weights=[1.0, 1.0, 2.0])

    with pytest.warns(UserWarning) as caught:
        retyped = Weighted.with_catspec(CatSpec(enums={"code": ["a", "b"]}))

    # Narrowing the domain invalidates both, and each is reported on its own.
    messages = [str(w.message) for w in caught]
    assert any("dropping choices ['c']" in m for m in messages), messages
    assert any("dropping 3 weight" in m for m in messages), messages
    assert retyped.spec.columns["code"].weights is None


def test_with_catspec_drops_choices_the_new_dtype_cannot_hold():
    """Carrying them into the new ColSpec raised a confusing error instead."""

    class Narrow(FrameSpec):
        code = ColSpec(pl.String, choices=["a", "b", "c"])

    with pytest.warns(UserWarning, match=r"dropping choices \['c'\]"):
        retyped = Narrow.with_catspec(CatSpec(enums={"code": ["a", "b"]}))
    assert retyped.spec.columns["code"].choices is None
    assert_roundtrip(retyped)


def test_with_catspec_keeps_choices_the_new_dtype_still_covers():
    class Subset(FrameSpec):
        code = ColSpec(pl.String, choices=["a", "b"])

    retyped = Subset.with_catspec(CatSpec(enums={"code": ["a", "b", "c"]}))
    assert retyped.spec.columns["code"].choices == ("a", "b")
    assert set(assert_roundtrip(retyped)["code"].unique()) <= {"a", "b"}


def test_with_catspec_keeps_weights_that_still_fit():
    class Weighted(FrameSpec):
        code = ColSpec(pl.String, choices=["a", "b", "c"], weights=[1.0, 1.0, 2.0])

    retyped = Weighted.with_catspec(CatSpec(enums={"code": ["a", "b", "c"]}))
    assert retyped.spec.columns["code"].weights == (1.0, 1.0, 2.0)


@pytest.mark.parametrize("name", sorted(FORMATS))
def test_every_format_survives_a_file_round_trip(name, tmp_path):
    """generate -> validate -> to_yaml -> from_yaml -> generate -> validate.

    The sampler and the check for a format are one declaration in
    `polspec.formats`; the file is the one place a column's format is
    reduced to its name alone, so this is where the name has to be enough.
    """
    spec_cls = _spec_for(f"file_{name}", ColSpec(pl.String, format=name))
    assert_roundtrip(spec_cls)
    path = tmp_path / f"{name}.yaml"
    spec_cls.to_yaml(path)
    reloaded = FrameSpec.from_yaml(path)
    assert reloaded.spec.columns["c"].format == name
    assert_roundtrip(reloaded).equals(assert_roundtrip(spec_cls))


def test_yaml_roundtrip_preserves_the_property(tmp_path):
    path = tmp_path / "spec.yaml"
    MixedSpec.to_yaml(path)
    assert_roundtrip(FrameSpec.from_yaml(path))


def test_profiled_spec_roundtrips():
    source = MixedSpec.generate(ROWS, seed=SEED)
    assert_roundtrip(FrameSpec.from_dataframe(source))


# ---------------------------------------------------------------------------
# Boundaries of the property.
#
# Three things sit outside it on purpose. `__checks__` and `ColSpec.validators`
# wrap arbitrary polars expressions, and generation cannot in general produce
# data satisfying an arbitrary predicate, so it does not try. A self-
# referencing foreign key is the third and a different shape of boundary: what
# it promises is satisfied exactly, and the thing it does *not* promise is one
# people expect anyway.
#
# The tests below record those decisions so a future change to any of them is
# a deliberate one.
# ---------------------------------------------------------------------------


class CheckedSpec(FrameSpec):
    lo = ColSpec(pl.Int64, bounds=(0, 10))
    hi = ColSpec(pl.Int64, bounds=(0, 10))
    __checks__ = [Check(pl.col("hi") >= pl.col("lo"), name="hi_gte_lo")]


def test_frame_checks_are_validation_only():
    """__checks__ constrain validation; generation makes no attempt to satisfy them."""
    df = CheckedSpec.generate(ROWS, seed=SEED)
    CheckedSpec.validate(df, validate_checks=False)
    with pytest.raises(Exception, match="hi_gte_lo"):
        CheckedSpec.validate(df)


def test_column_validators_are_validation_only():
    """ColSpec.validators behave the same way: honoured on validate, not on generate."""
    spec_cls = _spec_for(
        "unsatisfiable_validator",
        ColSpec(pl.Int64, bounds=(1, 10), validators=[pl.col("c") > 100]),
    )
    df = spec_cls.generate(ROWS, seed=SEED)
    spec_cls.validate(df, validate_validators=False)
    with pytest.raises(Exception, match="validator"):
        spec_cls.validate(df)


@pytest.mark.parametrize(
    "name, well_formed",
    [
        ("email", "nobody@example.invalid"),
        ("ipv4", "0.0.0.0"),  # noqa: S104
        ("hostname", "localhost"),
        ("uuid4", "00000000-0000-4000-8000-000000000000"),
    ],
)
def test_a_format_promises_syntax_not_existence(name, well_formed):
    """`format="email"` is a well-formed address, not a deliverable one.

    Generation produces values of the right shape and validation checks the
    shape; neither side looks anything up. So a value that is syntactically
    perfect and points nowhere is accepted, and that is the boundary
    `limitations.md` states.
    """
    spec_cls = _spec_for(f"syntax_{name}", ColSpec(pl.String, format=name))
    spec_cls.validate(pl.DataFrame({"c": [well_formed]}))


def test_a_pattern_is_validation_only():
    """`pattern=` is the honest half of the split `format=` made: any regex
    can be checked, only a curated set can be generated. So a String column
    with a pattern is filled with ordinary random text, and the round trip
    holds only with the check switched off -- exactly as for `validators`.
    """
    spec_cls = _spec_for("patterned", ColSpec(pl.String, pattern=r"^[A-Z]{3}-\d{4}$"))
    df = spec_cls.generate(ROWS, seed=SEED)
    spec_cls.validate(df, validate_pattern=False)
    with pytest.raises(Exception, match="not matching pattern"):
        spec_cls.validate(df)


def test_a_seed_name_survives_a_rename_and_an_insertion():
    """A column's values depend on its seed name and nothing else.

    Its own seed comes from that name, and so does the seed of the rule pass
    that rewrites it -- keyed, not drawn in declaration order -- so a rules
    column inserted ahead of another leaves it alone, and a rules column
    renamed with `seed_name` keeps its rule's draw as well as its own.
    """
    rule = [ColRule(when=col("flag"), choices=tuple(range(50, 100)))]

    class Original(FrameSpec):
        flag = ColSpec(pl.Boolean)
        ruled = ColSpec(pl.Int64, bounds=(0, 100), rules=rule)

    class Renamed(FrameSpec):
        flag = ColSpec(pl.Boolean)
        renamed = ColSpec(pl.Int64, bounds=(0, 100), rules=rule, seed_name="ruled")

    class Inserted(FrameSpec):
        flag = ColSpec(pl.Boolean)
        inserted = ColSpec(pl.Int64, bounds=(0, 100), rules=rule)
        ruled = ColSpec(pl.Int64, bounds=(0, 100), rules=rule)

    original = Original.generate(ROWS, seed=SEED)["ruled"]
    assert Renamed.generate(ROWS, seed=SEED)["renamed"].equals(original)
    assert Inserted.generate(ROWS, seed=SEED)["ruled"].equals(original)


class HierarchySpec(FrameSpec):
    """A parent/child table: every row points at another row of the same table.

    `parent` is not nullable, which is what makes the test below deterministic
    rather than a matter of luck -- see its docstring.
    """

    ref = ColSpec(pl.Int64, unique=True, bounds=(1, 10_000))
    parent = ColSpec(pl.Int64)
    __foreign_keys__ = [ForeignKey("parent", references="self", ref_columns="ref")]


def _reaches_a_cycle(parent_of: dict, start) -> bool:
    """Whether walking parents from `start` ever revisits a row.

    Floyd's tortoise and hare, so a chain that never terminates is detected
    without holding the path.
    """
    slow = fast = start
    while True:
        slow = parent_of[slow]
        fast = parent_of[parent_of[fast]]
        if slow == fast:
            return True


def test_a_self_referencing_key_is_referential_not_acyclic():
    """A self-referencing ForeignKey guarantees every parent exists. It does
    not guarantee the result is a tree, and here it certainly is not.

    The key samples parents from the frame as it stands, which builds a random
    functional graph rather than a hierarchy. With `parent` non-nullable every
    row has an outgoing edge, and a finite graph in which every node has one
    must contain a cycle -- so this holds for every seed, not just this one.

    Both halves are asserted: the promise generation makes is kept, and the
    promise it does not make is visibly not kept. `validate()` accepts the
    result either way, because no part of a spec can currently say "acyclic".
    """
    df = HierarchySpec.generate(ROWS, seed=SEED)

    # The promise: every parent is a row of this same frame, and the keys are
    # distinct. This is what the ForeignKey and `unique=True` declare.
    refs = set(df["ref"].to_list())
    assert df["ref"].n_unique() == df.height
    assert set(df["parent"].to_list()) <= refs
    HierarchySpec.validate(df)

    # The boundary: parents form cycles, and nothing objects.
    parent_of = dict(zip(df["ref"].to_list(), df["parent"].to_list(), strict=True))
    assert all(_reaches_a_cycle(parent_of, r) for r in refs)


def test_a_hand_built_cycle_still_validates():
    """The boundary from the other side: cyclic data is not a finding.

    Constructed rather than generated, so this pins `validate()`'s behaviour
    even if generation stops producing cycles of its own accord.
    """
    cyclic = pl.DataFrame({"ref": [1, 2, 3], "parent": [2, 1, 3]})
    HierarchySpec.validate(cyclic)


# ---------------------------------------------------------------------------
# Which dtypes the property covers
#
# Every dtype that holds data (`cases.EVERY_DTYPE`), nested to any depth --
# and the census below fails the day Polars adds a dtype that list does not
# name, which is how Int128, UInt128 and Float16 went unnoticed through 0.9.0.
# ---------------------------------------------------------------------------

# Dtypes that hold no data of their own to generate: a column of nothing,
# arbitrary Python objects, a placeholder, and the extension mechanism.
# Declared and validated by dtype; generate() refuses each by name.
NOT_DATA = {"Null", "Object", "Unknown", "Extension", "BaseExtension"}

# Dtypes that hold data polspec does not generate yet: declared, validated,
# profiled and written to a spec file by dtype, refused by name at generate().
# Empty -- `Map`, which arrived with Polars 2, was the last -- and kept for
# the next dtype Polars adds.
NOT_YET: set[str] = set()


def test_every_polars_dtype_is_generated_or_named_as_not_data():
    import inspect as _inspect

    # By each class's own name, so an alias (`Utf8` is `String`) counts once.
    exported = {
        obj.__name__
        for obj in vars(pl).values()
        if _inspect.isclass(obj)
        and issubclass(obj, pl.DataType)
        and obj is not pl.DataType
        and not _inspect.isabstract(obj)
        and not obj.__name__.endswith("Type")
    }
    covered = {
        (dtype if isinstance(dtype, type) else type(dtype)).__name__
        for dtype in EVERY_DTYPE
    }
    assert exported - covered - NOT_DATA - NOT_YET == set(), (
        "a Polars dtype neither generates nor is named as holding no data, or "
        "as not generated yet"
    )


@pytest.mark.parametrize("name", sorted(NOT_DATA - {"Extension", "BaseExtension"}))
def test_a_dtype_that_holds_no_data_is_refused_by_name(name):
    spec = TableSpec("NotData", {"c": ColSpec(getattr(pl, name))})
    with pytest.raises(SpecError, match="cannot generate data for dtype"):
        generate(spec, ROWS, seed=SEED)


@pytest.mark.parametrize("dtype", EVERY_DTYPE, ids=str)
def test_every_dtype_generates_and_validates(dtype):
    """There is no dtype left that declares and cannot be generated."""
    spec = TableSpec("Every", {"c": ColSpec(dtype)})
    df = generate(spec, ROWS, seed=SEED)
    assert df.schema["c"] == ColSpec(dtype).dtype
    validate(spec, df)


def test_a_time_zoned_datetime_completes_the_round_trip():
    """The one people assume is on the list above. It is not."""
    spec_cls = _spec_for(
        "tz_datetime", ColSpec(pl.Datetime("us", "UTC"), nullable=True)
    )
    df = spec_cls.generate(ROWS, seed=SEED)
    assert df["c"].dtype == pl.Datetime("us", "UTC")
    spec_cls.validate(df)
