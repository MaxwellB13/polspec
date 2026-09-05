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

Two constraint kinds sit deliberately outside the property, because
generation makes no claim to satisfy them; see the "boundaries" section at
the bottom for the tests that pin that down.
"""

import datetime as dt

import polars as pl
import pytest
from polspec import (
    CatSpec,
    Check,
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSpec,
    GenerationError,
    SpecError,
)

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
# Column matrix -- one declared column per case, round-tripped in isolation.
# ---------------------------------------------------------------------------

COLUMN_CASES: dict[str, ColSpec] = {
    # integers
    "int8": ColSpec(pl.Int8),
    "int16": ColSpec(pl.Int16),
    "int32": ColSpec(pl.Int32),
    "int64": ColSpec(pl.Int64),
    "uint8": ColSpec(pl.UInt8),
    "uint16": ColSpec(pl.UInt16),
    "uint32": ColSpec(pl.UInt32),
    "uint64": ColSpec(pl.UInt64),
    "int64_bounded": ColSpec(pl.Int64, bounds=(-100, 100)),
    "int64_nullable": ColSpec(pl.Int64, nullable=True, null_probability=0.3),
    "int64_always_null": ColSpec(pl.Int64, nullable=True, null_probability=1.0),
    "int64_lower_open": ColSpec(pl.Int64, bounds=(0, None)),
    "int64_upper_open": ColSpec(pl.Int64, bounds=(None, 0)),
    "int64_lower_open_nullable": ColSpec(pl.Int64, bounds=(0, None), nullable=True),
    # floats
    "float32": ColSpec(pl.Float32),
    "float64": ColSpec(pl.Float64),
    "float32_bounded": ColSpec(pl.Float32, bounds=(-1.0, 1.0)),
    "float64_bounded": ColSpec(pl.Float64, bounds=(-2.5, 2.5)),
    "float64_lower_open": ColSpec(pl.Float64, bounds=(0.0, None)),
    # boolean
    "bool": ColSpec(pl.Boolean),
    "bool_weighted": ColSpec(pl.Boolean, weights=[0.3, 0.7]),
    # text and bytes
    "string": ColSpec(pl.String),
    "string_len": ColSpec(pl.String, string_length=(3, 8)),
    "string_choices": ColSpec(pl.String, choices=["a", "b", "c"]),
    "string_choices_weighted": ColSpec(pl.String, choices={"a": 1.0, "b": 2.0}),
    "binary": ColSpec(pl.Binary),
    "binary_len": ColSpec(pl.Binary, string_length=(2, 4)),
    # temporal
    "date": ColSpec(pl.Date),
    "date_bounded": ColSpec(
        pl.Date, bounds=(dt.date(2020, 1, 1), dt.date(2021, 12, 31))
    ),
    "date_lower_open": ColSpec(pl.Date, bounds=(dt.date(2020, 1, 1), None)),
    "time": ColSpec(pl.Time),
    "datetime_us": ColSpec(pl.Datetime("us")),
    "datetime_ns": ColSpec(pl.Datetime("ns")),
    "datetime_bounded": ColSpec(
        pl.Datetime("us"),
        bounds=(dt.datetime(2020, 1, 1), dt.datetime(2021, 1, 1)),
    ),
    "duration_ms": ColSpec(pl.Duration("ms")),
    "duration_us": ColSpec(pl.Duration("us")),
    # enum / categorical
    "enum": ColSpec(pl.Enum(["x", "y", "z"])),
    "enum_weighted": ColSpec(pl.Enum(["x", "y", "z"]), weights=[1.0, 2.0, 3.0]),
    "enum_nullable": ColSpec(pl.Enum(["x", "y"]), nullable=True),
    "enum_choices_subset": ColSpec(pl.Enum(["x", "y", "z"]), choices=["x", "y"]),
    "categorical": ColSpec(pl.Categorical),
    "categorical_choices": ColSpec(pl.Categorical, choices=["p", "q"]),
    "categorical_u8": ColSpec(
        pl.Categorical(pl.Categories("roundtrip_u8", physical=pl.UInt8))
    ),
    "categorical_u16": ColSpec(
        pl.Categorical(pl.Categories("roundtrip_u16", physical=pl.UInt16))
    ),
    # distributions (each bounded, so validate has something to check)
    "int64_normal": ColSpec(
        pl.Int64,
        bounds=(0, 1_000),
        distribution="normal",
        distribution_params={"mean": 500.0, "std": 100.0},
    ),
    "int64_poisson": ColSpec(
        pl.Int64,
        bounds=(0, 100),
        distribution="poisson",
        distribution_params={"lambda": 4.0},
    ),
    "float64_uniform": ColSpec(pl.Float64, bounds=(0.0, 1.0), distribution="uniform"),
    "float64_lognormal": ColSpec(
        pl.Float64,
        bounds=(0.0, 100.0),
        distribution="lognormal",
        distribution_params={"mean": 1.0, "std": 0.5},
    ),
    "float64_exponential": ColSpec(
        pl.Float64,
        bounds=(0.0, 50.0),
        distribution="exponential",
        distribution_params={"rate": 0.5},
    ),
    "float64_gamma": ColSpec(
        pl.Float64,
        bounds=(0.0, 100.0),
        distribution="gamma",
        distribution_params={"shape": 2.0, "scale": 2.0},
    ),
    "float64_beta": ColSpec(
        pl.Float64,
        bounds=(0.0, 1.0),
        distribution="beta",
        distribution_params={"alpha": 2.0, "beta": 5.0},
    ),
    # a column validator that the declared bounds already imply
    "validator_implied_by_bounds": ColSpec(
        pl.Int64, bounds=(1, 10), validators=[pl.col("c") > 0]
    ),
    # bounds beyond 2**53 cross the boundary as integers, so nothing rounds
    "int64_bounds_above_2_53": ColSpec(
        pl.Int64, bounds=(9_007_199_254_740_990, 9_007_199_254_740_999)
    ),
    "uint64_bounds_at_the_maximum": ColSpec(
        pl.UInt64, bounds=(18_446_744_073_709_551_600, 18_446_744_073_709_551_615)
    ),
    # typed choices: no string round-trip on the way back from the engine
    "datetime_choices": ColSpec(
        pl.Datetime("us"),
        choices=[dt.datetime(2024, 1, 1, 12), dt.datetime(2025, 6, 30, 8, 30)],
    ),
    "binary_choices": ColSpec(pl.Binary, choices=[b"\x00\x01", b"\xff"]),
    # unique columns are drawn without replacement, from a domain with room
    "unique_int": ColSpec(pl.Int16, unique=True),
    "unique_bounded": ColSpec(pl.Int64, bounds=(1, 1_000), unique=True),
    "unique_string": ColSpec(pl.String, unique=True),
    "unique_choices": ColSpec(
        pl.String, choices=[f"c{i}" for i in range(400)], unique=True
    ),
    "unique_nullable": ColSpec(
        pl.Int32, bounds=(1, 400), unique=True, nullable=True, null_probability=0.3
    ),
}


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
        rules=[ColRule(when={"column": "g", "equals": "x"}, choices=[7])],
    )


class MultiRuleSpec(FrameSpec):
    g = ColSpec(pl.Enum(["x", "y", "z"]))
    v = ColSpec(
        pl.String,
        choices=["a", "b", "c"],
        rules=[
            ColRule(when={"column": "g", "equals": "x"}, choices=["a"]),
            ColRule(when={"column": "g", "in": ["y", "z"]}, choices=["b"]),
        ],
    )


class ChainedRuleSpec(FrameSpec):
    a = ColSpec(pl.Enum(["x", "y"]))
    b = ColSpec(
        pl.Enum(["x", "y"]),
        rules=[ColRule(when={"column": "a", "equals": "x"}, choices=["y"])],
    )
    c = ColSpec(
        pl.Enum(["x", "y"]),
        rules=[ColRule(when={"column": "b", "equals": "y"}, choices=["x"])],
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
                rules=[ColRule(when={"column": "b", "equals": "x"}, choices=["y"])],
            )
            b = ColSpec(
                pl.Enum(["x", "y"]),
                rules=[ColRule(when={"column": "a", "equals": "x"}, choices=["y"])],
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


def test_lazy_roundtrips():
    lf = MixedSpec.generate(ROWS, seed=SEED, lazy=True)
    assert MixedSpec.validate(lf).collect().height == ROWS


def test_every_batch_roundtrips():
    total = 0
    for batch in MixedSpec.generate_batches(500, batch_size=128, seed=SEED):
        MixedSpec.validate(batch)
        total += batch.height
    assert total == 500


def test_zero_rows_roundtrips():
    assert_roundtrip(MixedSpec, n=0)


@pytest.mark.xfail(
    strict=True,
    reason="C12: generate(0, method='cartesian') emits the whole coverage set "
    "instead of an empty frame, unlike generate(0) and the sink_* methods",
)
def test_zero_rows_cartesian_is_empty():
    assert MixedSpec.generate(0, method="cartesian", seed=SEED).height == 0


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
# These two constraint kinds wrap arbitrary polars expressions. Generation
# cannot in general produce data satisfying an arbitrary predicate, and does
# not try to -- so they are excluded from the property above by design, not by
# oversight. The tests below record that decision so a future change to it is
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
