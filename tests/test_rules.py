"""`ColRule`: conditional values applied as a vectorised pass after generation.

Covers what a rule may declare, which rows it touches, the supported `when`
operators, and that rules behave the same under `method="cartesian"`.
"""

import polars as pl
import pytest
from polspec import (
    Bound,
    ColRule,
    ColSpec,
    FrameSpec,
)

# ---------------------------------------------------------------------------
# single rules
# ---------------------------------------------------------------------------


class RuledSource(FrameSpec):
    enum_1 = ColSpec(dtype=pl.Enum(["A", "B", "C"]), nullable=False)
    enum_2 = ColSpec(
        dtype=pl.Enum(["X", "Y", "Z"]),
        nullable=False,
        rules=(
            ColRule(when={"column": "enum_1", "in": ["A", "B"]}, choices=["X", "Y"]),
        ),
    )


def test_rule_restricts_matching_rows():
    df = RuledSource.generate(2_000, seed=42)
    violations = df.filter(
        pl.col("enum_1").is_in(["A", "B"]) & ~pl.col("enum_2").is_in(["X", "Y"])
    )
    assert violations.height == 0


def test_rule_does_not_affect_unmatched_rows():
    df = RuledSource.generate(5_000, seed=42)
    unrestricted = set(df.filter(pl.col("enum_1") == "C")["enum_2"].unique().to_list())
    assert unrestricted == {"X", "Y", "Z"}


def test_rule_is_deterministic_given_seed():
    df_a = RuledSource.generate(1_000, seed=7)
    df_b = RuledSource.generate(1_000, seed=7)
    assert df_a.equals(df_b)


class MultiRuleSource(FrameSpec):
    enum_1 = ColSpec(dtype=pl.Enum(["A", "B", "C"]), nullable=False)
    enum_2 = ColSpec(
        dtype=pl.Enum(["X", "Y", "Z"]),
        nullable=False,
        rules=(
            ColRule(when={"column": "enum_1", "equals": "A"}, choices=["X"]),
            ColRule(when={"column": "enum_1", "in": ["A", "B"]}, choices=["Y"]),
        ),
    )


def test_multiple_rules_first_match_wins():
    df = MultiRuleSource.generate(3_000, seed=3)
    # enum_1=="A" matches both rules; the first (choices=["X"]) must win.
    a_values = set(df.filter(pl.col("enum_1") == "A")["enum_2"].unique().to_list())
    assert a_values == {"X"}
    # enum_1=="B" only matches the second rule.
    b_values = set(df.filter(pl.col("enum_1") == "B")["enum_2"].unique().to_list())
    assert b_values == {"Y"}


class NumericRuleSource(FrameSpec):
    enum_1 = ColSpec(dtype=pl.Enum(["A", "B", "C"]), nullable=False)
    int_1 = ColSpec(
        dtype=pl.Int64,
        bounds=Bound(-100, 100),
        nullable=False,
        rules=(ColRule(when={"column": "enum_1", "equals": "C"}, choices=[0, 1, 2]),),
    )


def test_rule_on_numeric_column():
    df = NumericRuleSource.generate(3_000, seed=5)
    c_values = set(df.filter(pl.col("enum_1") == "C")["int_1"].to_list())
    assert c_values <= {0, 1, 2}


def test_rule_applies_under_cartesian_method():
    df = RuledSource.generate(n=1, method="cartesian", seed=1)
    violations = df.filter(
        pl.col("enum_1").is_in(["A", "B"]) & ~pl.col("enum_2").is_in(["X", "Y"])
    )
    assert violations.height == 0


def test_rule_with_zero_rows():
    df = RuledSource.generate(0, seed=1)
    assert df.height == 0
    assert df.schema == RuledSource.schema()


# ---------------------------------------------------------------------------
# what a rule may declare
# ---------------------------------------------------------------------------


def test_colrule_rejects_choices_outside_enum_categories():
    with pytest.raises(ValueError, match="not among this column's Enum categories"):
        ColSpec(
            dtype=pl.Enum(["X", "Y", "Z"]),
            rules=(
                ColRule(
                    when={"column": "enum_1", "equals": "A"}, choices=["NOT_A_CATEGORY"]
                ),
            ),
        )


def test_colrule_rejects_malformed_when():
    with pytest.raises(TypeError, match="must be a predicate"):
        ColRule(when="not a dict", choices=["X"])


def test_colrule_rejects_ambiguous_when():
    with pytest.raises(ValueError, match="exactly one of"):
        ColRule(when={"column": "x", "equals": 1, "in": [1, 2]}, choices=["X"])


def test_colrule_rejects_empty_choices():
    with pytest.raises(ValueError, match="must not be empty"):
        ColRule(when={"column": "x", "equals": 1}, choices=[])


# ---------------------------------------------------------------------------
# when operators
# ---------------------------------------------------------------------------


def test_expanded_rule_operations():
    class ComplexRules(FrameSpec):
        score = ColSpec(dtype=pl.Int32, bounds=(0, 100), nullable=False)
        optional_val = ColSpec(dtype=pl.Float64, bounds=(0.0, 10.0), nullable=True)
        tier = ColSpec(
            dtype=pl.String,
            nullable=False,
            rules=(
                ColRule(when={"column": "score", "lt": 50}, choices=["Low"]),
                ColRule(when={"column": "score", "between": [50, 79]}, choices=["Mid"]),
                ColRule(when={"column": "score", "gte": 80}, choices=["High"]),
            ),
        )
        status = ColSpec(
            dtype=pl.String,
            nullable=False,
            rules=(
                ColRule(
                    when={"column": "optional_val", "is_null": True},
                    choices=["Missing"],
                ),
                ColRule(
                    when={"column": "optional_val", "is_not_null": True},
                    choices=["Present"],
                ),
            ),
        )

    df = ComplexRules.generate(5_000, seed=42)

    low_check = df.filter(pl.col("score") < 50)["tier"].unique().to_list()
    assert low_check == ["Low"]

    mid_check = df.filter(pl.col("score").is_between(50, 79))["tier"].unique().to_list()
    assert mid_check == ["Mid"]

    high_check = df.filter(pl.col("score") >= 80)["tier"].unique().to_list()
    assert high_check == ["High"]

    missing_check = (
        df.filter(pl.col("optional_val").is_null())["status"].unique().to_list()
    )
    assert missing_check == ["Missing"]

    present_check = (
        df.filter(pl.col("optional_val").is_not_null())["status"].unique().to_list()
    )
    assert present_check == ["Present"]


def test_rule_referencing_unknown_column_raises():
    with pytest.raises(ValueError, match="references unknown column"):

        class BadRule(FrameSpec):
            x = ColSpec(
                dtype=pl.Int32,
                rules=(
                    ColRule(when={"column": "non_existent", "equals": 1}, choices=[0]),
                ),
            )


def test_colrule_weighted_choices():
    class RuleWeightedSpec(FrameSpec):
        segment = ColSpec(dtype=pl.Enum(["standard", "premium"]))
        reward = ColSpec(
            dtype=pl.String,
            rules=(
                ColRule(
                    when={"column": "segment", "equals": "standard"},
                    choices={"voucher": 0.9, "gift": 0.1},
                ),
                ColRule(
                    when={"column": "segment", "equals": "premium"},
                    choices=["gift", "vip_pass"],
                    weights=[0.3, 0.7],
                ),
            ),
        )

    df = RuleWeightedSpec.generate(30_000, seed=42)
    standard_df = df.filter(pl.col("segment") == "standard")
    voucher_ratio = (
        standard_df.filter(pl.col("reward") == "voucher").height / standard_df.height
    )
    assert 0.88 <= voucher_ratio <= 0.92

    premium_df = df.filter(pl.col("segment") == "premium")
    vip_ratio = (
        premium_df.filter(pl.col("reward") == "vip_pass").height / premium_df.height
    )
    assert 0.67 <= vip_ratio <= 0.73
