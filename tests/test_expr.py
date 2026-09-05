"""`polspec.col()`: predicates that evaluate like Polars and survive a file.

Each node is checked three ways: it evaluates to the same mask as the Polars
expression it stands for, it round-trips through its data form, and its
Python source evaluates back to an equal predicate.
"""

import datetime as dt

import polars as pl
import pytest
from polspec import Check, ColRule, ColSpec, FrameSpec, SpecError, col
from polspec.expr import Pred, from_data, lit

FRAME = pl.DataFrame(
    {
        "a": [1, 2, 3, None, 5],
        "b": [5, 4, 3, 2, 1],
        "s": ["alpha", "beta", None, "delta", "alphabet"],
        "d": [
            dt.date(2024, 1, 1),
            dt.date(2024, 6, 1),
            None,
            dt.date(2025, 1, 1),
            dt.date(2023, 1, 1),
        ],
    }
)

CASES = [
    (col("a") == 3, pl.col("a") == 3),
    (col("a") != 3, pl.col("a") != 3),
    (col("a") < col("b"), pl.col("a") < pl.col("b")),
    (col("a") <= 2, pl.col("a") <= 2),
    (col("a") > 2, pl.col("a") > 2),
    (col("a") >= 3, pl.col("a") >= 3),
    (col("a") + col("b") == 6, (pl.col("a") + pl.col("b")) == 6),
    (col("a") - 1 > 1, (pl.col("a") - 1) > 1),
    (col("a") * 2 >= 6, (pl.col("a") * 2) >= 6),
    (col("b") / 2 < 2, (pl.col("b") / 2) < 2),
    (10 - col("a") > 7, (10 - pl.col("a")) > 7),
    ((col("a") > 1) & (col("b") > 1), (pl.col("a") > 1) & (pl.col("b") > 1)),
    ((col("a") > 4) | (col("b") > 4), (pl.col("a") > 4) | (pl.col("b") > 4)),
    (~(col("a") > 2), ~(pl.col("a") > 2)),
    (col("a").is_in([1, 5]), pl.col("a").is_in([1, 5])),
    (col("a").is_null(), pl.col("a").is_null()),
    (col("a").is_not_null(), pl.col("a").is_not_null()),
    (col("a").is_between(2, 3), pl.col("a").is_between(2, 3)),
    (col("s").str.contains("alpha"), pl.col("s").str.contains("alpha", literal=True)),
    (col("s").str.starts_with("al"), pl.col("s").str.starts_with("al")),
    (col("s").str.ends_with("ta"), pl.col("s").str.ends_with("ta")),
    (col("s").str.matches(r"^a.*t$"), pl.col("s").str.contains(r"^a.*t$")),
    (col("s").str.len_chars() > 4, pl.col("s").str.len_chars() > 4),
    (col("d") >= dt.date(2024, 6, 1), pl.col("d") >= dt.date(2024, 6, 1)),
    (lit(1) < col("a"), pl.lit(1) < pl.col("a")),
]


@pytest.mark.parametrize("pred, expected", CASES, ids=[repr(p) for p, _ in CASES])
def test_evaluates_like_polars(pred: Pred, expected: pl.Expr):
    got = FRAME.select(pred.to_expr().alias("m"))["m"].to_list()
    want = FRAME.select(expected.alias("m"))["m"].to_list()
    assert got == want


@pytest.mark.parametrize("pred", [p for p, _ in CASES], ids=[repr(p) for p, _ in CASES])
def test_round_trips_through_data_and_source(pred: Pred):
    data = pred.to_data()
    assert from_data(data).equals(pred)
    assert from_data(data).to_data() == data
    # Source evaluates back to an equal predicate.
    rebuilt = eval(pred.to_source(), {"col": col, "lit": lit, "datetime": dt})  # noqa: S307
    assert rebuilt.equals(pred)
    assert repr(pred) == pred.to_source()


def test_data_form_is_plain_yaml_values():
    import yaml

    pred = (col("total") >= col("subtotal")) & col("status").is_in(["NEW", "PAID"])
    text = yaml.safe_dump(pred.to_data())
    assert from_data(yaml.safe_load(text)).equals(pred)
    assert pred.to_data() == {
        "and": [
            {"ge": [{"col": "total"}, {"col": "subtotal"}]},
            {"is_in": [{"col": "status"}, ["NEW", "PAID"]]},
        ]
    }


def test_root_names_literals_and_rename():
    pred = (col("total") >= col("subtotal") * 2) & col("d").is_between(
        dt.date(2024, 1, 1), dt.date(2025, 1, 1)
    )
    assert pred.root_names() == {"total", "subtotal", "d"}
    assert pred.literals() == [2, dt.date(2024, 1, 1), dt.date(2025, 1, 1)]
    renamed = pred.rename({"total": "amount", "d": "when"})
    assert renamed.root_names() == {"amount", "subtotal", "when"}
    assert renamed.to_source() == (
        "(col('amount') >= (col('subtotal') * 2)) & "
        "(col('when').is_between(datetime.date(2024, 1, 1), datetime.date(2025, 1, 1)))"
    )


def test_predicates_have_no_truth_value():
    with pytest.raises(TypeError, match="no truth value"):
        bool(col("a") == 1)
    assert (col("a") == 1).equals(col("a") == 1)
    assert not (col("a") == 1).equals(col("a") == 2)
    assert hash(col("a") == 1) == hash(col("a") == 1)


def test_bad_operands_and_data_are_spec_errors():
    with pytest.raises(SpecError, match="operand must be"):
        _ = col("a") == object()
    with pytest.raises(SpecError, match="non-empty column name"):
        col("")
    with pytest.raises(SpecError, match="unknown operation"):
        from_data({"xor": [1, 2]})
    with pytest.raises(SpecError, match="exactly one key"):
        from_data({"eq": [1, 2], "ne": [1, 2]})
    with pytest.raises(SpecError, match="two-element list"):
        from_data({"eq": [1]})


# ---------------------------------------------------------------------------
# Predicates inside Check, validators and ColRule
# ---------------------------------------------------------------------------


def test_check_accepts_a_predicate_and_names_it_readably():
    check = Check(col("total") >= col("subtotal"))
    assert check.pred is not None
    assert check.name == "col('total') >= col('subtotal')"
    assert isinstance(check.expr, pl.Expr)
    raw = Check(pl.col("total") >= pl.col("subtotal"))
    assert raw.pred is None


def test_validators_accept_predicates():
    spec = ColSpec(pl.String, validators=[col("email").str.contains("@")])
    assert spec.validators[0].pred is not None
    with pytest.raises(SpecError, match="references other column"):

        class Bad(FrameSpec):
            email = ColSpec(pl.String, validators=[col("other").str.contains("@")])


def test_colrule_when_accepts_a_multi_column_predicate():
    class Spec(FrameSpec):
        region = ColSpec(pl.Enum(["UK", "US"]))
        qty = ColSpec(pl.Int64, bounds=(1, 100))
        carrier = ColSpec(
            pl.Enum(["RM", "UPS", "DHL"]),
            rules=[
                ColRule(
                    when=(col("region") == "UK") & (col("qty") > 50), choices=["RM"]
                ),
                ColRule(when=col("region") == "US", choices=["UPS"]),
            ],
        )

    df = Spec.generate(500, seed=1)
    heavy_uk = df.filter((pl.col("region") == "UK") & (pl.col("qty") > 50))
    assert heavy_uk["carrier"].unique().to_list() == ["RM"]
    assert df.filter(pl.col("region") == "US")["carrier"].unique().to_list() == ["UPS"]
    Spec.validate(df)


def test_legacy_dict_conditions_become_predicates():
    rule = ColRule(when=col("a").is_between(1, 5), choices=["x"])
    assert rule.when.equals(col("a").is_between(1, 5))
    assert ColRule(when=~col("a").is_in([1]), choices=["x"]).when.equals(
        ~col("a").is_in([1])
    )
    assert ColRule(when=col("a").is_null(), choices=["x"]).when.equals(
        col("a").is_null()
    )
    assert ColRule(when=col("a") >= 2, choices=["x"]).when.equals(col("a") >= 2)
    # The one-column dict is gone; `col()` is the only spelling.
    with pytest.raises(SpecError, match="must be a predicate built with col"):
        ColRule(when={"column": "a", "gte": 2}, choices=["x"])
    with pytest.raises(SpecError, match="must reference at least one column"):
        ColRule(when=lit(True) == lit(True), choices=["x"])
    with pytest.raises(SpecError, match="must be a predicate"):
        ColRule(when="not a dict", choices=["x"])


def test_colrules_compare_structurally():
    a = ColRule(when=col("a") == 1, choices=["x"])
    b = ColRule(when=col("a") == 1, choices=["x"])
    c = ColRule(when=col("a") == 2, choices=["x"])
    assert a == b
    assert a != c
    assert ColSpec(pl.String, rules=[a]) == ColSpec(pl.String, rules=[b])
    assert ColSpec(pl.String, rules=[a]) != ColSpec(pl.String, rules=[c])
