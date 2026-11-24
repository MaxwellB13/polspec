"""`assert_valid` and `check_pipeline`: the check at the end of a pipeline
test, failing as an assertion with the findings and rows in its message,
and the whole test -- inputs, call, check -- in one call."""

import polars as pl
import pytest
from polspec import (
    ColSpec,
    ForeignKey,
    FrameSpec,
    TableSpec,
    ValidationError,
    assert_valid,
    check_pipeline,
)


class Customers(FrameSpec):
    id = ColSpec(pl.Int64, bounds=(1, 1_000), unique=True)
    tier = pl.Enum(["free", "pro"])


class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, 1_000_000), unique=True)
    customer_id = ColSpec(pl.Int64, bounds=(1, 1_000))
    total = ColSpec(pl.Float64, bounds=(0.0, 500.0))
    __foreign_keys__ = [
        ForeignKey("customer_id", references=Customers, ref_columns="id")
    ]


class Spend(FrameSpec):
    customer_id = ColSpec(pl.Int64, unique=True)
    spend = ColSpec(pl.Float64, bounds=(0.0, None))
    __foreign_keys__ = [
        ForeignKey("customer_id", references=Customers, ref_columns="id")
    ]


def summarise(customers: pl.DataFrame, orders: pl.DataFrame) -> pl.DataFrame:
    return orders.group_by("customer_id").agg(spend=pl.col("total").sum())


TOTALS = TableSpec(
    "Totals", {"total": ColSpec(pl.Float64, bounds=(0, 100)), "id": pl.Int64}
)


# ---------------------------------------------------------------------------
# assert_valid
# ---------------------------------------------------------------------------


def test_a_frame_that_meets_its_spec_comes_back_validated():
    frame = pl.DataFrame({"total": [10.0], "id": [1], "extra": ["x"]})
    out = assert_valid(TOTALS, frame, extra_cols="drop")
    assert out.columns == ["total", "id"]


def test_a_lazy_frame_comes_back_lazy():
    lazy = pl.LazyFrame({"total": [10.0], "id": [1]})
    assert isinstance(assert_valid(TOTALS, lazy), pl.LazyFrame)


def test_a_failure_is_an_assertion_not_an_error():
    """A runner reports an `AssertionError` as a failed test; a
    `ValidationError` would be an error in the test itself."""
    with pytest.raises(AssertionError) as raised:
        assert_valid(TOTALS, pl.DataFrame({"total": [250.0], "id": [1]}))
    assert not isinstance(raised.value, ValidationError)
    assert raised.value.__cause__ is None


def test_the_message_names_the_spec_each_finding_and_the_failing_rows():
    frame = pl.DataFrame({"total": [10.0, 250.0, 300.0], "id": [1, None, 3]})
    with pytest.raises(AssertionError) as raised:
        assert_valid(TOTALS, frame)
    message = str(raised.value)
    assert message.startswith("Totals: 2 finding(s) in 3 row(s)")
    assert "  - total__bounds: " in message
    assert "  - id__null: " in message or "  - id__nullability" in message
    assert "First failing rows:" in message
    assert "250.0" in message and "300.0" in message


def test_the_message_shows_at_most_ten_failing_rows():
    frame = pl.DataFrame(
        {"total": [float(1_000 + i) for i in range(50)], "id": range(50)}
    )
    with pytest.raises(AssertionError) as raised:
        assert_valid(TOTALS, frame)
    rows = str(raised.value).split("First failing rows:")[1]
    assert "1009.0" in rows
    assert "1010.0" not in rows


def test_a_finding_without_rows_shows_no_table():
    with pytest.raises(AssertionError) as raised:
        assert_valid(TOTALS, pl.DataFrame({"total": [10.0]}))
    assert "missing_columns" in str(raised.value)
    assert "First failing rows:" not in str(raised.value)


def test_it_takes_validate_options_and_references():
    assert_valid(TOTALS, pl.DataFrame({"total": [250.0], "id": [1]}), bounds=False)
    customers = Customers.generate(10, seed=1)
    orders = Orders.generate(50, seed=1, references={Customers: customers})
    assert_valid(Orders, orders, references={Customers: customers})
    with pytest.raises(AssertionError, match="no matching parent record"):
        assert_valid(Orders, orders, references={Customers: customers.head(1)})


# ---------------------------------------------------------------------------
# check_pipeline
# ---------------------------------------------------------------------------


def test_a_pipeline_is_generated_run_and_checked_in_one_call():
    result, inputs = check_pipeline(
        summarise,
        inputs={"customers": Customers, "orders": Orders},
        output=Spend,
        n={"customers": 100, "orders": 1_000},
    )
    assert (inputs["customers"].height, inputs["orders"].height) == (100, 1_000)
    # The orders were generated against the customers, and the output's own
    # foreign key resolved against the generated customers too.
    assert set(inputs["orders"]["customer_id"]) <= set(inputs["customers"]["id"])
    assert set(result["customer_id"]) <= set(inputs["customers"]["id"])


def test_one_row_count_is_every_inputs_and_a_frame_passes_as_it_is():
    customers = Customers.generate(20, seed=9)
    _, inputs = check_pipeline(
        summarise,
        inputs={"customers": customers, "orders": Orders},
        output=Spend,
        n=200,
        references={Customers: customers},
    )
    assert inputs["customers"] is customers
    assert set(inputs["orders"]["customer_id"]) <= set(customers["id"])


def test_the_seed_makes_the_same_inputs_every_run():
    def run(seed):
        return check_pipeline(
            summarise,
            inputs={"customers": Customers, "orders": Orders},
            output=Spend,
            n=200,
            seed=seed,
        )[1]

    assert run(3)["orders"].equals(run(3)["orders"])
    assert not run(3)["orders"].equals(run(4)["orders"])


def test_a_pipeline_that_breaks_its_contract_fails_as_an_assertion():
    def careless(customers, orders):
        return summarise(customers, orders).with_columns(spend=-pl.col("spend"))

    with pytest.raises(AssertionError, match=r"^Spend: 1 finding\(s\)"):
        check_pipeline(
            careless,
            inputs={"customers": Customers, "orders": Orders},
            output=Spend,
            n=100,
        )


def test_a_lazy_pipeline_is_checked_lazily():
    def lazy(customers, orders):
        return summarise(customers, orders).lazy()

    result, _ = check_pipeline(
        lazy, inputs={"customers": Customers, "orders": Orders}, output=Spend, n=100
    )
    assert isinstance(result, pl.LazyFrame)


@pytest.mark.parametrize(
    ("n", "message"),
    [
        ({"customers": 10, "orders": 10, "refunds": 10}, r"n names \['refunds'\]"),
        ({"customers": 10}, r"no row count for \['orders'\]"),
    ],
)
def test_a_row_count_for_no_input_or_none_for_one_is_named(n, message):
    with pytest.raises(TypeError, match=message):
        check_pipeline(
            summarise,
            inputs={"customers": Customers, "orders": Orders},
            output=Spend,
            n=n,
        )


def test_the_frame_first_says_the_spec_comes_first():
    """1.x took `(df, spec)`; 2.0 takes the spec first, as every verb does."""
    with pytest.raises(TypeError, match=r"assert_valid\(spec, df\)"):
        assert_valid(pl.DataFrame({"total": [10.0]}), TOTALS)
