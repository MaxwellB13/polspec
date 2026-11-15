"""`strategy`: frames a spec accepts, drawn by Hypothesis from a row count and
a seed -- valid by construction, and shrinking to the fewest rows that
fail."""

import subprocess
import sys

import polars as pl
import pytest
from hypothesis import find, given, settings
from polspec import ColSpec, ForeignKey, FrameSpec, generate, strategy, validate


class Customers(FrameSpec):
    id = ColSpec(pl.Int64, bounds=(1, 1_000), unique=True)


class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, 1_000_000), unique=True)
    customer_id = ColSpec(pl.Int64, bounds=(1, 1_000))
    status = ColSpec(pl.Enum, choices={"NEW": 0.7, "PAID": 0.3})
    total = ColSpec(pl.Float64, bounds=(0.0, 500.0), nullable=True)
    __foreign_keys__ = [
        ForeignKey("customer_id", references=Customers, ref_columns="id")
    ]
    __order_by__ = "order_id"


CUSTOMERS = Customers.generate(200, seed=1)
PARENTS = {Customers: CUSTOMERS}


@settings(max_examples=60, deadline=None)
@given(orders=strategy(Orders, max_rows=300, references=PARENTS))
def test_every_frame_drawn_meets_its_spec(orders):
    validate(Orders, orders, references=PARENTS)
    assert orders.height <= 300


@settings(max_examples=30, deadline=None)
@given(orders=strategy(Orders, min_rows=5, max_rows=8, references=PARENTS))
def test_frames_hold_a_row_count_in_range(orders):
    assert 5 <= orders.height <= 8


def test_a_failure_shrinks_to_the_fewest_rows_that_fail():
    """The question fails once a frame holds a PAID order; Hypothesis
    reports the smallest frame that does."""

    def holds_a_paid_order(orders: pl.DataFrame) -> bool:
        return (orders["status"] == "PAID").any()

    smallest = find(
        strategy(Orders, max_rows=200, references=PARENTS),
        holds_a_paid_order,
        settings=settings(max_examples=500, database=None),
    )
    assert smallest.height == 1


def test_each_frame_is_one_generate_call():
    drawn = find(
        strategy(Orders, min_rows=40, max_rows=40, references=PARENTS),
        lambda frame: True,
        settings=settings(database=None),
    )
    # The smallest draw is seed 0.
    assert drawn.equals(generate(Orders, 40, seed=0, references=PARENTS))


@settings(max_examples=20, deadline=None)
@given(
    orders=strategy(Orders, max_rows=50, references=PARENTS, fixed={"status": "NEW"})
)
def test_fixed_pins_a_column_in_every_frame(orders):
    assert orders["status"].eq("NEW").all()


@pytest.mark.parametrize(("low", "high"), [(-1, 10), (10, 5)])
def test_a_row_range_that_is_negative_or_empty_is_refused(low, high):
    with pytest.raises(ValueError, match="0 <= min_rows <= max_rows"):
        strategy(Orders, min_rows=low, max_rows=high)


def test_without_hypothesis_it_says_how_to_install_it():
    probe = (
        "import sys; sys.modules['hypothesis'] = None; import polspec, polars as pl\n"
        "try:\n"
        "    polspec.strategy(polspec.TableSpec('T', {'a': pl.Int64}))\n"
        "except ImportError as error:\n"
        "    print(error)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert "pip install hypothesis" in out.stdout
