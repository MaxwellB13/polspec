"""`checks=True`: generated rows that keep the spec's checks and validators.

The rows kept are the first rows of the seed's one frame that pass every
check, so a frame whose rows all pass is the frame `checks=False` gives,
and every claim a subset of rows keeps -- uniqueness, a foreign key --
still holds, across a stream's batches too.
"""

from __future__ import annotations

import itertools

import polars as pl
import pytest
from polspec import (
    Check,
    ColSpec,
    ForeignKey,
    FrameSpec,
    GenerationError,
    Hierarchy,
    Registry,
    SpecError,
    col,
)

SEED = 7


class Orders(FrameSpec):
    """About half the rows a free draw makes fail the check, and a few the
    validator, so keeping them takes a longer read."""

    id = ColSpec(pl.Int64, unique=True)
    status = ColSpec(pl.Enum(["NEW", "PAID"]))
    subtotal = ColSpec(pl.Float64, bounds=(0, 100))
    total = ColSpec(pl.Float64, bounds=(0, 110), validators=[pl.col("total") > 5])
    __checks__ = [Check(col("total") >= col("subtotal"), name="total_gte_subtotal")]


class Passing(FrameSpec):
    """Checks every row a free draw makes already passes."""

    id = ColSpec(pl.Int64, unique=True)
    amount = ColSpec(pl.Int64, bounds=(0, 10), validators=[pl.col("amount") >= 0])
    __checks__ = [Check(col("amount") <= 10, name="at_most_ten")]


def _passes(spec: type[FrameSpec], frame: pl.DataFrame) -> bool:
    return spec.inspect(frame).passed


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def test_without_checks_the_frame_breaks_them():
    """What the rest of this file fixes: a free draw fails its own checks."""
    assert not _passes(Orders, Orders.generate(5_000, seed=SEED))


def test_kept_rows_pass_every_check_and_validator():
    frame = Orders.generate(5_000, seed=SEED, checks=True)
    assert frame.height == 5_000
    assert _passes(Orders, frame)


def test_a_frame_whose_rows_all_pass_is_the_frame_checks_false_gives():
    assert Passing.generate(5_000, seed=SEED, checks=True).equals(
        Passing.generate(5_000, seed=SEED)
    )


def test_a_spec_with_no_checks_is_generated_as_it_always_is():
    class Plain(FrameSpec):
        a = ColSpec(pl.Int64, bounds=(0, 10))

    assert Plain.generate(100, seed=SEED, checks=True).equals(
        Plain.generate(100, seed=SEED)
    )


def test_the_same_seed_keeps_the_same_rows():
    one = Orders.generate(2_000, seed=SEED, checks=True)
    assert one.equals(Orders.generate(2_000, seed=SEED, checks=True))


def test_kept_rows_are_rows_of_the_seeds_frame():
    """The first rows that pass, in order: each kept row is a row of a
    longer free draw from the same seed."""
    kept = Orders.generate(1_000, seed=SEED, checks=True)
    longer = Orders.generate(4_000, seed=SEED)
    assert longer.join(kept, on=kept.columns, how="semi").height == 1_000


def test_zero_rows_is_an_empty_typed_frame():
    frame = Orders.generate(0, seed=SEED, checks=True)
    assert frame.height == 0 and frame.schema == Orders.schema()


def test_a_check_that_ignores_nulls_keeps_a_null_and_one_that_does_not_drops_it():
    def spec(ignore_nulls: bool) -> type[FrameSpec]:
        class Rated(FrameSpec):
            rate = ColSpec(
                pl.Float64, bounds=(0, 1), nullable=True, null_probability=0.3
            )
            __checks__ = [
                Check(col("rate") < 0.5, name="low", ignore_nulls=ignore_nulls)
            ]

        return Rated

    lenient, strict = spec(ignore_nulls=True), spec(ignore_nulls=False)
    assert lenient.generate(2_000, seed=SEED, checks=True)["rate"].null_count() > 0
    assert strict.generate(2_000, seed=SEED, checks=True)["rate"].null_count() == 0


def test_an_ordered_spec_is_sorted_after_its_rows_are_kept():
    class Ordered(Orders):
        __order_by__ = ["subtotal"]

    frame = Ordered.generate(2_000, seed=SEED, checks=True)
    assert frame["subtotal"].is_sorted()
    assert _passes(Ordered, frame)


def test_a_foreign_key_still_points_at_its_parent():
    class Customers(FrameSpec):
        id = ColSpec(pl.Int64, unique=True, bounds=(1, 500))

    class Sales(FrameSpec):
        customer_id = ColSpec(pl.Int64)
        amount = ColSpec(pl.Int64, bounds=(-50, 50))
        __checks__ = [Check(col("amount") > 0, name="positive")]
        __foreign_keys__ = [
            ForeignKey("customer_id", references=Customers, ref_columns="id")
        ]

    parent = Customers.generate(500, seed=1)
    frame = Sales.generate(
        2_000, seed=SEED, references={Customers: parent}, checks=True
    )
    assert set(frame["customer_id"]) <= set(parent["id"])
    assert Sales.inspect(frame, references={Customers: parent}).passed


def test_a_scalar_pin_is_kept():
    frame = Orders.generate(1_000, seed=SEED, fixed={"status": "PAID"}, checks=True)
    assert frame["status"].unique().to_list() == ["PAID"]
    assert _passes(Orders, frame)


# ---------------------------------------------------------------------------
# What cannot be kept
# ---------------------------------------------------------------------------


def test_a_check_that_rejects_nearly_every_row_is_refused_by_name():
    class Impossible(FrameSpec):
        a = ColSpec(pl.Int64, bounds=(0, 10))
        __checks__ = [Check(col("a") > 100, name="too_big")]

    with pytest.raises(GenerationError, match=r"check 'too_big' rejected 100\.0%"):
        Impossible.generate(100, seed=SEED, checks=True)


def test_a_validator_is_named_when_it_rejects_too_much():
    class Picky(FrameSpec):
        a = ColSpec(pl.Int64, bounds=(0, 1000), validators=[pl.col("a") == 3])

    with pytest.raises(GenerationError, match="'a''s validator"):
        Picky.generate(1_000, seed=SEED, checks=True)


def test_a_domain_too_small_for_the_rows_read_says_checks_needed_them():
    """Ten distinct values and ten rows: every row has to pass, since there
    is no eleventh value for a row that replaces a rejected one."""

    class Tight(FrameSpec):
        code = ColSpec(pl.Int8, bounds=(0, 9), unique=True)
        __checks__ = [Check(col("code") != 3, name="not_three")]

    with pytest.raises(GenerationError, match="checks=True read"):
        Tight.generate(10, seed=SEED, checks=True)


def test_cartesian_is_refused():
    with pytest.raises(SpecError, match="coverage set"):
        Orders.generate(100, method="cartesian", checks=True)


def test_a_hierarchy_is_refused():
    class Tree(FrameSpec):
        ref = ColSpec(pl.Int64)
        parent = ColSpec(pl.Int64)
        __hierarchy__ = Hierarchy(child="ref", parent="parent", max_depth=3)
        __checks__ = [Check(col("ref") >= 0, name="non_negative")]

    with pytest.raises(SpecError, match="children without a parent"):
        Tree.generate(100, seed=SEED, checks=True)


def test_a_key_into_its_own_rows_is_refused():
    class Linked(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        previous = ColSpec(pl.Int64, nullable=True)
        __foreign_keys__ = [ForeignKey("previous", references="self", ref_columns="id")]
        __checks__ = [Check(col("id") >= 0, name="non_negative")]

    with pytest.raises(SpecError, match="its own rows"):
        Linked.generate(100, seed=SEED, checks=True)


def test_a_pin_of_every_value_is_refused():
    with pytest.raises(SpecError, match="pin a single value instead"):
        Orders.generate(
            3, seed=SEED, fixed={"status": ["NEW", "PAID", "NEW"]}, checks=True
        )


def test_a_refusal_only_applies_to_a_spec_that_has_checks():
    class Covered(FrameSpec):
        flag = ColSpec(pl.Boolean)

    assert Covered.generate(4, method="cartesian", seed=SEED, checks=True).equals(
        Covered.generate(4, method="cartesian", seed=SEED)
    )


# ---------------------------------------------------------------------------
# Batches, scans and sinks
# ---------------------------------------------------------------------------


def test_every_batch_keeps_the_checks_and_a_unique_column_holds_across_them():
    batches = list(
        Orders.generate_batches(25_000, batch_size=4_000, seed=SEED, checks=True)
    )
    assert [b.height for b in batches] == [4_000] * 6 + [1_000]
    whole = pl.concat(batches)
    assert _passes(Orders, whole)


def test_batches_whose_rows_all_pass_are_the_batches_checks_false_gives():
    kept = Passing.generate_batches(9_000, batch_size=2_000, seed=SEED, checks=True)
    free = Passing.generate_batches(9_000, batch_size=2_000, seed=SEED)
    assert all(a.equals(b) for a, b in zip(kept, free, strict=True))


def test_batches_keep_the_rows_generate_keeps():
    """Each batch reads on from where the last stopped, so the batches are
    the first rows of the frame that pass, as `generate` keeps them -- for a
    spec no pass rewrites, row for row."""
    whole = Orders.generate(10_000, seed=SEED, checks=True)
    batched = pl.concat(
        Orders.generate_batches(10_000, batch_size=3_000, seed=SEED, checks=True)
    )
    assert batched.equals(whole)


def test_a_scan_keeps_the_checks_and_projects_as_the_whole_scan_holds_it():
    lf = Orders.scan(12_000, seed=SEED, batch_size=5_000, checks=True)
    whole = lf.collect()
    assert whole.height == 12_000 and _passes(Orders, whole)
    names = whole.columns
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            assert lf.select(subset).collect().equals(whole.select(subset)), subset


def test_a_scans_head_is_the_head_of_the_whole_scan():
    lf = Orders.scan(12_000, seed=SEED, batch_size=5_000, checks=True)
    assert lf.head(7).collect().equals(lf.collect().head(7))


def test_a_sink_writes_rows_that_keep_the_checks(tmp_path):
    path = tmp_path / "orders.parquet"
    Orders.sink_parquet(path, 8_000, batch_size=3_000, seed=SEED, checks=True)
    assert _passes(Orders, pl.read_parquet(path))


def test_a_stream_refuses_what_generate_refuses():
    with pytest.raises(SpecError, match="coverage set"):
        Orders.scan(10, method="cartesian", checks=True)
    with pytest.raises(SpecError, match="coverage set"):
        Orders.generate_batches(10, method="cartesian", checks=True)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


class Customers(FrameSpec):
    id = ColSpec(pl.Int64, unique=True, bounds=(1, 100_000))
    credit = ColSpec(pl.Int64, bounds=(-100, 100))
    __checks__ = [Check(col("credit") >= 0, name="in_credit")]


class Invoices(FrameSpec):
    customer_id = ColSpec(pl.Int64)
    amount = ColSpec(pl.Int64, bounds=(-100, 100))
    __checks__ = [Check(col("amount") > 0, name="positive")]
    __foreign_keys__ = [
        ForeignKey("customer_id", references=Customers, ref_columns="id")
    ]


def test_generate_all_keeps_every_specs_checks():
    registry = Registry(Customers, Invoices)
    frames = registry.generate_all(2_000, seed=SEED, checks=True)
    assert all(report.passed for report in registry.inspect_all(frames).values())


def test_scan_all_keeps_every_specs_checks():
    registry = Registry(Customers, Invoices)
    frames = {
        name: lf.collect()
        for name, lf in registry.scan_all(2_000, seed=SEED, checks=True).items()
    }
    assert all(report.passed for report in registry.inspect_all(frames).values())


# ---------------------------------------------------------------------------
# @check
# ---------------------------------------------------------------------------


def test_a_decorated_function_is_a_check_named_after_it():
    from polspec import check

    class Priced(FrameSpec):
        subtotal = ColSpec(pl.Float64, bounds=(0, 100))
        total = ColSpec(pl.Float64, bounds=(0, 110))
        __checks__ = [Check(col("subtotal") >= 0, name="listed")]

        @check
        def total_gte_subtotal():
            return col("total") >= col("subtotal")

        @check(name="capped", description="no total past 105", ignore_nulls=False)
        def anything():
            return col("total") <= 105

    checks = Priced.spec.checks
    assert [c.name for c in checks] == ["listed", "total_gte_subtotal", "capped"]
    assert checks[2].description == "no total past 105"
    assert checks[2].ignore_nulls is False
    assert "total_gte_subtotal" not in vars(Priced)
    assert _passes(Priced, Priced.generate(1_000, seed=SEED, checks=True))


def test_a_decorated_predicate_is_written_to_a_spec_file(tmp_path):
    from polspec import check, from_yaml

    class Priced(FrameSpec):
        subtotal = ColSpec(pl.Float64, bounds=(0, 100))
        total = ColSpec(pl.Float64, bounds=(0, 110))

        @check(description="tax never makes a total smaller")
        def total_gte_subtotal():
            return col("total") >= col("subtotal")

    path = tmp_path / "priced.yaml"
    Priced.to_yaml(path)
    assert from_yaml(path).checks == Priced.spec.checks


def test_a_subclass_inherits_decorated_checks():
    from polspec import check

    class Base(FrameSpec):
        a = ColSpec(pl.Int64, bounds=(0, 10))

        @check
        def small():
            return col("a") < 8

    class Child(Base):
        b = ColSpec(pl.Int64)

    assert [c.name for c in Child.spec.checks] == ["small"]


def test_two_checks_of_one_name_are_refused():
    from polspec import check

    with pytest.raises(SpecError, match="two checks named 'small'"):

        class Twice(FrameSpec):
            a = ColSpec(pl.Int64)
            __checks__ = [Check(col("a") < 8, name="small")]

            @check
            def small():
                return col("a") < 5


def test_a_decorated_function_must_return_a_condition():
    from polspec import check

    with pytest.raises(SpecError, match="must return a polars Expr"):

        class Wrong(FrameSpec):
            a = ColSpec(pl.Int64)

            @check
            def nothing():
                return "a < 5"
