"""A drawn order: `order="drawn"` gives the rows their order by drawing the
first `order_by` column already in order, each row's value from its
position, so the streaming verbs can produce an ordered spec. *Since 2.2.*

Through the public API only: the engine's half is `generation/test_engine.py`.
"""

import datetime as dt
import warnings

import polars as pl
import polspec
import pytest
from polspec import (
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
    generate_batches,
)

SEED = 3
# Past the 65,536 rows a column's table holds, so the stratified draw runs.
BIG = 150_000


class Events(FrameSpec):
    TS = ColSpec(
        pl.Datetime("us"), bounds=(dt.datetime(2024, 1, 1), dt.datetime(2025, 1, 1))
    )
    USER = ColSpec(pl.Int64, bounds=(1, 10_000))
    KIND = ColSpec(pl.String, choices=["click", "view", "buy"])
    __order_by__ = "TS"
    __order__ = "drawn"


# ---------------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1_000, BIG])
def test_the_frame_is_in_order_and_validates(n):
    frame = Events.generate(n, seed=SEED)
    assert frame.height == n
    assert frame["TS"].is_sorted()
    Events.validate(frame)


@pytest.mark.parametrize("n", [1_000, BIG])
def test_the_other_columns_are_what_an_unordered_spec_draws(n):
    """Every column is seeded by its own name, so drawing the order column
    in order moves no other: they are the unordered spec's, row for row."""
    unordered = Events.spec.with_order_by()
    assert (
        Events.generate(n, seed=SEED)
        .drop("TS")
        .equals(generate(unordered, n, seed=SEED).drop("TS"))
    )


def test_a_small_frame_holds_the_values_a_sort_would():
    """Up to 65,536 rows, the order column is the column's own draw,
    sorted: the values `order="sorted"` gives it, in their places."""
    sorted_spec = Events.spec.with_order_by("TS", order="sorted")
    drawn = Events.generate(5_000, seed=SEED)
    assert drawn["TS"].equals(generate(sorted_spec, 5_000, seed=SEED)["TS"])


def test_descending_and_nulls_last():
    spec = TableSpec(
        "T",
        {
            "x": ColSpec(pl.Float64, nullable=True, null_probability=0.2),
            "y": ColSpec(pl.Int8),
        },
        order_by="x",
        descending=True,
        order="drawn",
    )
    frame = generate(spec, BIG, seed=SEED)
    values = frame["x"].drop_nulls()
    assert values.is_sorted(descending=True)
    assert frame["x"].is_null().arg_true().min() == values.len()
    assert frame["x"].null_count() == round(0.2 * BIG)
    polspec.validate(spec, frame)


@pytest.mark.parametrize(
    "column",
    [
        ColSpec(pl.Int32, bounds=(0, 50)),
        ColSpec(pl.Float32, distribution="gamma"),
        ColSpec(pl.Date),
        ColSpec(pl.Duration("ms")),
        ColSpec(pl.Time),
        ColSpec(pl.Decimal(10, 2), bounds=(0, 100)),
        ColSpec(pl.Boolean),
        ColSpec(pl.String, choices={"a": 0.7, "b": 0.2, "c": 0.1}),
        ColSpec(pl.Enum(["high", "mid", "low"])),
        ColSpec(pl.Categorical),
        ColSpec(pl.Int64, unique=True),
        ColSpec(pl.Datetime("ns"), unique=True),
    ],
    ids=lambda c: repr(c.dtype),
)
def test_every_kind_of_column_it_holds_streams_in_order(column):
    spec = TableSpec(
        "T", {"k": column, "v": ColSpec(pl.Int16)}, order_by="k", order="drawn"
    )
    whole = generate(spec, BIG, seed=SEED)
    assert whole["k"].is_sorted()
    batches = pl.concat(list(generate_batches(spec, BIG, batch_size=40_000, seed=SEED)))
    assert batches.equals(whole)
    polspec.validate(spec, whole)


def test_a_unique_first_column_orders_the_others_too():
    """With no ties in the first column, the frame is in order by every
    `order_by` column, whatever the others hold."""
    spec = TableSpec(
        "T",
        {"id": ColSpec(pl.Int64, unique=True), "ts": ColSpec(pl.Date)},
        order_by=["id", "ts"],
        order="drawn",
    )
    polspec.validate(spec, generate(spec, BIG, seed=SEED))


def test_a_rule_reading_the_order_column_reads_it_in_order():
    spec = TableSpec(
        "T",
        {
            "score": ColSpec(pl.Int64, bounds=(0, 100)),
            "band": ColSpec(
                pl.String,
                choices=["low", "high"],
                rules=(ColRule(when=col("score") >= 50, choices=["high"]),),
            ),
        },
        order_by="score",
        order="drawn",
    )
    frame = generate(spec, 20_000, seed=SEED)
    assert frame["score"].is_sorted()
    polspec.validate(spec, frame)
    # Streamed too: each batch's rule reads the column drawn for its rows.
    batches = generate_batches(spec, 20_000, batch_size=3_000, seed=SEED)
    assert pl.concat(list(batches)).equals(frame)


def test_a_key_into_the_order_column_points_at_its_drawn_values():
    spec = TableSpec(
        "T",
        {
            "id": ColSpec(pl.Int64, unique=True),
            "parent": ColSpec(pl.Int64),
        },
        foreign_keys=[ForeignKey("parent", references="self", ref_columns="id")],
        order_by="id",
        order="drawn",
    )
    polspec.validate(spec, generate(spec, 5_000, seed=SEED))
    for batch in generate_batches(spec, 5_000, batch_size=1_000, seed=SEED):
        assert batch["parent"].is_in(batch["id"]).all()


# ---------------------------------------------------------------------------
# Streaming: every batch is the rows generate() makes there
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 777, 40_000])
def test_batches_are_the_whole_frame(batch_size):
    n = 3_000 if batch_size == 1 else BIG
    whole = Events.generate(n, seed=SEED)
    batches = Events.generate_batches(n, batch_size=batch_size, seed=SEED)
    assert pl.concat(list(batches)).equals(whole)


def test_a_scan_and_its_projections_are_the_whole_frame():
    whole = Events.generate(BIG, seed=SEED)
    lazy = Events.scan(BIG, seed=SEED, batch_size=33_333)
    assert lazy.collect().equals(whole)
    assert lazy.select("TS").collect().equals(whole.select("TS"))
    assert lazy.select("USER").collect().equals(whole.select("USER"))
    assert lazy.head(10).collect().equals(whole.head(10))
    assert (
        lazy.filter(pl.col("KIND") == "buy")
        .collect()
        .equals(whole.filter(pl.col("KIND") == "buy"))
    )


def test_a_sink_writes_the_whole_frame(tmp_path):
    path = tmp_path / "events.parquet"
    Events.sink_parquet(path, BIG, seed=SEED, batch_size=50_000)
    assert pl.read_parquet(path).equals(Events.generate(BIG, seed=SEED))


def test_a_sorted_order_still_refuses_to_stream_and_names_the_way_out():
    class Sorted(Events):
        __order__ = "sorted"

    with pytest.raises(SpecError, match="Declare order='drawn'"):
        Sorted.scan(100)


def test_a_sorted_order_that_cannot_be_drawn_says_why():
    class Sorted(FrameSpec):
        name = ColSpec(pl.String)
        __order_by__ = "name"

    with pytest.raises(SpecError, match="free-text value's order"):
        list(Sorted.generate_batches(100, batch_size=10))


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("columns", "kwargs", "reason"),
    [
        ({"x": ColSpec(pl.String)}, {}, "free-text"),
        ({"x": ColSpec(pl.List(pl.Int64))}, {}, "a column of single values"),
        ({"x": ColSpec(pl.Float64, nan_probability=0.1)}, {}, "nan_probability"),
        (
            {"x": ColSpec(pl.Int64), "y": ColSpec(pl.Int64)},
            {"order_by": ["x", "y"]},
            "rows that tie on it",
        ),
        (
            {
                "x": ColSpec(pl.Int64, unique=True, nullable=True),
                "y": ColSpec(pl.Int64),
            },
            {"order_by": ["x", "y"]},
            "rows that tie on it",
        ),
        (
            {
                "flag": ColSpec(pl.Boolean),
                "x": ColSpec(
                    pl.Int64,
                    choices=[1, 2, 3],
                    rules=(ColRule(when=col("flag"), choices=[1]),),
                ),
            },
            {},
            "writes it after the columns are drawn",
        ),
        (
            {"x": ColSpec(pl.Int64), "y": ColSpec(pl.Int64)},
            {"unique_together": [("x", "y")]},
            "the composite key",
        ),
    ],
)
def test_a_column_it_cannot_draw_is_refused_at_declaration(columns, kwargs, reason):
    kwargs = {"order_by": "x", **kwargs}
    with pytest.raises(SpecError, match=reason):
        TableSpec("T", columns, order="drawn", **kwargs)


def test_an_order_needs_columns_and_a_known_name():
    with pytest.raises(SpecError, match="order_by is empty"):
        TableSpec("T", {"x": ColSpec(pl.Int64)}, order="drawn")
    with pytest.raises(SpecError, match="'sorted' or 'drawn'"):
        TableSpec("T", {"x": ColSpec(pl.Int64)}, order_by="x", order="random")  # ty: ignore[invalid-argument-type]


def test_the_coverage_method_is_refused():
    with pytest.raises(SpecError, match="method='cartesian'"):
        Events.generate(10, method="cartesian")
    with pytest.raises(SpecError, match="method='cartesian'"):
        Events.scan(10, method="cartesian")


def test_the_order_column_cannot_be_pinned_and_others_can():
    with pytest.raises(GenerationError, match="cannot pin 'TS'"):
        Events.generate(10, seed=SEED, fixed={"TS": dt.datetime(2024, 6, 1)})
    frame = Events.generate(1_000, seed=SEED, fixed={"KIND": "buy"})
    assert frame["TS"].is_sorted() and frame["KIND"].unique().to_list() == ["buy"]


def test_checks_are_not_kept_beside_it():
    class Checked(Events):
        __checks__ = [Check(col("USER") > 5_000, name="half")]

    with pytest.warns(UserWarning, match="drawn over exactly n rows"):
        frame = Checked.generate(1_000, seed=SEED)
    assert frame["TS"].is_sorted()
    with pytest.raises(SpecError, match="checks=True cannot keep"):
        Checked.generate(1_000, seed=SEED, checks=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Checked.generate(10, seed=SEED, checks=False)


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


def test_a_subclass_can_change_how_its_parents_order_is_given():
    class Sorted(Events):
        __order__ = "sorted"

    assert Sorted.spec.order_by == ("TS",) and Sorted.spec.order == "sorted"
    assert Events.spec.order == "drawn"


def test_an_order_without_columns_is_refused_in_a_class_body():
    with pytest.raises(SpecError, match="order_by is empty"):

        class Bare(FrameSpec):
            x = ColSpec(pl.Int64)
            __order__ = "drawn"


def test_it_survives_a_spec_file_and_python(tmp_path):
    path = tmp_path / "events.yaml"
    polspec.to_yaml(Events.spec, path)
    assert "order: drawn" in path.read_text(encoding="utf-8")
    assert polspec.from_yaml(path) == Events.spec
    assert "__order__ = 'drawn'" in polspec.to_python(Events.spec)
    plain = Events.spec.with_order_by("TS", order="sorted")
    assert "order" not in polspec.to_dict(plain)


def test_reshaping_keeps_or_drops_it_with_the_order():
    assert Events.spec.rename({"TS": "when"}).order == "drawn"
    assert Events.spec.drop("TS").order == "sorted"
    assert Events.spec.select("USER", "KIND").order == "sorted"
    assert Events.spec.with_order_by("USER").order == "drawn"
    assert Events.spec.with_order_by().order == "sorted"


def test_a_diff_reports_it_as_compatible():
    sorted_spec = Events.spec.with_order_by("TS", order="sorted")
    findings = polspec.diff(sorted_spec, Events.spec).findings
    assert [(f.key, f.severity) for f in findings] == [("order", "compatible")]
