"""`scan()`: a LazyFrame that generates rows as they are collected.

The property that makes the projection pushdown free rather than a trade:
every column is seeded by its name and every pass by what it is for, so
dropping a column's neighbours cannot move it. A scan is batched, so it
carries `generate_batches`' terms -- no hierarchy, a composite key distinct
within a batch, and the passes that rewrite a column drawn per batch.
"""

from __future__ import annotations

import itertools

import polars as pl
import pytest
from polspec import (
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSpec,
    GenerationError,
    Hierarchy,
    Registry,
    SpecError,
    col,
    scan,
)

ROWS, SEED = 250_000, 1


class Source(FrameSpec):
    """Plain columns, a ruled column that reads one, and a composite key, so
    the closure has something to follow."""

    flag = ColSpec(pl.Boolean)
    ruled = ColSpec(
        pl.Int64,
        bounds=(0, 100),
        rules=[ColRule(when=col("flag"), choices=tuple(range(50, 100)))],
    )
    plain = ColSpec(pl.String, string_length=(3, 6))
    a = ColSpec(pl.Int64, bounds=(0, 5))
    b = ColSpec(pl.Int64, bounds=(0, 1_000_000))
    __unique_together__ = [("a", "b")]


# ---------------------------------------------------------------------------
# The projection property
# ---------------------------------------------------------------------------


def test_projecting_cannot_change_what_a_column_holds():
    """Every subset of the columns, against the same scan collected whole."""
    lf = Source.scan(ROWS, seed=SEED)
    whole = lf.collect()
    names = whole.columns
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            assert lf.select(subset).collect().equals(whole.select(subset)), subset


class Covered(FrameSpec):
    """Three coverage dimensions and a filler, so dropping any of them would
    change the product the others are laid out in."""

    kind = ColSpec(pl.Enum(["x", "y", "z"]))
    flag = ColSpec(pl.Boolean)
    amount = ColSpec(pl.Int64, bounds=(-5, 5))
    note = ColSpec(pl.String, string_length=(3, 6))


def test_projecting_a_cartesian_scan_cannot_change_what_a_column_holds():
    """The coverage set is the product of every dimension, so a projected
    cartesian scan is the whole one, projected -- across batches, and past
    the coverage set into the padding."""
    lf = Covered.scan(200, seed=SEED, method="cartesian", batch_size=7)
    whole = lf.collect()
    names = whole.columns
    for size in range(1, len(names) + 1):
        for subset in itertools.combinations(names, size):
            assert lf.select(subset).collect().equals(whole.select(subset)), subset


def test_a_column_no_pass_rewrites_is_the_frame_generate_would_make():
    lf = Source.scan(ROWS, seed=SEED)
    eager = Source.generate(ROWS, seed=SEED)
    plain = ["flag", "plain"]
    assert lf.select(plain).collect().equals(eager.select(plain))


@pytest.mark.parametrize(
    "column",
    [
        "plain",
        "ruled",  # its rule reads `flag`, so `flag` is generated with it
        "a",  # a composite key is repaired as a group, so `b` comes too
        "flag",
    ],
)
def test_one_column_scans_as_the_whole_scan_holds_it(column):
    """Projected alone, a column still gets what its passes read."""
    lf = Source.scan(ROWS, seed=SEED)
    assert lf.select(column).collect().equals(lf.collect().select(column))


def test_a_foreign_key_column_scans_against_its_parent():
    class Parent(FrameSpec):
        id = ColSpec(pl.Int64, unique=True, bounds=(1, 500))

    class Child(FrameSpec):
        parent_id = ColSpec(pl.Int64)
        note = ColSpec(pl.String)
        __foreign_keys__ = [
            ForeignKey("parent_id", references=Parent, ref_columns="id")
        ]

    parent = Parent.generate(500, seed=2)
    got = Child.scan(5_000, seed=SEED, references={Parent: parent}).collect()
    assert set(got["parent_id"].to_list()) <= set(parent["id"].to_list())


# ---------------------------------------------------------------------------
# What reaches the reader
# ---------------------------------------------------------------------------


# A column whose domain holds ten values, so drawing more than ten rows of
# it raises: a scan that succeeds with it never drew more rows of it than
# that -- or, projected away, any.
_TIGHT = ColSpec(pl.Int8, bounds=(0, 9), unique=True)


def test_head_generates_only_the_rows_asked_for():
    spec = Source.spec.with_columns({"tight": _TIGHT})
    out = scan(spec, 50_000_000, seed=SEED).head(3).collect()
    assert out.height == 3


def test_a_column_projected_away_is_not_generated():
    spec = Source.spec.with_columns({"tight": _TIGHT})
    lf = scan(spec, 1_000, seed=SEED)
    assert lf.select("plain").collect().height == 1_000
    with pytest.raises(GenerationError, match="'tight' is unique"):
        lf.collect()


def test_a_projection_keeps_what_its_columns_depend_on():
    """`ruled` reads `flag`: projected alone, it holds what it holds in the
    whole frame."""
    lf = Source.scan(1_000, seed=SEED)
    assert lf.select("ruled").collect().equals(lf.collect().select("ruled"))


def test_batch_size_is_the_callers_when_given():
    frame = Source.scan(1_000, seed=SEED, batch_size=250).collect()
    assert [len(chunk) for chunk in frame["plain"].get_chunks()] == [250] * 4


def test_a_predicate_filters_rows_that_were_drawn_it_never_narrows_the_draw():
    """`n` rows are generated and the matching ones kept. Drawing only
    matching rows would change what a rate or a `unique` column means."""
    lf = Source.scan(10_000, seed=SEED)
    got = lf.filter(pl.col("a") == 1).collect()
    assert got.equals(lf.collect().filter(pl.col("a") == 1))
    assert 0 < got.height < 10_000


def test_a_predicate_on_a_column_the_projection_drops_still_generates_it():
    lf = Source.scan(5_000, seed=SEED)
    got = lf.filter(pl.col("a") == 1).select("plain").collect()
    assert got.columns == ["plain"]
    assert got.equals(lf.collect().filter(pl.col("a") == 1).select("plain"))


# ---------------------------------------------------------------------------
# Terms, refusals and the sinks
# ---------------------------------------------------------------------------


def test_a_whole_layout_hierarchy_is_refused_where_it_is_asked_for():
    """2.1: the default layout streams; only 2.0's whole-frame one cannot."""

    class Tree(FrameSpec):
        ref = ColSpec(pl.Int64)
        parent = ColSpec(pl.Int64)
        __hierarchy__ = Hierarchy(
            child="ref", parent="parent", max_depth=3, layout="whole"
        )

    with pytest.raises(SpecError, match="which scan cannot produce"):
        Tree.scan(10, seed=SEED)


def test_bad_arguments_are_refused_before_a_lazyframe_exists():
    with pytest.raises(ValueError, match="n must be >= 0"):
        Source.scan(-1)
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        Source.scan(10, batch_size=0)
    with pytest.raises(ValueError, match="Unknown method"):
        Source.scan(10, method="spiral")  # type: ignore[arg-type]


def test_the_schema_is_the_specs_before_anything_is_generated():
    assert Source.scan(10**12, seed=SEED).collect_schema() == Source.schema()


def test_zero_rows_gives_an_empty_typed_frame():
    got = Source.scan(0, seed=SEED).collect()
    assert got.height == 0 and got.schema == Source.schema()


def test_a_scan_sinks_to_parquet_in_batches(tmp_path):
    path = tmp_path / "rows.parquet"
    Source.scan(50_000, seed=SEED, batch_size=10_000).sink_parquet(path)
    back = pl.read_parquet(path)
    assert back.height == 50_000
    assert back.equals(Source.scan(50_000, seed=SEED, batch_size=10_000).collect())


def test_cartesian_covers_before_it_pads():
    class Covered(FrameSpec):
        e = ColSpec(pl.Enum(["x", "y", "z"]))
        b = ColSpec(pl.Boolean)

    got = Covered.scan(100, seed=SEED, method="cartesian").collect()
    assert got.height >= 6
    assert got.head(6).unique().height == 6


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_scan_all_threads_parents_and_agrees_with_generate_all():
    class Customers(FrameSpec):
        id = ColSpec(pl.Int64, unique=True, bounds=(1, 1_000))

    class Orders(FrameSpec):
        order_id = ColSpec(pl.Int64, bounds=(1, 100_000))
        customer_id = ColSpec(pl.Int64)
        __foreign_keys__ = [
            ForeignKey("customer_id", references=Customers, ref_columns="id")
        ]

    registry = Registry(Customers, Orders)
    lazy = registry.scan_all(200, seed=3)
    assert {k: type(v).__name__ for k, v in lazy.items()} == {
        "Customers": "LazyFrame",
        "Orders": "LazyFrame",
    }
    frames = {name: lf.collect() for name, lf in lazy.items()}
    assert set(frames["Orders"]["customer_id"]) <= set(frames["Customers"]["id"])
    # A parent is generated eagerly, so it is the frame generate_all makes.
    assert frames["Customers"].equals(registry.generate_all(200, seed=3)["Customers"])


def test_scan_is_exported_as_a_function_over_a_tablespec():
    assert scan(Source.spec, 10, seed=SEED).collect().height == 10


# ---------------------------------------------------------------------------
# What a scan says about itself in explain(), where Polars lets it
# ---------------------------------------------------------------------------


class Named(FrameSpec):
    id = ColSpec(pl.Int64, unique=True)
    total = ColSpec(pl.Float64)


def _register_standing_in(monkeypatch) -> dict:
    """A `register_io_source` that records the explain labels it was given."""
    seen: dict = {}

    def register(
        io_source,
        *,
        schema,
        validate_schema=False,
        is_pure=False,
        explain_name=None,
        explain_detail=None,
    ):
        seen.update(explain_name=explain_name, explain_detail=explain_detail)
        return pl.LazyFrame(schema=schema)

    monkeypatch.setattr(pl.io.plugins, "register_io_source", register)
    return seen


def test_the_scan_is_given_explain_labels(monkeypatch):
    seen = _register_standing_in(monkeypatch)
    Named.scan(1_000_000, seed=100, batch_size=50_000)
    assert seen["explain_name"] == "polspec: Named"
    assert seen["explain_detail"] == "1,000,000 rows, seed=100, batches of 50,000"


def test_the_detail_says_what_was_asked_for(monkeypatch):
    seen = _register_standing_in(monkeypatch)
    Named.scan(12_345, method="cartesian")
    assert seen["explain_detail"] == (
        "12,345 rows, unseeded, batches chosen by polars, method=cartesian"
    )


def test_explain_names_the_scan():
    plan = Named.scan(1_000_000, seed=100).select("id").explain()
    assert plan.startswith("PYTHON[polspec: Named] SCAN")
    assert "INFO: 1,000,000 rows, seed=100, batches chosen by polars" in plan
    assert "PROJECT 1/2 COLUMNS" in plan
