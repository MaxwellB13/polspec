"""`scan()`: a LazyFrame that generates rows as they are collected.

The property that makes the projection pushdown free rather than a trade:
every column is seeded by its name and every pass by what it is for, so
dropping a column's neighbours cannot move it. A scan is batched, so it
carries `generate_batches`' terms -- no hierarchy, uniqueness within a
batch, and passes drawn per batch.
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
    Hierarchy,
    Registry,
    SpecError,
    col,
    scan,
)
from polspec.generation.scan import _takes_explain_labels, needed_columns

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


def test_the_closure_follows_what_a_pass_reads():
    spec = Source.spec
    assert needed_columns(spec, frozenset({"plain"})) == ["plain"]
    # A ruled column pulls in the column its `when` names.
    assert needed_columns(spec, frozenset({"ruled"})) == ["flag", "ruled"]
    # A composite key is repaired as a group, so one member pulls in the rest.
    assert needed_columns(spec, frozenset({"a"})) == ["a", "b"]
    assert needed_columns(spec, frozenset({"flag"})) == ["flag"]


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


def _spy(monkeypatch) -> list[tuple[tuple[str, ...], int, int | None]]:
    """Records (columns, rows, batch_size) for each call the reader makes."""
    import polspec.generation as generation

    seen: list[tuple[tuple[str, ...], int, int | None]] = []
    real = generation.generate_batches

    def spy(spec, n, **kwargs):
        seen.append((tuple(spec.columns), n, kwargs.get("batch_size")))
        yield from real(spec, n, **kwargs)

    monkeypatch.setattr(generation, "generate_batches", spy)
    return seen


def test_head_generates_only_the_rows_asked_for(monkeypatch):
    seen = _spy(monkeypatch)
    out = Source.scan(50_000_000, seed=SEED).select("plain").head(3).collect()
    assert out.height == 3
    assert seen == [(("plain",), 3, 100_000)]


def test_a_projection_reaches_the_reader_as_its_closure(monkeypatch):
    seen = _spy(monkeypatch)
    Source.scan(1_000, seed=SEED).select("ruled").head(10).collect()
    assert seen == [(("flag", "ruled"), 10, 100_000)]


def test_batch_size_is_the_callers_when_given_and_polars_hint_otherwise(monkeypatch):
    seen = _spy(monkeypatch)
    Source.scan(1_000, seed=SEED, batch_size=250).collect()
    assert seen[0][2] == 250


def test_a_predicate_filters_rows_that_were_drawn_it_never_narrows_the_draw(
    monkeypatch,
):
    """`n` rows are generated and the matching ones kept. Drawing only
    matching rows would change what a rate or a `unique` column means."""
    seen = _spy(monkeypatch)
    lf = Source.scan(10_000, seed=SEED)
    got = lf.filter(pl.col("a") == 1).collect()
    assert seen[0][1] == 10_000, "every row is drawn, then filtered"
    assert got.equals(lf.collect().filter(pl.col("a") == 1))
    assert 0 < got.height < 10_000


def test_a_predicate_on_a_column_the_projection_drops_still_generates_it(
    monkeypatch,
):
    seen = _spy(monkeypatch)
    lf = Source.scan(5_000, seed=SEED)
    got = lf.filter(pl.col("a") == 1).select("plain").collect()
    assert seen[0][0] == ("plain", "a", "b"), "the predicate's column joins the closure"
    assert got.columns == ["plain"]
    assert got.equals(lf.collect().filter(pl.col("a") == 1).select("plain"))


# ---------------------------------------------------------------------------
# Terms, refusals and the sinks
# ---------------------------------------------------------------------------


def test_a_hierarchy_is_refused_where_it_is_asked_for():
    class Tree(FrameSpec):
        ref = ColSpec(pl.Int64)
        parent = ColSpec(pl.Int64)
        __hierarchy__ = Hierarchy(child="ref", parent="parent", max_depth=3)

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


def _register_standing_in(monkeypatch, accepts_labels: bool) -> dict:
    """A `register_io_source` that records what it was given, with or
    without the explain parameters Polars 2 added -- so both branches run on
    whichever Polars is installed."""
    seen: dict = {}

    if accepts_labels:

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

    else:

        def register(io_source, *, schema, validate_schema=False, is_pure=False):
            seen["called"] = True
            return pl.LazyFrame(schema=schema)

    monkeypatch.setattr(pl.io.plugins, "register_io_source", register)
    return seen


def test_a_polars_that_takes_explain_labels_is_given_them(monkeypatch):
    seen = _register_standing_in(monkeypatch, accepts_labels=True)
    Named.scan(1_000_000, seed=100, batch_size=50_000)
    assert seen["explain_name"] == "polspec: Named"
    assert seen["explain_detail"] == "1,000,000 rows, seed=100, batches of 50,000"


def test_a_polars_without_them_is_called_exactly_as_before(monkeypatch):
    """Polars 1 has no explain parameters and would refuse them; the call
    made there is the one polspec has always made."""
    seen = _register_standing_in(monkeypatch, accepts_labels=False)
    Named.scan(10, seed=1)
    assert seen == {"called": True}


def test_the_detail_says_what_was_asked_for(monkeypatch):
    seen = _register_standing_in(monkeypatch, accepts_labels=True)
    Named.scan(12_345, method="cartesian")
    assert seen["explain_detail"] == (
        "12,345 rows, unseeded, batches chosen by polars, method=cartesian"
    )


@pytest.mark.skipif(
    not _takes_explain_labels(pl.io.plugins.register_io_source),
    reason="this Polars gives an IO source no name in explain()",
)
def test_explain_names_the_scan_where_polars_can():
    plan = Named.scan(1_000_000, seed=100).select("id").explain()
    assert plan.startswith("PYTHON[polspec: Named] SCAN")
    assert "INFO: 1,000,000 rows, seed=100, batches chosen by polars" in plan
    assert "PROJECT 1/2 COLUMNS" in plan
