"""`order_by`: rows that come sorted.

A spec declares the columns its rows are sorted by, as `DataFrame.sort`
takes them. `generate()` sorts whole rows last, so every value keeps the
others in its row and every claim about a row's columns still holds;
`validate()` reports each row that belongs before the one above it; the
streaming verbs refuse, as an order is a property of the whole frame.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from polspec import (
    Check,
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSpec,
    Hierarchy,
    SpecError,
    TableSpec,
    col,
    diff,
    from_yaml,
    generate,
    generate_batches,
    inspect,
    profile,
    scan,
    synthesize,
    to_markdown,
    to_python,
    to_yaml,
    validate,
)


class Events(FrameSpec):
    __order_by__ = ["user", "ts"]
    __descending__ = [False, True]
    user = ColSpec(pl.Int64, bounds=(1, 20))
    ts = ColSpec(
        pl.Datetime("us"),
        bounds=(dt.datetime(2024, 1, 1), dt.datetime(2024, 3, 1)),
        nullable=True,
        null_probability=0.1,
    )
    status = ColSpec(pl.Enum(["NEW", "PAID"]))
    paid = ColSpec(
        pl.Boolean, rules=[ColRule(when=col("status") == "PAID", choices=[True])]
    )


def _sorted(df: pl.DataFrame, spec: TableSpec) -> pl.DataFrame:
    return df.sort(
        list(spec.order_by),
        descending=list(spec.descending),
        nulls_last=True,
        maintain_order=True,
    )


# ---------------------------------------------------------------------------
# Declaring
# ---------------------------------------------------------------------------


def test_an_order_is_declared_as_sort_takes_it():
    spec = Events.spec
    assert spec.order_by == ("user", "ts")
    assert spec.descending == (False, True)
    # One bool for every column, and a bare name for one column.
    one = TableSpec("T", {"a": ColSpec(pl.Int8)}, order_by="a", descending=True)
    assert (one.order_by, one.descending) == (("a",), (True,))
    assert TableSpec("T", {"a": ColSpec(pl.Int8)}).order_by == ()


@pytest.mark.parametrize(
    ("order", "message"),
    [
        ({"order_by": ["missing"]}, "unknown column 'missing'"),
        ({"order_by": ["a", "a"]}, "names 'a' twice"),
        ({"order_by": ["a"], "descending": [True, False]}, "give one per column"),
        ({"descending": True}, "order_by is empty"),
        ({"order_by": ["items"]}, "a column of single values"),
        ({"order_by": 3}, "a column name or a sequence"),
    ],
)
def test_an_order_that_cannot_hold_is_refused(order, message):
    columns = {"a": ColSpec(pl.Int8), "items": ColSpec(pl.List(pl.Int8))}
    with pytest.raises(SpecError, match=message):
        TableSpec("T", columns, **order)


def test_descending_without_an_order_is_refused_in_a_class_body():
    with pytest.raises(SpecError, match="needs an __order_by__"):

        class Loose(FrameSpec):
            __descending__ = True
            a = ColSpec(pl.Int8)


def test_a_subclass_inherits_its_parents_order_unless_it_declares_one():
    class More(Events):
        note = ColSpec(pl.String)

    class Reordered(Events):
        __order_by__ = ["ts"]

    assert (More.spec.order_by, More.spec.descending) == (
        Events.spec.order_by,
        Events.spec.descending,
    )
    assert (Reordered.spec.order_by, Reordered.spec.descending) == (("ts",), (False,))


def test_the_order_follows_its_columns_through_drop_select_and_rename():
    spec = Events.spec
    assert spec.rename({"user": "account"}).order_by == ("account", "ts")
    # An order missing one of its columns is not the order declared.
    assert spec.drop("ts").order_by == ()
    assert spec.select("user", "status").order_by == ()
    assert spec.drop("paid").order_by == ("user", "ts")
    assert spec.with_order_by("status").order_by == ("status",)
    assert spec.with_order_by().order_by == ()


# ---------------------------------------------------------------------------
# Generating
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["random", "cartesian"])
def test_generated_rows_come_sorted_and_keep_every_claim(method):
    df = Events.generate(2_000, seed=5, method=method)
    assert df.equals(_sorted(df, Events.spec))
    # The rule still holds: whole rows moved, not one column's values.
    assert df.filter(pl.col("status") == "PAID")["paid"].all()
    # Nulls last, within each user.
    last = df.group_by("user", maintain_order=True).agg(pl.col("ts").last())
    assert last["ts"].null_count() > 0
    validate(Events.spec, df)


def test_every_claim_across_a_row_holds_in_order():
    """Checks, composite keys, a self-referencing foreign key and a
    hierarchy: each holds after the rows are sorted, because each is a claim
    about values in a row, not about where the row is."""

    class Tree(FrameSpec):
        __order_by__ = ["rank"]
        __descending__ = True
        __unique_together__ = [("a", "b")]
        __checks__ = [Check(col("a") <= col("b") + 100, name="a_le_b")]
        __hierarchy__ = Hierarchy(child="node", parent="up", max_depth=3)
        node = ColSpec(pl.String, string_length=(6, 6))
        up = ColSpec(pl.String)
        rank = ColSpec(pl.Float64, bounds=(0, 1), nan_probability=0.05)
        a = ColSpec(pl.Int16, bounds=(0, 60))
        b = ColSpec(pl.Int16, bounds=(0, 60))
        ref = ColSpec(pl.String, nullable=True, null_probability=0.3)
        __foreign_keys__ = [ForeignKey("ref", references="self", ref_columns="node")]

    df = Tree.generate(800, seed=9)
    assert df.equals(_sorted(df, Tree.spec))
    validate(Tree.spec, df)


def test_ties_keep_the_order_they_were_generated_in():
    """The sort is stable: rows that tie on the order keep the order the
    engine drew them in, so the same seed gives the same frame."""
    spec = TableSpec(
        "Ties",
        {"k": ColSpec(pl.Int8, bounds=(0, 2)), "v": ColSpec(pl.Int64)},
        order_by=["k"],
    )
    unordered = generate(spec.with_order_by(), 500, seed=4)
    ordered = generate(spec, 500, seed=4)
    assert ordered.equals(unordered.sort("k", maintain_order=True))


def test_an_order_does_not_change_the_values_drawn():
    """Only where rows sit: the same seed draws the same rows, ordered."""
    unordered = Events.spec.with_order_by()
    a = generate(unordered, 300, seed=1)
    b = generate(Events.spec, 300, seed=1)
    assert b.equals(_sorted(a, Events.spec))


@pytest.mark.parametrize(
    "stream",
    [
        lambda spec: list(generate_batches(spec, 100, batch_size=10)),
        lambda spec: scan(spec, 100).collect(),
    ],
    ids=["generate_batches", "scan"],
)
def test_streaming_an_ordered_spec_is_refused_by_name(stream):
    with pytest.raises(SpecError, match="declares order_by"):
        stream(Events.spec)


def test_a_sink_refuses_an_ordered_spec(tmp_path):
    with pytest.raises(SpecError, match="declares order_by"):
        Events.sink_parquet(tmp_path / "events.parquet", 100)


# ---------------------------------------------------------------------------
# Validating
# ---------------------------------------------------------------------------


def test_a_row_out_of_order_is_found():
    df = Events.generate(500, seed=2)
    swapped = pl.concat([df[1:2], df[0:1], df[2:]])
    report = inspect(Events.spec, swapped)
    (finding,) = [f for f in report.findings if f.code == "order"]
    assert finding.details == {"order_by": ["user", "ts"], "descending": [False, True]}
    assert "user, ts (descending)" in finding.message
    # The rows found are the ones that belong earlier.
    assert report.rows(finding).collect().height >= 1


def test_nulls_and_nan_sort_as_polars_sorts_them():
    spec = TableSpec(
        "F",
        {"x": ColSpec(pl.Float64, nullable=True, nan_probability=0.1)},
        order_by=["x"],
    )
    good = pl.DataFrame({"x": [-1.0, 0.0, 2.5, float("nan"), None]})
    assert not inspect(spec, good).findings
    bad = pl.DataFrame({"x": [-1.0, None, 0.0, float("nan"), 2.5]})
    (finding,) = inspect(spec, bad).findings
    assert finding.code == "order"


def test_the_order_check_can_be_skipped():
    df = Events.generate(200, seed=2).reverse()
    assert not inspect(Events.spec, df, order=False).findings
    assert [f.code for f in inspect(Events.spec, df).findings] == ["order"]


def test_an_order_over_a_column_the_frame_lacks_is_not_checked():
    df = Events.generate(50, seed=2).drop("ts")
    codes = {f.code for f in inspect(Events.spec, df).findings}
    assert codes == {"missing_columns"}


# ---------------------------------------------------------------------------
# Files and drift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("descending", [False, True, [True, False]])
def test_an_order_round_trips_through_yaml_and_python(tmp_path, descending):
    spec = Events.spec.with_order_by("user", "ts", descending=descending)
    to_yaml(spec, tmp_path / "events.yaml")
    assert from_yaml(tmp_path / "events.yaml") == spec
    to_python(spec, tmp_path / "events_spec.py")
    namespace: dict = {}
    exec((tmp_path / "events_spec.py").read_text(encoding="utf-8"), namespace)
    assert namespace[spec.name].spec == spec


def test_a_file_writes_one_flag_when_every_column_agrees(tmp_path):
    to_yaml(Events.spec.with_order_by("user", descending=True), tmp_path / "e.yaml")
    text = (tmp_path / "e.yaml").read_text(encoding="utf-8")
    assert "order_by:\n- user\ndescending: true\n" in text
    to_yaml(Events.spec.with_order_by("user"), tmp_path / "f.yaml")
    assert "descending" not in (tmp_path / "f.yaml").read_text(encoding="utf-8")


def test_adding_an_order_is_breaking_and_removing_one_compatible():
    plain = Events.spec.with_order_by()
    (added,) = diff(plain, Events.spec).findings
    assert (added.code, added.severity) == ("constraint_added", "breaking")
    (removed,) = diff(Events.spec, plain).findings
    assert (removed.code, removed.severity) == ("constraint_removed", "compatible")
    changed = diff(Events.spec, Events.spec.with_order_by("ts")).findings
    assert sorted(f.severity for f in changed) == ["breaking", "compatible"]


# ---------------------------------------------------------------------------
# Profiling and rendering
# ---------------------------------------------------------------------------


def _log(n: int = 3_000) -> pl.DataFrame:
    """A log: in time order, a user that repeats, an amount that does not
    sort."""
    start = dt.datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "user": [i % 7 for i in range(n)],
            "ts": [start + dt.timedelta(minutes=i) for i in range(n)],
            "amount": [float((i * 37) % 101) for i in range(n)],
        }
    )


@pytest.mark.parametrize("descending", [False, True])
def test_profile_declares_the_order_a_frame_comes_in(descending):
    log = _log().reverse() if descending else _log()
    spec = profile(log)
    assert (spec.order_by, spec.descending) == (("ts",), (descending,))
    validate(spec, log)


def test_profile_takes_no_order_from_chance_or_shuffled_rows():
    # Sorted, but by five values: an order that could be chance.
    few = pl.DataFrame({"k": sorted(i % 5 for i in range(500))})
    assert profile(few).order_by == ()
    shuffled = _log().sample(fraction=1.0, shuffle=True, seed=1)
    assert profile(shuffled).order_by == ()
    assert profile(_log(), detect_order=False).order_by == ()


def test_a_sorted_column_with_nulls_last_is_still_an_order():
    log = _log().with_columns(
        pl.when(pl.int_range(pl.len()) >= 2_900)
        .then(None)
        .otherwise(pl.col("ts"))
        .alias("ts")
    )
    assert profile(log).order_by == ("ts",)


def test_a_sample_keeps_the_sources_order():
    """`sample=` draws rows at random but keeps them in file order, so a
    sorted file's profile still sees its order."""
    assert profile(_log(), sample=500).order_by == ("ts",)


def test_synthesized_rows_come_in_the_sources_order():
    fake = synthesize(_log(), 1_000, seed=2)
    assert fake["ts"].is_sorted()


def test_from_dataframe_detects_an_order_only_when_asked():
    assert FrameSpec.from_dataframe(_log()).spec.order_by == ()
    asked = FrameSpec.from_dataframe(_log(), detect_order=True)
    assert asked.spec.order_by == ("ts",)


def test_schema_infer_declares_an_order_with_order(tmp_path):
    from polspec.cli import main

    source = tmp_path / "log.parquet"
    _log().write_parquet(source)
    plain, ordered = tmp_path / "plain.yaml", tmp_path / "ordered.yaml"
    assert main(["schema", "infer", str(source), "-o", str(plain)]) == 0
    assert main(["schema", "infer", str(source), "-o", str(ordered), "--order"]) == 0
    assert from_yaml(plain).order_by == ()
    assert from_yaml(ordered).order_by == ("ts",)


def test_the_markdown_names_the_order():
    text = to_markdown(Events.spec)
    assert "- **Row Order:** `user`, `ts` (descending)" in text


def test_order_detection_reads_the_head_first_and_still_checks_the_whole():
    """The head rules a column out cheaply; a column sorted only for its
    head is still not an order, because the survivors are checked whole."""
    from polspec.profiler import ORDER_PREFIX_ROWS

    n = ORDER_PREFIX_ROWS * 3
    sorted_then_not = list(range(ORDER_PREFIX_ROWS)) + list(
        range(n - ORDER_PREFIX_ROWS, 0, -1)
    )
    frame = pl.DataFrame(
        {"head_only": sorted_then_not, "noise": [(i * 7919) % 101 for i in range(n)]}
    )
    assert profile(frame).order_by == ()
    ordered = profile(frame.with_columns(pl.int_range(n).alias("id")))
    assert (ordered.order_by, ordered.descending) == (("id",), (False,))
