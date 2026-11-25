"""`Hierarchy`: a link table whose shape is declared rather than left to chance.

A self-referencing `ForeignKey` fills a column with values that exist and
promises nothing about the result -- see the boundary tests in
`test_roundtrip.py`. `Hierarchy` promises the shape: one parent per reference,
a bounded depth, and no cycles unless the caller asks for some.

The tests below check both halves of that. The forest is what generation is
for, and the faults are what the fixtures are for: a resolver that walks
parents without a visited set does not fail on a cycle, it runs forever, so
there has to be a way to build one on purpose.
"""

import polars as pl
import pytest
from polspec import (
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSpec,
    GenerationError,
    Hierarchy,
    SpecError,
    col,
)

ROWS = 5_000
SEED = 11
DEPTH = 5


class Links(FrameSpec):
    PARENT_REF = ColSpec(pl.String)
    CHILD_REF = ColSpec(pl.String)
    __hierarchy__ = Hierarchy(child="CHILD_REF", parent="PARENT_REF", max_depth=DEPTH)


def parent_map(df: pl.DataFrame) -> dict:
    return dict(
        zip(
            df["CHILD_REF"].to_list(),
            df["PARENT_REF"].to_list(),
            strict=True,
        )
    )


def depth_of(parent_of: dict, start, limit: int = 200) -> int:
    """Hops from `start` to its ultimate parent, or -1 if it never gets there."""
    node, hops = start, 0
    while node in parent_of:
        node = parent_of[node]
        hops += 1
        if hops > limit:
            return -1
    return hops


def cycle_nodes(parent_of: dict) -> set:
    """Every reference that sits on a loop (not merely below one)."""
    on_loop: set = set()
    for start in parent_of:
        seen: list = []
        node = start
        while node in parent_of:
            if node in seen:
                on_loop.update(seen[seen.index(node) :])
                break
            seen.append(node)
            node = parent_of[node]
            if len(seen) > 200:
                break
    return on_loop


# ---------------------------------------------------------------------------
# The forest
# ---------------------------------------------------------------------------


def test_every_reference_has_exactly_one_parent():
    """What makes "the ultimate parent" a single value rather than a set."""
    df = Links.generate(ROWS, seed=SEED)
    assert df.height == ROWS
    assert df["CHILD_REF"].n_unique() == ROWS


def test_every_chain_terminates_at_an_ultimate_parent():
    """No dangling reference, and nothing that walks forever."""
    df = Links.generate(ROWS, seed=SEED)
    parent_of = parent_map(df)
    children = set(df["CHILD_REF"].to_list())
    roots = {p for p in df["PARENT_REF"].to_list() if p not in children}

    assert roots, "there must be at least one ultimate parent"
    assert not roots & children, "an ultimate parent has no parent of its own"
    # -1 is `depth_of` giving up, which is what a loop or a dangling parent
    # would look like from here.
    assert all(depth_of(parent_of, c) > 0 for c in children)


def test_depth_reaches_max_depth_and_never_exceeds_it():
    """A ceiling nothing reaches would stop testing the boundary silently."""
    parent_of = parent_map(Links.generate(ROWS, seed=SEED))
    depths = [depth_of(parent_of, c) for c in parent_of]
    assert max(depths) == DEPTH
    assert min(depths) >= 1


def test_the_frame_is_not_ordered_by_depth():
    """Nothing downstream should be able to lean on the build order."""
    df = Links.generate(ROWS, seed=SEED)
    parent_of = parent_map(df)
    depths = [depth_of(parent_of, c) for c in df["CHILD_REF"].to_list()]
    assert depths != sorted(depths)


def test_a_clean_hierarchy_validates():
    df = Links.generate(ROWS, seed=SEED)
    assert Links.inspect(df).passed
    Links.validate(df)


def test_the_same_seed_gives_the_same_frame():
    a = Links.generate(ROWS, seed=SEED)
    b = Links.generate(ROWS, seed=SEED)
    assert a.equals(b)
    assert not a.equals(Links.generate(ROWS, seed=SEED + 1))


def test_roots_pins_the_number_of_ultimate_parents():
    class Pinned(FrameSpec):
        up = ColSpec(pl.String)
        down = ColSpec(pl.String)
        __hierarchy__ = Hierarchy(child="down", parent="up", max_depth=3, roots=7)

    df = Pinned.generate(2_000, seed=SEED)
    children = set(df["down"].to_list())
    roots = {p for p in df["up"].to_list() if p not in children}
    assert len(roots) == 7


def test_max_depth_of_one_is_a_flat_table():
    class Flat(FrameSpec):
        up = ColSpec(pl.String)
        down = ColSpec(pl.String)
        __hierarchy__ = Hierarchy(child="down", parent="up", max_depth=1)

    parent_of = parent_map(
        Flat.generate(500, seed=SEED).rename({"down": "CHILD_REF", "up": "PARENT_REF"})
    )
    assert {depth_of(parent_of, c) for c in parent_of} == {1}


def test_zero_rows():
    assert Links.generate(0, seed=SEED).height == 0


def test_too_few_rows_to_reach_the_depth_is_refused():
    with pytest.raises(GenerationError, match="at least 5 row"):
        Links.generate(3, seed=SEED)


# ---------------------------------------------------------------------------
# Faults, which are the point of the fixtures
# ---------------------------------------------------------------------------


def test_cycles_makes_exactly_the_number_asked_for():
    clean = parent_map(Links.generate(ROWS, seed=SEED))
    assert cycle_nodes(clean) == set()

    broken = parent_map(Links.generate(ROWS, seed=SEED, cycles=4))
    loops = cycle_nodes(broken)
    # Four loops, none sharing a reference with another.
    assert len(loops) == 4 * 4


def test_self_references_makes_exactly_the_number_asked_for():
    df = Links.generate(ROWS, seed=SEED, self_references=6)
    same = df.filter(pl.col("CHILD_REF") == pl.col("PARENT_REF"))
    assert same.height == 6


def test_faults_keep_the_row_count():
    assert Links.generate(ROWS, seed=SEED, cycles=3, self_references=2).height == ROWS


def test_validate_reports_injected_cycles():
    df = Links.generate(ROWS, seed=SEED, cycles=3)
    report = Links.inspect(df)
    assert not report.passed
    finding = report.by_code("hierarchy_cycle")[0]
    # Every row that never reaches an ultimate parent, which includes the rows
    # hanging below a loop -- they do not terminate either.
    assert finding.count >= 3 * 4
    assert report.rows(finding).collect().height == finding.count


def test_validate_reports_injected_self_references():
    df = Links.generate(ROWS, seed=SEED, self_references=5)
    report = Links.inspect(df)
    assert not report.passed
    assert report.by_code("hierarchy_cycle")


def test_faults_need_a_hierarchy_to_damage():
    class Plain(FrameSpec):
        a = ColSpec(pl.String)

    with pytest.raises(SpecError, match="declares no __hierarchy__"):
        Plain.generate(10, seed=SEED, cycles=1)


def test_cycles_need_room_to_close():
    class Flat(FrameSpec):
        up = ColSpec(pl.String)
        down = ColSpec(pl.String)
        __hierarchy__ = Hierarchy(child="down", parent="up", max_depth=1)

    with pytest.raises(GenerationError, match="max_depth 2 or more"):
        Flat.generate(500, seed=SEED, cycles=1)


def test_negative_faults_are_refused():
    with pytest.raises(ValueError, match=">= 0"):
        Links.generate(100, seed=SEED, cycles=-1)


# ---------------------------------------------------------------------------
# Validation of data polspec did not generate
# ---------------------------------------------------------------------------


class Chain(FrameSpec):
    PARENT = ColSpec(pl.String)
    CHILD = ColSpec(pl.String)
    __hierarchy__ = Hierarchy(child="CHILD", parent="PARENT", max_depth=3)


def test_a_chain_deeper_than_declared_is_reported():
    nodes = [f"n{i}" for i in range(7)]
    deep = pl.DataFrame({"PARENT": nodes[:-1], "CHILD": nodes[1:]})
    report = Chain.inspect(deep)
    finding = report.by_code("hierarchy_depth")[0]
    assert finding.count == 3  # the rows at depth 4, 5 and 6
    assert not report.by_code("hierarchy_cycle")


def test_a_reference_with_two_parents_is_reported():
    multi = pl.DataFrame({"PARENT": ["a", "b"], "CHILD": ["c", "c"]})
    report = Chain.inspect(multi)
    assert report.by_code("hierarchy_multi_parent")[0].count == 2


def test_validating_a_cycle_terminates():
    """The load-bearing one.

    These fixtures exist to be walked by code that might loop forever on them,
    and a validator that walked until it reached a root would be the first
    such casualty. Both checks are bounded, so this returns rather than hangs
    -- a test that fails by timing out rather than by asserting.
    """
    cyclic = pl.DataFrame({"PARENT": ["b", "c", "a"], "CHILD": ["a", "b", "c"]})
    report = Chain.inspect(cyclic)
    assert report.by_code("hierarchy_cycle")[0].count == 3


def test_a_shallow_frame_produces_no_findings():
    ok = pl.DataFrame({"PARENT": ["r", "r", "a"], "CHILD": ["a", "b", "c"]})
    assert Chain.inspect(ok).passed


def test_the_check_can_be_switched_off():
    cyclic = pl.DataFrame({"PARENT": ["b", "c", "a"], "CHILD": ["a", "b", "c"]})
    assert Chain.inspect(cyclic, hierarchy=False).passed


# ---------------------------------------------------------------------------
# Declaration-time refusals
# ---------------------------------------------------------------------------


def test_one_column_cannot_be_both_ends():
    with pytest.raises(SpecError, match="two columns"):
        Hierarchy(child="a", parent="a")


def test_branching_and_roots_are_mutually_exclusive():
    with pytest.raises(SpecError, match="not both"):
        Hierarchy(child="a", parent="b", branching=2.0, roots=5)


def test_max_depth_must_be_positive():
    with pytest.raises(SpecError, match="at least 1"):
        Hierarchy(child="a", parent="b", max_depth=0)


def test_unknown_columns_are_refused():
    with pytest.raises(SpecError, match="unknown column 'nope'"):

        class Bad(FrameSpec):
            a = ColSpec(pl.String)
            __hierarchy__ = Hierarchy(child="nope", parent="a")


def test_incompatible_dtypes_are_refused():
    with pytest.raises(SpecError, match="dtype-compatible"):

        class Bad(FrameSpec):
            up = ColSpec(pl.Int64)
            down = ColSpec(pl.String)
            __hierarchy__ = Hierarchy(child="down", parent="up")


def test_a_rule_on_a_link_column_is_refused():
    """Both write the column, and whichever ran second would undo the other."""
    with pytest.raises(SpecError, match="carries rules and is part of the hierarchy"):

        class Bad(FrameSpec):
            up = ColSpec(pl.String)
            down = ColSpec(
                pl.String,
                choices=["x", "y"],
                rules=(ColRule(when=col("up") == "a", choices=["x"]),),
            )
            __hierarchy__ = Hierarchy(child="down", parent="up")


def test_a_foreign_key_on_a_link_column_is_refused():
    with pytest.raises(SpecError, match="foreign-keyed and part of the hierarchy"):

        class Bad(FrameSpec):
            up = ColSpec(pl.String)
            down = ColSpec(pl.String)
            __hierarchy__ = Hierarchy(child="down", parent="up")
            __foreign_keys__ = [ForeignKey("down", references="self", ref_columns="up")]


def test_a_unique_parent_column_is_refused():
    """A reference with two children appears in the parent column twice."""
    with pytest.raises(SpecError, match="more than one child"):

        class Bad(FrameSpec):
            up = ColSpec(pl.String, unique=True)
            down = ColSpec(pl.String)
            __hierarchy__ = Hierarchy(child="down", parent="up")


# ---------------------------------------------------------------------------
# It is a property of the whole frame, so the streaming verbs refuse it
# ---------------------------------------------------------------------------


class WholeLinks(Links):
    __hierarchy__ = Hierarchy(
        child="CHILD_REF", parent="PARENT_REF", max_depth=5, layout="whole"
    )


@pytest.mark.parametrize("batch_size", [1, 77, 1_000, 5_000])
def test_a_hierarchy_streams_as_its_whole_frame(batch_size):
    """2.1: the windowed layout, the default, draws each row's place in the
    forest from its position alone, so every batch is `generate()`'s rows."""
    whole = Links.generate(5_000, seed=SEED)
    batches = Links.generate_batches(5_000, batch_size=batch_size, seed=SEED)
    assert pl.concat(list(batches)).equals(whole)
    assert Links.scan(5_000, seed=SEED, batch_size=batch_size).collect().equals(whole)


def test_a_sunk_hierarchy_is_its_whole_frame(tmp_path):
    whole = Links.generate(3_000, seed=SEED)
    Links.sink_parquet(tmp_path / "out.parquet", 3_000, seed=SEED, batch_size=700)
    assert pl.read_parquet(tmp_path / "out.parquet").equals(whole)


def test_a_projected_scan_holds_the_column_the_frame_holds():
    whole = Links.generate(2_000, seed=SEED)
    parents = Links.scan(2_000, seed=SEED, batch_size=300).select("PARENT_REF")
    assert parents.collect().to_series().equals(whole["PARENT_REF"])


def test_the_whole_layout_is_refused_by_every_stream(tmp_path):
    """`layout="whole"` is 2.0's builder, over the whole frame at once."""
    with pytest.raises(SpecError, match="layout='whole', which generate_batches"):
        list(WholeLinks.generate_batches(100, batch_size=10, seed=SEED))
    with pytest.raises(SpecError, match="layout='whole', which a sink"):
        WholeLinks.sink_parquet(tmp_path / "out.parquet", 100, seed=SEED)
    with pytest.raises(SpecError, match="layout='whole', which scan"):
        WholeLinks.scan(100, seed=SEED)


@pytest.mark.parametrize("layout", ["windowed", "whole"])
def test_both_layouts_draw_the_declared_shape(layout):
    """The same level sizes and roots, a chain of exactly `max_depth` and none
    longer, one parent per child, every parent a reference or a root."""

    class Tree(FrameSpec):
        ref = ColSpec(pl.Int64)
        up = ColSpec(pl.Int64)
        __hierarchy__ = Hierarchy(
            child="ref", parent="up", max_depth=4, roots=25, layout=layout
        )

    frame = Tree.generate(4_000, seed=SEED)
    assert frame["ref"].n_unique() == 4_000
    parent_of = dict(zip(frame["ref"].to_list(), frame["up"].to_list(), strict=True))
    roots = {p for p in frame["up"].to_list() if p not in parent_of}
    # The windowed layout gives every root a child, so exactly `roots` are
    # in the data; 2.0's whole layout picks roots with replacement, and a
    # root no reference picked is not there at all.
    if layout == "windowed":
        assert len(roots) == 25
    else:
        assert len(roots) <= 25

    def depth(ref):
        hops = 0
        while ref in parent_of:
            ref, hops = parent_of[ref], hops + 1
        return hops

    assert max(depth(r) for r in parent_of) == 4
    Tree.validate(frame)


def test_both_layouts_put_as_many_references_at_each_depth():
    """The level sizes come from `n` and the declaration alone, so the two
    layouts' forests hold the same number of references at every depth --
    the branching declared, laid out the same way."""

    def depths(layout):
        class Wide(FrameSpec):
            ref = ColSpec(pl.String)
            up = ColSpec(pl.String)
            __hierarchy__ = Hierarchy(
                child="ref", parent="up", max_depth=3, branching=4.0, layout=layout
            )

        frame = Wide.generate(20_000, seed=SEED)
        parent_of = dict(
            zip(frame["ref"].to_list(), frame["up"].to_list(), strict=True)
        )
        counts: dict[int, int] = {}
        for ref in parent_of:
            hops, node = 0, ref
            while node in parent_of:
                node, hops = parent_of[node], hops + 1
            counts[hops] = counts.get(hops, 0) + 1
        return counts

    assert depths("windowed") == depths("whole")


@pytest.mark.parametrize("batch_size", [1, 77, 1_000])
def test_damage_streams_as_the_whole_damaged_frame(batch_size):
    """2.1: `cycles=` and `self_references=` on every streaming verb. The
    edits are drawn from the seed alone, so each batch holds the damage the
    whole frame holds there, and the counts are exact however it is split."""
    broken = Links.generate(ROWS, seed=SEED, cycles=4, self_references=3)
    batches = Links.generate_batches(
        ROWS, batch_size=batch_size, seed=SEED, cycles=4, self_references=3
    )
    streamed = pl.concat(list(batches))
    assert streamed.equals(broken)
    lf = Links.scan(ROWS, seed=SEED, batch_size=batch_size, cycles=4, self_references=3)
    assert lf.collect().equals(broken)
    # Four loops of four, and each self-reference a loop of one.
    assert len(cycle_nodes(parent_map(streamed))) == 4 * 4 + 3
    same = streamed.filter(pl.col("CHILD_REF") == pl.col("PARENT_REF"))
    assert same.height == 3


def test_a_sunk_damaged_hierarchy_is_reported_by_validation(tmp_path):
    path = tmp_path / "broken.parquet"
    Links.sink_parquet(path, ROWS, seed=SEED, batch_size=500, cycles=3)
    frame = pl.read_parquet(path)
    assert frame.equals(Links.generate(ROWS, seed=SEED, cycles=3))
    report = Links.inspect(frame)
    assert report.by_code("hierarchy_cycle")[0].count >= 3 * 4


def test_streaming_damage_needs_a_hierarchy_to_damage():
    class Plain(FrameSpec):
        a = ColSpec(pl.Int64)

    with pytest.raises(SpecError, match="declares no __hierarchy__"):
        list(Plain.generate_batches(10, batch_size=5, seed=SEED, cycles=1))
    with pytest.raises(SpecError, match="declares no __hierarchy__"):
        Plain.scan(10, seed=SEED, self_references=1)


@pytest.mark.parametrize("rows", [1, 5, 4_097])
def test_a_scans_head_is_the_whole_frames_first_rows(rows):
    """A `head` reads fewer rows than the frame holds; the forest is laid out
    over all of them, so its first rows are `generate()`'s."""
    whole = Links.generate(10_000, seed=SEED)
    assert Links.scan(10_000, seed=SEED).head(rows).collect().equals(whole.head(rows))


# ---------------------------------------------------------------------------
# Structural operations carry it, or drop it deliberately
# ---------------------------------------------------------------------------


def test_rename_rewrites_the_declaration():
    renamed = Links.spec.rename({"CHILD_REF": "child_ref"})
    assert renamed.hierarchy.child == "child_ref"
    assert renamed.hierarchy.parent == "PARENT_REF"
    assert FrameSpec.from_spec(renamed).generate(200, seed=SEED).height == 200


def test_dropping_a_link_column_drops_the_hierarchy():
    """A hierarchy over a column that no longer exists is not a hierarchy."""
    assert Links.spec.drop("CHILD_REF").hierarchy is None


def test_selecting_both_columns_keeps_it():
    kept = Links.spec.select("PARENT_REF", "CHILD_REF")
    assert kept.hierarchy == Links.spec.hierarchy


def test_yaml_round_trip(tmp_path):
    path = tmp_path / "links.yaml"
    Links.to_yaml(path)
    assert FrameSpec.from_yaml(path).spec.hierarchy == Links.spec.hierarchy
