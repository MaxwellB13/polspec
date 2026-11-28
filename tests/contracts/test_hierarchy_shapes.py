"""A hierarchy's shape, over declarations nobody wrote down.

`features/test_hierarchy.py` holds the forest to its promises for a few
fixed declarations. This holds it to them for any: Hypothesis draws a
`Hierarchy` -- its depth, its branching or its roots, its layout, a string
or an integer reference -- a row count and a seed, and every frame must

- validate, with one parent per reference and a chain exactly `max_depth`
  long, none longer;
- hold exactly `roots` ultimate parents under the windowed layout (at most
  that many under the whole one, which picks them with replacement);
- stream as its whole frame, under the windowed layout, at any batch size;
- hold exactly the damage asked for, whole and streamed.

*Since 2.2; 2.1's windowed layout was checked on fixed declarations only.*
"""

from __future__ import annotations

import os

import polars as pl
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from polspec import (
    ColSpec,
    GenerationError,
    Hierarchy,
    TableSpec,
    generate,
    generate_batches,
    inspect,
    validate,
)

# A deep run's count for every property, as `test_properties.py` reads it.
DEEP_EXAMPLES = int(os.environ.get("POLSPEC_DEEP_EXAMPLES", "0"))


def _examples(usual: int) -> int:
    return DEEP_EXAMPLES or usual


SETTINGS = settings(
    max_examples=_examples(100),
    print_blob=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
_SEEDS = st.integers(0, 2**32)


@st.composite
def hierarchies(draw: st.DrawFn, layout: str | None = None) -> TableSpec:
    depth = draw(st.integers(1, 8))
    shape = draw(
        st.one_of(
            st.just({}),
            st.builds(lambda b: {"branching": b}, st.floats(1.0, 12.0)),
            st.builds(lambda r: {"roots": r}, st.integers(1, 60)),
        )
    )
    dtype = draw(st.sampled_from([pl.String, pl.Int64]))
    hierarchy = Hierarchy(
        child="ref",
        parent="up",
        max_depth=depth,
        layout=layout or draw(st.sampled_from(["windowed", "whole"])),
        **shape,
    )
    return TableSpec(
        "Drawn",
        {
            "ref": ColSpec(dtype),
            "up": ColSpec(dtype),
            "tag": ColSpec(pl.Int16, nullable=True),
        },
        hierarchy=hierarchy,
    )


def _rows(spec: TableSpec) -> st.SearchStrategy[int]:
    """At least `max_depth` rows, which a hierarchy needs to reach its depth."""
    assert spec.hierarchy is not None
    return st.integers(spec.hierarchy.max_depth, 600)


def _too_few_for_roots(spec: TableSpec, n: int) -> bool:
    """Whether the windowed layout refuses `n` rows: a child for each root
    and a chain `max_depth` deep need `roots + max_depth - 1`."""
    h = spec.hierarchy
    assert h is not None
    return (
        h.layout == "windowed" and h.roots is not None and n < h.roots + h.max_depth - 1
    )


def _parents(frame: pl.DataFrame) -> dict:
    return dict(zip(frame["ref"].to_list(), frame["up"].to_list(), strict=True))


def _depth(parent_of: dict, ref, limit: int) -> int:
    hops = 0
    while ref in parent_of:
        ref, hops = parent_of[ref], hops + 1
        if hops > limit:
            return -1
    return hops


def _on_loops(parent_of: dict) -> set:
    """Every reference on a loop, not merely below one."""
    on_loop: set = set()
    for start in parent_of:
        seen: list = []
        node = start
        while node in parent_of and len(seen) <= len(parent_of):
            if node in seen:
                on_loop.update(seen[seen.index(node) :])
                break
            seen.append(node)
            node = parent_of[node]
    return on_loop


def _streamed(spec: TableSpec, n: int, seed: int, batch_size: int, **damage):
    parts = list(generate_batches(spec, n, batch_size=batch_size, seed=seed, **damage))
    return pl.concat(parts)


@SETTINGS
@given(data=st.data(), seed=_SEEDS)
def test_a_drawn_hierarchy_has_its_declared_shape(data, seed):
    spec = data.draw(hierarchies())
    hierarchy = spec.hierarchy
    assert hierarchy is not None
    n = data.draw(_rows(spec))
    if _too_few_for_roots(spec, n):
        with pytest.raises(GenerationError, match="declare fewer roots"):
            generate(spec, n, seed=seed)
        return
    frame = generate(spec, n, seed=seed)

    assert frame.height == n
    assert frame["ref"].n_unique() == n  # one parent per reference
    parent_of = _parents(frame)
    depths = [_depth(parent_of, ref, hierarchy.max_depth) for ref in parent_of]
    assert max(depths) == hierarchy.max_depth
    assert min(depths) >= 1

    roots = {p for p in frame["up"].to_list() if p not in parent_of}
    if hierarchy.roots is not None:
        if hierarchy.layout == "windowed":
            assert len(roots) == hierarchy.roots
        else:
            assert len(roots) <= hierarchy.roots
    validate(spec, frame)


@SETTINGS
@given(data=st.data(), seed=_SEEDS, batch_size=st.integers(1, 200))
def test_a_drawn_windowed_hierarchy_streams_as_its_whole_frame(data, seed, batch_size):
    spec = data.draw(hierarchies(layout="windowed"))
    n = data.draw(_rows(spec))
    assume(not _too_few_for_roots(spec, n))
    whole = generate(spec, n, seed=seed)
    assert _streamed(spec, n, seed, batch_size).equals(whole)


@settings(SETTINGS, max_examples=_examples(80))
@given(
    data=st.data(),
    seed=_SEEDS,
    cycles=st.integers(0, 6),
    self_references=st.integers(0, 6),
    batch_size=st.integers(1, 200),
)
def test_a_drawn_hierarchy_holds_exactly_the_damage_asked_for(
    data, seed, cycles, self_references, batch_size
):
    spec = data.draw(hierarchies(layout="windowed"))
    hierarchy = spec.hierarchy
    assert hierarchy is not None
    assume(cycles == 0 or hierarchy.max_depth >= 2)
    n = data.draw(_rows(spec))
    assume(not _too_few_for_roots(spec, n))
    damage = {"cycles": cycles, "self_references": self_references}
    try:
        frame = generate(spec, n, seed=seed, **damage)
    except GenerationError as error:
        # Too few chains, or rows, for that much damage: a refusal that says
        # so, not a frame holding less than was asked for.
        assert "Could not build" in str(error) or "self-reference" in str(error)
        return

    assert frame.height == n
    selfs = frame.filter(pl.col("ref") == pl.col("up")).height
    assert selfs == self_references
    # A loop is min(3, max_depth - 1) hops closed into a ring; the self
    # references are loops of one.
    ring = min(3, hierarchy.max_depth - 1) + 1
    assert len(_on_loops(_parents(frame))) == cycles * ring + self_references
    if cycles or self_references:
        assert not inspect(spec, frame).passed
    else:
        validate(spec, frame)
    assert _streamed(spec, n, seed, batch_size, **damage).equals(frame)
