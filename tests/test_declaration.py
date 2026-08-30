"""Declaration-time contracts: what a spec accepts, and what it says when it can't.

The round-trip property in `test_roundtrip.py` covers constraints that survive
into generated data. These are the ones that never get that far -- a column
that silently fails to exist, a name collision that disables a method, a
`choices` list that quietly collapses. Each was a place polspec discarded what
the caller declared without saying so.
"""

import datetime as dt

import polars as pl
import pytest
from polspec import Bound, ColRule, ColSpec, FrameSpec, ValidationError


def _spec_for(column: ColSpec) -> type[FrameSpec]:
    """A single-column FrameSpec with the column as `c`."""
    return type("Declared", (FrameSpec,), {"__columns__": {"c": column}})


# ---------------------------------------------------------------------------
# C8 -- names that cannot be class attributes
# ---------------------------------------------------------------------------


def test_underscore_columns_are_dropped_from_the_class_body():
    """Documents the limit that makes __columns__ necessary.

    A leading underscore marks a spec's own internals, so the attribute scan
    has to skip it. Nothing is silently lost here: what the caller wrote is
    what they get.
    """

    class Spec(FrameSpec):
        _id = ColSpec(pl.Int64)
        val = ColSpec(pl.Int64)

    assert list(Spec._columns) == ["val"]


def test_underscore_columns_survive_via_explicit_declaration():
    class Spec(FrameSpec):
        __columns__ = {"_id": ColSpec(pl.Int64), "val": ColSpec(pl.Int64)}

    assert list(Spec._columns) == ["_id", "val"]
    assert Spec.generate(10, seed=1).columns == ["_id", "val"]


def test_profiling_keeps_underscore_columns():
    """C8: from_dataframe takes its names from data, so it must keep all of them."""
    source = pl.DataFrame({"_id": [1, 2, 3], "val": [1.0, 2.0, 3.0]})
    spec_cls = FrameSpec.from_dataframe(source)

    assert list(spec_cls._columns) == ["_id", "val"]
    assert spec_cls.generate(10, seed=1).columns == ["_id", "val"]


def test_yaml_roundtrip_keeps_underscore_columns(tmp_path):
    class Spec(FrameSpec):
        __columns__ = {"_id": ColSpec(pl.Int64), "val": ColSpec(pl.Int64)}

    path = tmp_path / "spec.yaml"
    Spec.to_yaml(path)
    assert list(FrameSpec.from_yaml(path)._columns) == ["_id", "val"]


def test_explicit_declarations_must_be_colspecs():
    with pytest.raises(TypeError, match="must be a ColSpec"):

        class Spec(FrameSpec):
            __columns__ = {"a": "not a colspec"}


def test_explicit_declarations_must_be_a_mapping():
    with pytest.raises(TypeError, match="must be a mapping"):

        class Spec(FrameSpec):
            __columns__ = [ColSpec(pl.Int64)]


# ---------------------------------------------------------------------------
# C9 -- names that collide with the FrameSpec API
# ---------------------------------------------------------------------------


def test_column_shadowing_a_method_warns():
    """C9: the shadowing itself is unavoidable; doing it in silence was the bug."""
    with pytest.warns(UserWarning, match=r"shadows FrameSpec\.schema"):

        class Spec(FrameSpec):
            schema = ColSpec(pl.Int64)
            other = ColSpec(pl.Int64)

    # The column is honoured -- only the accessor is lost.
    assert list(Spec._columns) == ["schema", "other"]
    assert Spec.generate(5, seed=1).columns == ["schema", "other"]


def test_shadowing_a_method_via_explicit_declaration_keeps_both():
    class Spec(FrameSpec):
        __columns__ = {"schema": ColSpec(pl.Int64), "tag": ColSpec(pl.String)}

    assert Spec.generate(5, seed=1).columns == ["schema", "tag"]
    # Both classmethods still resolve to the method, not to a ColSpec.
    assert Spec.schema() == pl.Schema({"schema": pl.Int64, "tag": pl.String})
    assert Spec.tag("anything") == []


def test_ordinary_column_names_do_not_warn(recwarn):
    class Spec(FrameSpec):
        name = ColSpec(pl.String)
        value = ColSpec(pl.Int64)

    assert [w for w in recwarn if "shadows" in str(w.message)] == []
    assert list(Spec._columns) == ["name", "value"]


def test_profiling_a_frame_with_reserved_names_does_not_warn(recwarn):
    """Names arriving from data go through __columns__, so nothing is shadowed."""
    source = pl.DataFrame({"schema": [1, 2], "tag": ["a", "b"]})
    spec_cls = FrameSpec.from_dataframe(source)

    assert [w for w in recwarn if "shadows" in str(w.message)] == []
    assert spec_cls.schema().names() == ["schema", "tag"]


# ---------------------------------------------------------------------------
# M5 -- choices that are distinct in Python but identical as strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "choices",
    [
        [1, "1", 2],
        ["a", "a"],
        [1.0, "1.0"],
        [True, "True"],
    ],
)
def test_colliding_choices_are_rejected(choices):
    with pytest.raises(ValueError, match="both render as"):
        ColSpec(pl.String, choices=choices)


def test_colliding_rule_choices_are_rejected():
    with pytest.raises(ValueError, match="both render as"):
        ColRule(when={"column": "g", "equals": "x"}, choices=[1, "1"])


def test_distinct_string_forms_are_accepted():
    spec = ColSpec(pl.String, choices=[1, 2, 3])
    assert spec.choices == (1, 2, 3)


def test_colliding_choices_would_have_misapplied_weights():
    """The reason this is an error and not a de-duplication.

    `[1, "1", 2]` used to collapse to two generated values while three weights
    were still applied positionally -- so the weight written for 2 landed on
    whichever of the colliding pair survived.
    """
    with pytest.raises(ValueError, match="both render as"):
        ColSpec(pl.String, choices=[1, "1", 2], weights=[10.0, 10.0, 1.0])


# ---------------------------------------------------------------------------
# Open-ended bounds -- `bounds=(0, None)`
#
# An open end means different things to the two consumers of `bounds`, and
# that asymmetry is the whole feature: `validate()` treats it as genuinely
# unconstrained, while `generate()` falls back to the same default it would
# use with no bounds at all, because it cannot sample an unbounded range.
# ---------------------------------------------------------------------------


def test_open_upper_bound_generates_only_non_negative():
    values = _spec_for(ColSpec(pl.Int64, bounds=(0, None))).generate(3000, seed=7)["c"]
    assert values.min() >= 0


def test_open_lower_bound_generates_only_non_positive():
    values = _spec_for(ColSpec(pl.Int64, bounds=(None, 0))).generate(3000, seed=7)["c"]
    assert values.max() <= 0


def test_open_end_generates_within_the_dtype_default():
    """Generation resolves the open end; it does not run away to the dtype max."""
    values = _spec_for(ColSpec(pl.Int64, bounds=(0, None))).generate(3000, seed=7)["c"]
    assert values.max() <= 1_000_000  # polspec's default wide-int bound

    # A dtype whose own default range is narrower keeps that narrower range.
    u8 = _spec_for(ColSpec(pl.UInt8, bounds=(0, None))).generate(3000, seed=7)["c"]
    assert u8.max() <= 255


def test_open_end_is_unconstrained_at_validation():
    """The point of the feature: no ceiling is invented for validation."""
    spec_cls = _spec_for(ColSpec(pl.Int64, bounds=(0, None)))

    # Far above anything generation would produce, and still valid.
    spec_cls.validate(pl.DataFrame({"c": [0, 10**9, 10**15]}))

    with pytest.raises(ValidationError, match=">= 0"):
        spec_cls.validate(pl.DataFrame({"c": [-1]}))


def test_open_lower_end_is_unconstrained_at_validation():
    spec_cls = _spec_for(ColSpec(pl.Int64, bounds=(None, 0)))
    spec_cls.validate(pl.DataFrame({"c": [0, -(10**15)]}))
    with pytest.raises(ValidationError, match="<= 0"):
        spec_cls.validate(pl.DataFrame({"c": [1]}))


def test_open_temporal_bound():
    spec_cls = _spec_for(ColSpec(pl.Date, bounds=(dt.date(2020, 1, 1), None)))
    assert spec_cls.generate(2000, seed=7)["c"].min() >= dt.date(2020, 1, 1)
    spec_cls.validate(pl.DataFrame({"c": [dt.date(2999, 1, 1)]}))
    with pytest.raises(ValidationError, match="out of bounds"):
        spec_cls.validate(pl.DataFrame({"c": [dt.date(2019, 12, 31)]}))


@pytest.mark.parametrize("bounds", [(None, None), Bound(None, None)])
def test_bounds_open_on_both_ends_normalizes_to_none(bounds):
    """One internal representation for 'unconstrained' keeps downstream guards honest."""
    assert ColSpec(pl.Int64, bounds=bounds).bounds is None


def test_closed_end_beyond_the_default_still_generates_above_it():
    """bounds=(2_000_000, None) sits past Int64's 1_000_000 default ceiling.

    Resolving the open end to that default would invert the range, which the
    engine would silently swap; it widens to the dtype limit instead.
    """
    values = _spec_for(ColSpec(pl.Int64, bounds=(2_000_000, None))).generate(
        2000, seed=7
    )["c"]
    assert values.min() >= 2_000_000


def test_string_length_still_requires_both_endpoints():
    """Open ends are scoped to `bounds`; string_length says so rather than crashing."""
    with pytest.raises(ValueError, match="requires both endpoints"):
        ColSpec(pl.String, string_length=(3, None))


def test_choices_are_checked_against_the_constrained_side_only():
    ColSpec(pl.Int64, choices=[5, 50], bounds=(0, None))  # no ceiling to exceed
    with pytest.raises(ValueError, match=r"fall outside this column's bounds >= 10"):
        ColSpec(pl.Int64, choices=[5, 50], bounds=(10, None))


def test_open_bounds_survive_yaml_roundtrip(tmp_path):
    class Spec(FrameSpec):
        n = ColSpec(pl.Int64, bounds=(0, None))

    path = tmp_path / "spec.yaml"
    Spec.to_yaml(path)
    assert FrameSpec.from_yaml(path)._columns["n"].bounds == Bound(0, None)


@pytest.mark.parametrize(
    ("bound", "rendered"),
    [
        (Bound(0, None), ">= 0"),
        (Bound(None, 100), "<= 100"),
        (Bound(0, 100), "[0, 100]"),
    ],
)
def test_bounds_render_readably(bound, rendered):
    assert str(bound) == rendered


def test_generated_docs_render_open_bounds():
    class Spec(FrameSpec):
        n = ColSpec(pl.Int64, bounds=(0, None))

    assert ">= 0" in Spec.to_markdown()
    assert "bounds: >= 0" in Spec.to_mermaid()
