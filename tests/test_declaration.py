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
from polspec import Bound, ColRule, ColSpec, FrameSpec, SpecError, ValidationError, col


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

    assert list(Spec.spec.columns) == ["val"]


def test_underscore_columns_survive_via_explicit_declaration():
    class Spec(FrameSpec):
        __columns__ = {"_id": ColSpec(pl.Int64), "val": ColSpec(pl.Int64)}

    assert list(Spec.spec.columns) == ["_id", "val"]
    assert Spec.generate(10, seed=1).columns == ["_id", "val"]


def test_profiling_keeps_underscore_columns():
    """C8: from_dataframe takes its names from data, so it must keep all of them."""
    source = pl.DataFrame({"_id": [1, 2, 3], "val": [1.0, 2.0, 3.0]})
    spec_cls = FrameSpec.from_dataframe(source)

    assert list(spec_cls.spec.columns) == ["_id", "val"]
    assert spec_cls.generate(10, seed=1).columns == ["_id", "val"]


def test_yaml_roundtrip_keeps_underscore_columns(tmp_path):
    class Spec(FrameSpec):
        __columns__ = {"_id": ColSpec(pl.Int64), "val": ColSpec(pl.Int64)}

    path = tmp_path / "spec.yaml"
    Spec.to_yaml(path)
    assert list(FrameSpec.from_yaml(path).spec.columns) == ["_id", "val"]


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


def test_column_named_after_a_method_keeps_both():
    """C9: the column is a column and the method is still the method.

    The metaclass takes ColSpec attributes out of the class namespace before
    the class exists, so nothing is shadowed and nothing needs to warn.
    """

    class Spec(FrameSpec):
        schema = ColSpec(pl.Int64)
        other = ColSpec(pl.Int64)

    assert list(Spec.spec.columns) == ["schema", "other"]
    assert Spec.generate(5, seed=1).columns == ["schema", "other"]
    # The method wins on attribute access; the column is reachable by name.
    assert Spec.schema() == pl.Schema({"schema": pl.Int64, "other": pl.Int64})
    assert Spec.col("schema") == ColSpec(pl.Int64)
    assert Spec.spec["schema"] == ColSpec(pl.Int64)
    # An ordinary column is still an attribute.
    assert Spec.other == ColSpec(pl.Int64)


def test_declaring_a_method_name_via_explicit_declaration_keeps_both():
    class Spec(FrameSpec):
        __columns__ = {"schema": ColSpec(pl.Int64), "tag": ColSpec(pl.String)}

    assert Spec.generate(5, seed=1).columns == ["schema", "tag"]
    assert Spec.schema() == pl.Schema({"schema": pl.Int64, "tag": pl.String})
    assert Spec.tag("anything") == []


def test_no_column_name_warns(recwarn):
    class Spec(FrameSpec):
        schema = ColSpec(pl.Int64)
        validate = ColSpec(pl.String)
        name = ColSpec(pl.String)

    assert [w for w in recwarn if "shadow" in str(w.message)] == []
    assert list(Spec.spec.columns) == ["schema", "validate", "name"]
    assert Spec.validate(Spec.generate(3, seed=1)).height == 3


def test_profiling_a_frame_with_reserved_names(recwarn):
    source = pl.DataFrame({"schema": [1, 2], "tag": ["a", "b"]})
    spec_cls = FrameSpec.from_dataframe(source)

    assert [w for w in recwarn if "shadow" in str(w.message)] == []
    assert spec_cls.schema().names() == ["schema", "tag"]


def test_unknown_attribute_is_still_an_attribute_error():
    class Spec(FrameSpec):
        a = ColSpec(pl.Int64)

    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        Spec.nope  # noqa: B018
    assert not hasattr(Spec, "nope")


# ---------------------------------------------------------------------------
# Duplicate choices -- as written, or once cast to the column's dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "choices",
    [
        [1, "1", 2],  # one value once cast to String
        ["a", "a"],
        [1.0, "1.0"],
    ],
)
def test_choices_that_collapse_to_one_value_are_rejected(choices):
    with pytest.raises(ValueError, match="same once cast to String"):
        ColSpec(pl.String, choices=choices)


def test_rule_choices_repeated_as_written_are_rejected():
    with pytest.raises(ValueError, match="duplicate values"):
        ColRule(when=col("g") == "x", choices=[1, 1])


def test_choices_distinct_in_their_dtype_are_accepted():
    # Values cross into the engine as indices into a typed domain, so choices
    # need only be distinct in the column's own dtype -- not as strings.
    assert ColSpec(pl.String, choices=[1, 2, 3]).choices == (1, 2, 3)
    assert ColSpec(pl.String, choices=[True, "True"]).choices == (True, "True")
    assert ColSpec(pl.Int64, choices=[1, 2]).choices == (1, 2)


def test_duplicate_choices_would_have_misapplied_weights():
    """The reason this is an error and not a de-duplication: weights are
    positional, so a repeated choice quietly doubles its share.
    """
    with pytest.raises(ValueError, match="same once cast"):
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
    assert FrameSpec.from_yaml(path).spec.columns["n"].bounds == Bound(0, None)


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


# ---------------------------------------------------------------------------
# col_name -- a class attribute's name diverges from the column's real name
#
# Declaring columns as class attributes requires a Python identifier, which
# can't hold a space or other punctuation a real dataset's header might use.
# `col_name` lets the attribute stay a clean identifier while the generated/
# validated DataFrame uses whatever name the data actually has.
# ---------------------------------------------------------------------------


def test_col_name_overrides_the_attribute_name():
    class Spec(FrameSpec):
        unit_price = ColSpec(pl.Float64, col_name="Unit Price", bounds=(0, 100))
        qty = ColSpec(pl.Int64, bounds=(1, 10))

    assert list(Spec.spec.columns) == ["Unit Price", "qty"]
    df = Spec.generate(50, seed=1)
    assert df.columns == ["Unit Price", "qty"]
    Spec.validate(df)


def test_col_name_rejects_empty_string():
    with pytest.raises(ValueError, match="must not be an empty string"):
        ColSpec(pl.Int64, col_name="")


def test_col_name_used_by_rules_and_unique_together():
    class Spec(FrameSpec):
        region = ColSpec(
            pl.String,
            col_name="Sales Region",
            choices=["east", "west"],
        )
        amount = ColSpec(
            pl.Int64,
            col_name="Sale Amount",
            bounds=(0, 100),
            rules=(
                ColRule(
                    when=col("Sales Region") == "east",
                    choices=[99],
                ),
            ),
        )
        ticket = ColSpec(pl.Int64, col_name="Ticket No", bounds=(1, 100_000))
        __unique_together__ = [("Sales Region", "Ticket No")]

    df = Spec.generate(200, seed=3)
    east_amounts = df.filter(pl.col("Sales Region") == "east")["Sale Amount"]
    assert (east_amounts == 99).all()
    assert df.select(["Sales Region", "Ticket No"]).n_unique() == 200


def test_a_rule_cannot_sit_on_a_composite_key_column():
    """A rule assigns from a fixed set, which is how two rows come to share a
    combination; the repair that separates them would undo the rule.
    """
    with pytest.raises(SpecError, match="carries rules and is part of the"):

        class Spec(FrameSpec):
            g = ColSpec(pl.Enum(["x", "y"]))
            v = ColSpec(
                pl.Int64,
                bounds=(0, 100),
                rules=(ColRule(when=col("g") == "x", choices=[99]),),
            )
            __unique_together__ = [("g", "v")]


def test_colliding_col_names_are_rejected():
    with pytest.raises(ValueError, match="both resolve to the column name 'x'"):

        class Spec(FrameSpec):
            a = ColSpec(pl.Int64, col_name="x")
            b = ColSpec(pl.Int64, col_name="x")


def test_col_name_conflicting_with_explicit_declaration_key_is_rejected():
    with pytest.raises(ValueError, match="conflicts with its __columns__ key"):

        class Spec(FrameSpec):
            __columns__ = {"foo": ColSpec(pl.Int64, col_name="bar")}


def test_col_name_matching_explicit_declaration_key_is_redundant_but_fine():
    class Spec(FrameSpec):
        __columns__ = {"foo": ColSpec(pl.Int64, col_name="foo")}

    assert list(Spec.spec.columns) == ["foo"]


def test_overriding_a_col_name_attribute_removes_the_renamed_column():
    class Base(FrameSpec):
        keep = ColSpec(pl.Int64, col_name="Keep Col")
        drop = ColSpec(pl.Int64, col_name="Drop Col")

    class Child(Base):
        drop = None

    assert list(Child.spec.columns) == ["Keep Col"]


def test_col_name_survives_yaml_roundtrip(tmp_path):
    """The YAML key already is the real column name, so col_name need not persist."""

    class Spec(FrameSpec):
        unit_price = ColSpec(pl.Float64, col_name="Unit Price", bounds=(0, 100))

    path = tmp_path / "spec.yaml"
    Spec.to_yaml(path)
    loaded = FrameSpec.from_yaml(path)
    assert list(loaded.spec.columns) == ["Unit Price"]
    assert loaded.generate(10, seed=1).columns == ["Unit Price"]
