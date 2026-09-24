"""`ColSpec(fields=…)`: what is claimed about a struct's field values.

The dtype is the schema -- every field's name and type comes from it -- and
`fields` is what is claimed about the values in it, so a struct of twenty
fields where one needs bounds spells one field. A field is a value, not a
column, so the claims that only mean something about a column among columns
are refused on one.

This file covers the declaration, what it round-trips to, how a struct is
generated, and how each field's claim is checked -- a finding names the
field (`point.lat`) and locates the column's rows.
"""

from __future__ import annotations

import polars as pl
import pytest
from polspec import ColRule, ColSpec, FrameSpec, SpecError, TableSpec, col
from polspec.drift import diff

POINT = pl.Struct({"lat": pl.Float64, "lon": pl.Float64, "label": pl.String})


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


def test_fields_describes_the_values_and_may_be_partial():
    spec = ColSpec(
        POINT,
        fields={"lat": ColSpec(pl.Float64, bounds=(-90, 90))},
        nullable=True,
    )
    assert list(spec.fields) == ["lat"]
    assert spec.fields["lat"].bounds is not None
    assert spec.nullable, "nullable describes the cell, as it does for a List"


def test_a_list_of_structs_takes_fields_because_it_describes_the_element():
    spec = ColSpec(
        pl.List(POINT),
        fields={"label": ColSpec(pl.String, string_length=(1, 8))},
        list_length=(1, 3),
    )
    assert spec.value_dtype == POINT
    assert list(spec.fields) == ["label"]


def test_fields_nests_to_any_depth():
    inner = pl.Struct({"x": pl.Int64})
    outer = pl.Struct({"point": inner, "name": pl.String})
    spec = ColSpec(
        outer,
        fields={
            "point": ColSpec(inner, fields={"x": ColSpec(pl.Int64, bounds=(0, 9))})
        },
    )
    assert spec.fields["point"].fields["x"].bounds is not None


def test_fields_belongs_to_a_struct():
    with pytest.raises(SpecError, match=r"only supported for pl\.Struct, got Int64"):
        ColSpec(pl.Int64, fields={"x": ColSpec(pl.Int64)})
    with pytest.raises(SpecError, match=r"only supported for pl\.Struct"):
        ColSpec(pl.List(pl.Int64), fields={"x": ColSpec(pl.Int64)})


def test_a_field_the_dtype_does_not_declare_is_a_typo():
    with pytest.raises(SpecError, match="names 'nope', which"):
        ColSpec(POINT, fields={"nope": ColSpec(pl.Int64)})


def test_the_dtype_is_the_schema_so_a_field_cannot_disagree_with_it():
    with pytest.raises(SpecError, match="declares Int64, but the struct declares"):
        ColSpec(POINT, fields={"lat": ColSpec(pl.Int64)})


def test_a_field_must_be_a_colspec():
    with pytest.raises(SpecError, match="must be a ColSpec, got str"):
        ColSpec(POINT, fields={"lat": "nope"})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    ("claim", "value"),
    [
        ("unique", True),
        ("seed_name", "other"),
        ("col_name", "other"),
        ("validators", pl.col("lat") > 0),
    ],
)
def test_a_field_is_a_value_not_a_column(claim, value):
    """Uniqueness spans rows, a seed name names a column, a col_name is the
    name in the frame, a validator is an expression over columns -- none of
    which a field inside a value has."""
    with pytest.raises(SpecError, match=f"sets {claim}, which has no meaning"):
        ColSpec(POINT, fields={"lat": ColSpec(pl.Float64, **{claim: value})})


def test_a_field_cannot_carry_rules():
    rule = [ColRule(when=col("flag"), choices=(1.0,))]
    with pytest.raises(SpecError, match="sets rules, which has no meaning"):
        ColSpec(POINT, fields={"lat": ColSpec(pl.Float64, rules=rule)})


def test_what_a_field_may_say_is_everything_about_a_value():
    """Every value-describing claim, checked by the rules that already
    govern it -- a field spec is a ColSpec, refusals included."""
    spec = ColSpec(
        POINT,
        fields={
            "lat": ColSpec(pl.Float64, bounds=(-90, 90), nullable=True),
            "lon": ColSpec(pl.Float64, distribution="normal"),
            "label": ColSpec(pl.String, format="hostname"),
        },
    )
    assert spec.fields["label"].format == "hostname"
    assert spec.fields["lon"].distribution == "normal"
    with pytest.raises(SpecError, match="both format='hostname' and string_length"):
        ColSpec(
            POINT,
            fields={
                "label": ColSpec(pl.String, format="hostname", string_length=(1, 20))
            },
        )


# ---------------------------------------------------------------------------
# What it round-trips to
# ---------------------------------------------------------------------------


class Nested(FrameSpec):
    point = ColSpec(
        POINT, fields={"lat": ColSpec(pl.Float64, bounds=(-90, 90))}, nullable=True
    )
    path = ColSpec(pl.List(pl.Struct({"x": pl.Int64})), list_length=(1, 3))
    deep = ColSpec(pl.List(pl.List(pl.Int64)))


def test_a_struct_column_round_trips_through_yaml_and_python(tmp_path):
    Nested.to_yaml(tmp_path / "s.yaml")
    text = (tmp_path / "s.yaml").read_text(encoding="utf-8")
    assert "Struct:" in text and "lat: Float64" in text
    assert "fields:" in text
    assert FrameSpec.from_yaml(tmp_path / "s.yaml").spec == Nested.spec

    Nested.to_python(tmp_path / "s.py")
    source = (tmp_path / "s.py").read_text(encoding="utf-8")
    assert "pl.Struct({'lat': pl.Float64" in source
    assert "fields={'lat': ColSpec(" in source
    namespace: dict = {}
    exec(source, namespace)
    assert namespace["Nested"].spec == Nested.spec


def test_a_struct_dtype_is_read_back_from_a_mapping_of_its_fields():
    from polspec.serialization.dtypes import dtype_from_data, dtype_to_data

    assert dtype_to_data(POINT) == {
        "Struct": {"lat": "Float64", "lon": "Float64", "label": "String"}
    }
    assert dtype_from_data(dtype_to_data(POINT)) == POINT
    from polspec.errors import SerializationError

    with pytest.raises(SerializationError, match=r"written as {Struct"):
        dtype_from_data({"Struct": "nope"})


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


def _with_fields(fields) -> TableSpec:
    return TableSpec("T", {"c": ColSpec(POINT, fields=fields)})


def test_diff_reports_a_field_described_added_removed_and_changed():
    described = _with_fields({"lat": ColSpec(pl.Float64, bounds=(-90, 90))})
    undescribed = _with_fields(None)

    (added,) = diff(undescribed, described).findings
    assert added.code == "column_added" and added.breaking
    assert added.details["field"] == "fields.lat"

    (removed,) = diff(described, undescribed).findings
    assert removed.code == "column_removed" and not removed.breaking

    widened = _with_fields({"lat": ColSpec(pl.Float64, bounds=(-180, 180))})
    (changed,) = diff(described, widened).findings
    assert changed.code == "field_changed"
    assert changed.details["field"] == "fields.lat"

    assert diff(described, described).unchanged


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _spec_for(column: ColSpec) -> type[FrameSpec]:
    return type("Structured", (FrameSpec,), {"__columns__": {"c": column}})


def test_a_struct_is_its_fields_gathered():
    spec_cls = _spec_for(
        ColSpec(POINT, fields={"lat": ColSpec(pl.Float64, bounds=(-90, 90))})
    )
    df = spec_cls.generate(500, seed=1)
    assert df.schema["c"] == POINT
    assert df["c"].null_count() == 0
    lat = df["c"].struct.field("lat")
    assert lat.min() >= -90 and lat.max() <= 90
    assert df["c"].struct.field("label").null_count() == 0
    spec_cls.validate(df)


def test_a_field_fields_omits_is_generated_from_its_dtype():
    """`fields` is partial: the rest are drawn as a column of their dtype."""
    spec_cls = _spec_for(ColSpec(POINT, fields={"lat": ColSpec(pl.Float64)}))
    df = spec_cls.generate(200, seed=1)
    assert df["c"].struct.field("label").str.len_chars().min() >= 1


def test_nullable_describes_the_cell_not_its_fields():
    spec_cls = _spec_for(ColSpec(POINT, nullable=True, null_probability=0.5))
    df = spec_cls.generate(2_000, seed=1)
    assert 800 < df["c"].null_count() < 1_200
    present = df["c"].drop_nulls()
    assert present.struct.field("lat").null_count() == 0


def test_a_field_is_null_where_its_own_declaration_says_so():
    """A field spec is a ColSpec, and its `nullable` is a claim like any
    other: generation honours it, inside the cells that are present."""
    spec_cls = _spec_for(
        ColSpec(
            POINT,
            fields={"label": ColSpec(pl.String, nullable=True, null_probability=0.5)},
        )
    )
    df = spec_cls.generate(2_000, seed=1)
    assert df["c"].null_count() == 0
    assert 800 < df["c"].struct.field("label").null_count() < 1_200
    assert df["c"].struct.field("lat").null_count() == 0
    spec_cls.validate(df)


def test_a_struct_nests_as_deep_as_its_dtype():
    inner = pl.Struct({"x": pl.Int64})
    outer = pl.Struct({"point": inner, "name": pl.String})
    spec_cls = _spec_for(
        ColSpec(
            outer,
            fields={
                "point": ColSpec(inner, fields={"x": ColSpec(pl.Int64, bounds=(0, 9))})
            },
        )
    )
    df = spec_cls.generate(200, seed=1)
    xs = df["c"].struct.field("point").struct.field("x")
    assert xs.min() >= 0 and xs.max() <= 9
    spec_cls.validate(df)


def test_a_list_of_structs_describes_its_element():
    spec_cls = _spec_for(
        ColSpec(
            pl.List(POINT),
            fields={"lat": ColSpec(pl.Float64, bounds=(0, 1))},
            list_length=(1, 3),
        )
    )
    df = spec_cls.generate(300, seed=1)
    assert df.schema["c"] == pl.List(POINT)
    lengths = df["c"].list.len()
    assert lengths.min() == 1 and lengths.max() == 3
    lat = df["c"].explode(empty_as_null=False).struct.field("lat")
    assert lat.min() >= 0 and lat.max() <= 1


def test_a_struct_of_a_list_is_a_column_inside_a_value():
    dtype = pl.Struct({"xs": pl.List(pl.Int64)})
    spec_cls = _spec_for(
        ColSpec(
            dtype,
            fields={
                "xs": ColSpec(pl.List(pl.Int64), bounds=(0, 9), list_length=(2, 2))
            },
        )
    )
    df = spec_cls.generate(200, seed=1)
    xs = df["c"].struct.field("xs")
    assert set(xs.list.len().unique().to_list()) == {2}
    assert xs.explode(empty_as_null=False).max() <= 9


def test_a_struct_column_is_seeded_by_name_like_any_other():
    """Renaming with `seed_name` keeps every field, and a field added beside
    another moves nothing: each is seeded under its parent by name."""
    before = _spec_for(ColSpec(POINT))
    renamed = type(
        "Renamed",
        (FrameSpec,),
        {"__columns__": {"d": ColSpec(POINT, seed_name="c")}},
    )
    assert renamed.generate(100, seed=5)["d"].equals(before.generate(100, seed=5)["c"])

    wider = pl.Struct(
        {"lat": pl.Float64, "lon": pl.Float64, "label": pl.String, "z": pl.Int64}
    )
    added = _spec_for(ColSpec(wider))
    original = before.generate(100, seed=5)["c"]
    grown = added.generate(100, seed=5)["c"]
    for field in ("lat", "lon", "label"):
        assert grown.struct.field(field).equals(original.struct.field(field)), field


def test_a_struct_column_is_a_window_under_batching():
    spec_cls = _spec_for(ColSpec(POINT, nullable=True))
    whole = spec_cls.generate(1_000, seed=7)["c"]
    batched = pl.concat(list(spec_cls.generate_batches(1_000, batch_size=250, seed=7)))
    assert batched["c"].equals(whole)


def test_a_struct_scans_and_sinks(tmp_path):
    spec_cls = _spec_for(ColSpec(POINT, nullable=True))
    path = tmp_path / "rows.parquet"
    spec_cls.scan(5_000, seed=1, batch_size=1_000).sink_parquet(path)
    assert pl.read_parquet(path).equals(
        spec_cls.scan(5_000, seed=1, batch_size=1_000).collect()
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _frame(values, dtype) -> pl.DataFrame:
    return pl.DataFrame({"c": values}, schema={"c": dtype})


def test_a_fields_claim_is_a_finding_named_for_the_field():
    spec_cls = _spec_for(
        ColSpec(
            POINT,
            fields={
                "lat": ColSpec(pl.Float64, bounds=(-90, 90)),
                "label": ColSpec(pl.String, string_length=(1, 4)),
            },
            nullable=True,
        )
    )
    df = _frame(
        [
            {"lat": 10.0, "lon": 0.0, "label": "ok"},
            {"lat": 120.0, "lon": 0.0, "label": "far too long"},
            None,
        ],
        POINT,
    )
    report = spec_cls.inspect(df)
    by_key = {f.key: f for f in report.findings}
    assert set(by_key) == {"c.lat__bounds", "c.label__len"}

    bounds = by_key["c.lat__bounds"]
    assert bounds.code == "bounds"
    assert bounds.columns == ("c",), "the rows are the column's"
    assert "Column 'c.lat'" in bounds.message
    assert bounds.samples == (120.0,)
    assert bounds.details["max_found"] == 120.0
    located = report.rows(bounds).collect()
    assert located["c"].struct.field("lat").to_list() == [120.0]


def test_a_null_field_is_reported_unless_the_field_is_nullable():
    df = _frame([{"lat": None, "lon": 1.0, "label": None}, None], POINT)
    strict = _spec_for(ColSpec(POINT, nullable=True))
    by_key = {f.key: f for f in strict.inspect(df).findings}
    assert set(by_key) == {"c.lat__null", "c.label__null"}
    assert by_key["c.lat__null"].code == "nullability"
    assert by_key["c.lat__null"].count == 1, "a null cell has no fields to be null"
    assert "non-nullable field" in by_key["c.lat__null"].message

    lenient = _spec_for(
        ColSpec(
            POINT,
            fields={
                "lat": ColSpec(pl.Float64, nullable=True),
                "label": ColSpec(pl.String, nullable=True),
            },
            nullable=True,
        )
    )
    assert lenient.inspect(df).passed


def test_a_nested_fields_claim_names_the_whole_path():
    inner = pl.Struct({"x": pl.Int64})
    outer = pl.Struct({"point": inner})
    spec_cls = _spec_for(
        ColSpec(
            outer,
            fields={
                "point": ColSpec(inner, fields={"x": ColSpec(pl.Int64, bounds=(0, 9))})
            },
        )
    )
    (finding,) = spec_cls.inspect(
        _frame([{"point": {"x": 1}}, {"point": {"x": 50}}], outer)
    ).findings
    assert finding.key == "c.point.x__bounds"
    assert "Column 'c.point.x'" in finding.message
    assert finding.samples == (50,)


def test_a_list_of_structs_reports_the_offending_lists():
    dtype = pl.List(pl.Struct({"lat": pl.Float64}))
    spec_cls = _spec_for(
        ColSpec(dtype, fields={"lat": ColSpec(pl.Float64, bounds=(-90, 90))})
    )
    report = spec_cls.inspect(
        _frame([[{"lat": 1.0}], [{"lat": 2.0}, {"lat": 95.0}], [{"lat": None}]], dtype)
    )
    by_key = {f.key: f for f in report.findings}
    assert set(by_key) == {"c.lat__bounds", "c.lat__null"}
    assert by_key["c.lat__bounds"].samples == ([{"lat": 2.0}, {"lat": 95.0}],)
    assert by_key["c.lat__bounds"].details["max_found"] == 95.0
    assert report.rows(by_key["c.lat__bounds"]).collect().height == 1


def test_a_list_inside_a_struct_carries_its_list_claims():
    dtype = pl.Struct({"xs": pl.List(pl.Int64)})
    spec_cls = _spec_for(
        ColSpec(
            dtype,
            fields={
                "xs": ColSpec(pl.List(pl.Int64), bounds=(0, 9), list_length=(1, 2))
            },
        )
    )
    report = spec_cls.inspect(
        _frame([{"xs": [1]}, {"xs": [1, 2, 3]}, {"xs": [50]}, {"xs": [None]}], dtype)
    )
    by_key = {f.key: f for f in report.findings}
    assert set(by_key) == {"c.xs__list_len", "c.xs__bounds", "c.xs__element_null"}
    assert by_key["c.xs__bounds"].samples == ([50],)
    assert by_key["c.xs__list_len"].samples == ([1, 2, 3],)


def test_a_list_of_lists_tells_its_two_levels_apart():
    dtype = pl.List(pl.List(pl.Int64))
    spec_cls = _spec_for(ColSpec(dtype))
    report = spec_cls.inspect(_frame([[[1], None], [[1, None]], [[2]]], dtype))
    by_key = {f.key: f for f in report.findings}
    assert set(by_key) == {"c__element_null", "c[]__element_null"}
    assert by_key["c__element_null"].samples == ([[1], None],)
    assert by_key["c[]__element_null"].samples == ([[1, None]],)


def test_a_struct_of_a_widened_field_is_compatible_unless_strict():
    wide = pl.Struct({"lat": pl.Int32, "lon": pl.Float64, "label": pl.String})
    df = _frame([{"lat": 1, "lon": 2.0, "label": "a"}], wide)
    spec_cls = _spec_for(ColSpec(POINT))
    assert spec_cls.inspect(df).passed
    (finding,) = spec_cls.inspect(df, strict_dtypes=True).findings
    assert finding.code == "dtype"


def test_a_struct_is_compatible_by_field_name_not_order():
    reordered = pl.Struct({"label": pl.String, "lon": pl.Float64, "lat": pl.Float64})
    df = _frame([{"label": "a", "lon": 2.0, "lat": 1.0}], reordered)
    assert _spec_for(ColSpec(POINT)).inspect(df).passed


def test_a_struct_missing_a_field_is_a_different_struct():
    narrow = pl.Struct({"lat": pl.Float64, "lon": pl.Float64})
    df = _frame([{"lat": 1.0, "lon": 2.0}], narrow)
    spec_cls = _spec_for(
        ColSpec(POINT, fields={"lat": ColSpec(pl.Float64, bounds=(0, 9))})
    )
    (finding,) = spec_cls.inspect(df).findings
    assert finding.code == "dtype"
