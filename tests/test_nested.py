"""`List` and `Array` columns: the value fields describe each element.

`ColSpec(pl.List(pl.Int64), bounds=(0, 10), list_length=(1, 5))` -- `bounds`,
`choices`, `format`, `pattern`, `string_length`, `distribution` and `weights`
describe an element; `list_length` the list; `nullable` the cell. Generation
draws the lengths and the elements as two engine columns and wraps one by
the other; validation lifts each element constraint through `list.eval`. The
round trip per inner dtype is pinned in `test_roundtrip.py`.
"""

from __future__ import annotations

import json

import polars as pl
import pytest
from polspec import (
    ColRule,
    ColSpec,
    FrameSpec,
    SpecError,
    TableSpec,
    ValidationError,
    col,
)
from polspec.drift import diff, drift


def _spec_for(column: ColSpec) -> type[FrameSpec]:
    return type("Nested", (FrameSpec,), {"__columns__": {"c": column}})


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


def test_the_value_fields_are_checked_against_the_element_dtype():
    spec = ColSpec(pl.List(pl.Int64), bounds=(0, 10), list_length=(1, 5))
    assert spec.value_dtype == pl.Int64()
    assert spec.list_length is not None and spec.list_length.closed() == (1, 5)
    with pytest.raises(SpecError, match="bounds is only supported for numeric"):
        ColSpec(pl.List(pl.String), bounds=(0, 1))
    with pytest.raises(SpecError, match=r"format is only supported for pl.String"):
        ColSpec(pl.List(pl.Int64), format="email")
    ColSpec(pl.List(pl.String), format="email")  # the element is a String
    ColSpec(pl.List(pl.Enum(["a", "b"])), choices=["a"])
    with pytest.raises(SpecError, match="not among this column's Enum"):
        ColSpec(pl.List(pl.Enum(["a", "b"])), choices=["z"])


def test_a_list_length_belongs_to_a_list():
    with pytest.raises(SpecError, match=r"only supported for pl\.List, got Int64"):
        ColSpec(pl.Int64, list_length=(1, 2))
    with pytest.raises(SpecError, match="an Array's length is part of its dtype"):
        ColSpec(pl.Array(pl.Int64, 2), list_length=(1, 2))
    with pytest.raises(SpecError, match="requires both endpoints"):
        ColSpec(pl.List(pl.Int64), list_length=(1, None))
    with pytest.raises(SpecError, match="non-negative"):
        ColSpec(pl.List(pl.Int64), list_length=(-1, 2))


def test_what_a_list_column_refuses():
    with pytest.raises(SpecError, match="a list is not drawn without replacement"):
        ColSpec(pl.List(pl.Int64), unique=True)
    with pytest.raises(SpecError, match="a list value would be a list of lists"):
        ColSpec(pl.List(pl.Int64), rules=[ColRule(when=col("x") > 1, choices=(1,))])
    # A list of lists declares and validates by dtype; only generate() objects,
    # and the value fields have nothing to describe on it.
    ColSpec(pl.List(pl.List(pl.Int64)))
    with pytest.raises(SpecError, match="bounds is only supported"):
        ColSpec(pl.List(pl.List(pl.Int64)), bounds=(0, 1))
    with pytest.raises(SpecError, match="cannot generate data for dtype"):
        _spec_for(ColSpec(pl.Array(pl.Struct({"a": pl.Int64}), 2))).generate(1)


def test_an_element_class_is_instantiated_like_a_dtype_class():
    assert ColSpec(pl.List(pl.Categorical)).dtype == pl.List(pl.Categorical())
    assert ColSpec(pl.Array(pl.Int64, 3)).dtype.inner == pl.Int64()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_lengths_and_elements_respect_the_declaration():
    spec_cls = _spec_for(ColSpec(pl.List(pl.Int64), bounds=(0, 9), list_length=(1, 3)))
    df = spec_cls.generate(500, seed=1)
    assert df.schema["c"] == pl.List(pl.Int64)
    lengths = df["c"].list.len()
    assert lengths.min() == 1 and lengths.max() == 3
    elements = df["c"].explode(empty_as_null=False)
    assert elements.min() >= 0 and elements.max() <= 9
    assert elements.null_count() == 0
    spec_cls.validate(df)


def test_the_default_length_is_zero_to_five():
    df = _spec_for(ColSpec(pl.List(pl.String))).generate(500, seed=2)
    lengths = df["c"].list.len()
    assert lengths.min() == 0 and lengths.max() == 5


def test_nullability_is_the_lists_and_never_the_elements():
    spec_cls = _spec_for(
        ColSpec(pl.List(pl.Int64), nullable=True, null_probability=0.5)
    )
    df = spec_cls.generate(1_000, seed=3)
    assert 300 < df["c"].null_count() < 700
    elements = df["c"].explode(empty_as_null=False)
    assert elements.drop_nulls().len() == df["c"].list.len().sum()


def test_an_array_takes_its_width_from_the_dtype():
    spec_cls = _spec_for(
        ColSpec(pl.Array(pl.Float64, 3), bounds=(0.0, 1.0), nullable=True)
    )
    df = spec_cls.generate(200, seed=4)
    assert df.schema["c"] == pl.Array(pl.Float64, 3)
    assert df["c"].null_count() > 0
    assert set(df["c"].drop_nulls().arr.len().unique().to_list()) == {3}
    spec_cls.validate(df)


def test_a_list_column_keeps_its_data_across_a_rename():
    before = _spec_for(ColSpec(pl.List(pl.Int64), list_length=(0, 4), nullable=True))
    renamed = ColSpec(
        pl.List(pl.Int64), list_length=(0, 4), nullable=True, seed_name="c"
    )
    after = type("After", (FrameSpec,), {"__columns__": {"d": renamed}})
    assert after.generate(100, seed=5)["d"].equals(before.generate(100, seed=5)["c"])


def test_a_list_is_a_filler_under_cartesian_coverage():
    class S(FrameSpec):
        flag = ColSpec(pl.Boolean)
        tags = ColSpec(pl.List(pl.Enum(["a", "b"])), list_length=(1, 2))

    df = S.generate(10, seed=6, method="cartesian")
    assert df.height == 10 and df.schema["tags"] == pl.List(pl.Enum(["a", "b"]))
    S.validate(df)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

BAD = pl.DataFrame(
    {
        "ids": [[1, 50], [], [None, 1], [1, 2, 3, 4]],
        "codes": [["AB"], ["ab"], ["ABC"], ["ZZ"]],
        "fixed": [[0.5, 0.5, 0.5], [2.0, 0.1, 0.1], [0.1] * 3, [0.2] * 3],
    },
    schema={
        "ids": pl.List(pl.Int64),
        "codes": pl.List(pl.String),
        "fixed": pl.Array(pl.Float64, 3),
    },
)


class Checked(FrameSpec):
    ids = ColSpec(pl.List(pl.Int64), bounds=(0, 9), list_length=(0, 3))
    codes = ColSpec(
        pl.List(pl.String),
        pattern=r"^[A-Z]{2}$",
        string_length=(2, 2),
        choices=["AB", "CD"],
    )
    fixed = ColSpec(pl.Array(pl.Float64, 3), bounds=(0.0, 1.0))


def test_each_element_constraint_reports_the_offending_lists():
    report = Checked.inspect(BAD)
    by_key = {f.key: f for f in report.findings}
    assert set(by_key) == {
        "ids__list_len",
        "ids__element_null",
        "ids__bounds",
        "codes__choices",
        "codes__len",
        "codes__pattern",
        "fixed__bounds",
    }
    assert by_key["ids__list_len"].code == "list_length"
    assert by_key["ids__list_len"].samples == ([1, 2, 3, 4],)
    assert by_key["ids__element_null"].code == "nullability"
    assert by_key["ids__element_null"].samples == ([None, 1],)
    assert by_key["ids__bounds"].details["max_found"] == 50
    assert by_key["ids__bounds"].samples == ([1, 50],)
    assert by_key["codes__choices"].count == 3
    assert by_key["codes__pattern"].samples == (["ab"], ["ABC"])
    assert by_key["fixed__bounds"].samples == ([2.0, 0.1, 0.1],)
    assert report.rows(by_key["ids__list_len"]).collect().height == 1


def test_a_list_of_the_wrong_element_dtype_is_a_dtype_finding_only():
    df = pl.DataFrame(
        {"ids": [["1"]], "codes": [["AB"]], "fixed": [[0.1] * 3]},
        schema={
            "ids": pl.List(pl.String),
            "codes": pl.List(pl.String),
            "fixed": pl.Array(pl.Float64, 3),
        },
    )
    assert [f.code for f in Checked.inspect(df).findings] == ["dtype"]
    with pytest.raises(ValidationError, match="dtype"):
        Checked.validate(df.with_columns(fixed=pl.col("fixed").arr.to_list()))


def test_a_list_of_a_wider_element_is_compatible_unless_strict():
    spec_cls = _spec_for(ColSpec(pl.List(pl.Int32)))
    df = pl.DataFrame({"c": [[1, 2]]}, schema={"c": pl.List(pl.Int64)})
    assert spec_cls.inspect(df).passed
    assert not spec_cls.inspect(df, strict_dtypes=True).passed


# ---------------------------------------------------------------------------
# Files, drift, profiling, reports
# ---------------------------------------------------------------------------


def test_nested_dtypes_round_trip_through_yaml_and_python(tmp_path):
    class S(FrameSpec):
        ids = ColSpec(
            pl.List(pl.Int64), bounds=(0, 9), list_length=(1, 3), nullable=True
        )
        tags = ColSpec(pl.List(pl.Enum(["a", "b"])), choices=["a"])
        grid = ColSpec(pl.Array(pl.Decimal(6, 2), 2), bounds=(0, "9.99"))

    S.to_yaml(tmp_path / "s.yaml")
    text = (tmp_path / "s.yaml").read_text(encoding="utf-8")
    assert "List: Int64" in text and "list_length:" in text
    assert "Array:" in text and "width: 2" in text
    assert FrameSpec.from_yaml(tmp_path / "s.yaml").spec == S.spec
    S.to_python(tmp_path / "s.py")
    source = (tmp_path / "s.py").read_text(encoding="utf-8")
    assert "pl.List(pl.Int64)" in source and "pl.Array(pl.Decimal(6, 2), 2)" in source
    namespace: dict = {}
    exec(source, namespace)
    assert namespace["S"].spec == S.spec


def test_diff_reports_a_list_length_change_like_a_string_length_change():
    old = TableSpec("T", {"c": ColSpec(pl.List(pl.Int64), list_length=(0, 5))})
    new = TableSpec("T", {"c": ColSpec(pl.List(pl.Int64), list_length=(0, 3))})
    (finding,) = diff(old, new).findings
    assert finding.code == "domain_narrowed" and finding.breaking
    assert finding.details["field"] == "list_length"
    wider = TableSpec("T", {"c": ColSpec(pl.List(pl.Int32))})
    (finding,) = diff(wider, TableSpec("T", {"c": ColSpec(pl.List(pl.Int64))})).findings
    assert finding.code == "dtype_changed" and not finding.breaking


def test_drift_measures_a_list_column_as_its_elements_and_its_length():
    spec_cls = _spec_for(ColSpec(pl.List(pl.Int64), bounds=(0, 9), list_length=(1, 3)))
    assert drift(spec_cls, spec_cls.generate(200, seed=7)).unchanged
    bad = pl.DataFrame({"c": [[1, 2, 3, 4, 50]] * 10}, schema={"c": pl.List(pl.Int64)})
    report = drift(spec_cls, bad)
    findings = {f.details.get("field"): f for f in report.breaking}
    assert set(findings) == {"bounds", "list_length"}
    assert findings["bounds"].details["max_found"] == 50
    assert findings["list_length"].details["max_found"] == 5
    json.loads(report.to_json())


def test_from_dataframe_declares_a_list_by_its_elements():
    df = pl.DataFrame(
        {"c": [[1, 2], [], None, [9]], "d": [[["x"]], [[]], [[]], [["y"]]]},
        schema={"c": pl.List(pl.Int64), "d": pl.List(pl.List(pl.String))},
    )
    spec = FrameSpec.from_dataframe(df, name="P").spec
    c = spec.columns["c"]
    assert c.dtype == pl.List(pl.Int64) and c.nullable
    assert c.bounds is not None and c.bounds.closed() == (1, 9)
    assert c.list_length is not None and c.list_length.closed() == (0, 2)
    # A list of lists is recorded faithfully and left to validation.
    assert spec.columns["d"].dtype == pl.List(pl.List(pl.String))


def test_reports_show_the_list_length():
    spec_cls = _spec_for(ColSpec(pl.List(pl.Int64), list_length=(1, 3)))
    assert "1..3 elements" in spec_cls.to_markdown()
