"""Specs as files: `to_yaml`/`from_yaml` and `to_python`.

The property under test is that a spec written out and read back generates
the same data for the same seed. The rest is what the writers warn about and
drop -- `__checks__`, cross-spec foreign keys, column validators -- since a
`polars.Expr` and a Python class have no representation in a standalone file.
"""

import polars as pl
import pytest
import yaml
from polspec import (
    Bound,
    Check,
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSpec,
    ValidationError,
)
from polspec.serialization import _colspec_to_yaml

# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


class YamlSource(FrameSpec):
    string_1 = ColSpec(dtype=pl.String, nullable=False)
    enum_1 = ColSpec(dtype=pl.Enum(["mammal", "reptile", "insect"]), nullable=True)
    int_1 = ColSpec(dtype=pl.Int64, bounds=Bound(-100, 100), nullable=True)
    float_1 = ColSpec(
        dtype=pl.Float64,
        bounds=(-2_000, 2_000),
        nullable=True,
        rules=(
            ColRule(
                when={"column": "enum_1", "in": ["mammal", "reptile"]},
                choices=[0.0, 1.0],
            ),
        ),
    )


def test_yaml_roundtrip_generates_identical_data(tmp_path):
    yaml_path = tmp_path / "spec.yaml"
    YamlSource.to_yaml(source=yaml_path)
    Loaded = FrameSpec.from_yaml(source=yaml_path)

    assert Loaded.schema() == YamlSource.schema()
    df_original = YamlSource.generate(500, seed=42)
    df_loaded = Loaded.generate(500, seed=42)
    assert df_original.equals(df_loaded)


def test_yaml_roundtrip_preserves_column_order():
    yaml_text = yaml.safe_dump(
        {
            "name": "X",
            "columns": {
                name: _colspec_to_yaml(spec)
                for name, spec in YamlSource._columns.items()
            },
        },
        sort_keys=False,
    )
    parsed = yaml.safe_load(yaml_text)
    assert list(parsed["columns"].keys()) == list(YamlSource._columns.keys())


def test_yaml_roundtrip_preserves_rule_behavior(tmp_path):
    yaml_path = tmp_path / "spec.yaml"
    YamlSource.to_yaml(source=yaml_path)
    Loaded = FrameSpec.from_yaml(source=yaml_path)

    df = Loaded.generate(2_000, seed=1)
    restricted = df.filter(pl.col("enum_1").is_in(["mammal", "reptile"]))[
        "float_1"
    ].drop_nulls()
    assert set(restricted.to_list()) <= {0.0, 1.0}


def test_yaml_file_is_plain_readable_yaml(tmp_path):
    yaml_path = tmp_path / "spec.yaml"
    YamlSource.to_yaml(source=yaml_path)
    text = yaml_path.read_text()
    # no python-object tags (!!python/..., etc) -- must be safe_load-able
    # plain data, and should read like the sketch from the design discussion
    assert "!!python" not in text
    assert "Enum:" in text
    assert "bounds:" in text


def test_to_yaml_requires_columns(tmp_path):
    class Empty(FrameSpec):
        pass

    with pytest.raises(ValueError, match="declares no ColSpec columns"):
        Empty.to_yaml(source=tmp_path / "spec.yaml")


def test_from_yaml_requires_columns(tmp_path):
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text(yaml.safe_dump({"name": "Empty", "columns": {}}))
    with pytest.raises(ValueError, match="declares no columns"):
        FrameSpec.from_yaml(source=yaml_path)


def test_yaml_defaults_are_omitted_for_compactness(tmp_path):
    class Minimal(FrameSpec):
        plain_int = ColSpec(dtype=pl.Int32)

    yaml_path = tmp_path / "spec.yaml"
    Minimal.to_yaml(source=yaml_path)
    parsed = yaml.safe_load(yaml_path.read_text())
    col = parsed["columns"]["plain_int"]
    # nullable=False and the default null_probability shouldn't be written
    assert col == {"dtype": "Int32"}


def test_distributions_and_weights_yaml_roundtrip(tmp_path):
    class FullFeatureSpec(FrameSpec):
        norm = ColSpec(
            dtype=pl.Float64,
            distribution="normal",
            distribution_params={"mean": 42.0, "std": 3.5},
            bounds=(0.0, 100.0),
        )
        enum_col = ColSpec(
            dtype=pl.Enum(["X", "Y", "Z"]),
            weights=[0.5, 0.3, 0.2],
        )
        rule_col = ColSpec(
            dtype=pl.String,
            rules=(
                ColRule(
                    when={"column": "enum_col", "equals": "X"},
                    choices=["A", "B"],
                    weights=[0.8, 0.2],
                ),
            ),
        )

    yaml_file = tmp_path / "full_spec.yaml"
    FullFeatureSpec.to_yaml(source=yaml_file)

    Loaded = FrameSpec.from_yaml(source=yaml_file)
    assert Loaded.schema() == FullFeatureSpec.schema()

    df1 = FullFeatureSpec.generate(1_000, seed=123)
    df2 = Loaded.generate(1_000, seed=123)
    assert df1.equals(df2)


# ---------------------------------------------------------------------------
# what the writers warn about and drop
# ---------------------------------------------------------------------------


def test_to_yaml_warns_when_checks_are_not_persisted(tmp_path):
    class CheckedSpec(FrameSpec):
        a = ColSpec(pl.Int64)
        b = ColSpec(pl.Int64)
        __checks__ = [Check(pl.col("a") > pl.col("b"), name="a_gt_b")]

    yaml_path = tmp_path / "checked_spec.yaml"
    with pytest.warns(UserWarning, match="a_gt_b"):
        CheckedSpec.to_yaml(yaml_path)

    Loaded = FrameSpec.from_yaml(yaml_path)
    assert Loaded.checks() == ()


def test_yaml_roundtrip_preserves_unique_together(tmp_path):
    class CompoundSpec(FrameSpec):
        tenant_id = ColSpec(pl.String, nullable=False)
        user_id = ColSpec(pl.Int64, nullable=False)
        email = ColSpec(pl.String, unique=True, nullable=False)
        __unique_together__ = [("tenant_id", "user_id")]

    yaml_path = tmp_path / "compound_spec.yaml"
    CompoundSpec.to_yaml(yaml_path)

    LoadedCompound = FrameSpec.from_yaml(yaml_path)
    assert LoadedCompound.unique_together() == (("tenant_id", "user_id"),)
    assert LoadedCompound._columns["email"].unique is True

    # Test validation with loaded spec
    df_valid = pl.DataFrame(
        {
            "tenant_id": ["T1", "T2"],
            "user_id": [1, 1],
            "email": ["u1@t1.com", "u1@t2.com"],
        }
    )
    res = LoadedCompound.validate(df_valid)
    assert res.height == 2

    df_dup = pl.DataFrame(
        {
            "tenant_id": ["T1", "T1"],
            "user_id": [1, 1],
            "email": ["u1@t1.com", "u2@t1.com"],
        }
    )
    with pytest.raises(ValidationError, match="Composite unique key"):
        LoadedCompound.validate(df_dup)


def test_yaml_nested_directory_creation_and_utf8(tmp_path):
    class UnicodeYamlSpec(FrameSpec):
        user_id = ColSpec(pl.Int64, unique=True)
        comment = ColSpec(pl.String, choices=["café", "naïve", "🚀"])

    nested_file = tmp_path / "nested" / "subfolder" / "spec.yaml"
    UnicodeYamlSpec.to_yaml(nested_file)
    assert nested_file.exists()

    LoadedSpec = FrameSpec.from_yaml(nested_file)
    assert LoadedSpec._columns["comment"].choices == ("café", "naïve", "🚀")


def test_to_yaml_warns_when_column_validators_are_not_persisted(tmp_path):
    class ValidatedSpec(FrameSpec):
        price = ColSpec(pl.Float64, validators=[pl.col("price") > 0])

    yaml_path = tmp_path / "validated_spec.yaml"
    with pytest.warns(UserWarning, match="price"):
        ValidatedSpec.to_yaml(yaml_path)

    Loaded = FrameSpec.from_yaml(yaml_path)
    assert Loaded._columns["price"].validators == ()


def _exec_python_spec(path):
    """Runs a file `to_python` wrote and returns its module namespace."""
    namespace: dict = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace


def test_to_python_roundtrip_generates_identical_data(tmp_path):
    py_path = tmp_path / "spec.py"
    YamlSource.to_python(py_path)
    Loaded = _exec_python_spec(py_path)["YamlSource"]

    assert Loaded.schema() == YamlSource.schema()
    df_original = YamlSource.generate(500, seed=42)
    df_loaded = Loaded.generate(500, seed=42)
    assert df_original.equals(df_loaded)


def test_to_python_preserves_rule_behavior(tmp_path):
    py_path = tmp_path / "spec.py"
    YamlSource.to_python(py_path)
    Loaded = _exec_python_spec(py_path)["YamlSource"]

    df = Loaded.generate(2_000, seed=1)
    restricted = df.filter(pl.col("enum_1").is_in(["mammal", "reptile"]))[
        "float_1"
    ].drop_nulls()
    assert set(restricted.to_list()) <= {0.0, 1.0}


def test_to_python_writes_plain_importable_source(tmp_path):
    py_path = tmp_path / "spec.py"
    YamlSource.to_python(py_path)
    text = py_path.read_text(encoding="utf-8")
    assert "import polars as pl" in text
    assert "from polspec import" in text
    assert "class YamlSource(FrameSpec):" in text
    assert "__columns__ = {" in text


def test_to_python_handles_non_identifier_column_names(tmp_path):
    class RawNamed(FrameSpec):
        __columns__ = {
            "Unit Price": ColSpec(pl.Float64, bounds=(0.0, 100.0)),
            "_id": ColSpec(pl.Int64, unique=True),
        }

    py_path = tmp_path / "raw_named.py"
    RawNamed.to_python(py_path)
    Loaded = _exec_python_spec(py_path)["RawNamed"]

    assert set(Loaded._columns) == {"Unit Price", "_id"}
    df = Loaded.generate(10, seed=1)
    Loaded.validate(df)


def test_to_python_requires_columns(tmp_path):
    class Empty(FrameSpec):
        pass

    with pytest.raises(ValueError, match="declares no ColSpec columns"):
        Empty.to_python(tmp_path / "spec.py")


def test_to_python_warns_when_checks_are_not_persisted(tmp_path):
    class CheckedSpec(FrameSpec):
        a = ColSpec(pl.Int64)
        b = ColSpec(pl.Int64)
        __checks__ = [Check(pl.col("a") > pl.col("b"), name="a_gt_b")]

    py_path = tmp_path / "checked_spec.py"
    with pytest.warns(UserWarning, match="a_gt_b"):
        CheckedSpec.to_python(py_path)

    Loaded = _exec_python_spec(py_path)["CheckedSpec"]
    assert Loaded.checks() == ()


def test_to_python_warns_when_foreign_keys_to_other_specs_are_not_persisted(tmp_path):
    class ParentSpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)

    class ChildSpec(FrameSpec):
        customer_id = ColSpec(pl.Int64)
        __foreign_keys__ = [
            ForeignKey("customer_id", references=ParentSpec, ref_columns="id")
        ]

    py_path = tmp_path / "child_spec.py"
    with pytest.warns(UserWarning, match="fk_customer_id__ParentSpec"):
        ChildSpec.to_python(py_path)

    Loaded = _exec_python_spec(py_path)["ChildSpec"]
    assert Loaded.foreign_keys() == ()


def test_to_python_warns_when_column_validators_are_not_persisted(tmp_path):
    class ValidatedSpec(FrameSpec):
        price = ColSpec(pl.Float64, validators=[pl.col("price") > 0])

    py_path = tmp_path / "validated_spec.py"
    with pytest.warns(UserWarning, match="price"):
        ValidatedSpec.to_python(py_path)

    Loaded = _exec_python_spec(py_path)["ValidatedSpec"]
    assert Loaded._columns["price"].validators == ()


def test_to_python_preserves_unique_together_and_self_foreign_keys(tmp_path):
    class HierarchySpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        parent_id = ColSpec(pl.Int64, nullable=True)
        tenant_id = ColSpec(pl.Int32)
        __unique_together__ = [("tenant_id", "id")]
        __foreign_keys__ = [
            ForeignKey("parent_id", references="self", ref_columns="id")
        ]

    py_path = tmp_path / "hierarchy_spec.py"
    HierarchySpec.to_python(py_path)
    Loaded = _exec_python_spec(py_path)["HierarchySpec"]

    assert Loaded.unique_together() == (("tenant_id", "id"),)
    assert Loaded.foreign_keys() == (
        ForeignKey("parent_id", references="self", ref_columns="id"),
    )

    df_valid = pl.DataFrame(
        {"id": [1, 2, 3], "parent_id": [None, 1, 1], "tenant_id": [1, 1, 1]}
    )
    assert Loaded.validate(df_valid).height == 3


def test_to_python_renders_categorical_and_datetime_dtypes(tmp_path):
    import datetime

    class TemporalSpec(FrameSpec):
        __columns__ = {
            "ts": ColSpec(
                pl.Datetime(time_unit="us"),
                bounds=(0, 10_000_000),
            ),
            "d": ColSpec(
                pl.Date,
                bounds=(datetime.date(2020, 1, 1), datetime.date(2024, 1, 1)),
            ),
            "cat": ColSpec(pl.Categorical()),
        }

    py_path = tmp_path / "temporal_spec.py"
    TemporalSpec.to_python(py_path)
    text = py_path.read_text(encoding="utf-8")
    assert "import datetime" in text

    Loaded = _exec_python_spec(py_path)["TemporalSpec"]
    assert Loaded.schema() == TemporalSpec.schema()
    df = Loaded.generate(20, seed=1)
    Loaded.validate(df)


def test_dtype_to_python_renders_time_zone():
    from polspec.serialization import _dtype_to_python

    src = _dtype_to_python(pl.Datetime(time_unit="us", time_zone="UTC"))
    assert src == "pl.Datetime(time_unit='us', time_zone='UTC')"
