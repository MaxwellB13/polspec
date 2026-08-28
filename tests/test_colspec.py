import polars as pl
import pytest
import yaml
from polspec import Bound, ColRule, ColSpec, DfSpec
from polspec.colspec import _colspec_to_yaml


class DataSource(DfSpec):
    string_1 = ColSpec(dtype=pl.String, nullable=False)
    enum_1 = ColSpec(dtype=pl.Enum(["mammal", "reptile", "insect"]), nullable=True)
    int_1 = ColSpec(dtype=pl.Int64, bounds=Bound(-100, 100), nullable=True)
    float_1 = ColSpec(dtype=pl.Float64, bounds=(-2_000, 2_000), nullable=True)


def test_generate_matches_schema():
    df = DataSource.generate(1_000, seed=42)
    assert df.height == 1_000
    assert df.schema == DataSource.schema()


def test_generate_is_deterministic_given_seed():
    df_a = DataSource.generate(500, seed=7)
    df_b = DataSource.generate(500, seed=7)
    assert df_a.equals(df_b)


def test_generate_varies_without_fixed_seed_reuse():
    df_a = DataSource.generate(500, seed=1)
    df_b = DataSource.generate(500, seed=2)
    assert not df_a.equals(df_b)


def test_non_nullable_column_has_no_nulls():
    df = DataSource.generate(2_000, seed=3)
    assert df["string_1"].null_count() == 0


def test_int_bounds_respected():
    df = DataSource.generate(5_000, seed=4)
    values = df["int_1"].drop_nulls()
    assert values.min() >= -100
    assert values.max() <= 100


def test_float_bounds_respected():
    df = DataSource.generate(5_000, seed=5)
    values = df["float_1"].drop_nulls()
    assert values.min() >= -2_000
    assert values.max() <= 2_000


def test_enum_values_within_categories():
    df = DataSource.generate(2_000, seed=6)
    categories = set(df["enum_1"].dtype.categories.to_list())
    assert categories <= {"mammal", "reptile", "insect"}


def test_zero_rows():
    df = DataSource.generate(0, seed=1)
    assert df.height == 0
    assert df.schema == DataSource.schema()


def test_large_dataset_generation():
    df = DataSource.generate(1_000_000, seed=42)
    assert df.height == 1_000_000


class Bounded8(DfSpec):
    small_int = ColSpec(dtype=pl.Int8, nullable=False)


def test_fixed_width_int_default_bounds_fit_dtype():
    # Int8 has no explicit bounds here; the default must fall inside
    # [-128, 127] or the cast back to Int8 in generate() would fail.
    df = Bounded8.generate(10_000, seed=9)
    assert df["small_int"].min() >= -128
    assert df["small_int"].max() <= 127


class CoverageSource(DfSpec):
    string_1 = ColSpec(dtype=pl.String, nullable=False)
    enum_1 = ColSpec(dtype=pl.Enum(["mammal", "reptile", "insect"]), nullable=True)
    enum_2 = ColSpec(dtype=pl.Enum(["red", "blue"]), nullable=False)
    int_1 = ColSpec(dtype=pl.Int64, bounds=Bound(-100, 100), nullable=True)
    float_1 = ColSpec(dtype=pl.Float64, bounds=(-2_000, 2_000), nullable=True)


def _expected_coverage_size(spec_cls) -> int:
    # (enum_1: 3 categories + null) * (enum_2: 2 categories) *
    # (int_1: negative/zero/positive/null) * (float_1: negative/zero/positive/null)
    return 4 * 2 * 4 * 4


def test_cartesian_covers_every_enum_combination():
    df = CoverageSource.generate(n=1, method="cartesian", seed=42)
    combos = df.select("enum_1", "enum_2").unique()
    # 3 categories + null, times 2 categories
    assert combos.height == 4 * 2


def test_cartesian_covers_numeric_sign_and_null_buckets():
    df = CoverageSource.generate(n=1, method="cartesian", seed=42)
    int_signs = {
        (v is None, None if v is None else (v > 0) - (v < 0))
        for v in df["int_1"].to_list()
    }
    assert int_signs == {(True, None), (False, -1), (False, 0), (False, 1)}


def test_cartesian_n_is_a_minimum_not_exact():
    coverage_size = _expected_coverage_size(CoverageSource)
    df_small = CoverageSource.generate(n=1, method="cartesian", seed=1)
    assert df_small.height == coverage_size

    df_large = CoverageSource.generate(n=coverage_size * 10, method="cartesian", seed=1)
    assert df_large.height == coverage_size * 10
    # coverage rows must still all be present among the padded output
    combos = df_large.select("enum_1", "enum_2").unique().height
    assert combos == 4 * 2


def test_cartesian_respects_schema_and_non_nullable_columns():
    df = CoverageSource.generate(n=5_000, method="cartesian", seed=3)
    assert df.schema == CoverageSource.schema()
    assert df["string_1"].null_count() == 0
    assert df["enum_2"].null_count() == 0


def test_cartesian_is_deterministic_given_seed():
    df_a = CoverageSource.generate(n=2_000, method="cartesian", seed=11)
    df_b = CoverageSource.generate(n=2_000, method="cartesian", seed=11)
    assert df_a.equals(df_b)


def test_cartesian_requires_at_least_one_coverage_column():
    class StringsOnly(DfSpec):
        a = ColSpec(dtype=pl.String, nullable=False)

    with pytest.raises(
        ValueError, match="at least one Enum, Boolean, or bounded numeric"
    ):
        StringsOnly.generate(n=10, method="cartesian")


def test_cartesian_size_cap_raises_before_allocating():
    class Huge(DfSpec):
        e1 = ColSpec(dtype=pl.Enum([f"v{i}" for i in range(200)]))
        e2 = ColSpec(dtype=pl.Enum([f"v{i}" for i in range(200)]))
        e3 = ColSpec(dtype=pl.Enum([f"v{i}" for i in range(200)]))
        e4 = ColSpec(dtype=pl.Enum([f"v{i}" for i in range(200)]))

    with pytest.raises(ValueError, match="safety cap"):
        Huge.generate(n=10, method="cartesian")


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        CoverageSource.generate(n=10, method="bogus")


class RuledSource(DfSpec):
    enum_1 = ColSpec(dtype=pl.Enum(["A", "B", "C"]), nullable=False)
    enum_2 = ColSpec(
        dtype=pl.Enum(["X", "Y", "Z"]),
        nullable=False,
        rules=(
            ColRule(when={"column": "enum_1", "in": ["A", "B"]}, choices=["X", "Y"]),
        ),
    )


def test_rule_restricts_matching_rows():
    df = RuledSource.generate(2_000, seed=42)
    violations = df.filter(
        pl.col("enum_1").is_in(["A", "B"]) & ~pl.col("enum_2").is_in(["X", "Y"])
    )
    assert violations.height == 0


def test_rule_does_not_affect_unmatched_rows():
    df = RuledSource.generate(5_000, seed=42)
    unrestricted = set(df.filter(pl.col("enum_1") == "C")["enum_2"].unique().to_list())
    assert unrestricted == {"X", "Y", "Z"}


def test_rule_is_deterministic_given_seed():
    df_a = RuledSource.generate(1_000, seed=7)
    df_b = RuledSource.generate(1_000, seed=7)
    assert df_a.equals(df_b)


class MultiRuleSource(DfSpec):
    enum_1 = ColSpec(dtype=pl.Enum(["A", "B", "C"]), nullable=False)
    enum_2 = ColSpec(
        dtype=pl.Enum(["X", "Y", "Z"]),
        nullable=False,
        rules=(
            ColRule(when={"column": "enum_1", "equals": "A"}, choices=["X"]),
            ColRule(when={"column": "enum_1", "in": ["A", "B"]}, choices=["Y"]),
        ),
    )


def test_multiple_rules_first_match_wins():
    df = MultiRuleSource.generate(3_000, seed=3)
    # enum_1=="A" matches both rules; the first (choices=["X"]) must win.
    a_values = set(df.filter(pl.col("enum_1") == "A")["enum_2"].unique().to_list())
    assert a_values == {"X"}
    # enum_1=="B" only matches the second rule.
    b_values = set(df.filter(pl.col("enum_1") == "B")["enum_2"].unique().to_list())
    assert b_values == {"Y"}


class NumericRuleSource(DfSpec):
    enum_1 = ColSpec(dtype=pl.Enum(["A", "B", "C"]), nullable=False)
    int_1 = ColSpec(
        dtype=pl.Int64,
        bounds=Bound(-100, 100),
        nullable=False,
        rules=(ColRule(when={"column": "enum_1", "equals": "C"}, choices=[0, 1, 2]),),
    )


def test_rule_on_numeric_column():
    df = NumericRuleSource.generate(3_000, seed=5)
    c_values = set(df.filter(pl.col("enum_1") == "C")["int_1"].to_list())
    assert c_values <= {0, 1, 2}


def test_rule_applies_under_cartesian_method():
    df = RuledSource.generate(n=1, method="cartesian", seed=1)
    violations = df.filter(
        pl.col("enum_1").is_in(["A", "B"]) & ~pl.col("enum_2").is_in(["X", "Y"])
    )
    assert violations.height == 0


def test_rule_with_zero_rows():
    df = RuledSource.generate(0, seed=1)
    assert df.height == 0
    assert df.schema == RuledSource.schema()


def test_colrule_rejects_choices_outside_enum_categories():
    with pytest.raises(ValueError, match="not among this column's Enum categories"):
        ColSpec(
            dtype=pl.Enum(["X", "Y", "Z"]),
            rules=(
                ColRule(
                    when={"column": "enum_1", "equals": "A"}, choices=["NOT_A_CATEGORY"]
                ),
            ),
        )


def test_colrule_rejects_malformed_when():
    with pytest.raises(TypeError, match="must be a dict"):
        ColRule(when="not a dict", choices=["X"])


def test_colrule_rejects_ambiguous_when():
    with pytest.raises(ValueError, match="exactly one of"):
        ColRule(when={"column": "x", "equals": 1, "in": [1, 2]}, choices=["X"])


def test_colrule_rejects_empty_choices():
    with pytest.raises(ValueError, match="must not be empty"):
        ColRule(when={"column": "x", "equals": 1}, choices=[])


class YamlSource(DfSpec):
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
    Loaded = DfSpec.from_yaml(source=yaml_path)

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
    Loaded = DfSpec.from_yaml(source=yaml_path)

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
    class Empty(DfSpec):
        pass

    with pytest.raises(ValueError, match="declares no ColSpec columns"):
        Empty.to_yaml(source=tmp_path / "spec.yaml")


def test_from_yaml_requires_columns(tmp_path):
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text(yaml.safe_dump({"name": "Empty", "columns": {}}))
    with pytest.raises(ValueError, match="declares no columns"):
        DfSpec.from_yaml(source=yaml_path)


def test_yaml_defaults_are_omitted_for_compactness(tmp_path):
    class Minimal(DfSpec):
        plain_int = ColSpec(dtype=pl.Int32)

    yaml_path = tmp_path / "spec.yaml"
    Minimal.to_yaml(source=yaml_path)
    parsed = yaml.safe_load(yaml_path.read_text())
    col = parsed["columns"]["plain_int"]
    # nullable=False and the default null_probability shouldn't be written
    assert col == {"dtype": "Int32"}


class AllTypesSource(DfSpec):
    i8 = ColSpec(dtype=pl.Int8, bounds=(-50, 50), nullable=True)
    i16 = ColSpec(dtype=pl.Int16, bounds=(-1000, 1000), nullable=False)
    i32 = ColSpec(dtype=pl.Int32, bounds=(-100000, 100000), nullable=True)
    i64 = ColSpec(dtype=pl.Int64, bounds=(-10000000, 10000000), nullable=False)
    u8 = ColSpec(dtype=pl.UInt8, bounds=(0, 200), nullable=True)
    u16 = ColSpec(dtype=pl.UInt16, bounds=(10, 50000), nullable=False)
    u32 = ColSpec(dtype=pl.UInt32, bounds=(100, 3000000), nullable=True)
    u64 = ColSpec(dtype=pl.UInt64, bounds=(1000, 50000000), nullable=False)
    f32 = ColSpec(dtype=pl.Float32, bounds=(-10.5, 10.5), nullable=True)
    f64 = ColSpec(dtype=pl.Float64, bounds=(-1000.5, 1000.5), nullable=False)
    b = ColSpec(dtype=pl.Boolean, nullable=True)
    s = ColSpec(dtype=pl.String, nullable=False)


def test_all_native_dtypes_generation_and_bounds():
    df = AllTypesSource.generate(5_000, seed=123)
    assert df.schema == AllTypesSource.schema()
    assert df.height == 5_000

    assert df["i8"].drop_nulls().min() >= -50
    assert df["i8"].drop_nulls().max() <= 50

    assert df["i16"].null_count() == 0
    assert df["i16"].min() >= -1000
    assert df["i16"].max() <= 1000

    assert df["u8"].drop_nulls().min() >= 0
    assert df["u8"].drop_nulls().max() <= 200

    assert df["u16"].null_count() == 0
    assert df["u16"].min() >= 10
    assert df["u16"].max() <= 50000

    assert df["f32"].drop_nulls().min() >= -10.5
    assert df["f32"].drop_nulls().max() <= 10.5

    assert df["b"].null_count() > 0
    assert set(df["b"].drop_nulls().to_list()) <= {True, False}


def test_concurrent_generation_releases_gil():
    import concurrent.futures

    def worker(seed):
        return AllTypesSource.generate(10_000, seed=seed)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, seed) for seed in range(4)]
        results = [f.result() for f in futures]

    for df in results:
        assert df.height == 10_000
        assert df.schema == AllTypesSource.schema()


def test_narrow_float_bounds_in_cartesian():
    class NarrowFloat(DfSpec):
        flag = ColSpec(dtype=pl.Boolean, nullable=False)
        narrow_pos = ColSpec(dtype=pl.Float64, bounds=(0.0, 1e-12), nullable=False)
        narrow_neg = ColSpec(dtype=pl.Float64, bounds=(-1e-12, 0.0), nullable=False)

    df = NarrowFloat.generate(n=1, method="cartesian", seed=42)
    assert df["narrow_pos"].min() >= 0.0
    assert df["narrow_pos"].max() <= 1e-12
    assert df["narrow_neg"].min() >= -1e-12
    assert df["narrow_neg"].max() <= 0.0


def test_expanded_rule_operations():
    class ComplexRules(DfSpec):
        score = ColSpec(dtype=pl.Int32, bounds=(0, 100), nullable=False)
        optional_val = ColSpec(dtype=pl.Float64, bounds=(0.0, 10.0), nullable=True)
        tier = ColSpec(
            dtype=pl.String,
            nullable=False,
            rules=(
                ColRule(when={"column": "score", "lt": 50}, choices=["Low"]),
                ColRule(when={"column": "score", "between": [50, 79]}, choices=["Mid"]),
                ColRule(when={"column": "score", "gte": 80}, choices=["High"]),
            ),
        )
        status = ColSpec(
            dtype=pl.String,
            nullable=False,
            rules=(
                ColRule(
                    when={"column": "optional_val", "is_null": True},
                    choices=["Missing"],
                ),
                ColRule(
                    when={"column": "optional_val", "is_not_null": True},
                    choices=["Present"],
                ),
            ),
        )

    df = ComplexRules.generate(5_000, seed=42)

    low_check = df.filter(pl.col("score") < 50)["tier"].unique().to_list()
    assert low_check == ["Low"]

    mid_check = df.filter(pl.col("score").is_between(50, 79))["tier"].unique().to_list()
    assert mid_check == ["Mid"]

    high_check = df.filter(pl.col("score") >= 80)["tier"].unique().to_list()
    assert high_check == ["High"]

    missing_check = (
        df.filter(pl.col("optional_val").is_null())["status"].unique().to_list()
    )
    assert missing_check == ["Missing"]

    present_check = (
        df.filter(pl.col("optional_val").is_not_null())["status"].unique().to_list()
    )
    assert present_check == ["Present"]


def test_rule_referencing_unknown_column_raises():
    with pytest.raises(ValueError, match="references unknown column"):

        class BadRule(DfSpec):
            x = ColSpec(
                dtype=pl.Int32,
                rules=(
                    ColRule(when={"column": "non_existent", "equals": 1}, choices=[0]),
                ),
            )


def test_subclass_attribute_overriding():
    class BaseSpec(DfSpec):
        a = ColSpec(dtype=pl.Int32)
        b = ColSpec(dtype=pl.String)

    class DerivedSpec(BaseSpec):
        b = None  # removes column b
        c = ColSpec(dtype=pl.Float64)

    assert "b" not in DerivedSpec._columns
    assert set(DerivedSpec._columns.keys()) == {"a", "c"}
    df = DerivedSpec.generate(100, seed=42)
    assert set(df.columns) == {"a", "c"}


def test_temporal_and_binary_dtypes_and_yaml(tmp_path):
    class TemporalAndBinary(DfSpec):
        d = ColSpec(dtype=pl.Date, nullable=False)
        t = ColSpec(dtype=pl.Time, nullable=True)
        dt = ColSpec(dtype=pl.Datetime(time_unit="us"), nullable=False)
        dur = ColSpec(dtype=pl.Duration(time_unit="ms"), nullable=True)
        b = ColSpec(dtype=pl.Binary, nullable=False)
        cat = ColSpec(dtype=pl.Categorical, nullable=False)

    df = TemporalAndBinary.generate(1_000, seed=42)
    assert df.schema["d"] == pl.Date
    assert df.schema["t"] == pl.Time
    assert df.schema["dt"] == pl.Datetime(time_unit="us")
    assert df.schema["dur"] == pl.Duration(time_unit="ms")
    assert df.schema["b"] == pl.Binary
    assert isinstance(df.schema["cat"], pl.Categorical)

    # Test YAML serialization
    yaml_path = tmp_path / "temporal.yaml"
    TemporalAndBinary.to_yaml(source=yaml_path)
    Loaded = DfSpec.from_yaml(source=yaml_path)
    assert Loaded.schema() == TemporalAndBinary.schema()
    df_loaded = Loaded.generate(500, seed=42)
    df_orig = TemporalAndBinary.generate(500, seed=42)
    assert df_orig.equals(df_loaded)
