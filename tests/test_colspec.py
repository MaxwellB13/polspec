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


class DataSource(FrameSpec):
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


class Bounded8(FrameSpec):
    small_int = ColSpec(dtype=pl.Int8, nullable=False)


def test_fixed_width_int_default_bounds_fit_dtype():
    # Int8 has no explicit bounds here; the default must fall inside
    # [-128, 127] or the cast back to Int8 in generate() would fail.
    df = Bounded8.generate(10_000, seed=9)
    assert df["small_int"].min() >= -128
    assert df["small_int"].max() <= 127


class CoverageSource(FrameSpec):
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
    class StringsOnly(FrameSpec):
        a = ColSpec(dtype=pl.String, nullable=False)

    with pytest.raises(
        ValueError, match="at least one Enum, Boolean, or bounded numeric"
    ):
        StringsOnly.generate(n=10, method="cartesian")


def test_cartesian_size_cap_raises_before_allocating():
    class Huge(FrameSpec):
        e1 = ColSpec(dtype=pl.Enum([f"v{i}" for i in range(200)]))
        e2 = ColSpec(dtype=pl.Enum([f"v{i}" for i in range(200)]))
        e3 = ColSpec(dtype=pl.Enum([f"v{i}" for i in range(200)]))
        e4 = ColSpec(dtype=pl.Enum([f"v{i}" for i in range(200)]))

    with pytest.raises(ValueError, match="safety cap"):
        Huge.generate(n=10, method="cartesian")


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        CoverageSource.generate(n=10, method="bogus")


class RuledSource(FrameSpec):
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


class MultiRuleSource(FrameSpec):
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


class NumericRuleSource(FrameSpec):
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


class AllTypesSource(FrameSpec):
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
    class NarrowFloat(FrameSpec):
        flag = ColSpec(dtype=pl.Boolean, nullable=False)
        narrow_pos = ColSpec(dtype=pl.Float64, bounds=(0.0, 1e-12), nullable=False)
        narrow_neg = ColSpec(dtype=pl.Float64, bounds=(-1e-12, 0.0), nullable=False)

    df = NarrowFloat.generate(n=1, method="cartesian", seed=42)
    assert df["narrow_pos"].min() >= 0.0
    assert df["narrow_pos"].max() <= 1e-12
    assert df["narrow_neg"].min() >= -1e-12
    assert df["narrow_neg"].max() <= 0.0


def test_expanded_rule_operations():
    class ComplexRules(FrameSpec):
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

        class BadRule(FrameSpec):
            x = ColSpec(
                dtype=pl.Int32,
                rules=(
                    ColRule(when={"column": "non_existent", "equals": 1}, choices=[0]),
                ),
            )


def test_subclass_attribute_overriding():
    class BaseSpec(FrameSpec):
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
    class TemporalAndBinary(FrameSpec):
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
    Loaded = FrameSpec.from_yaml(source=yaml_path)
    assert Loaded.schema() == TemporalAndBinary.schema()
    df_loaded = Loaded.generate(500, seed=42)
    df_orig = TemporalAndBinary.generate(500, seed=42)
    assert df_orig.equals(df_loaded)


def test_statistical_distributions():
    class DistSpec(FrameSpec):
        norm_float = ColSpec(
            dtype=pl.Float64,
            distribution="normal",
            distribution_params={"mean": 100.0, "std": 15.0},
        )
        norm_int = ColSpec(
            dtype=pl.Int32,
            distribution="normal",
            distribution_params={"mean": 50.0, "std": 5.0},
            bounds=(0, 100),
        )
        lognorm_float = ColSpec(
            dtype=pl.Float64,
            distribution="lognormal",
            distribution_params={"mean": 0.0, "std": 0.5},
        )
        exp_float = ColSpec(
            dtype=pl.Float64,
            distribution="exponential",
            distribution_params={"rate": 0.1},
        )
        poisson_int = ColSpec(
            dtype=pl.Int64,
            distribution="poisson",
            distribution_params={"lambda": 10.0},
        )
        gamma_float = ColSpec(
            dtype=pl.Float64,
            distribution="gamma",
            distribution_params={"shape": 2.0, "scale": 2.0},
        )
        beta_float = ColSpec(
            dtype=pl.Float64,
            distribution="beta",
            distribution_params={"alpha": 2.0, "beta": 5.0},
        )

    df = DistSpec.generate(50_000, seed=42)
    assert df.height == 50_000

    # Normal float verification (mean ~ 100, std ~ 15)
    mean_val = df["norm_float"].mean()
    std_val = df["norm_float"].std()
    assert 99.0 <= mean_val <= 101.0
    assert 14.5 <= std_val <= 15.5

    # Normal int bounded verification
    assert df["norm_int"].min() >= 0
    assert df["norm_int"].max() <= 100
    assert 48.0 <= df["norm_int"].mean() <= 52.0

    # Lognormal verification (strictly positive)
    assert df["lognorm_float"].min() > 0.0

    # Exponential verification (strictly positive, mean ~ 1/rate = 10)
    assert df["exp_float"].min() >= 0.0
    assert 9.0 <= df["exp_float"].mean() <= 11.0

    # Poisson verification (mean ~ 10)
    assert 9.5 <= df["poisson_int"].mean() <= 10.5

    # Gamma verification (mean = shape * scale = 4.0)
    assert df["gamma_float"].min() >= 0.0
    assert 3.8 <= df["gamma_float"].mean() <= 4.2

    # Beta verification (values in [0, 1], mean = alpha / (alpha + beta) = 2/7 ~ 0.2857)
    assert df["beta_float"].min() >= 0.0
    assert df["beta_float"].max() <= 1.0
    assert 0.27 <= df["beta_float"].mean() <= 0.30


def test_distribution_validation_and_errors():
    with pytest.raises(ValueError, match="Unsupported distribution"):
        ColSpec(dtype=pl.Float64, distribution="invalid_dist")

    with pytest.raises(ValueError, match="Normal distribution std must be positive"):
        ColSpec(
            dtype=pl.Float64, distribution="normal", distribution_params={"std": -1.0}
        )

    with pytest.raises(
        ValueError, match="Exponential distribution scale must be positive"
    ):
        ColSpec(
            dtype=pl.Float64,
            distribution="exponential",
            distribution_params={"scale": -2.0},
        )


def test_weighted_choices_enum_and_categorical():
    class WeightedSpec(FrameSpec):
        enum_col = ColSpec(
            dtype=pl.Enum(["rare", "common"]),
            weights=[0.1, 0.9],
        )
        cat_dict_col = ColSpec(
            dtype=pl.Categorical,
            choices={"A": 0.8, "B": 0.2},
        )
        str_choices_col = ColSpec(
            dtype=pl.String,
            choices=["first", "second"],
            weights=[0.75, 0.25],
        )
        int_choices_col = ColSpec(
            dtype=pl.Int64,
            choices=[100, 200, 300],
            weights=[0.7, 0.2, 0.1],
        )
        bool_col = ColSpec(
            dtype=pl.Boolean,
            weights=[0.8, 0.2],  # [p_false, p_true] -> 20% True
        )

    df = WeightedSpec.generate(50_000, seed=42)
    assert df.height == 50_000

    # Enum verification: common ~ 90%, rare ~ 10%
    counts = df["enum_col"].value_counts()
    common_ratio = counts.filter(pl.col("enum_col") == "common")["count"][0] / 50_000
    assert 0.88 <= common_ratio <= 0.92

    # Categorical verification: A ~ 80%, B ~ 20%
    cat_counts = df["cat_dict_col"].value_counts()
    a_ratio = cat_counts.filter(pl.col("cat_dict_col") == "A")["count"][0] / 50_000
    assert 0.78 <= a_ratio <= 0.82

    # String choices verification: first ~ 75%
    str_counts = df["str_choices_col"].value_counts()
    first_ratio = (
        str_counts.filter(pl.col("str_choices_col") == "first")["count"][0] / 50_000
    )
    assert 0.73 <= first_ratio <= 0.77

    # Int choices verification: 100 ~ 70%, 200 ~ 20%, 300 ~ 10%
    int_counts = df["int_choices_col"].value_counts()
    int_100_ratio = (
        int_counts.filter(pl.col("int_choices_col") == 100)["count"][0] / 50_000
    )
    int_300_ratio = (
        int_counts.filter(pl.col("int_choices_col") == 300)["count"][0] / 50_000
    )
    assert 0.68 <= int_100_ratio <= 0.72
    assert 0.08 <= int_300_ratio <= 0.12

    # Bool verification: True ~ 20%
    true_ratio = df["bool_col"].sum() / 50_000
    assert 0.18 <= true_ratio <= 0.22


def test_colrule_weighted_choices():
    class RuleWeightedSpec(FrameSpec):
        segment = ColSpec(dtype=pl.Enum(["standard", "premium"]))
        reward = ColSpec(
            dtype=pl.String,
            rules=(
                ColRule(
                    when={"column": "segment", "equals": "standard"},
                    choices={"voucher": 0.9, "gift": 0.1},
                ),
                ColRule(
                    when={"column": "segment", "equals": "premium"},
                    choices=["gift", "vip_pass"],
                    weights=[0.3, 0.7],
                ),
            ),
        )

    df = RuleWeightedSpec.generate(30_000, seed=42)
    standard_df = df.filter(pl.col("segment") == "standard")
    voucher_ratio = (
        standard_df.filter(pl.col("reward") == "voucher").height / standard_df.height
    )
    assert 0.88 <= voucher_ratio <= 0.92

    premium_df = df.filter(pl.col("segment") == "premium")
    vip_ratio = (
        premium_df.filter(pl.col("reward") == "vip_pass").height / premium_df.height
    )
    assert 0.67 <= vip_ratio <= 0.73


def test_weights_validation_errors():
    with pytest.raises(ValueError, match="must match length of choices"):
        ColSpec(dtype=pl.String, choices=["a", "b"], weights=[1.0])

    with pytest.raises(ValueError, match="must match number of Enum categories"):
        ColSpec(dtype=pl.Enum(["a", "b", "c"]), weights=[0.5, 0.5])

    with pytest.raises(
        ValueError, match="Boolean weights must be a 2-element sequence"
    ):
        ColSpec(dtype=pl.Boolean, weights=[0.5, 0.3, 0.2])

    with pytest.raises(ValueError, match="Weights must all be non-negative"):
        ColSpec(dtype=pl.Enum(["a", "b"]), weights=[-0.1, 1.1])

    with pytest.raises(ValueError, match="Sum of weights must be positive"):
        ColSpec(dtype=pl.Enum(["a", "b"]), weights=[0.0, 0.0])

    with pytest.raises(
        ValueError,
        match="Cannot specify both a dict for choices and an explicit weights parameter",
    ):
        ColSpec(dtype=pl.String, choices={"a": 0.5, "b": 0.5}, weights=[0.5, 0.5])

    with pytest.raises(
        ValueError,
        match="Cannot specify both a dict for choices and an explicit weights parameter",
    ):
        ColRule(
            when={"column": "x", "equals": 1},
            choices={"a": 0.5, "b": 0.5},
            weights=[0.5, 0.5],
        )


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


def test_cartesian_with_custom_choices():
    class ChoiceCartesian(FrameSpec):
        size = ColSpec(dtype=pl.String, choices=["S", "M", "L"])
        color = ColSpec(dtype=pl.String, choices=["Red", "Blue"], nullable=True)

    df = ChoiceCartesian.generate(n=1, method="cartesian", seed=42)
    # 3 sizes * (2 colors + null) = 9 combinations
    assert df.height == 9
    assert df.select("size", "color").unique().height == 9


def test_from_dataframe_basic():
    from datetime import UTC, date, datetime

    source_df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "price": [10.5, 20.0, 15.25, 30.0, 25.5],
            "is_active": [True, False, True, True, False],
            "created_date": [
                date(2023, 1, 1),
                date(2023, 6, 15),
                date(2023, 12, 31),
                date(2023, 3, 10),
                date(2023, 8, 20),
            ],
            "created_at": [
                datetime(2023, 1, 1, 10, 0, tzinfo=UTC),
                datetime(2023, 6, 15, 12, 30, tzinfo=UTC),
                datetime(2023, 12, 31, 23, 59, tzinfo=UTC),
                datetime(2023, 3, 10, 8, 15, tzinfo=UTC),
                datetime(2023, 8, 20, 14, 45, tzinfo=UTC),
            ],
            "category": pl.Series(
                ["electronics", "clothing", "electronics", "food", "food"],
                dtype=pl.Enum(["electronics", "clothing", "food"]),
            ),
            "notes": [
                "short note",
                "a slightly longer note here",
                "abc",
                "tiny",
                "medium note text",
            ],
        }
    )

    Profiled = FrameSpec.from_dataframe(
        source_df, name="StoreProfile", max_unique_enum=2
    )
    assert Profiled.__name__ == "StoreProfile"

    cols = Profiled._columns
    assert cols["id"].dtype == pl.Int64
    assert cols["id"].bounds.min == 1
    assert cols["id"].bounds.max == 5
    assert not cols["id"].nullable

    assert cols["price"].dtype == pl.Float64
    assert cols["price"].bounds.min == 10.5
    assert cols["price"].bounds.max == 30.0

    assert cols["is_active"].dtype == pl.Boolean

    assert cols["created_date"].dtype == pl.Date
    assert cols["created_at"].dtype == pl.Datetime("us", "UTC")

    assert isinstance(cols["category"].dtype, pl.Enum)
    assert cols["notes"].dtype == pl.String
    assert cols["notes"].string_length.min == 3
    assert cols["notes"].string_length.max == 27

    # Generate from profiled spec
    gen_df = Profiled.generate(100, seed=42)
    assert gen_df.height == 100
    assert gen_df.schema["id"] == pl.Int64
    assert gen_df.schema["created_date"] == pl.Date
    assert gen_df["id"].min() >= 1
    assert gen_df["id"].max() <= 5


def test_from_dataframe_weights_and_enums():
    # 80% cat, 20% dog
    species = ["cat"] * 800 + ["dog"] * 200
    # 90% True, 10% False
    flags = [True] * 900 + [False] * 100

    df = pl.DataFrame(
        {
            "species": species,
            "flag": flags,
        }
    )

    ProfiledWeighted = FrameSpec.from_dataframe(df, weights=True, max_unique_enum=10)
    cols = ProfiledWeighted._columns

    # Species should be converted to Enum with categories ["cat", "dog"] and weights [0.8, 0.2]
    assert isinstance(cols["species"].dtype, pl.Enum)
    assert cols["species"].dtype.categories.to_list() == ["cat", "dog"]
    assert cols["species"].weights == pytest.approx((0.8, 0.2), abs=1e-4)

    # Boolean weights: [p_false, p_true] -> [0.1, 0.9]
    assert cols["flag"].dtype == pl.Boolean
    assert cols["flag"].weights == pytest.approx((0.1, 0.9), abs=1e-4)

    # Generate and verify empirical convergence
    gen = ProfiledWeighted.generate(20_000, seed=42)
    cat_ratio = (gen["species"] == "cat").sum() / 20_000
    true_ratio = gen["flag"].sum() / 20_000
    assert 0.78 <= cat_ratio <= 0.82
    assert 0.88 <= true_ratio <= 0.92


def test_from_dataframe_max_unique_threshold():
    df = pl.DataFrame(
        {
            "low_card": ["A", "B", "C", "A", "B"] * 20,
            "high_card": [f"user_{i}" for i in range(100)],
        }
    )

    # max_unique = 5 -> low_card (3 unique) becomes Enum, high_card (100 unique) stays String
    Spec1 = FrameSpec.from_dataframe(df, max_unique_enum=5)
    assert isinstance(Spec1._columns["low_card"].dtype, pl.Enum)
    assert Spec1._columns["high_card"].dtype == pl.String

    # Using alias max_unique
    Spec2 = FrameSpec.from_dataframe(df, max_unique=2)
    # low_card has 3 unique > 2, so it remains String
    assert Spec2._columns["low_card"].dtype == pl.String


def test_from_dataframe_calculate_bounds_toggle():
    df = pl.DataFrame(
        {
            "num": [10, 20, 30, 40, 50],
            "txt": ["hello", "world", "longer text here", "a", "bc"],
        }
    )

    SpecWithBounds = FrameSpec.from_dataframe(
        df, calculate_bounds=True, max_unique_enum=0
    )
    assert SpecWithBounds._columns["num"].bounds == Bound(10, 50)
    assert SpecWithBounds._columns["txt"].string_length == Bound(1, 16)

    SpecNoBounds = FrameSpec.from_dataframe(
        df, calculate_bounds=False, max_unique_enum=0
    )
    assert SpecNoBounds._columns["num"].bounds is None
    assert SpecNoBounds._columns["txt"].string_length is None

    # Test alias bounds=False
    SpecNoBoundsAlias = FrameSpec.from_dataframe(df, bounds=False, max_unique_enum=0)
    assert SpecNoBoundsAlias._columns["num"].bounds is None
    assert SpecNoBoundsAlias._columns["txt"].string_length is None


def test_from_dataframe_nullability_and_edge_cases():
    df = pl.DataFrame(
        {
            "with_nulls": [1, None, 3, None, 5],
            "no_nulls": [10, 20, 30, 40, 50],
            "all_nulls": [None, None, None, None, None],
        },
        schema={"with_nulls": pl.Int64, "no_nulls": pl.Int64, "all_nulls": pl.Float64},
    )

    Spec = FrameSpec.from_dataframe(df)
    cols = Spec._columns

    assert cols["with_nulls"].nullable is True
    assert cols["with_nulls"].null_probability == pytest.approx(0.4, abs=1e-4)
    assert cols["with_nulls"].bounds == Bound(1, 5)

    assert cols["no_nulls"].nullable is False
    assert cols["no_nulls"].null_probability == 0.0
    assert cols["no_nulls"].bounds == Bound(10, 50)

    assert cols["all_nulls"].nullable is True
    assert cols["all_nulls"].null_probability == 1.0
    assert cols["all_nulls"].bounds is None

    # Non-dataframe raises TypeError
    with pytest.raises(TypeError, match="Expected pl.DataFrame"):
        FrameSpec.from_dataframe([{"a": 1}])  # type: ignore[arg-type]

    # Empty dataframe (0 rows)
    empty_df = pl.DataFrame({"a": [], "b": []}, schema={"a": pl.Int32, "b": pl.String})
    EmptySpec = FrameSpec.from_dataframe(empty_df)
    assert EmptySpec.schema() == empty_df.schema
    assert not EmptySpec._columns["a"].nullable


def test_from_dataframe_temporal_and_binary(tmp_path):
    from datetime import time, timedelta

    df = pl.DataFrame(
        {
            "t": [time(8, 0), time(12, 30), time(18, 45)],
            "dur": [
                timedelta(seconds=10),
                timedelta(seconds=60),
                timedelta(seconds=120),
            ],
            "bin": [b"hello", b"polars", b"data"],
        },
        schema={
            "t": pl.Time,
            "dur": pl.Duration("ms"),
            "bin": pl.Binary,
        },
    )

    Spec = FrameSpec.from_dataframe(df)
    cols = Spec._columns

    assert cols["t"].dtype == pl.Time
    assert cols["t"].bounds is not None
    assert cols["dur"].dtype == pl.Duration("ms")
    assert cols["dur"].bounds is not None
    assert cols["bin"].dtype == pl.Binary
    assert cols["bin"].string_length == Bound(4, 6)

    # Roundtrip through YAML
    yaml_path = tmp_path / "temporal_profile.yaml"
    Spec.to_yaml(yaml_path)
    LoadedSpec = FrameSpec.from_yaml(yaml_path)
    assert LoadedSpec.schema() == Spec.schema()

    # Generate from LoadedSpec
    gen = LoadedSpec.generate(100, seed=42)
    assert gen.height == 100
    assert gen.schema == df.schema


def test_modular_subpackage_imports():
    from polspec import (
        Bound,
        ColRule,
        ColSpec,
        FrameSchema,
        FrameSpec,
        ValidationError,
        profile_dataframe,
    )
    from polspec.bound import Bound as BoundDirect
    from polspec.engine import _generate_cartesian, _generate_random
    from polspec.framespec import (
        FrameSchema as FrameSchemaDirect,
    )
    from polspec.framespec import (
        FrameSpec as FrameSpecDirect,
    )
    from polspec.profiler import profile_dataframe as profile_dataframe_direct
    from polspec.rules import ColRule as ColRuleDirect
    from polspec.serialization import _colspec_from_yaml, _colspec_to_yaml
    from polspec.spec import ColSpec as ColSpecDirect
    from polspec.validation import (
        ValidationError as ValidationErrorDirect,
    )
    from polspec.validation import (
        _validate_dataframe,
    )

    assert Bound is BoundDirect
    assert ColRule is ColRuleDirect
    assert ColSpec is ColSpecDirect
    assert FrameSpec is FrameSpecDirect
    assert FrameSchema is FrameSchemaDirect
    assert FrameSchema is FrameSpec
    assert ValidationError is ValidationErrorDirect
    assert profile_dataframe is profile_dataframe_direct
    assert callable(_colspec_to_yaml)
    assert callable(_colspec_from_yaml)
    assert callable(_generate_random)
    assert callable(_generate_cartesian)
    assert callable(_validate_dataframe)


def test_colspec_and_framespec_tags(tmp_path):
    # 1. ColSpec initialization with tags
    c1 = ColSpec(pl.Int64, tags="aggregate")
    assert c1.tags == ("aggregate",)

    c2 = ColSpec(pl.Float64, tags=["aggregate", "index"])
    assert c2.tags == ("aggregate", "index")

    c3 = ColSpec(pl.String, tags=("feature", "metadata"))
    assert c3.tags == ("feature", "metadata")

    c4 = ColSpec(pl.Boolean, tags="")
    assert c4.tags == ()

    c5 = ColSpec(pl.Date, tags=None)
    assert c5.tags == ()

    c6 = ColSpec(pl.Time, tags=["dup", "dup", "unique"])
    assert c6.tags == ("dup", "unique")

    with pytest.raises(TypeError, match="ColSpec.tags must be a string or sequence"):
        ColSpec(pl.Int64, tags=123)  # type: ignore[arg-type]

    # 2. FrameSpec.tag query
    class TaggedSpec(FrameSpec):
        id_col = ColSpec(pl.Int64, tags="index")
        agg_val = ColSpec(pl.Float64, tags=["aggregate", "metric"])
        total_sum = ColSpec(pl.Float64, tags=["aggregate", "index"])
        feature_1 = ColSpec(pl.String, tags="feature")
        untagged = ColSpec(pl.Boolean)

    # Single tag
    assert TaggedSpec.tag("aggregate") == ["agg_val", "total_sum"]
    assert TaggedSpec.tag("index") == ["id_col", "total_sum"]
    assert TaggedSpec.tag("feature") == ["feature_1"]
    assert TaggedSpec.tag("non_existent") == []

    # Sequence of tags (match="any" by default)
    assert TaggedSpec.tag(["aggregate", "feature"]) == [
        "agg_val",
        "total_sum",
        "feature_1",
    ]
    assert TaggedSpec.tag(["index", "feature"]) == ["id_col", "total_sum", "feature_1"]

    # Multiple positional arguments (match="any" by default)
    assert TaggedSpec.tag("aggregate", "feature") == [
        "agg_val",
        "total_sum",
        "feature_1",
    ]
    assert TaggedSpec.tag("index", "feature") == ["id_col", "total_sum", "feature_1"]

    # Multiple tags with match="all"
    assert TaggedSpec.tag(["aggregate", "index"], match="all") == ["total_sum"]
    assert TaggedSpec.tag("aggregate", "index", match="all") == ["total_sum"]
    assert TaggedSpec.tag(["aggregate", "feature"], match="all") == []
    assert TaggedSpec.tag("aggregate", "feature", match="all") == []

    # Empty tags returns empty list
    assert TaggedSpec.tag() == []

    # Invalid tag type
    with pytest.raises(TypeError, match="Tag must be a string or sequence"):
        TaggedSpec.tag(123)  # type: ignore[arg-type]

    # Invalid match mode
    with pytest.raises(ValueError, match="Invalid match mode"):
        TaggedSpec.tag(["aggregate"], match="invalid")  # type: ignore[arg-type]

    # FrameSpec.tags should not exist
    assert not hasattr(TaggedSpec, "tags")

    # 3. YAML serialization roundtrip preserving tags
    yaml_file = tmp_path / "tagged_spec.yaml"
    TaggedSpec.to_yaml(yaml_file)
    LoadedTagged = FrameSpec.from_yaml(yaml_file)
    assert LoadedTagged.tag("aggregate") == ["agg_val", "total_sum"]
    assert LoadedTagged.tag(["index", "feature"]) == [
        "id_col",
        "total_sum",
        "feature_1",
    ]
    assert LoadedTagged._columns["id_col"].tags == ("index",)
    assert LoadedTagged._columns["agg_val"].tags == ("aggregate", "metric")


def test_check_construction_rejects_non_expr():
    with pytest.raises(TypeError, match="Check expr must be a polars Expr"):
        Check("col(a) > col(b)")  # type: ignore[arg-type]


def test_checks_declaration_rejects_non_check_items():
    with pytest.raises(TypeError, match="Items in __checks__ must be Check instances"):

        class BadSpec(FrameSpec):
            x = ColSpec(pl.Int64)
            __checks__ = ["not_a_check"]  # type: ignore[list-item]


def test_check_anonymous_naming_defaults_to_expr_repr():
    class AnonSpec(FrameSpec):
        id_col = ColSpec(pl.Int64, unique=True, nullable=True)
        val = ColSpec(pl.Int64)

        __checks__ = [Check(pl.col("val") > 0)]

    check_obj = AnonSpec.checks()[0]
    assert "val" in check_obj.name


def test_checks_declaration_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate Check name 'positive'"):

        class DupCheckSpec(FrameSpec):
            a = ColSpec(pl.Int64)
            b = ColSpec(pl.Int64)
            __checks__ = [
                Check(pl.col("a") > 0, name="positive"),
                Check(pl.col("b") > 0, name="positive"),
            ]


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


def test_unique_together_declaration_variations_and_validation():
    # Single tuple of column names
    class SingleTupleSpec(FrameSpec):
        tenant_id = ColSpec(pl.Int32)
        account_id = ColSpec(pl.Int32)
        __unique_together__ = ("tenant_id", "account_id")

    assert SingleTupleSpec.unique_together() == (("tenant_id", "account_id"),)

    # Reference to non-existent column raises ValueError
    with pytest.raises(ValueError, match="references unknown column 'missing_col'"):

        class BadUniqueSpec(FrameSpec):
            a = ColSpec(pl.Int64)
            __unique_together__ = [("a", "missing_col")]


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


def test_framespec_to_markdown_and_to_mermaid(tmp_path):
    class CustomerSpec(FrameSpec):
        customer_id = ColSpec(
            pl.Int64, unique=True, bounds=Bound(1, 1_000_000), tags="index"
        )
        tier = ColSpec(
            pl.Enum(["BRONZE", "SILVER", "GOLD"]), nullable=False, tags="segment"
        )
        score = ColSpec(pl.Float64, bounds=Bound(0.0, 100.0), nullable=True)
        country = ColSpec(pl.String, choices=["US", "UK", "DE", "FR"], tags="geo")
        created_date = ColSpec(pl.Date, nullable=False, tags="temporal")

        __unique_together__ = [("customer_id", "country")]
        __checks__ = [
            Check(
                pl.col("score") >= 0.0,
                name="score_non_negative",
                description="Credit score must be non-negative if present",
            )
        ]

    # 1. to_markdown() without path
    md_str = CustomerSpec.to_markdown(title="Customer Data Dictionary")
    assert "# Customer Data Dictionary" in md_str
    assert "## Overview" in md_str
    assert "| `customer_id` |" in md_str
    assert "| `tier` |" in md_str
    assert "score_non_negative" in md_str
    assert "['customer_id', 'country']" in md_str
    assert "`index`" in md_str
    assert "`segment`" in md_str

    # 2. to_markdown() with file path
    md_file = tmp_path / "customer_dict.md"
    written_md = CustomerSpec.to_markdown(md_file)
    assert md_file.exists()
    assert md_file.read_text(encoding="utf-8") == written_md

    # 3. to_mermaid() without path
    mermaid_str = CustomerSpec.to_mermaid()
    assert "erDiagram" in mermaid_str
    assert "CustomerSpec {" in mermaid_str
    assert "Int64 customer_id PK" in mermaid_str
    assert "Enum tier" in mermaid_str
    assert "tags: [segment]" in mermaid_str
    assert "bounds: [1, 1000000]" in mermaid_str

    # 4. to_mermaid() with file path
    mermaid_file = tmp_path / "customer_erd.mmd"
    written_mermaid = CustomerSpec.to_mermaid(mermaid_file)
    assert mermaid_file.exists()
    assert mermaid_file.read_text(encoding="utf-8") == written_mermaid

    # 5. to_mermaid() with quotes in choices / tags / fk names
    class QuotedSpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        flag = ColSpec(pl.String, choices=['a"1', 'b"2'], tags=['geo"zone'])
        __foreign_keys__ = [ForeignKey("id", references="self", name='quoted"fk')]

    quoted_mmd = QuotedSpec.to_mermaid()
    # Ensure double quotes inside attributes are replaced to avoid breaking mermaid ER syntax
    assert "choices: [a'1, b'2]" in quoted_mmd
    assert "tags: [geo'zone]" in quoted_mmd
    assert "quoted'fk" in quoted_mmd


def test_yaml_nested_directory_creation_and_utf8(tmp_path):
    class UnicodeYamlSpec(FrameSpec):
        user_id = ColSpec(pl.Int64, unique=True)
        comment = ColSpec(pl.String, choices=["café", "naïve", "🚀"])

    nested_file = tmp_path / "nested" / "subfolder" / "spec.yaml"
    UnicodeYamlSpec.to_yaml(nested_file)
    assert nested_file.exists()

    LoadedSpec = FrameSpec.from_yaml(nested_file)
    assert LoadedSpec._columns["comment"].choices == ("café", "naïve", "🚀")


# =====================================================================
# Foreign Keys (ForeignKey / __foreign_keys__)
# =====================================================================


class CustomerFkSpec(FrameSpec):
    id = ColSpec(pl.Int64, unique=True)
    code = ColSpec(pl.String, unique=True)


def test_foreign_key_declaration_variations():
    # Single column, ref_columns defaults to the same name
    class SameNameFk(FrameSpec):
        id = ColSpec(pl.Int64)
        __foreign_keys__ = [ForeignKey("id", references=CustomerFkSpec)]

    fk = SameNameFk.foreign_keys()[0]
    assert fk.columns == ("id",)
    assert fk.ref_columns == ("id",)
    assert fk.references is CustomerFkSpec
    assert fk.name == "fk_id__CustomerFkSpec"

    # Explicit ref_columns and name
    class RenamedFk(FrameSpec):
        customer_id = ColSpec(pl.Int64)
        __foreign_keys__ = [
            ForeignKey(
                "customer_id",
                references=CustomerFkSpec,
                ref_columns="id",
                name="customer_fk",
            )
        ]

    fk2 = RenamedFk.foreign_keys()[0]
    assert fk2.columns == ("customer_id",)
    assert fk2.ref_columns == ("id",)
    assert fk2.name == "customer_fk"

    # Composite key
    class CompositeFk(FrameSpec):
        a = ColSpec(pl.Int64)
        b = ColSpec(pl.String)
        __foreign_keys__ = [
            ForeignKey(
                ["a", "b"], references=CustomerFkSpec, ref_columns=["id", "code"]
            )
        ]

    fk3 = CompositeFk.foreign_keys()[0]
    assert fk3.columns == ("a", "b")
    assert fk3.ref_columns == ("id", "code")

    # Self reference
    class SelfFk(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        parent_id = ColSpec(pl.Int64, nullable=True)
        __foreign_keys__ = [
            ForeignKey("parent_id", references="self", ref_columns="id")
        ]

    fk4 = SelfFk.foreign_keys()[0]
    assert fk4.references == "self"
    assert fk4.name == "fk_parent_id__self"


def test_foreign_key_mismatched_ref_columns_length_raises():
    with pytest.raises(ValueError, match="must have the same length"):
        ForeignKey(["a", "b"], references=CustomerFkSpec, ref_columns=["id"])


def test_foreign_key_rejects_invalid_references_type():
    with pytest.raises(TypeError, match="must be a FrameSpec subclass"):
        ForeignKey("a", references=123)  # type: ignore[arg-type]


def test_foreign_key_declaration_rejects_unknown_local_column():
    with pytest.raises(ValueError, match="unknown local column 'missing'"):

        class BadLocal(FrameSpec):
            a = ColSpec(pl.Int64)
            __foreign_keys__ = [ForeignKey("missing", references=CustomerFkSpec)]


def test_foreign_key_declaration_rejects_unknown_ref_column():
    with pytest.raises(
        ValueError, match="unknown column 'missing' on 'CustomerFkSpec'"
    ):

        class BadRef(FrameSpec):
            a = ColSpec(pl.Int64)
            __foreign_keys__ = [
                ForeignKey("a", references=CustomerFkSpec, ref_columns="missing")
            ]


def test_foreign_key_declaration_rejects_dtype_mismatch():
    with pytest.raises(ValueError, match="not dtype-compatible"):

        class BadDtype(FrameSpec):
            a = ColSpec(pl.String)
            __foreign_keys__ = [
                ForeignKey("a", references=CustomerFkSpec, ref_columns="id")
            ]


def test_foreign_key_declaration_allows_string_enum_categorical_bucket():
    # String/Enum/Categorical are treated as one compatible domain bucket.
    class EnumChild(FrameSpec):
        code = ColSpec(pl.Enum(["A", "B"]))
        __foreign_keys__ = [
            ForeignKey("code", references=CustomerFkSpec, ref_columns="code")
        ]

    assert EnumChild.foreign_keys()[0].columns == ("code",)


def test_foreign_key_declaration_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate ForeignKey name 'dup'"):

        class DupFk(FrameSpec):
            a = ColSpec(pl.Int64)
            b = ColSpec(pl.String)
            __foreign_keys__ = [
                ForeignKey(
                    "a", references=CustomerFkSpec, ref_columns="id", name="dup"
                ),
                ForeignKey(
                    "b", references=CustomerFkSpec, ref_columns="code", name="dup"
                ),
            ]


def test_foreign_key_inheritance_deduplicates_and_accumulates():
    class BaseFkSpec(FrameSpec):
        a = ColSpec(pl.Int64)
        __foreign_keys__ = [
            ForeignKey("a", references=CustomerFkSpec, ref_columns="id")
        ]

    class ExtendedFkSpec(BaseFkSpec):
        b = ColSpec(pl.String)
        __foreign_keys__ = [
            ForeignKey("b", references=CustomerFkSpec, ref_columns="code")
        ]

    assert len(ExtendedFkSpec.foreign_keys()) == 2
    fk_names = [fk.name for fk in ExtendedFkSpec.foreign_keys()]
    assert fk_names == ["fk_a__CustomerFkSpec", "fk_b__CustomerFkSpec"]


def test_foreign_key_yaml_roundtrip_self_reference(tmp_path):
    class HierarchySpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        parent_id = ColSpec(pl.Int64, nullable=True)
        __foreign_keys__ = [
            ForeignKey("parent_id", references="self", ref_columns="id")
        ]

    yaml_path = tmp_path / "hierarchy_spec.yaml"
    HierarchySpec.to_yaml(yaml_path)
    assert "foreign_keys" in yaml_path.read_text()

    Loaded = FrameSpec.from_yaml(yaml_path)
    assert Loaded.foreign_keys() == (
        ForeignKey("parent_id", references="self", ref_columns="id"),
    )

    df_valid = pl.DataFrame({"id": [1, 2, 3], "parent_id": [None, 1, 1]})
    assert Loaded.validate(df_valid).height == 3

    df_invalid = pl.DataFrame({"id": [1, 2], "parent_id": [1, 999]})
    with pytest.raises(ValidationError, match="ForeignKey"):
        Loaded.validate(df_invalid)


def test_to_yaml_warns_when_foreign_keys_to_other_specs_are_not_persisted(tmp_path):
    class ChildSpec(FrameSpec):
        customer_id = ColSpec(pl.Int64)
        __foreign_keys__ = [
            ForeignKey("customer_id", references=CustomerFkSpec, ref_columns="id")
        ]

    yaml_path = tmp_path / "child_spec.yaml"
    with pytest.warns(UserWarning, match="fk_customer_id__CustomerFkSpec"):
        ChildSpec.to_yaml(yaml_path)

    Loaded = FrameSpec.from_yaml(yaml_path)
    assert Loaded.foreign_keys() == ()


def test_framespec_to_markdown_and_to_mermaid_with_foreign_keys():
    class OrderFkSpec(FrameSpec):
        order_id = ColSpec(pl.Int64, unique=True)
        customer_id = ColSpec(pl.Int64, nullable=True)
        __foreign_keys__ = [
            ForeignKey("customer_id", references=CustomerFkSpec, ref_columns="id")
        ]

    md = OrderFkSpec.to_markdown()
    assert "**Foreign Keys:** 1 key(s)" in md
    assert "### Foreign Keys" in md
    assert "fk_customer_id__CustomerFkSpec" in md
    assert "['customer_id']" in md
    assert "CustomerFkSpec.['id']" in md

    mmd = OrderFkSpec.to_mermaid()
    assert "Int64 customer_id FK" in mmd
    assert 'CustomerFkSpec ||--o{ OrderFkSpec : "fk_customer_id__CustomerFkSpec"' in mmd

    class SelfFkSpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        parent_id = ColSpec(pl.Int64, nullable=True)
        __foreign_keys__ = [
            ForeignKey("parent_id", references="self", ref_columns="id")
        ]

    self_mmd = SelfFkSpec.to_mermaid()
    assert 'SelfFkSpec ||--o{ SelfFkSpec : "fk_parent_id__self"' in self_mmd


# =====================================================================
# Foreign Keys -- referentially-consistent generation
# =====================================================================


class GenCustomerSpec(FrameSpec):
    id = ColSpec(pl.Int64, unique=True)
    name = ColSpec(pl.String)


class GenOrderSpec(FrameSpec):
    order_id = ColSpec(pl.Int64, unique=True)
    customer_id = ColSpec(pl.Int64, nullable=True, null_probability=0.3)
    __foreign_keys__ = [
        ForeignKey("customer_id", references=GenCustomerSpec, ref_columns="id")
    ]


def test_generate_foreign_key_samples_from_parent_when_references_supplied():
    customers = GenCustomerSpec.generate(5, seed=1)
    customer_ids = set(customers["id"].to_list())

    orders = GenOrderSpec.generate(200, seed=2, references={GenCustomerSpec: customers})
    non_null = [v for v in orders["customer_id"].to_list() if v is not None]

    assert non_null  # sanity: at least some non-null rows given the sample size
    assert all(v in customer_ids for v in non_null)
    # No validation error against the very parent it was sampled from.
    GenOrderSpec.validate(orders, references={GenCustomerSpec: customers})


def test_generate_foreign_key_accepts_lazyframe_reference():
    customers = GenCustomerSpec.generate(5, seed=1)
    customer_ids = set(customers["id"].to_list())

    orders = GenOrderSpec.generate(
        50, seed=2, references={GenCustomerSpec: customers.lazy()}
    )
    non_null = [v for v in orders["customer_id"].to_list() if v is not None]
    assert non_null
    assert all(v in customer_ids for v in non_null)


def test_generate_foreign_key_without_references_uses_free_generation():
    # No `references` supplied: runs fine and behaves exactly as if the FK
    # weren't declared (deterministic given the same seed).
    orders_a = GenOrderSpec.generate(50, seed=2)
    orders_b = GenOrderSpec.generate(50, seed=2)
    assert orders_a.height == 50
    assert orders_a.equals(orders_b)


def test_generate_foreign_key_respects_null_probability():
    customers = GenCustomerSpec.generate(5, seed=1)
    orders = GenOrderSpec.generate(
        2_000, seed=7, references={GenCustomerSpec: customers}
    )
    null_frac = orders["customer_id"].null_count() / orders.height
    # null_probability=0.3 on a 2000-row sample; loose tolerance for randomness.
    assert 0.2 < null_frac < 0.4


def test_generate_foreign_key_self_reference():
    class EmployeeGenSpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        manager_id = ColSpec(pl.Int64, nullable=True, null_probability=0.3)
        __foreign_keys__ = [
            ForeignKey("manager_id", references="self", ref_columns="id")
        ]

    emps = EmployeeGenSpec.generate(100, seed=3)
    ids = set(emps["id"].to_list())
    non_null_mgrs = [v for v in emps["manager_id"].to_list() if v is not None]
    assert non_null_mgrs
    assert all(v in ids for v in non_null_mgrs)


def test_generate_foreign_key_composite_samples_jointly():
    class RegionGenSpec(FrameSpec):
        tenant = ColSpec(pl.Int64)
        region_id = ColSpec(pl.Int64)
        __unique_together__ = [("tenant", "region_id")]

    class StoreGenSpec(FrameSpec):
        tenant = ColSpec(pl.Int64)
        region_id = ColSpec(pl.Int64)
        __foreign_keys__ = [
            ForeignKey(
                ["tenant", "region_id"],
                references=RegionGenSpec,
                ref_columns=["tenant", "region_id"],
            )
        ]

    regions = pl.DataFrame({"tenant": [1, 1, 2], "region_id": [10, 20, 10]})
    valid_pairs = {(1, 10), (1, 20), (2, 10)}

    stores = StoreGenSpec.generate(100, seed=4, references={RegionGenSpec: regions})
    pairs = set(zip(stores["tenant"].to_list(), stores["region_id"].to_list()))
    assert pairs <= valid_pairs
    StoreGenSpec.validate(stores, references={RegionGenSpec: regions})


def test_generate_foreign_key_unique_column_samples_without_replacement():
    class ProfileGenSpec(FrameSpec):
        user_id = ColSpec(pl.Int64, unique=True)
        __foreign_keys__ = [
            ForeignKey("user_id", references=GenCustomerSpec, ref_columns="id")
        ]

    customers = GenCustomerSpec.generate(10, seed=1)
    customer_ids = set(customers["id"].to_list())

    # Parent has exactly as many rows as requested: a clean one-to-one mapping
    # should come out as a permutation of the parent ids, with no duplicates.
    profiles = ProfileGenSpec.generate(
        10, seed=5, references={GenCustomerSpec: customers}
    )
    uids = profiles["user_id"].to_list()
    assert len(set(uids)) == len(uids)
    assert set(uids) == customer_ids


def test_generate_foreign_key_empty_parent_raises():
    empty_customers = pl.DataFrame({"id": pl.Series([], dtype=pl.Int64)})
    with pytest.raises(ValueError, match="cannot generate values"):
        GenOrderSpec.generate(10, references={GenCustomerSpec: empty_customers})


def test_generate_foreign_key_parent_with_nulls_filters_them():
    parent_df = pl.DataFrame({"id": [1, None, 2, None, 3]})

    class ChildGenSpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        customer_id = ColSpec(pl.Int64, nullable=False)
        __foreign_keys__ = [
            ForeignKey("customer_id", references=GenCustomerSpec, ref_columns="id")
        ]

    df = ChildGenSpec.generate(20, seed=42, references={GenCustomerSpec: parent_df})
    assert df["customer_id"].null_count() == 0
    assert set(df["customer_id"].to_list()).issubset({1, 2, 3})


def test_generate_foreign_key_parent_all_nulls_raises():
    all_null_parent = pl.DataFrame({"id": [None, None]})

    class ChildGenSpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        customer_id = ColSpec(pl.Int64, nullable=False)
        __foreign_keys__ = [
            ForeignKey("customer_id", references=GenCustomerSpec, ref_columns="id")
        ]

    with pytest.raises(ValueError, match="cannot generate values"):
        ChildGenSpec.generate(10, references={GenCustomerSpec: all_null_parent})


def test_generate_batches_and_sink_thread_foreign_key_references(tmp_path):
    customers = GenCustomerSpec.generate(5, seed=1)
    customer_ids = set(customers["id"].to_list())

    batches = list(
        GenOrderSpec.generate_batches(
            50, batch_size=10, seed=6, references={GenCustomerSpec: customers}
        )
    )
    all_vals = [v for b in batches for v in b["customer_id"].to_list() if v is not None]
    assert all_vals
    assert all(v in customer_ids for v in all_vals)

    out_path = tmp_path / "orders.csv"
    GenOrderSpec.sink_csv(out_path, 30, seed=6, references={GenCustomerSpec: customers})
    sunk = pl.read_csv(out_path)
    sunk_vals = [v for v in sunk["customer_id"].to_list() if v is not None]
    assert sunk_vals
    assert all(v in customer_ids for v in sunk_vals)


# =====================================================================
# Column-level validators (ColSpec.validators)
# =====================================================================


def test_colspec_validators_accepts_bare_expr_and_wraps_in_check():
    spec = ColSpec(pl.Int64, validators=[pl.col("x") % 2 == 0])
    assert len(spec.validators) == 1
    assert isinstance(spec.validators[0], Check)


def test_colspec_validators_accepts_single_expr_without_a_list():
    spec = ColSpec(pl.Int64, validators=pl.col("x") > 0)
    assert len(spec.validators) == 1


def test_colspec_validators_accepts_single_check_without_a_list():
    chk = Check(pl.col("x") > 0, name="positive")
    spec = ColSpec(pl.Int64, validators=chk)
    assert spec.validators == (chk,)


def test_colspec_validators_accepts_check_with_name_and_description():
    spec = ColSpec(
        pl.Float64,
        nullable=True,
        validators=[
            Check(
                pl.col("score") >= 0,
                name="score_non_negative",
                description="Scores can't be negative",
            )
        ],
    )
    assert spec.validators[0].name == "score_non_negative"
    assert spec.validators[0].description == "Scores can't be negative"


def test_colspec_validators_default_empty():
    assert ColSpec(pl.Int64).validators == ()


def test_colspec_validators_rejects_invalid_item_type():
    with pytest.raises(TypeError, match="must be a polars Expr or Check"):
        ColSpec(pl.Int64, validators=["not an expr"])  # type: ignore[list-item]


def test_colspec_validators_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate validator name 'dup'"):
        ColSpec(
            pl.Int64,
            validators=[
                Check(pl.col("x") > 0, name="dup"),
                Check(pl.col("x") < 100, name="dup"),
            ],
        )


def test_colspec_validators_allows_reusing_identical_check():
    # Repeating the exact same Check (same name/expr) is allowed -- it's a
    # duplicate-name collision only when the *definitions* differ.
    chk = Check(pl.col("x") > 0, name="positive")
    spec = ColSpec(pl.Int64, validators=[chk, chk])
    assert spec.validators == (chk, chk)


def test_framespec_rejects_validator_referencing_another_column():
    with pytest.raises(ValueError, match="references other column"):

        class BadSpec(FrameSpec):
            a = ColSpec(pl.Int64, validators=[pl.col("b") > 0])
            b = ColSpec(pl.Int64)


def test_framespec_allows_validator_referencing_only_its_own_column():
    class GoodSpec(FrameSpec):
        a = ColSpec(pl.Int64, validators=[pl.col("a") > 0])

    assert len(GoodSpec._columns["a"].validators) == 1


def test_to_yaml_warns_when_column_validators_are_not_persisted(tmp_path):
    class ValidatedSpec(FrameSpec):
        price = ColSpec(pl.Float64, validators=[pl.col("price") > 0])

    yaml_path = tmp_path / "validated_spec.yaml"
    with pytest.warns(UserWarning, match="price"):
        ValidatedSpec.to_yaml(yaml_path)

    Loaded = FrameSpec.from_yaml(yaml_path)
    assert Loaded._columns["price"].validators == ()


def test_framespec_to_markdown_lists_column_validators():
    class ValidatedSpec(FrameSpec):
        price = ColSpec(
            pl.Float64,
            validators=[
                Check(
                    pl.col("price") > 0,
                    name="price_positive",
                    description="Price must be positive",
                )
            ],
        )

    md = ValidatedSpec.to_markdown()
    assert "### Column Validators" in md
    assert "Column `price`" in md
    assert "price_positive" in md
    assert "Price must be positive" in md
