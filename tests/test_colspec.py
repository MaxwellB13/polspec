import polars as pl
import pytest
import yaml
from polspec import Bound, ColRule, ColSpec, FrameSpec
from polspec.colspec import _colspec_to_yaml


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

    Profiled = FrameSpec.from_dataframe(source_df, name="StoreProfile", max_unique_enum=2)
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

    SpecWithBounds = FrameSpec.from_dataframe(df, calculate_bounds=True, max_unique_enum=0)
    assert SpecWithBounds._columns["num"].bounds == Bound(10, 50)
    assert SpecWithBounds._columns["txt"].string_length == Bound(1, 16)

    SpecNoBounds = FrameSpec.from_dataframe(df, calculate_bounds=False, max_unique_enum=0)
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
    from polspec import Bound, ColRule, ColSpec, FrameSchema, FrameSpec, ValidationError, profile_dataframe
    from polspec.bound import Bound as BoundDirect
    from polspec.framespec import FrameSchema as FrameSchemaDirect, FrameSpec as FrameSpecDirect
    from polspec.engine import _generate_cartesian, _generate_random
    from polspec.profiler import profile_dataframe as profile_dataframe_direct
    from polspec.rules import ColRule as ColRuleDirect
    from polspec.serialization import _colspec_from_yaml, _colspec_to_yaml
    from polspec.spec import ColSpec as ColSpecDirect
    from polspec.validation import ValidationError as ValidationErrorDirect, _validate_dataframe

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
