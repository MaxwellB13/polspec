"""Generation: random and cartesian methods, dtype coverage, distributions, weights.

Everything here asks one question of `generate()`: does the frame it returns
have the shape, domain and reproducibility the spec declared? Rules, foreign
keys and serialization each have their own file.
"""

import polars as pl
import pytest
from polspec import (
    Bound,
    ColRule,
    ColSpec,
    FrameSpec,
)


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


# ---------------------------------------------------------------------------
# cartesian coverage
# ---------------------------------------------------------------------------


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
        ValueError, match="at least one non-unique Enum, Boolean, or bounded numeric"
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


# ---------------------------------------------------------------------------
# every native dtype
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# distributions and weights
# ---------------------------------------------------------------------------


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


def test_cartesian_with_custom_choices():
    class ChoiceCartesian(FrameSpec):
        size = ColSpec(dtype=pl.String, choices=["S", "M", "L"])
        color = ColSpec(dtype=pl.String, choices=["Red", "Blue"], nullable=True)

    df = ChoiceCartesian.generate(n=1, method="cartesian", seed=42)
    # 3 sizes * (2 colors + null) = 9 combinations
    assert df.height == 9
    assert df.select("size", "color").unique().height == 9
