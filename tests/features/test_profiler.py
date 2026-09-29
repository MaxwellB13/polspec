"""`FrameSpec.from_dataframe`: inferring a spec by profiling existing data."""

import warnings

import polars as pl
import pytest
from polspec import (
    Bound,
    ColSpec,
    FrameSpec,
    TableSpec,
    generate,
    inspect,
    profile_dataframe,
)
from polspec.formats import FORMATS
from polspec.profiler import ENUM_MIN_REPEATS, FORMAT_MIN_DISTINCT


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

    cols = Profiled.spec.columns
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
    cols = ProfiledWeighted.spec.columns

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
    assert isinstance(Spec1.spec.columns["low_card"].dtype, pl.Enum)
    assert Spec1.spec.columns["high_card"].dtype == pl.String

    # Using alias max_unique
    Spec2 = FrameSpec.from_dataframe(df, max_unique_enum=2)
    # low_card has 3 unique > 2, so it remains String
    assert Spec2.spec.columns["low_card"].dtype == pl.String


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
    assert SpecWithBounds.spec.columns["num"].bounds == Bound(10, 50)
    assert SpecWithBounds.spec.columns["txt"].string_length == Bound(1, 16)

    SpecNoBounds = FrameSpec.from_dataframe(
        df, calculate_bounds=False, max_unique_enum=0
    )
    assert SpecNoBounds.spec.columns["num"].bounds is None
    assert SpecNoBounds.spec.columns["txt"].string_length is None


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
    cols = Spec.spec.columns

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
    with pytest.raises(TypeError, match=r"Expected pl\.DataFrame"):
        FrameSpec.from_dataframe([{"a": 1}])  # type: ignore[arg-type]

    # Empty dataframe (0 rows)
    empty_df = pl.DataFrame({"a": [], "b": []}, schema={"a": pl.Int32, "b": pl.String})
    EmptySpec = FrameSpec.from_dataframe(empty_df)
    assert EmptySpec.schema() == empty_df.schema
    assert not EmptySpec.spec.columns["a"].nullable


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
    cols = Spec.spec.columns

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


# ---------------------------------------------------------------------------
# NaN and infinity: bounds from the finite values, and a warning
# ---------------------------------------------------------------------------


def _profiled_with_warnings(df: pl.DataFrame, **options) -> tuple[dict, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        columns = profile_dataframe(df, **options)
    return columns, [str(w.message) for w in caught]


def test_a_float_is_bounded_by_its_finite_values_and_its_others_are_named():
    """An infinity used to make profiling fail -- a bound must be finite.
    The bounds are the finite values', the NaN share is recorded, and each
    column holding an infinity says so."""
    inf = float("inf")
    df = pl.DataFrame(
        {
            "f": [1.5, inf, float("nan"), 3.0, -inf],
            "clean": [1.0, 2.0, 3.0, 4.0, 5.0],
            "point": [{"v": float("nan")}, {"v": 1.0}, {"v": 2.0}, None, {"v": 2.0}],
            "readings": [[1.0, inf], [2.0], [], None, [0.5]],
        }
    )
    columns, said = _profiled_with_warnings(df)
    assert columns["f"].bounds == Bound(1.5, 3.0)
    assert columns["point"].fields["v"].bounds == Bound(1.0, 2.0)
    assert columns["readings"].bounds == Bound(0.5, 2.0)
    assert columns["f"].nan_probability == pytest.approx(1 / 5)
    assert columns["point"].fields["v"].nan_probability == pytest.approx(1 / 4)
    assert columns["readings"].nan_probability == 0.0
    assert columns["clean"].nan_probability == 0.0
    assert said == [
        f"Column '{where}' holds {infinities} infinite value(s), which its "
        "bounds -- its finite values' -- leave out, so validating this data "
        "against the spec reports them. Open the bound on that side (None) if "
        "infinities belong there."
        for where, infinities in (("f", 2), ("readings", 1))
    ]


def test_a_float_with_no_finite_value_has_no_bounds_and_no_warning():
    """No bounds, so nothing is left outside them: validation accepts the
    infinity, the NaN share is recorded, and there is nothing to warn about."""
    df = pl.DataFrame({"f": [float("nan"), float("inf")]})
    columns, said = _profiled_with_warnings(df)
    assert columns["f"].bounds is None
    assert columns["f"].nan_probability == 0.5
    assert said == []


def test_no_bounds_asked_for_means_no_warning():
    df = pl.DataFrame({"f": [1.0, float("nan")]})
    columns, said = _profiled_with_warnings(df, calculate_bounds=False)
    assert columns["f"].bounds is None
    assert said == []


def test_from_dataframe_records_a_nan_share_and_warns_only_for_an_infinity():
    spec = FrameSpec.from_dataframe(pl.DataFrame({"f": [1.0, float("nan")]}))
    assert (spec.col("f").bounds, spec.col("f").nan_probability) == (
        Bound(1.0, 1.0),
        0.5,
    )
    with pytest.warns(UserWarning, match="Column 'f' holds 1 infinite"):
        FrameSpec.from_dataframe(pl.DataFrame({"f": [1.0, float("inf")]}))


# ---------------------------------------------------------------------------
# Names and dtypes at the edge
# ---------------------------------------------------------------------------


def test_a_column_with_the_empty_name_profiles_with_its_weights():
    df = pl.DataFrame({"": ["x", "y", "x", "x", "y", "x"], "b": [1, 2, 3, 4, 5, 6]})
    columns = profile_dataframe(df, weights=True)
    assert columns[""].dtype == pl.Enum(["x", "y"])
    assert columns[""].weights == pytest.approx((2 / 3, 1 / 3))


def test_a_null_column_profiles_as_nothing_but_nulls():
    """However many rows -- none included, where a rate cannot be measured."""
    null = pl.Series([None, None], dtype=pl.Null)
    for df in (pl.DataFrame({"z": null}), pl.DataFrame({"z": null}).head(0)):
        column = profile_dataframe(df)["z"]
        assert (column.dtype, column.nullable, column.null_probability) == (
            pl.Null,
            True,
            1.0,
        )


# ---------------------------------------------------------------------------
# An Enum needs repeats
# ---------------------------------------------------------------------------


def test_text_of_distinct_values_stays_text_however_few():
    """Forty names in forty rows used to become an `Enum` of those forty,
    and the spec rejected every other name. Values have to repeat -- each
    at least twice on average -- to be categories."""
    names = [f"person_{i}" for i in range(40)]
    column = profile_dataframe(pl.DataFrame({"name": names}))["name"]
    assert column.dtype == pl.String
    assert column.string_length == Bound(8, 9)


@pytest.mark.parametrize(
    "values, narrowed",
    [
        (["a", "b", "a", "b"], True),  # two values, each twice: exactly enough
        (["a", "b", "c", "a", "b"], False),  # three values in five rows
        (["a", "b", "c", "a", "b", "c"], True),
        (["only"], False),  # one row is no evidence of a category
        (["x"] * 50, True),
    ],
)
def test_an_enum_needs_each_value_twice_on_average(values, narrowed):
    column = profile_dataframe(pl.DataFrame({"c": values}))["c"]
    assert isinstance(column.dtype, pl.Enum) == narrowed
    assert ENUM_MIN_REPEATS == 2


def test_nulls_do_not_count_as_repeats():
    values = ["a", None, "b", None, None, None]
    assert profile_dataframe(pl.DataFrame({"c": values}))["c"].dtype == pl.String


def test_a_categorical_of_distinct_values_stays_categorical():
    series = pl.Series("c", ["a", "b", "c"], dtype=pl.Categorical)
    column = profile_dataframe(series.to_frame())["c"]
    assert column.dtype == pl.Categorical()


def test_a_lists_elements_need_repeats_too():
    df = pl.DataFrame({"tags": [["red", "blue"], ["green"], ["gold", "teal"]]})
    assert profile_dataframe(df)["tags"].dtype == pl.List(pl.String)
    repeated = pl.DataFrame({"tags": [["red", "blue"], ["red"], ["blue", "red"]]})
    assert profile_dataframe(repeated)["tags"].dtype == pl.List(
        pl.Enum(["blue", "red"])
    )


# ---------------------------------------------------------------------------
# A text column names its format
# ---------------------------------------------------------------------------


def _values_of(format_name: str, n: int) -> list[str]:
    spec = TableSpec("S", {"c": ColSpec(pl.String, format=format_name)})
    return generate(spec, n, seed=1)["c"].to_list()


@pytest.mark.parametrize("format_name", list(FORMATS))
def test_a_column_of_one_format_is_named_for_it(format_name):
    """Every format polspec can generate, recognised from its own values --
    before a finite one could narrow to an Enum of the codes seen."""
    values = _values_of(format_name, 200)
    column = profile_dataframe(pl.DataFrame({"c": values}))["c"]
    assert (column.dtype, column.format, column.extra_values) == (
        pl.String,
        format_name,
        None,
    )
    assert column.string_length is None  # a format fixes its own lengths
    spec = TableSpec("T", {"c": column})
    assert not inspect(spec, pl.DataFrame({"c": values}))


def test_too_few_distinct_values_are_not_evidence_of_a_format():
    few = _values_of("email", FORMAT_MIN_DISTINCT - 1) * 3
    column = profile_dataframe(pl.DataFrame({"c": few}))["c"]
    assert column.format is None
    assert isinstance(column.dtype, pl.Enum)


def test_a_hostname_needs_a_dot():
    """A single word passes the hostname check -- `localhost` is one -- so a
    column of first names would be hostnames; inference asks for more."""
    names = [f"name{letter}" for letter in "abcdefghijklmnopqrstuvwxyz"] * 2
    assert profile_dataframe(pl.DataFrame({"c": names}))["c"].format is None


def _near_miss(stand_ins: int, repeats: int) -> list[str]:
    countries = _values_of("iso_country", 2_000)[:1_000]
    extra = [f"stand-in {i}" for i in range(stand_ins) for _ in range(repeats)]
    return countries + extra


def test_a_near_miss_keeps_its_stand_ins_as_extra_values():
    """The case `extra_values` was built for: ISO countries, and a few rows
    of `UK (ISO)`."""
    values = _values_of("iso_country", 2_000)[:1_000] + ["UK (ISO)"] * 6
    column = profile_dataframe(pl.DataFrame({"c": values}))["c"]
    assert column.format == "iso_country"
    assert dict(column.extra_values) == {"UK (ISO)": pytest.approx(6 / 1_006)}
    assert not inspect(TableSpec("T", {"c": column}), pl.DataFrame({"c": values}))


@pytest.mark.parametrize(
    "stand_ins, repeats, named",
    [
        (5, 2, True),  # five stand-ins, each twice: 1% of rows
        (6, 1, False),  # more than five
        (1, 1, False),  # a stand-in seen once is a typo, not a convention
        (1, 20, False),  # 2% of rows: more than a near miss
    ],
)
def test_a_near_miss_is_a_few_repeated_stand_ins(stand_ins, repeats, named):
    values = _near_miss(stand_ins, repeats)
    column = profile_dataframe(pl.DataFrame({"c": values}))["c"]
    assert (column.format == "iso_country") is named


def test_a_replaced_column_is_named_a_format_but_keeps_no_stand_ins():
    """`replace=` keeps a column's values out of the spec: the format is a
    shape, not values, so it stays; stand-ins are values of the source."""
    exact = _values_of("email", 100)
    column = profile_dataframe(pl.DataFrame({"c": exact}), replace=["c"])["c"]
    assert column.format == "email"
    near = _values_of("iso_country", 2_000)[:1_000] + ["UK (ISO)"] * 6
    column = profile_dataframe(pl.DataFrame({"c": near}), replace=["c"])["c"]
    assert column.format is None and column.dtype == pl.String


def test_formats_false_profiles_text_as_before():
    values = _values_of("email", 100)
    column = profile_dataframe(pl.DataFrame({"c": values}), formats=False)["c"]
    assert column.format is None and column.string_length is not None


def test_a_categorical_column_is_not_named_a_format():
    """A format describes a String; a Categorical keeps its dtype."""
    values = pl.Series(_values_of("email", 100), dtype=pl.Categorical)
    column = profile_dataframe(values.to_frame("c"))["c"]
    assert column.format is None


def test_a_lists_elements_and_a_structs_fields_name_their_format():
    emails = _values_of("email", 90)
    frame = pl.DataFrame(
        {
            "l": [emails[i : i + 3] for i in range(0, 90, 3)],
            "s": [{"host": h} for h in _values_of("hostname", 30)],
        }
    )
    columns = profile_dataframe(frame)
    assert columns["l"].format == "email"
    assert columns["s"].fields["host"].format == "hostname"


def test_a_unique_key_of_uuids_is_both():
    ids = _values_of("uuid4", 150)
    column = profile_dataframe(pl.DataFrame({"id": ids}), detect_unique=True)["id"]
    assert (column.format, column.unique) == ("uuid4", True)


def test_stand_ins_are_listed_most_frequent_first_every_run():
    """`value_counts` keeps no order; the spec file written from the same
    data must not change between runs."""
    countries = _values_of("iso_country", 2_000)[:1_000]
    values = countries + ["N/A"] * 2 + ["UK (ISO)"] * 4 + ["TBD"] * 2
    orders = {
        tuple(profile_dataframe(pl.DataFrame({"c": values}))["c"].extra_values)
        for _ in range(20)
    }
    assert orders == {("UK (ISO)", "N/A", "TBD")}
