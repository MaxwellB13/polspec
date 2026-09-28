"""`validate()` and the frame's shape: extra and missing columns, dtypes,
strictness and casting, lazy and streaming input, ordering, empty frames.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from polspec import (
    Bound,
    Check,
    ColRule,
    ColSpec,
    FrameSpec,
    TableSpec,
    ValidationError,
    col,
    inspect,
    validate,
)


class ProduceInventory(FrameSpec):
    id = ColSpec(dtype=pl.Int64, bounds=Bound(1, 10_000), nullable=False)
    category = ColSpec(
        dtype=pl.Enum(["fruit", "vegetable", "meat"]),
        nullable=False,
    )
    quantity = ColSpec(
        dtype=pl.Int32,
        bounds=Bound(0, 500),
        nullable=False,
    )
    price = ColSpec(
        dtype=pl.Float64,
        bounds=Bound(0.01, 100.0),
        nullable=True,
    )
    code = ColSpec(
        dtype=pl.String,
        string_length=Bound(3, 5),
        nullable=True,
    )


def test_validation_success():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "category": ["fruit", "vegetable", "meat"],
            "quantity": [10, 50, 100],
            "price": [1.99, 2.50, None],
            "code": ["APP", "CAR", "BEEF"],
        }
    )

    result = ProduceInventory.validate(df)
    assert isinstance(result, pl.DataFrame)
    assert result.height == 3


def test_validation_collects_all_column_errors():
    # Multiple errors across various columns simultaneously:
    # 1. 'id' has a null value (non-nullable)
    # 2. 'category' has invalid value 'chicken'
    # 3. 'quantity' has an out-of-bounds value (999 > 500)
    # 4. 'price' has a negative out-of-bounds value (-10.0 < 0.01)
    # 5. 'code' has string length violation ("TOOLONG" > 5 chars)
    df = pl.DataFrame(
        {
            "id": [1, None, 3, 4],
            "category": ["fruit", "vegetable", "meat", "chicken"],
            "quantity": [10, 20, 999, 40],
            "price": [1.0, 2.0, 3.0, -10.0],
            "code": ["AAA", "BBB", "CCC", "TOOLONG"],
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        ProduceInventory.validate(df)

    err = exc_info.value
    assert len(err.errors) == 5
    err_str = str(err)
    assert "non-nullable column contains 1 null value(s)" in err_str
    assert "invalid value(s) not in allowed choices/categories" in err_str
    assert "quantity" in err_str and "out of bounds" in err_str
    assert "price" in err_str and "out of bounds" in err_str
    assert "code" in err_str and "string length outside" in err_str


def test_validation_extra_cols():
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "category": ["fruit", "vegetable"],
            "quantity": [10, 20],
            "price": [1.0, 2.0],
            "code": ["APP", "CAR"],
            "extra_1": ["x", "y"],
            "extra_2": [100, 200],
        }
    )

    # 1. extra_cols="raise" (default)
    with pytest.raises(ValidationError) as exc_info:
        ProduceInventory.validate(df, extra_cols="raise")
    assert "Extra columns found that are not in schema: ['extra_1', 'extra_2']" in str(
        exc_info.value
    )

    # 2. extra_cols="drop"
    res_drop = ProduceInventory.validate(df, extra_cols="drop")
    assert res_drop.columns == ["id", "category", "quantity", "price", "code"]
    assert "extra_1" not in res_drop.columns
    assert "extra_2" not in res_drop.columns

    # 3. extra_cols="allow"
    res_allow = ProduceInventory.validate(df, extra_cols="allow")
    assert "extra_1" in res_allow.columns
    assert "extra_2" in res_allow.columns


def test_validation_missing_cols():
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "category": ["fruit", "vegetable"],
            "quantity": [10, 20],
        }
    )

    # 1. missing_cols="raise" (default)
    with pytest.raises(ValidationError) as exc_info:
        ProduceInventory.validate(df, missing_cols="raise")
    assert "Missing required columns in DataFrame: ['price', 'code']" in str(
        exc_info.value
    )

    # 2. missing_cols="add"
    res_add = ProduceInventory.validate(df, missing_cols="add")
    assert "price" in res_add.columns
    assert "code" in res_add.columns
    assert res_add["price"].null_count() == 2
    assert res_add["code"].null_count() == 2

    # 3. missing_cols="allow"
    res_allow = ProduceInventory.validate(df, missing_cols="allow")
    assert res_allow.columns == ["id", "category", "quantity"]


def test_validation_dtype_mismatch():
    df = pl.DataFrame(
        {
            "id": ["one", "two"],  # Expected Int64
            "category": ["fruit", "vegetable"],
            "quantity": [10, 20],
            "price": [1.0, 2.0],
            "code": ["APP", "CAR"],
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        ProduceInventory.validate(df)

    assert "Column 'id': expected dtype Int64, got String" in str(exc_info.value)


def test_a_wrong_dtype_on_a_column_with_choices_is_reported_not_raised():
    """A column with a closed domain must not turn a dtype error into a crash.

    Comparing values against `choices` of another type is not a noisy finding
    -- Polars refuses to compile the `is_in` at all -- so the domain check has
    to sit behind the dtype check, not in front of it. `inspect()` promises
    never to raise for a bad frame, and this is the frame most likely to
    arrive: a column read back from CSV or JSON as the wrong type.
    """

    class Statuses(FrameSpec):
        status = ColSpec(pl.String, choices=["NEW", "PAID"])

    report = Statuses.inspect(pl.DataFrame({"status": [1, 2]}))

    assert [f.code for f in report.findings] == ["dtype"]
    assert "expected dtype String, got Int64" in report.findings[0].message

    with pytest.raises(ValidationError, match="expected dtype String"):
        Statuses.validate(pl.DataFrame({"status": [1, 2]}))


def test_validation_strict_dtypes_and_cast():
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "category": ["fruit", "vegetable"],
            "quantity": [10, 20],  # Int64 by default in Polars, expected Int32
            "price": [1.0, 2.0],
            "code": ["APP", "CAR"],
        }
    )

    # 1. By default, compatible integers are accepted
    res = ProduceInventory.validate(df)
    assert res.height == 2

    # 2. strict_dtypes=True rejects Int64 when Int32 is expected
    with pytest.raises(ValidationError) as exc_info:
        ProduceInventory.validate(df, strict_dtypes=True)
    assert "Column 'quantity': expected dtype Int32, got Int64" in str(exc_info.value)

    # 3. cast=True converts types to expected schema
    res_cast = ProduceInventory.validate(df, cast=True)
    assert res_cast.schema["quantity"] == pl.Int32
    assert res_cast.schema["category"] == pl.Enum(["fruit", "vegetable", "meat"])


def test_validation_lazyframe_and_streaming():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "category": ["fruit", "vegetable", "meat"],
            "quantity": [10, 20, 30],
            "price": [1.0, 2.0, 3.0],
            "code": ["AAA", "BBB", "CCC"],
        }
    )
    lf = df.lazy()

    # Success case returning LazyFrame
    res_lf = ProduceInventory.validate(lf)
    assert isinstance(res_lf, pl.LazyFrame)
    collected = res_lf.collect()
    assert collected.height == 3

    # Streaming success
    res_stream = ProduceInventory.validate(lf, streaming=True)
    assert isinstance(res_stream, pl.LazyFrame)
    assert res_stream.collect().height == 3

    # Failure case with LazyFrame
    invalid_lf = pl.DataFrame(
        {
            "id": [1, 2],
            "category": ["fruit", "alien_food"],
            "quantity": [10, 20],
            "price": [1.0, 2.0],
            "code": ["AAA", "BBB"],
        }
    ).lazy()

    with pytest.raises(ValidationError) as exc_info:
        ProduceInventory.validate(invalid_lf, streaming=True)

    assert "alien_food" in str(exc_info.value)


def test_validation_empty_dataframe():
    df_empty = pl.DataFrame(
        schema={
            "id": pl.Int64,
            "category": pl.Enum(["fruit", "vegetable", "meat"]),
            "quantity": pl.Int32,
            "price": pl.Float64,
            "code": pl.String,
        }
    )
    res = ProduceInventory.validate(df_empty)
    assert res.height == 0

    # Lazy empty
    res_lazy = ProduceInventory.validate(df_empty.lazy())
    assert res_lazy.collect().height == 0


def test_validation_invalid_options():
    df = pl.DataFrame(
        {
            "id": [1],
            "category": ["fruit"],
            "quantity": [10],
            "price": [1.0],
            "code": ["APP"],
        }
    )

    with pytest.raises(ValueError, match="extra_cols must be one of"):
        ProduceInventory.validate(df, extra_cols="invalid_option")  # type: ignore

    with pytest.raises(ValueError, match="missing_cols must be one of"):
        ProduceInventory.validate(df, missing_cols="invalid_option")  # type: ignore


def test_validation_column_ordering():
    class OrderedSpec(FrameSpec):
        first = ColSpec(dtype=pl.Int64)
        second = ColSpec(dtype=pl.String)
        third = ColSpec(dtype=pl.Float64)

    # Input DataFrame has columns in scrambled order and an extra column
    df = pl.DataFrame(
        {
            "third": [3.0],
            "extra": ["foo"],
            "first": [1],
            "second": ["bar"],
        }
    )

    res = OrderedSpec.validate(df, extra_cols="allow")
    assert res.columns == ["first", "second", "third", "extra"]


# ---------------------------------------------------------------------------
# A temporal stands in for its own kind only
# ---------------------------------------------------------------------------

_DATE = dt.date(2020, 6, 1)
_TEMPORAL_DATA = {
    "Date": pl.Series([_DATE]),
    "Datetime": pl.Series([dt.datetime(2020, 6, 1, 12)]),
    "Datetime(ms)": pl.Series([dt.datetime(2020, 6, 1, 12)]).cast(pl.Datetime("ms")),
    "Datetime(UTC)": pl.Series([dt.datetime(2020, 6, 1, 12)]).dt.replace_time_zone(
        "UTC"
    ),
    "Datetime(Asia/Tokyo)": pl.Series(
        [dt.datetime(2020, 6, 1, 12)]
    ).dt.replace_time_zone("Asia/Tokyo"),
    "Time": pl.Series([dt.time(12)]),
    "Duration": pl.Series([dt.timedelta(hours=12)]),
    "Duration(ms)": pl.Series([dt.timedelta(hours=12)]).cast(pl.Duration("ms")),
}

# (declared dtype, bounds, the data kinds that stand in for it)
_TEMPORAL_DECLARED = {
    "Date": (
        pl.Date,
        (dt.date(2020, 1, 1), dt.date(2021, 1, 1)),
        {"Date", "Datetime", "Datetime(ms)", "Datetime(UTC)", "Datetime(Asia/Tokyo)"},
    ),
    "Datetime": (
        pl.Datetime("us"),
        (dt.datetime(2020, 1, 1), dt.datetime(2021, 1, 1)),
        {"Date", "Datetime", "Datetime(ms)"},
    ),
    "Datetime(UTC)": (
        pl.Datetime("us", "UTC"),
        (
            dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2021, 1, 1, tzinfo=dt.UTC),
        ),
        {"Date", "Datetime(UTC)", "Datetime(Asia/Tokyo)"},
    ),
    "Time": (pl.Time, (dt.time(1), dt.time(23)), {"Time"}),
    "Duration": (
        pl.Duration("us"),
        (dt.timedelta(0), dt.timedelta(days=1)),
        {"Duration", "Duration(ms)"},
    ),
}


@pytest.mark.parametrize("with_bounds", [True, False], ids=["bounded", "unbounded"])
@pytest.mark.parametrize("declared", list(_TEMPORAL_DECLARED))
def test_a_temporal_stands_in_only_for_its_own_kind(declared, with_bounds):
    """A date and a datetime stand in for each other -- text hands one back
    as the other -- and a time of day or a duration for nothing else. Before
    0.13.1 any temporal stood in for any other: a `Time` on a bounded `Date`
    made `inspect()` raise Polars' cast error, and on an unbounded one
    passed. Naive and zoned datetimes are an instant and a wall-clock
    reading, so neither stands in for the other; two zones do.
    """
    dtype, bounds, accepted = _TEMPORAL_DECLARED[declared]
    spec = TableSpec("T", {"t": ColSpec(dtype, bounds=bounds if with_bounds else None)})
    for kind, values in _TEMPORAL_DATA.items():
        report = inspect(spec, pl.DataFrame({"t": values}))
        codes = [f.code for f in report.findings]
        if kind in accepted:
            assert codes == [], (declared, kind, report.findings)
        else:
            assert codes == ["dtype"], (declared, kind, report.findings)


def test_a_zoned_datetime_casts_to_the_declared_zone():
    """Two zones stand in for each other because they hold the same
    instants; `cast=True` then shows them in the declared one."""
    spec = TableSpec("T", {"t": ColSpec(pl.Datetime("us", "UTC"))})
    tokyo = pl.DataFrame({"t": _TEMPORAL_DATA["Datetime(Asia/Tokyo)"]})
    cast = validate(spec, tokyo, cast=True)
    assert cast.schema["t"] == pl.Datetime("us", "UTC")
    assert cast["t"].dt.epoch("us").equals(tokyo["t"].dt.epoch("us"))


def test_strict_dtypes_still_tells_two_zones_apart():
    spec = TableSpec("T", {"t": ColSpec(pl.Datetime("us", "UTC"))})
    tokyo = pl.DataFrame({"t": _TEMPORAL_DATA["Datetime(Asia/Tokyo)"]})
    report = inspect(spec, tokyo, strict_dtypes=True)
    assert [f.code for f in report.findings] == ["dtype"]


# ---------------------------------------------------------------------------
# Claims over a column that arrived as another dtype. Each was found by
# `test_inspect_answers_for_data_of_any_dtype` making `inspect()` raise.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", [["x", "y"], [1, 0]], ids=["String", "Int64"])
def test_a_rule_reading_a_column_of_the_wrong_dtype_is_skipped(flag):
    """`flag` is a `dtype` finding; a rule keyed on it cannot be evaluated,
    and used to be compiled anyway -- Polars refused `&` on text."""
    spec = TableSpec(
        "T",
        {
            "flag": ColSpec(pl.Boolean),
            "v": ColSpec(
                pl.Int64,
                choices=[0, 1],
                rules=[ColRule(when=col("flag"), choices=[0])],
            ),
        },
        checks=[Check(col("flag") | (col("v") == 1), name="flag_or_one")],
    )
    report = inspect(spec, pl.DataFrame({"flag": flag, "v": [1, 1]}))
    assert [f.key for f in report.findings] == ["flag__dtype"]


def test_a_check_on_a_date_holds_for_the_datetime_standing_in_for_it():
    """A `Datetime` stands in for a declared `Date`; `is_in` refused a list
    of dates against it."""
    spec = TableSpec(
        "T",
        {"d": ColSpec(pl.Date)},
        checks=[Check(col("d").is_in([_DATE]), name="the_day")],
    )
    on_the_day = pl.DataFrame({"d": [dt.datetime(2020, 6, 1)]}).cast(
        {"d": pl.Datetime("ms")}
    )
    assert not inspect(spec, on_the_day)
    other_day = on_the_day.with_columns(pl.col("d") + dt.timedelta(days=1))
    assert [f.key for f in inspect(spec, other_day).findings] == ["check:the_day"]


def test_choices_are_asked_as_the_column_would_be_cast():
    """A `Datetime` column's choices, against a `Date` standing in for it:
    `is_in` refused datetimes against dates. Membership is asked of the
    column cast to what it declares -- what `cast=True` hands back -- so a
    clean report stays clean once cast."""
    midnight = dt.datetime(2020, 6, 1)
    spec = TableSpec(
        "T",
        {
            "t": ColSpec(pl.Datetime("us"), choices=[midnight]),
            "r": ColSpec(
                pl.Datetime("us"),
                rules=[ColRule(when=col("t") == midnight, choices=[midnight])],
            ),
        },
    )
    as_dates = pl.DataFrame({"t": [_DATE], "r": [_DATE]})
    assert not inspect(spec, as_dates)
    assert not inspect(spec, validate(spec, as_dates, cast=True))
    wrong_day = pl.DataFrame({"t": [dt.date(2020, 6, 2)], "r": [_DATE]})
    assert [f.key for f in inspect(spec, wrong_day).findings] == ["t__choices"]


def test_temporal_bounds_are_measured_as_the_column_would_be_cast():
    """A bound a microsecond past midnight, on a `Datetime` column arriving
    as a `Date`: measured as a date, the bound rounded to the day and the
    report was clean -- then `cast=True` made midnight, outside it. Found by
    the arrival property's re-inspection of the cast frame."""
    edge = dt.datetime(2000, 1, 1, 0, 0, 0, 1)
    spec = TableSpec("T", {"t": ColSpec(pl.Datetime("us"), bounds=(edge, edge))})
    as_dates = pl.DataFrame({"t": [dt.date(2000, 1, 1)]})
    assert [f.key for f in inspect(spec, as_dates).findings] == ["t__bounds"]

    midnight = TableSpec(
        "T", {"t": ColSpec(pl.Datetime("us"), bounds=(dt.datetime(2000, 1, 1), edge))}
    )
    assert not inspect(midnight, as_dates)
    assert not inspect(midnight, validate(midnight, as_dates, cast=True))
