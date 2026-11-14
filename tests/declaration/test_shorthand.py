"""Writing less: a bare dtype as a column, and dates and times in bounds
written as ISO text -- in a class body, a `TableSpec` and a spec file."""

import datetime as dt

import polars as pl
import polspec
import pytest
from polspec import ColSpec, FrameSpec, SpecError, TableSpec

# ---------------------------------------------------------------------------
# A bare dtype is a column
# ---------------------------------------------------------------------------


def test_a_bare_dtype_in_a_class_body_is_a_column_of_it():
    class Loans(FrameSpec):
        loan_id = ColSpec(pl.Int64, unique=True)
        product = pl.Enum(["mortgage", "card"])
        opened = pl.Date
        lgd = pl.Float64()

    assert Loans.schema() == pl.Schema(
        {
            "loan_id": pl.Int64,
            "product": pl.Enum(["mortgage", "card"]),
            "opened": pl.Date,
            "lgd": pl.Float64,
        }
    )
    assert Loans.spec["opened"] == ColSpec(pl.Date)


def test_a_bare_dtype_overrides_an_inherited_column():
    class Base(FrameSpec):
        opened = ColSpec(pl.Date, nullable=True)

    class Precise(Base):
        opened = pl.Datetime("us")

    assert Precise.spec["opened"] == ColSpec(pl.Datetime("us"))


def test_a_bare_dtype_in_columns_and_in_a_tablespec():
    class Odd(FrameSpec):
        __columns__ = {"Unit Price": pl.Float64}

    assert Odd.schema() == pl.Schema({"Unit Price": pl.Float64})
    spec = TableSpec("T", {"a": pl.Int64, "b": pl.List(pl.String)})
    assert spec["a"] == ColSpec(pl.Int64)
    assert spec["b"] == ColSpec(pl.List(pl.String))


def test_a_bare_dtype_generates_and_validates_like_its_colspec():
    short = TableSpec("T", {"a": pl.Int32, "b": pl.String})
    long = TableSpec("T", {"a": ColSpec(pl.Int32), "b": ColSpec(pl.String)})
    assert short == long
    frame = polspec.generate(short, 100, seed=1)
    assert frame.equals(polspec.generate(long, 100, seed=1))
    polspec.validate(short, frame)


def test_a_spec_file_reads_a_column_written_as_its_dtype_name():
    spec = polspec.from_dict({"name": "T", "columns": {"a": "Int64", "b": "Date"}})
    assert spec == TableSpec("T", {"a": pl.Int64, "b": pl.Date})


def test_a_spec_file_is_still_written_in_the_long_form():
    """So 1.0 and 1.1, which read no short form, read what 1.2 writes."""
    text = polspec.to_yaml(TableSpec("T", {"a": pl.Int64}))
    assert "a:\n    dtype: Int64" in text


# ---------------------------------------------------------------------------
# ISO text in temporal bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "text", "value"),
    [
        (
            pl.Date,
            ("2020-01-01", "2021-06-30"),
            (dt.date(2020, 1, 1), dt.date(2021, 6, 30)),
        ),
        (
            pl.Datetime("us"),
            ("2020-01-01", "2020-02-01T12:30:00"),
            (dt.datetime(2020, 1, 1), dt.datetime(2020, 2, 1, 12, 30)),
        ),
        (pl.Time, ("09:00", "17:30:00.5"), (dt.time(9), dt.time(17, 30, 0, 500_000))),
        (pl.List(pl.Date), ("2020-01-01", None), (dt.date(2020, 1, 1), None)),
    ],
    ids=["date", "datetime", "time", "list of dates"],
)
def test_iso_text_in_bounds_is_read_as_the_value_it_spells(dtype, text, value):
    bounds = ColSpec(dtype, bounds=text).bounds
    assert (bounds.min, bounds.max) == value


def test_text_and_typed_endpoints_mix():
    bounds = ColSpec(pl.Date, bounds=("2020-01-01", dt.date(2021, 1, 1))).bounds
    assert (bounds.min, bounds.max) == (dt.date(2020, 1, 1), dt.date(2021, 1, 1))


def test_text_bounds_generate_the_same_frame_typed_ones_do():
    text = TableSpec("T", {"d": ColSpec(pl.Date, bounds=("2020-01-01", "2021-01-01"))})
    typed = TableSpec(
        "T", {"d": ColSpec(pl.Date, bounds=(dt.date(2020, 1, 1), dt.date(2021, 1, 1)))}
    )
    assert text == typed
    assert polspec.generate(text, 500, seed=3).equals(
        polspec.generate(typed, 500, seed=3)
    )


def test_text_bounds_validate_on_every_polars():
    """Kept as text, they were compared by casting text to a date, which
    Polars 2 refuses."""
    spec = TableSpec("T", {"d": ColSpec(pl.Date, bounds=("2020-01-01", "2021-01-01"))})
    report = polspec.inspect(spec, pl.DataFrame({"d": [dt.date(2022, 1, 1)]}))
    assert [f.code for f in report.findings] == ["bounds"]


def test_a_spec_file_reads_quoted_dates_as_dates():
    spec = polspec.from_dict(
        {
            "name": "T",
            "columns": {"d": {"dtype": "Date", "bounds": ["2020-01-01", None]}},
        }
    )
    assert spec["d"].bounds.min == dt.date(2020, 1, 1)
    assert "- 2020-01-01\n" in polspec.to_yaml(spec)


@pytest.mark.parametrize(
    ("dtype", "text", "message"),
    [
        (pl.Date, "31/01/2024", "not an ISO date"),
        (pl.Time, "9am", "not an ISO time"),
        (pl.Int64, "1", "is text; the bounds of a Int64 column are its own values"),
        (pl.Duration("us"), "1d", "Write a timedelta"),
    ],
)
def test_text_that_is_not_an_iso_value_of_the_column_is_refused(dtype, text, message):
    with pytest.raises(SpecError, match=message):
        ColSpec(dtype, bounds=(text, None))


def test_a_datetime_on_a_date_column_is_refused_by_name():
    with pytest.raises(
        SpecError, match=r"is a datetime, and a Date column's bounds are dates"
    ):
        ColSpec(pl.Date, bounds=(dt.datetime(2020, 1, 1), None))


def test_a_decimal_still_takes_its_bounds_as_text():
    bounds = ColSpec(pl.Decimal(10, 2), bounds=("1.25", "9")).bounds
    assert str(bounds.min) == "1.25"
