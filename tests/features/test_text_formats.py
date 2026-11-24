"""Dates, datetimes and times written as text: read with `parse_text`, held
to their `text_format`, written back in it; and why a column stays text."""

from __future__ import annotations

import datetime as dt

import polars as pl
import polspec
import pytest
from polspec import ColSpec, FrameSpec, SpecError, TableSpec

# One column per way of writing, and the format it is read with.
WRITTEN = [
    (["30 Sep 2031", "31 Dec 2049", "05 Mar 2027"], "%d %b %Y", pl.Date),
    (["1 Jan 2030", "15 Jan 2030", "3 Feb 2031"], "%-d %b %Y", pl.Date),
    (["30 September 2031", "1 January 2030"], "%-d %B %Y", pl.Date),
    (["30-Sep-2031", "05-Mar-2027"], "%d-%b-%Y", pl.Date),
    (["Sep 30, 2031", "Mar 05, 2027"], "%b %d, %Y", pl.Date),
    (["30/09/2031", "13/02/2030"], "%d/%m/%Y", pl.Date),
    (["09/30/2031", "02/13/2030"], "%m/%d/%Y", pl.Date),
    (["2031/09/30", "2030/02/13"], "%Y/%m/%d", pl.Date),
    (["30.09.2031", "13.02.2030"], "%d.%m.%Y", pl.Date),
    (["30 Sep 2031 14:05", "05 Mar 2027 09:30"], "%d %b %Y %H:%M", pl.Datetime),
    (["30/09/2031 14:05:59", "13/02/2030 09:30:00"], "%d/%m/%Y %H:%M:%S", pl.Datetime),
    (["14:05", "09:30"], "%H:%M", pl.Time),
    (["2:05 PM", "11:30 AM"], "%-I:%M %p", pl.Time),
]


def _frame(values: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"when": values * 10})


@pytest.mark.parametrize(
    ("values", "fmt", "dtype"), WRITTEN, ids=[f for _, f, _ in WRITTEN]
)
def test_text_is_read_with_its_format_and_validates_as_it(values, fmt, dtype):
    raw = _frame(values)
    spec = polspec.profile(raw, parse_text=True)
    column = spec["when"]
    assert isinstance(column.dtype, dtype)
    assert column.text_format == fmt
    polspec.validate(spec, raw, parse_text=True)


@pytest.mark.parametrize(
    ("values", "fmt", "dtype"), WRITTEN, ids=[f for _, f, _ in WRITTEN]
)
def test_a_csv_written_in_a_format_reads_back_and_a_fake_one_is_written_in_it(
    values, fmt, dtype, tmp_path
):
    source = tmp_path / "source.csv"
    _frame(values).write_csv(source)
    raw = pl.read_csv(source, infer_schema=False)
    spec = polspec.profile(raw, parse_text=True)
    read = polspec.read(spec, source)
    assert isinstance(read["when"].dtype, dtype)

    fake = tmp_path / "fake.csv"
    polspec.sink_csv(spec, fake, 200, seed=1)
    text = pl.read_csv(fake, infer_schema=False)["when"]
    again = text.str.strptime(dtype if dtype != pl.Datetime else pl.Datetime("us"), fmt)
    assert again.null_count() == 0
    assert (again.dt.strftime(fmt) == text).all()
    polspec.validate(spec, polspec.read(spec, fake))


def test_an_iso_date_needs_no_format_and_profiles_as_it_always_did():
    spec = polspec.profile(_frame(["2031-09-30", "2030-01-01"]), parse_text=True)
    assert spec["when"].dtype == pl.Date
    assert spec["when"].text_format is None


def test_a_date_and_month_that_could_be_either_way_round_stay_text_and_say_so():
    column = polspec.profile(_frame(["01/02/2031", "03/04/2030"]), parse_text=True)[
        "when"
    ]
    assert column.dtype != pl.Date
    assert "'%d/%m/%Y' and '%m/%d/%Y'" in column.description
    assert "text_format" in column.description


def test_a_day_past_twelve_settles_which_way_round():
    spec = polspec.profile(_frame(["01/02/2031", "13/04/2030"]), parse_text=True)
    assert spec["when"].text_format == "%d/%m/%Y"


@pytest.mark.parametrize(
    "values",
    [
        ["30 Feb 2031", "31 Dec 2049"],
        ["hello world", "not a date"],
        ["20240131", "12345678"],
    ],
    ids=["an impossible date", "free text", "digits"],
)
def test_what_is_no_date_is_not_read_as_one(values):
    column = polspec.profile(_frame(values), parse_text=True)["when"]
    assert column.dtype not in (pl.Date, pl.Datetime, pl.Time)
    assert column.text_format is None


def test_without_parse_text_a_date_says_its_format_and_takes_no_pattern():
    days = [dt.date(2030, 1, 1) + dt.timedelta(days=i) for i in range(400)]
    column = polspec.profile(
        pl.DataFrame({"when": [d.strftime("%d %b %Y") for d in days]})
    )["when"]
    assert column.dtype == pl.String
    assert column.pattern is None
    assert "parses as Date ('%d %b %Y')" in column.description


@pytest.mark.parametrize(
    ("values", "why"),
    [
        (
            [f"{i},{i:03d}.50" for i in range(1, 900)] + ["0.25", "75.5"],
            "899 value(s) do not parse as a number, such as ['1,001.50'",
        ),
        (["1500.25", "0.5"] * 100 + ["N/A"] * 5, "5 value(s) do not parse as a number"),
        (["30 Sep 2031"] * 50 + ["unknown"], "do not parse as a date ('%d %b %Y')"),
    ],
    ids=["thousands commas", "a stand-in", "a stray among dates"],
)
def test_a_column_that_nearly_parses_says_what_kept_it_as_text(values, why):
    column = polspec.profile(pl.DataFrame({"c": values}), parse_text=True)["c"]
    assert column.description.startswith("Stays text: ")
    assert why in column.description


@pytest.mark.parametrize(
    ("dtype", "fmt", "match"),
    [
        (pl.Time, "%d %b", "a time cannot be written with"),
        (pl.Date, "%H:%M", "a date cannot be written with"),
        (pl.String, "%d", "needs a Date, Datetime or Time column"),
        (pl.List(pl.Date), "%d/%m/%Y", "needs a Date, Datetime or Time column"),
        (pl.Date, "plain", "has no directive"),
        (pl.Date, "", "non-empty string"),
    ],
)
def test_a_text_format_its_column_cannot_be_written_in_is_refused(dtype, fmt, match):
    with pytest.raises(SpecError, match=match):
        ColSpec(dtype, text_format=fmt)


def test_a_declared_format_reads_a_file_and_refuses_text_written_another_way():
    class Bonds(FrameSpec):
        maturity = ColSpec(pl.Date, text_format="%d %b %Y")

    good = pl.DataFrame({"maturity": ["30 Sep 2031", "31 Dec 2049"]})
    assert Bonds.validate(good, parse_text=True)["maturity"].to_list() == [
        dt.date(2031, 9, 30),
        dt.date(2049, 12, 31),
    ]
    report = Bonds.inspect(
        pl.DataFrame({"maturity": ["30 Sep 2031", "2031-09-30"]}), parse_text=True
    )
    (finding,) = report
    assert finding.code == "dtype"
    assert "written as '%d %b %Y'" in finding.message
    assert finding.samples == ("2031-09-30",)


def test_a_format_survives_a_spec_file_and_python_source(tmp_path):
    spec = TableSpec("Bonds", {"maturity": ColSpec(pl.Date, text_format="%d %b %Y")})
    polspec.to_yaml(spec, tmp_path / "bonds.yaml")
    assert polspec.from_yaml(tmp_path / "bonds.yaml") == spec
    source = polspec.to_python(spec)
    assert "text_format='%d %b %Y'" in source


def test_a_changed_format_is_reported_by_diff():
    old = TableSpec("B", {"m": ColSpec(pl.Date, text_format="%d %b %Y")})
    new = TableSpec("B", {"m": ColSpec(pl.Date, text_format="%d/%m/%Y")})
    assert [f.columns for f in polspec.diff(old, new)] == [("m",)]


def test_in_memory_and_in_parquet_a_date_stays_a_date(tmp_path):
    spec = TableSpec("B", {"m": ColSpec(pl.Date, text_format="%d %b %Y")})
    assert polspec.generate(spec, 5, seed=1)["m"].dtype == pl.Date
    polspec.sink_parquet(spec, tmp_path / "m.parquet", 5, seed=1)
    assert pl.read_parquet(tmp_path / "m.parquet")["m"].dtype == pl.Date


def test_ndjson_writes_the_format_too(tmp_path):
    spec = TableSpec("B", {"m": ColSpec(pl.Date, text_format="%d %b %Y")})
    polspec.sink_ndjson(spec, tmp_path / "m.ndjson", 20, seed=1)
    text = pl.read_ndjson(tmp_path / "m.ndjson")["m"]
    assert text.dtype == pl.String
    assert text.str.to_date("%d %b %Y", strict=True).null_count() == 0


def test_text_a_declared_format_nearly_fits_is_read_or_reported_never_a_crash():
    """Polars 1.39 panicked reading `30 Sep 2031` with `%d %B %Y`. Whatever
    the Polars, validation returns: the value reads, or it is reported as
    text that does not."""
    spec = TableSpec("B", {"m": ColSpec(pl.Date, text_format="%d %B %Y")})
    frame = pl.DataFrame({"m": ["30 Sep 2031", "30 September 2031"]})
    report = polspec.inspect(spec, frame, parse_text=True)
    assert all(f.code == "dtype" and "30 Sep 2031" in f.samples for f in report)
