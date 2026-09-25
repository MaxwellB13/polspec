"""`read()`: a data file in the terms a spec declares -- the front door to
validating a file someone hands you.

The extension picks the reader, Polars' options pass through, and one thing
is done in the spec's name: a declared date or time that arrived as text is
parsed, when every value parses. Nothing else is cast.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from polspec import ColSpec, FrameSpec, read


class Events(FrameSpec):
    day = ColSpec(pl.Date, bounds=(dt.date(2024, 1, 1), dt.date(2025, 1, 1)))
    at = ColSpec(pl.Datetime("us"))
    zoned = ColSpec(pl.Datetime("ms", "Europe/London"))
    clock = ColSpec(pl.Time)
    ref = ColSpec(pl.String)
    status = ColSpec(pl.Enum(["NEW", "PAID"]))


ROW = (
    "2024-06-01,2024-01-01T00:00:00,2024-01-01T00:00:00+0000,12:00:00,2024-06-01,NEW\n"
)
HEADER = "day,at,zoned,clock,ref,status\n"


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_declared_date_or_time_is_read_as_one(tmp_path):
    path = _write(tmp_path, "events.csv", HEADER + ROW)
    df = Events.read(path)
    assert df.schema["day"] == pl.Date
    assert df.schema["at"] == pl.Datetime("us")
    assert df.schema["zoned"] == pl.Datetime("ms", "Europe/London")
    assert df.schema["clock"] == pl.Time
    # Only what the spec declares as a date: date-shaped text stays text, and
    # an Enum arrives as the String a CSV holds -- validation accepts both.
    assert df.schema["ref"] == pl.String
    assert df.schema["status"] == pl.String
    Events.validate(df)


def test_a_date_that_does_not_parse_stays_text_and_says_so(tmp_path):
    path = _write(
        tmp_path,
        "events.csv",
        HEADER
        + ROW
        + ROW.replace("2024-06-01,2024-01-01T", "not a date,2024-01-01T", 1),
    )
    df = Events.read(path)
    assert df.schema["day"] == pl.String
    (finding,) = Events.inspect(df).findings
    assert finding.code == "dtype" and finding.key == "day__dtype"


def test_a_parsed_date_is_checked_like_any_date(tmp_path):
    path = _write(
        tmp_path, "events.csv", HEADER + ROW.replace("2024-06-01", "2023-06-01", 1)
    )
    (finding,) = Events.inspect(Events.read(path)).findings
    assert finding.key == "day__bounds"


def test_reader_options_pass_through_and_a_tsv_is_tab_separated(tmp_path):
    semi = _write(tmp_path, "events.csv", (HEADER + ROW).replace(",", ";"))
    assert Events.read(semi, separator=";").schema["day"] == pl.Date
    tabbed = _write(tmp_path, "events.tsv", (HEADER + ROW).replace(",", "\t"))
    assert Events.read(tabbed).schema["day"] == pl.Date
    # A caller's own separator still wins over the TSV default.
    piped = _write(tmp_path, "piped.tsv", (HEADER + ROW).replace(",", "|"))
    assert Events.read(piped, separator="|").height == 1


@pytest.mark.parametrize(
    "suffix", [".parquet", ".ndjson", ".json", ".arrow", ".PARQUET"]
)
def test_every_format_the_cli_reads(tmp_path, suffix):
    frame = Events.generate(20, seed=1)
    path = tmp_path / f"events{suffix}"
    writer = {
        ".parquet": frame.write_parquet,
        ".ndjson": frame.write_ndjson,
        ".json": frame.write_json,
        ".arrow": frame.write_ipc,
    }[suffix.lower()]
    writer(path)
    Events.validate(Events.read(path))


def test_an_unknown_extension_names_the_ones_it_knows(tmp_path):
    with pytest.raises(ValueError, match=r"don't know how to read '\.xlsx'.*\.csv"):
        read(Events, tmp_path / "events.xlsx")


def test_the_function_and_the_facade_read_alike(tmp_path):
    path = _write(tmp_path, "events.csv", HEADER + ROW)
    assert read(Events.spec, path).equals(Events.read(path))
    assert read(Events, str(path)).equals(Events.read(path))


def test_read_then_validate_cast_is_the_typed_frame(tmp_path):
    path = _write(tmp_path, "events.csv", HEADER + ROW)
    typed = Events.validate(Events.read(path), cast=True)
    assert typed.schema["status"] == pl.Enum(["NEW", "PAID"])
    assert typed.schema["day"] == pl.Date
