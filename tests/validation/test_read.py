"""`read()`: a data file in the terms a spec declares -- the front door to
validating a file someone hands you.

The extension picks the reader, Polars' options pass through, and one thing
is done in the spec's name: a declared type the file could not hold -- a
date, time, decimal or duration held as text, a fixed-size array held as a
list, a column of nothing but nulls -- is read back as declared, when every
value reads exactly. Nothing else is cast.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
from cases import COLUMN_CASES
from polspec import (
    ColSpec,
    FrameSpec,
    GenerationError,
    TableSpec,
    generate,
    inspect,
    read,
    sink_ndjson,
    validate,
)


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


# ---------------------------------------------------------------------------
# What polspec writes as JSON, it reads back
# ---------------------------------------------------------------------------


def _json_cannot_hold(column: ColSpec) -> str | None:
    """Why a column cannot survive NDJSON at all -- lost on the way out, so
    nothing on the way in can bring it back."""
    if "Binary" in str(column.dtype):
        return "JSON has no bytes: sink_ndjson refuses it by name"
    if column.nan_probability:
        return "JSON has no NaN: Polars writes it as null"
    return None


@pytest.mark.parametrize("case", sorted(COLUMN_CASES))
def test_what_sink_ndjson_writes_read_reads_back_exactly(tmp_path, case):
    """Before 0.15.1 a `Decimal` came back as `"1.25"`, a `Duration` as
    `"PT3600S"`, an `Array` as a list, a list of dates as text and a column
    of nulls as `Null` -- each failing validation of polspec's own output."""
    column = COLUMN_CASES[case]
    if (why := _json_cannot_hold(column)) is not None:
        pytest.skip(why)
    spec = TableSpec("Case", {"c": column})
    path = tmp_path / "case.ndjson"
    sink_ndjson(spec, path, 300, seed=11)
    frame = read(spec, path)
    assert not inspect(spec, frame)
    assert validate(spec, frame, cast=True).equals(generate(spec, 300, seed=11))


def test_sink_ndjson_refuses_bytes_by_name_instead_of_panicking(tmp_path):
    """Before 0.15.1 a `Binary` column panicked inside Polars' JSON writer,
    an exception `except Exception` does not catch."""
    spec = TableSpec("T", {"blob": ColSpec(pl.List(pl.Binary))})
    path = tmp_path / "t.ndjson"
    with pytest.raises(GenerationError, match=r"'blob' is List\(Binary\).*Parquet"):
        sink_ndjson(spec, path, 10, seed=1)
    assert not path.exists()


def test_categories_are_written_as_text_and_read_back(tmp_path):
    """A list of a few hundred categories panicked Polars' JSON writer."""
    spec = TableSpec(
        "T",
        {
            "tags": ColSpec(pl.List(pl.Enum(["x", "y"])), list_length=(1, 1)),
            "kind": ColSpec(pl.Categorical, choices=["a", "b"]),
        },
    )
    path = tmp_path / "t.ndjson"
    sink_ndjson(spec, path, 500, seed=1)
    assert path.read_text(encoding="utf-8").startswith('{"tags":["')
    frame = read(spec, path)
    assert not inspect(spec, frame)
    assert validate(spec, frame, cast=True).equals(generate(spec, 500, seed=1))


def _ndjson(tmp_path, values: dict[str, list]) -> Path:
    path = tmp_path / "data.ndjson"
    pl.DataFrame(values).write_ndjson(path)
    return path


def test_a_decimal_is_read_from_its_digits_and_never_rounded(tmp_path):
    spec = TableSpec("T", {"d": ColSpec(pl.Decimal(10, 2))})
    exact = read(spec, _ndjson(tmp_path, {"d": ["1.25", "-0.5", "+7", None]}))
    assert exact.schema["d"] == pl.Decimal(10, 2)
    assert [str(v) for v in exact["d"].drop_nulls()] == ["1.25", "-0.50", "7.00"]
    # A cast would round 1.255 to 1.26: a value that was never in the file.
    rounded = read(spec, _ndjson(tmp_path, {"d": ["1.25", "1.255"]}))
    assert rounded.schema["d"] == pl.String
    (finding,) = inspect(spec, rounded).findings
    assert finding.code == "dtype"


@pytest.mark.parametrize(
    ("unit", "text", "expected"),
    [
        ("us", "P0D", dt.timedelta(0)),
        ("us", "PT86400S", dt.timedelta(days=1)),
        ("us", "-PT0.5S", dt.timedelta(seconds=-0.5)),
        ("us", "-PT0.000001S", dt.timedelta(microseconds=-1)),
        ("ms", "PT1.5S", dt.timedelta(seconds=1.5)),
        ("ns", "PT0.000001S", dt.timedelta(microseconds=1)),
    ],
)
def test_a_duration_is_read_from_the_iso_text_polars_writes(
    tmp_path, unit, text, expected
):
    spec = TableSpec("T", {"d": ColSpec(pl.Duration(unit))})
    frame = read(spec, _ndjson(tmp_path, {"d": [text, None]}))
    assert frame.schema["d"] == pl.Duration(unit)
    assert frame["d"][0] == expected


def test_a_duration_at_the_edge_of_its_unit_reads_to_the_nanosecond(tmp_path):
    spec = TableSpec("T", {"d": ColSpec(pl.Duration("ns"))})
    edges = pl.Series([-(2**63), 2**63 - 1], dtype=pl.Int64)
    path = tmp_path / "edges.ndjson"
    pl.DataFrame({"d": edges.cast(pl.Duration("ns"))}).write_ndjson(path)
    assert read(spec, path)["d"].cast(pl.Int64).to_list() == edges.to_list()


@pytest.mark.parametrize(
    "text",
    [
        "PT0.0000001S",  # finer than the microseconds the column holds
        "PT1H",  # ISO 8601, but not the form Polars writes
        "3600",
        "PT99999999999999S",  # past what a Duration holds
    ],
)
def test_a_duration_that_does_not_read_exactly_stays_text(tmp_path, text):
    spec = TableSpec("T", {"d": ColSpec(pl.Duration("us"))})
    frame = read(spec, _ndjson(tmp_path, {"d": ["PT1S", text]}))
    assert frame.schema["d"] == pl.String


def test_an_array_is_read_back_from_a_list_only_when_every_list_fits(tmp_path):
    spec = TableSpec("T", {"a": ColSpec(pl.Array(pl.Int64, 2), nullable=True)})
    fits = read(spec, _ndjson(tmp_path, {"a": [[1, 2], None, [3, 4]]}))
    assert fits.schema["a"] == pl.Array(pl.Int64, 2)
    ragged = read(spec, _ndjson(tmp_path, {"a": [[1, 2], [3]]}))
    assert ragged.schema["a"] == pl.List(pl.Int64)
    assert inspect(spec, ragged).findings[0].code == "dtype"


def test_nested_text_is_read_inside_lists_and_structs(tmp_path):
    spec = TableSpec(
        "T",
        {
            "days": ColSpec(pl.List(pl.Date)),
            "money": ColSpec(
                pl.Struct({"amount": pl.Decimal(10, 2), "note": pl.String})
            ),
        },
    )
    path = _ndjson(
        tmp_path,
        {
            "days": [["2024-01-02", "2024-02-03"], []],
            "money": [
                {"amount": "1.50", "note": "2024-01-02"},
                {"amount": "-3", "note": "x"},
            ],
        },
    )
    frame = read(spec, path)
    assert frame.schema["days"] == pl.List(pl.Date)
    # The declared field is read; the String one keeps its date-shaped text.
    assert frame.schema["money"] == pl.Struct(
        {"amount": pl.Decimal(10, 2), "note": pl.String}
    )
    assert frame["money"][0] == {"amount": Decimal("1.50"), "note": "2024-01-02"}
    assert not inspect(spec, frame)


def test_a_column_of_nothing_but_nulls_is_read_as_declared(tmp_path):
    spec = TableSpec("T", {"n": ColSpec(pl.Int64, nullable=True)})
    frame = read(spec, _ndjson(tmp_path, {"n": [None, None]}))
    assert frame.schema["n"] == pl.Int64
