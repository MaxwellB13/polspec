"""`parse_text`: profiling a frame whose types were switched off, where every
column arrived as text -- and the note a text column gets without it."""

import datetime as dt
import random

import polars as pl
import polspec
import pytest
from cases import COLUMN_CASES
from polspec import FrameSpec
from polspec.cli import main

NOTE = "Every value is text that parses as "


def _profiled(df: pl.DataFrame, **options) -> polspec.TableSpec:
    return polspec.profile(df, **options)


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        (["true", "FALSE", " True ", "false"], pl.Boolean),
        (["1", "-20", "+3", "40"], pl.Int64),
        (["0.45", "0", "1e-3", "-2.5"], pl.Float64),
        (["2024-01-31", "1999-12-01", "2024-02-29", "2000-01-01"], pl.Date),
        (
            ["2024-01-31T09:30:00", "2024-02-01T10:00:00", "2024-02-02T11:15:30"],
            pl.Datetime,
        ),
    ],
    ids=["boolean", "int64", "float64", "date", "datetime"],
)
def test_each_type_text_can_hold_is_read_as_it(values, dtype):
    column = _profiled(pl.DataFrame({"c": values}), parse_text=True)["c"]
    assert column.dtype.base_type() == dtype
    assert column.description is None


def test_integers_come_before_floats_and_a_decimal_makes_them_floats():
    ints = _profiled(pl.DataFrame({"c": ["1", "2", "3"]}), parse_text=True)["c"]
    mixed = _profiled(pl.DataFrame({"c": ["1", "2", "3.5"]}), parse_text=True)["c"]
    assert (ints.dtype, mixed.dtype) == (pl.Int64, pl.Float64)


@pytest.mark.parametrize(
    "values",
    [
        ["007", "123", "456"],
        ["-01", "5", "6"],
        ["00.5", "1.5", "2.5"],
        [str(10**20 + i) for i in range(3)],
        ["1", "2", "two"],
        ["2024-01-31", "31/01/2024", "2024-02-01"],
    ],
    ids=[
        "leading zero",
        "signed leading zero",
        "float leading zero",
        "past int64",
        "a word",
        "not iso",
    ],
)
def test_text_that_is_a_code_or_not_all_one_type_stays_text(values):
    """A leading zero, or an integer no Int64 holds, makes a code: parsing
    would lose the zero, or round it into a float."""
    column = _profiled(pl.DataFrame({"c": values}), parse_text=True)["c"]
    assert column.dtype == pl.String
    assert column.description is None


def test_empty_text_is_a_null():
    values = ["0.45", "", "   ", None, "0.2"]
    column = _profiled(pl.DataFrame({"c": values}), parse_text=True)["c"]
    assert column.dtype == pl.Float64
    assert column.nullable
    assert column.null_probability == pytest.approx(3 / 5)


def test_a_column_of_nothing_but_blanks_stays_text():
    column = _profiled(pl.DataFrame({"c": ["", " ", ""]}), parse_text=True)["c"]
    assert column.dtype == pl.String


def test_without_parse_text_the_column_stays_text_and_says_what_it_would_be():
    spec = _profiled(pl.DataFrame({"n": ["1", "2"], "s": ["a", "b"]}))
    assert spec["n"].dtype == pl.String
    assert spec["n"].description == (
        f"{NOTE}Int64; profile with parse_text=True (--parse-text) to declare "
        "it as one."
    )
    assert spec["s"].description is None


def test_the_note_survives_a_spec_file(tmp_path):
    spec = _profiled(pl.DataFrame({"n": ["1", "2"]}))
    polspec.to_yaml(spec, tmp_path / "spec.yaml")
    again = polspec.from_yaml(tmp_path / "spec.yaml")
    assert again["n"].description.startswith(NOTE)


def test_a_replaced_column_is_neither_parsed_nor_noted():
    """Its values are not carried over, and a number's extremes would be."""
    df = pl.DataFrame({"salary": ["50000", "61000", "72000"]})
    column = _profiled(df, parse_text=True, replace=["salary"])["salary"]
    assert column.dtype == pl.String
    assert column.description is None
    assert _profiled(df, replace=["salary"])["salary"].description is None


def test_order_is_read_from_the_parsed_values():
    """Text sorts `"10"` before `"9"`; the numbers it holds are in order."""
    df = pl.DataFrame({"id": [str(i) for i in range(1, 41)]})
    assert _profiled(df, parse_text=True).order_by == ("id",)
    assert _profiled(df).order_by == ()


def test_synthesize_generates_the_parsed_types_and_the_source_types_elsewhere():
    df = pl.DataFrame(
        {
            "LGD": [f"0.{i:02d}" for i in range(1, 61)],
            "tier": ["gold", "silver", "bronze"] * 20,
        }
    )
    fake = polspec.synthesize(df, 50, seed=1, parse_text=True)
    assert fake.schema == pl.Schema({"LGD": pl.Float64, "tier": pl.String})


def test_from_dataframe_takes_it_too():
    schema = FrameSpec.from_dataframe(
        pl.DataFrame({"opened": ["2024-01-31", "2024-02-01"]}), parse_text=True
    ).schema()
    assert schema == pl.Schema({"opened": pl.Date})


def test_a_csv_read_as_text_profiles_as_its_numbers(tmp_path):
    """The case that asked for this: a CSV too unclean to type on reading,
    read with `infer_schema=False`, whose LGD column is decimals, zeros,
    blanks and nulls -- profiled, before, as hostnames."""
    rng = random.Random(1)
    lines = ["loan_id,LGD,opened"]
    for i in range(1, 301):
        lgd = rng.choice([f"{rng.random():.4f}", "0", ""])
        lines.append(f"{i},{lgd},2024-01-{1 + i % 28:02d}")
    path = tmp_path / "loans.csv"
    path.write_text("\n".join(lines), encoding="utf-8")

    raw = pl.read_csv(path, infer_schema=False)
    spec = polspec.profile(raw, parse_text=True)
    assert spec.schema() == pl.Schema(
        {"loan_id": pl.Int64, "LGD": pl.Float64, "opened": pl.Date}
    )
    lgd = spec["LGD"]
    assert lgd.nullable and lgd.format is None
    assert lgd.bounds.min >= 0.0 and lgd.bounds.max < 1.0


def test_schema_infer_takes_parse_text(tmp_path, capsys):
    data = tmp_path / "loans.parquet"
    pl.DataFrame(
        {"loan_id": ["1", "2", "3"], "LGD": ["0.45", None, "0.2"]}
    ).write_parquet(data)

    plain, parsed = tmp_path / "plain.yaml", tmp_path / "parsed.yaml"
    assert main(["schema", "infer", str(data), "-o", str(plain)]) == 0
    assert "--parse-text reads them" in capsys.readouterr().err
    assert main(["schema", "infer", str(data), "-o", str(parsed), "--parse-text"]) == 0
    assert "note:" not in capsys.readouterr().err

    assert FrameSpec.from_yaml(plain).schema()["LGD"] == pl.String
    assert FrameSpec.from_yaml(parsed).schema() == pl.Schema(
        {"loan_id": pl.Int64, "LGD": pl.Float64}
    )


def test_synthesize_on_the_command_line_takes_parse_text(tmp_path):
    data, fake = tmp_path / "loans.parquet", tmp_path / "fake.parquet"
    pl.DataFrame(
        {"when": [str(dt.date(2024, 1, d)) for d in range(1, 29)]}
    ).write_parquet(data)
    command = ["synthesize", str(data), "-o", str(fake), "--seed", "1", "--parse-text"]
    assert main(command) == 0
    assert pl.read_parquet(fake).schema == pl.Schema({"when": pl.Date})


# ---------------------------------------------------------------------------
# Validating text: `validate(parse_text=True)` and `read()`
# ---------------------------------------------------------------------------


def _loans_csv(tmp_path, *, rows: int = 300, bad: str | None = None):
    rng = random.Random(1)
    lines = ["loan_id,LGD,opened,flag"]
    for i in range(1, rows + 1):
        lgd = rng.choice([f"{rng.random():.4f}", "0", ""])
        if bad is not None and i == 5:
            lgd = bad
        flag = rng.choice(["true", "FALSE"])
        lines.append(f"{i},{lgd},2024-01-{1 + i % 28:02d},{flag}")
    path = tmp_path / "loans.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


TYPED = pl.Schema(
    {"loan_id": pl.Int64, "LGD": pl.Float64, "opened": pl.Date, "flag": pl.Boolean}
)


def test_a_text_frame_validates_against_its_own_profile(tmp_path):
    raw = pl.read_csv(_loans_csv(tmp_path), infer_schema=False)
    spec = polspec.profile(raw, parse_text=True)
    assert polspec.validate(spec, raw, parse_text=True).schema == TYPED
    lazy = polspec.validate(spec, raw.lazy(), parse_text=True)
    assert lazy.collect().schema == TYPED


def test_without_parse_text_text_is_still_the_wrong_dtype(tmp_path):
    raw = pl.read_csv(_loans_csv(tmp_path), infer_schema=False)
    spec = polspec.profile(raw, parse_text=True)
    report = polspec.inspect(spec, raw)
    assert [f.code for f in report.findings] == ["dtype"] * 4


def test_a_value_that_does_not_parse_is_a_dtype_finding_naming_it(tmp_path):
    raw = pl.read_csv(_loans_csv(tmp_path, bad="N/A"), infer_schema=False)
    spec = polspec.TableSpec(
        "Loans", {"LGD": polspec.ColSpec(pl.Float64, nullable=True)}
    )
    (finding,) = polspec.inspect(spec, raw.select("LGD"), parse_text=True).findings
    assert finding.code == "dtype"
    assert (finding.count, finding.samples) == (1, ("N/A",))
    assert finding.details["unparsed"] == 1
    assert "do not parse as Float64" in finding.message


def test_a_declared_narrow_integer_is_judged_by_its_width():
    """Text is read as an Int64, so a value past an Int8 is the dtype's
    range, as for any wider integer standing in -- not a parse failure."""
    spec = polspec.TableSpec("T", {"n": polspec.ColSpec(pl.Int8)})
    report = polspec.inspect(spec, pl.DataFrame({"n": ["1", "300"]}), parse_text=True)
    assert [f.code for f in report.findings] == ["dtype"]
    assert report.findings[0].samples == (300,)


def test_a_declared_type_parses_a_leading_zero():
    """The leading-zero rule is the profiler guessing; a declared integer
    says the column is numbers."""
    spec = polspec.TableSpec("T", {"n": polspec.ColSpec(pl.Int64)})
    out = polspec.validate(spec, pl.DataFrame({"n": ["007", "10"]}), parse_text=True)
    assert out["n"].to_list() == [7, 10]


def test_empty_text_is_filled_by_a_default():
    spec = polspec.TableSpec("T", {"n": polspec.ColSpec(pl.Int64, default=0)})
    out = polspec.validate(
        spec, pl.DataFrame({"n": ["1", " ", ""]}), parse_text=True, fill_defaults=True
    )
    assert out["n"].to_list() == [1, 0, 0]


def test_read_types_a_file_read_as_text(tmp_path):
    path = _loans_csv(tmp_path)
    spec = polspec.profile(pl.read_csv(path, infer_schema=False), parse_text=True)
    assert polspec.read(spec, path, infer_schema=False).schema == TYPED


def test_validate_on_the_command_line_names_what_does_not_parse(tmp_path, capsys):
    path = _loans_csv(tmp_path, bad="N/A")
    spec_path = tmp_path / "loans.yaml"
    polspec.to_yaml(
        polspec.TableSpec("Loans", {"LGD": polspec.ColSpec(pl.Float64, nullable=True)}),
        spec_path,
    )
    assert main(["validate", str(spec_path), str(path), "--allow-extra"]) == 1
    assert "such as ['N/A']" in capsys.readouterr().out


SCALARS = {
    name: column
    for name, column in COLUMN_CASES.items()
    if column.dtype in (pl.Boolean, pl.Date, pl.Time)
    or column.dtype.is_numeric()
    or isinstance(column.dtype, pl.Datetime)
}


@pytest.mark.parametrize("name", sorted(SCALARS))
def test_generated_data_written_as_text_validates_with_parse_text(name):
    """What `generate()` makes, written out as text, is read back as what
    it was by the one parser `read()` and `validate()` share."""
    spec = polspec.TableSpec("T", {"c": SCALARS[name]})  # validators read "c"
    data = polspec.generate(spec, 200, seed=1)
    column = pl.col("c")
    # Polars casts a Time to text without its fraction of a second.
    written = (
        column.dt.to_string("%H:%M:%S%.f")
        if data.schema["c"] == pl.Time
        else column.cast(pl.String)
    )
    text = data.select(written)
    out = polspec.validate(spec, text, parse_text=True, cast=True)
    assert out.equals(data)
