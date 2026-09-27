"""The case catalogue the contract tests share, said once.

`COLUMN_CASES` is one declared column per case, covering every field a
column can carry; `EVERY_DTYPE` is every Polars dtype that holds data,
nested to any depth. `contracts/test_roundtrip.py` holds each to the
round-trip property through `generate()`, and `contracts/test_streaming.py`
through `generate_batches()` and `scan()` -- one list, so a case added for
one is held to both.

On `pythonpath` (see `[tool.pytest.ini_options]`), like `helpers`.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import polars as pl
from polspec import ColSpec, col
from polspec.dtypes import MAP

# ---------------------------------------------------------------------------
# Column matrix -- one declared column per case, round-tripped in isolation.
# ---------------------------------------------------------------------------

COLUMN_CASES: dict[str, ColSpec] = {
    # integers
    "int8": ColSpec(pl.Int8),
    "int16": ColSpec(pl.Int16),
    "int32": ColSpec(pl.Int32),
    "int64": ColSpec(pl.Int64),
    "uint8": ColSpec(pl.UInt8),
    "uint16": ColSpec(pl.UInt16),
    "uint32": ColSpec(pl.UInt32),
    "uint64": ColSpec(pl.UInt64),
    "int64_bounded": ColSpec(pl.Int64, bounds=(-100, 100)),
    "int64_nullable": ColSpec(pl.Int64, nullable=True, null_probability=0.3),
    "int64_always_null": ColSpec(pl.Int64, nullable=True, null_probability=1.0),
    "int64_lower_open": ColSpec(pl.Int64, bounds=(0, None)),
    "int64_upper_open": ColSpec(pl.Int64, bounds=(None, 0)),
    "int64_lower_open_nullable": ColSpec(pl.Int64, bounds=(0, None), nullable=True),
    # floats
    "float32": ColSpec(pl.Float32),
    "float64": ColSpec(pl.Float64),
    "float32_bounded": ColSpec(pl.Float32, bounds=(-1.0, 1.0)),
    "float64_bounded": ColSpec(pl.Float64, bounds=(-2.5, 2.5)),
    "float64_lower_open": ColSpec(pl.Float64, bounds=(0.0, None)),
    # decimals: drawn as the physical integer, scaled back to the declared type
    "decimal": ColSpec(pl.Decimal(10, 2)),
    "decimal_bounded": ColSpec(pl.Decimal(10, 2), bounds=(0, "99.99")),
    "decimal_wide": ColSpec(pl.Decimal(38, 6), bounds=(-1, 1), nullable=True),
    "decimal_no_scale": ColSpec(pl.Decimal(5, 0), bounds=(None, 100)),
    "decimal_choices": ColSpec(
        pl.Decimal(4, 1), choices=[Decimal("0.5"), Decimal("1.5")]
    ),
    # lists and arrays: the value fields describe each element
    "list_int": ColSpec(pl.List(pl.Int64)),
    "list_int_bounded": ColSpec(pl.List(pl.Int64), bounds=(-5, 5), list_length=(1, 4)),
    "list_empty_only": ColSpec(pl.List(pl.Int64), list_length=(0, 0)),
    "list_nullable": ColSpec(pl.List(pl.Float64), nullable=True, null_probability=0.4),
    "list_enum": ColSpec(pl.List(pl.Enum(["x", "y"]))),
    "list_choices_weighted": ColSpec(pl.List(pl.String), choices={"a": 1.0, "b": 3.0}),
    "list_format": ColSpec(pl.List(pl.String), format="uuid4", list_length=(2, 2)),
    "list_string_len": ColSpec(pl.List(pl.String), string_length=(1, 3)),
    "list_date": ColSpec(
        pl.List(pl.Date), bounds=(dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    ),
    "list_decimal": ColSpec(pl.List(pl.Decimal(6, 2)), bounds=(0, 10)),
    "list_bool": ColSpec(pl.List(pl.Boolean), weights=[0.2, 0.8]),
    "list_binary": ColSpec(pl.List(pl.Binary), string_length=(2, 2)),
    "list_categorical": ColSpec(pl.List(pl.Categorical), choices=["p", "q"]),
    "list_distribution": ColSpec(
        pl.List(pl.Float64), distribution="normal", bounds=(0.0, 1.0)
    ),
    "array_float": ColSpec(pl.Array(pl.Float64, 3), bounds=(0.0, 1.0)),
    "array_nullable": ColSpec(
        pl.Array(pl.Int64, 2), nullable=True, null_probability=0.3
    ),
    "array_enum": ColSpec(pl.Array(pl.Enum(["a", "b"]), 4)),
    # boolean
    "bool": ColSpec(pl.Boolean),
    "bool_weighted": ColSpec(pl.Boolean, weights=[0.3, 0.7]),
    # text and bytes
    "string": ColSpec(pl.String),
    "string_len": ColSpec(pl.String, string_length=(3, 8)),
    "string_choices": ColSpec(pl.String, choices=["a", "b", "c"]),
    "string_choices_weighted": ColSpec(pl.String, choices={"a": 1.0, "b": 2.0}),
    "binary": ColSpec(pl.Binary),
    "binary_len": ColSpec(pl.Binary, string_length=(2, 4)),
    # formats: a String column generated to satisfy its own validator
    "format_uuid4": ColSpec(pl.String, format="uuid4"),
    "format_email": ColSpec(pl.String, format="email"),
    "format_ipv4": ColSpec(pl.String, format="ipv4"),
    "format_ipv6": ColSpec(pl.String, format="ipv6"),
    "format_mac": ColSpec(pl.String, format="mac"),
    "format_hostname": ColSpec(pl.String, format="hostname"),
    "format_iso_country": ColSpec(pl.String, format="iso_country"),
    "format_iso_currency": ColSpec(pl.String, format="iso_currency"),
    "format_nullable": ColSpec(
        pl.String, format="email", nullable=True, null_probability=0.3
    ),
    # a format widened with the values a column holds beside it
    "format_extended_finite": ColSpec(
        pl.String, format="iso_country", extra_values=["UK (ISO)", "UK (ISLANDS)"]
    ),
    "format_extended_template": ColSpec(
        pl.String, format="ipv4", extra_values={"NOT AVAILABLE": 0.05, "INVALID": 0.01}
    ),
    "format_extended_nullable": ColSpec(
        pl.String,
        format="email",
        extra_values=["n/a"],
        nullable=True,
        null_probability=0.3,
    ),
    "list_format_extended": ColSpec(
        pl.List(pl.String), format="uuid4", extra_values=["none"], list_length=(1, 3)
    ),
    "format_with_validator": ColSpec(
        pl.String, format="email", validators=[col("c").str.contains("@")]
    ),
    # temporal
    "date": ColSpec(pl.Date),
    "date_bounded": ColSpec(
        pl.Date, bounds=(dt.date(2020, 1, 1), dt.date(2021, 12, 31))
    ),
    "date_lower_open": ColSpec(pl.Date, bounds=(dt.date(2020, 1, 1), None)),
    "time": ColSpec(pl.Time),
    "datetime_us": ColSpec(pl.Datetime("us")),
    "datetime_ns": ColSpec(pl.Datetime("ns")),
    "datetime_bounded": ColSpec(
        pl.Datetime("us"),
        bounds=(dt.datetime(2020, 1, 1), dt.datetime(2021, 1, 1)),
    ),
    "duration_ms": ColSpec(pl.Duration("ms")),
    "duration_us": ColSpec(pl.Duration("us")),
    # enum / categorical
    "enum": ColSpec(pl.Enum(["x", "y", "z"])),
    "enum_weighted": ColSpec(pl.Enum(["x", "y", "z"]), weights=[1.0, 2.0, 3.0]),
    "enum_nullable": ColSpec(pl.Enum(["x", "y"]), nullable=True),
    "enum_choices_subset": ColSpec(pl.Enum(["x", "y", "z"]), choices=["x", "y"]),
    "categorical": ColSpec(pl.Categorical),
    "categorical_choices": ColSpec(pl.Categorical, choices=["p", "q"]),
    "categorical_u8": ColSpec(
        pl.Categorical(pl.Categories("roundtrip_u8", physical=pl.UInt8))
    ),
    "categorical_u16": ColSpec(
        pl.Categorical(pl.Categories("roundtrip_u16", physical=pl.UInt16))
    ),
    # distributions (each bounded, so validate has something to check)
    "int64_normal": ColSpec(
        pl.Int64,
        bounds=(0, 1_000),
        distribution="normal",
        distribution_params={"mean": 500.0, "std": 100.0},
    ),
    "int64_poisson": ColSpec(
        pl.Int64,
        bounds=(0, 100),
        distribution="poisson",
        distribution_params={"lambda": 4.0},
    ),
    "float64_uniform": ColSpec(pl.Float64, bounds=(0.0, 1.0), distribution="uniform"),
    "float64_lognormal": ColSpec(
        pl.Float64,
        bounds=(0.0, 100.0),
        distribution="lognormal",
        distribution_params={"mean": 1.0, "std": 0.5},
    ),
    "float64_exponential": ColSpec(
        pl.Float64,
        bounds=(0.0, 50.0),
        distribution="exponential",
        distribution_params={"rate": 0.5},
    ),
    "float64_gamma": ColSpec(
        pl.Float64,
        bounds=(0.0, 100.0),
        distribution="gamma",
        distribution_params={"shape": 2.0, "scale": 2.0},
    ),
    "float64_beta": ColSpec(
        pl.Float64,
        bounds=(0.0, 1.0),
        distribution="beta",
        distribution_params={"alpha": 2.0, "beta": 5.0},
    ),
    # a column validator that the declared bounds already imply
    "validator_implied_by_bounds": ColSpec(
        pl.Int64, bounds=(1, 10), validators=[pl.col("c") > 0]
    ),
    # bounds beyond 2**53 cross the boundary as integers, so nothing rounds
    "int64_bounds_above_2_53": ColSpec(
        pl.Int64, bounds=(9_007_199_254_740_990, 9_007_199_254_740_999)
    ),
    "uint64_bounds_at_the_maximum": ColSpec(
        pl.UInt64, bounds=(18_446_744_073_709_551_600, 18_446_744_073_709_551_615)
    ),
    # typed choices: no string round-trip on the way back from the engine
    "datetime_choices": ColSpec(
        pl.Datetime("us"),
        choices=[dt.datetime(2024, 1, 1, 12), dt.datetime(2025, 6, 30, 8, 30)],
    ),
    "binary_choices": ColSpec(pl.Binary, choices=[b"\x00\x01", b"\xff"]),
    # unique columns are drawn without replacement, from a domain with room
    "unique_int": ColSpec(pl.Int16, unique=True),
    "unique_bounded": ColSpec(pl.Int64, bounds=(1, 1_000), unique=True),
    "unique_string": ColSpec(pl.String, unique=True),
    "unique_choices": ColSpec(
        pl.String, choices=[f"c{i}" for i in range(400)], unique=True
    ),
    "unique_nullable": ColSpec(
        pl.Int32, bounds=(1, 400), unique=True, nullable=True, null_probability=0.3
    ),
    "unique_format": ColSpec(pl.String, format="uuid4", unique=True),
}


# ---------------------------------------------------------------------------
# Every dtype that holds data, nested to any depth -- and the census in
# `test_roundtrip.py` fails the day Polars adds a dtype this list does not
# name, which is how Int128, UInt128 and Float16 went unnoticed through 0.9.0.
# ---------------------------------------------------------------------------

EVERY_DTYPE = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.Int128,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.UInt128,
    pl.Float16,
    pl.Float32,
    pl.Float64,
    pl.Decimal(10, 2),
    pl.Boolean,
    pl.String,
    pl.Binary,
    pl.Date,
    pl.Time,
    pl.Datetime("us"),
    pl.Duration("ms"),
    pl.Enum(["a", "b"]),
    pl.Categorical,
    pl.List(pl.Int64),
    pl.Array(pl.Float64, 3),
    pl.Struct({"a": pl.Int64, "b": pl.String}),
    pl.Struct({}),
    # Nested to two levels, every way round.
    pl.List(pl.List(pl.Int64)),
    pl.List(pl.Struct({"a": pl.Int64})),
    pl.Array(pl.Struct({"a": pl.Int64}), 2),
    pl.Struct({"xs": pl.List(pl.Int64)}),
    pl.Struct({"inner": pl.Struct({"a": pl.Int64})}),
]
# Polars 2's `Map`, only where it exists: it is covered there, and on Polars
# 1 there is no such dtype for the census to find.
if MAP is not None:
    EVERY_DTYPE += [
        MAP(pl.String, pl.Int64),
        MAP(pl.Int64, pl.Struct({"a": pl.List(pl.Int8)})),
        pl.List(MAP(pl.Boolean, pl.String)),
    ]
