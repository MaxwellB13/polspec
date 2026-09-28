"""`validate()` and what a value may be: choices, bounds (and turning them
off), string lengths, rules, and the samples a finding carries.
"""

from __future__ import annotations

import datetime
import decimal

import polars as pl
import pytest
from polspec import (
    Bound,
    ColRule,
    ColSpec,
    FrameSpec,
    SpecError,
    TableSpec,
    ValidationError,
    col,
    generate,
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


def test_validation_enum_invalid_value_chicken():
    # User's example: enum ["fruit", "vegetable", "meat"] receives "chicken"
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "category": ["fruit", "vegetable", "meat", "chicken"],
            "quantity": [10, 20, 30, 40],
            "price": [1.0, 2.0, 3.0, 4.0],
            "code": ["AAA", "BBB", "CCC", "DDD"],
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        ProduceInventory.validate(df)

    err_msg = str(exc_info.value)
    assert "category" in err_msg
    assert "chicken" in err_msg
    assert "allowed choices/categories" in err_msg
    assert len(exc_info.value.errors) >= 1


def test_validation_colrule():
    class RuleSpec(FrameSpec):
        status = ColSpec(dtype=pl.Enum(["active", "inactive"]), nullable=False)
        discount = ColSpec(
            dtype=pl.Float64,
            bounds=Bound(0.0, 1.0),
            nullable=False,
            rules=(
                ColRule(
                    when=col("status") == "inactive",
                    choices=[0.0],
                ),
            ),
        )

    # Valid: inactive has discount 0.0
    valid_df = pl.DataFrame(
        {
            "status": ["active", "inactive", "active"],
            "discount": [0.25, 0.0, 0.50],
        }
    )
    RuleSpec.validate(valid_df)

    # Invalid: inactive has discount 0.80 violating rule choices [0.0]
    invalid_df = pl.DataFrame(
        {
            "status": ["active", "inactive", "active"],
            "discount": [0.25, 0.80, 0.50],
        }
    )
    with pytest.raises(ValidationError) as exc_info:
        RuleSpec.validate(invalid_df)

    assert "violating ColRule" in str(exc_info.value)


def test_validation_temporal_bounds():
    import datetime

    class DateSpec(FrameSpec):
        d = ColSpec(
            dtype=pl.Date,
            bounds=Bound(datetime.date(2023, 1, 1), datetime.date(2023, 12, 31)),
        )

    valid_df = pl.DataFrame(
        {"d": [datetime.date(2023, 6, 15), datetime.date(2023, 1, 1)]}
    )
    assert DateSpec.validate(valid_df).height == 2

    invalid_df = pl.DataFrame(
        {"d": [datetime.date(2022, 12, 31), datetime.date(2023, 6, 15)]}
    )
    with pytest.raises(ValidationError) as exc:
        DateSpec.validate(invalid_df)
    assert "out of bounds" in str(exc.value)


def test_validation_rule_precedence():
    class TierSpec(FrameSpec):
        tier = ColSpec(dtype=pl.String)
        amount = ColSpec(
            dtype=pl.Int64,
            rules=(
                ColRule(when=col("tier") == "gold", choices=[100]),
                ColRule(when=col("tier").is_in(["gold", "silver"]), choices=[50]),
            ),
        )

    # For "gold", Rule 1 matches so amount must be 100.
    # Because Rule 1 matched, Rule 2 should NOT fail "gold" for not being 50.
    df = pl.DataFrame(
        {
            "tier": ["gold", "silver"],
            "amount": [100, 50],
        }
    )
    validated = TierSpec.validate(df)
    assert validated.height == 2


def test_a_null_condition_does_not_excuse_the_rules_after_it():
    """Validation must read a null `when` the way generation writes one.

    Generation folds a null condition to False before testing it *and* before
    accumulating it into the claimed mask, so a row the first rule does not
    match is still offered to the second. If validation lets the null through
    instead, Kleene logic turns `~claimed` null on that row and every later
    rule is silently excused there -- on a row generation did rewrite.
    """

    class Shipments(FrameSpec):
        tier = ColSpec(pl.String, nullable=True, choices=["gold"])
        express = ColSpec(pl.Boolean)
        carrier = ColSpec(
            pl.String,
            choices=["RM", "UPS", "DHL"],
            rules=(
                ColRule(when=col("tier") == "gold", choices=["RM"]),
                ColRule(when=col("express"), choices=["UPS"]),
            ),
        )

    # Row 0's `tier` is null, so rule 1's condition is null there; rule 2 still
    # applies, and "DHL" violates it exactly as it does on row 1.
    df = pl.DataFrame(
        {
            "tier": [None, "gold"],
            "express": [True, True],
            "carrier": ["DHL", "DHL"],
        },
        schema={"tier": pl.String, "express": pl.Boolean, "carrier": pl.String},
    )
    report = Shipments.inspect(df)

    violations = report.by_code("rule")
    assert [f.count for f in violations] == [1, 1]
    assert report.rows(violations[1]).collect().height == 1

    # And the same spec's own generated data agrees with the same check, which
    # is the property the two implementations exist to keep.
    generated = Shipments.generate(2_000, seed=7)
    assert generated["tier"].null_count() > 0
    assert Shipments.inspect(generated).passed


def test_validation_multibyte_string_length():
    class UnicodeSpec(FrameSpec):
        text = ColSpec(dtype=pl.String, string_length=Bound(2, 3))

    # "🚀🌟" is 2 characters (len_chars=2, len_bytes=8)
    df_valid = pl.DataFrame({"text": ["🚀🌟", "abc"]})
    assert UnicodeSpec.validate(df_valid).height == 2

    # "🚀🌟🎉✨" is 4 characters (len_chars=4) -> violates max 3
    df_invalid = pl.DataFrame({"text": ["🚀🌟🎉✨"]})
    with pytest.raises(ValidationError) as exc:
        UnicodeSpec.validate(df_invalid)
    assert "string length outside" in str(exc.value)


# ---------------------------------------------------------------------------
# validate_bounds
# ---------------------------------------------------------------------------


class Ranged(FrameSpec):
    total = ColSpec(pl.Float64, bounds=(0, 100))
    code = ColSpec(pl.String, string_length=(2, 3))
    ids = ColSpec(pl.List(pl.Int64), bounds=(0, 9), list_length=(1, 2))
    point = ColSpec(
        pl.Struct({"lat": pl.Float64}),
        fields={"lat": ColSpec(pl.Float64, bounds=(-90, 90))},
    )


OUT_OF_RANGE = pl.DataFrame(
    {
        "total": [500.0],
        "code": ["toolong"],
        "ids": [[50, 1, 2]],
        "point": [{"lat": 200.0}],
    },
    schema=Ranged.schema(),
)


def test_validate_bounds_false_turns_off_every_bounds_check_and_nothing_else():
    """Column, list element and struct field bounds all go; the length
    claims have codes of their own and stay."""
    assert {f.key for f in Ranged.inspect(OUT_OF_RANGE).findings} == {
        "total__bounds",
        "code__len",
        "ids__bounds",
        "ids__list_len",
        "point.lat__bounds",
    }
    report = Ranged.inspect(OUT_OF_RANGE, validate_bounds=False)
    assert {f.key for f in report.findings} == {"code__len", "ids__list_len"}
    assert report.options.bounds is False


def test_validate_bounds_is_an_option_like_the_others():
    from polspec import ValidationOptions, inspect

    in_range = OUT_OF_RANGE.with_columns(
        code=pl.lit("ok"), ids=pl.lit([1], dtype=pl.List(pl.Int64))
    )
    with pytest.raises(ValidationError, match="out of bounds"):
        Ranged.validate(in_range)
    assert Ranged.validate(in_range, validate_bounds=False).height == 1
    assert inspect(Ranged.spec, in_range, validate_bounds=False).passed
    assert Ranged.inspect(in_range, options=ValidationOptions(bounds=False)).passed


def test_a_finding_on_a_zoned_datetime_carries_its_samples():
    """A finding's samples are Python values, and a zoned one needs the IANA
    database -- which Windows does not ship, so polspec depends on `tzdata`
    there. Without it Polars panics, past any `except Exception`."""
    import datetime

    class Zoned(FrameSpec):
        at = ColSpec(
            pl.Datetime("us", "Europe/London"),
            bounds=(datetime.datetime(2024, 1, 1), datetime.datetime(2025, 1, 1)),
        )

    df = pl.DataFrame({"at": [datetime.datetime(2030, 6, 1)]}).with_columns(
        pl.col("at").dt.replace_time_zone("Europe/London")
    )
    (finding,) = Zoned.inspect(df).findings
    assert finding.key == "at__bounds"
    assert finding.samples[0].year == 2030


# ---------------------------------------------------------------------------
# Choices compare in the column's own dtype
# ---------------------------------------------------------------------------

_TYPED_CHOICES = [
    # int choices on a float column: Polars 2 refuses to coerce them lossily
    (pl.Float64, [1, 2], 9.0),
    # a Python datetime is microseconds; the column is milliseconds
    (pl.Datetime("ms"), [datetime.datetime(2024, 1, 1)], datetime.datetime(2030, 1, 1)),
    # ...and on a nanosecond column the list rejected the column's own data
    (pl.Datetime("ns"), [datetime.datetime(2024, 1, 1)], datetime.datetime(2030, 1, 1)),
    # a Python Decimal list arrives at full precision
    (
        pl.Decimal(10, 2),
        [decimal.Decimal("1.50"), decimal.Decimal("2.25")],
        decimal.Decimal("9.99"),
    ),
    (pl.Int8, [1, 2, 3], 9),
]


@pytest.mark.parametrize(("dtype", "choices", "outside"), _TYPED_CHOICES, ids=str)
def test_choices_are_compared_in_the_columns_own_dtype(dtype, choices, outside):
    """A Python list of choices reaches `is_in` at its widest type -- an int
    list on a float column, microseconds on a millisecond column, a Decimal
    at full precision -- which Polars will not compare (strictly so from
    Polars 2). Typed as the column, the choices validate what they should
    and reject what they should, on either line."""
    spec = TableSpec("T", {"c": ColSpec(dtype, choices=choices)})
    df = generate(spec, 100, seed=1)
    assert inspect(spec, df).passed
    bad = df.with_columns(pl.lit(outside).cast(dtype).alias("c"))
    assert [f.key for f in inspect(spec, bad).findings] == ["c__choices"]


def test_a_rules_choices_are_compared_in_the_columns_own_dtype():
    spec = TableSpec(
        "R",
        {
            "flag": ColSpec(pl.Boolean),
            "v": ColSpec(
                pl.Float64,
                choices=[1, 2, 3],
                rules=[ColRule(when=col("flag"), choices=(1,))],
            ),
        },
    )
    df = generate(spec, 200, seed=1)
    assert inspect(spec, df).passed


# ---------------------------------------------------------------------------
# A value the declared dtype cannot hold
# ---------------------------------------------------------------------------

_INTEGER_LIMITS = {
    pl.Int8: (-(2**7), 2**7 - 1),
    pl.Int16: (-(2**15), 2**15 - 1),
    pl.Int32: (-(2**31), 2**31 - 1),
    pl.UInt8: (0, 2**8 - 1),
    pl.UInt16: (0, 2**16 - 1),
    pl.UInt32: (0, 2**32 - 1),
}


def _dtype_findings(report) -> list[str]:
    return [f.key for f in report.findings if f.code == "dtype"]


@pytest.mark.parametrize("declared", list(_INTEGER_LIMITS), ids=str)
@pytest.mark.parametrize("arrives_as", [pl.Int64, pl.UInt64, pl.Int128], ids=str)
def test_an_integer_the_declared_width_cannot_hold_is_a_dtype_finding(
    declared, arrives_as
):
    """Any integer width stands in for a declared integer; its values still
    have to fit. Before 0.13.1 `Int8` holding 1000 passed, and
    `validate(cast=True)` -- which only runs on a frame that passed -- then
    raised Polars' conversion error.
    """
    lo, hi = _INTEGER_LIMITS[declared]
    spec = TableSpec("T", {"n": ColSpec(declared, nullable=True)})
    edges = [v for v in (lo, hi) if _fits(v, arrives_as)]
    inside = pl.DataFrame({"n": pl.Series([*edges, None], dtype=arrives_as)})
    assert not inspect(spec, inside)
    assert validate(spec, inside, cast=True)["n"].dtype == declared

    beyond = [v for v in (lo - 1, hi + 1) if _fits(v, arrives_as)]
    outside = pl.DataFrame({"n": pl.Series([hi, *beyond], dtype=arrives_as)})
    report = inspect(spec, outside)
    assert _dtype_findings(report) == ["n__dtype_range"]
    (finding,) = report.findings
    assert finding.count == len(beyond)
    assert sorted(finding.samples) == sorted(beyond)
    assert f"{declared} cannot hold ({lo}..{hi})" in finding.message


def _fits(value: int, dtype: pl.DataType) -> bool:
    return value >= 0 or not dtype.is_unsigned_integer()


def test_a_wider_integer_that_holds_nothing_past_the_declared_one_adds_no_check():
    """`Int8` arriving as `Int8`, or as a narrower `UInt8` for an `Int16`,
    cannot hold a value its declaration cannot."""
    from polspec.validation.constraints._values import _unholdable

    assert _unholdable(pl.col("n"), pl.Int16, pl.UInt8) is None
    assert _unholdable(pl.col("n"), pl.Int64, pl.Int64) is None


@pytest.mark.parametrize(
    "data, unholdable",
    [
        # A float is rounded to two places first: 999.995 becomes 1000.00.
        (pl.Series([999.99, 999.994, -999.994]), []),
        (pl.Series([999.995, 1000.0, -1000.0, float("inf")]), None),
        (pl.Series([float("nan")]), None),
        (pl.Series([999, -999], dtype=pl.Int64), []),
        (pl.Series([1000, -1000], dtype=pl.Int64), None),
        (pl.Series([100, 255], dtype=pl.UInt8), []),
        (
            pl.Series(["999.994", "-999.994"]).cast(pl.Decimal(10, 3)),
            [],
        ),
        (
            pl.Series(["999.995", "-1000.000"]).cast(pl.Decimal(10, 3)),
            None,
        ),
        (pl.Series(["99.99"]).cast(pl.Decimal(4, 2)), []),
    ],
    ids=lambda v: str(v.dtype) if isinstance(v, pl.Series) else "",
)
def test_a_decimal_holds_magnitudes_under_its_integer_digits(data, unholdable):
    """A `Decimal(5, 2)` holds magnitudes under 1000, however the column
    arrives: as a float, an integer or a wider Decimal. `None` in the table
    means every value is one it cannot hold."""
    spec = TableSpec("T", {"d": ColSpec(pl.Decimal(5, 2))})
    frame = pl.DataFrame({"d": data})
    report = inspect(spec, frame)
    if unholdable == []:
        assert not report, report
        assert validate(spec, frame, cast=True)["d"].dtype == pl.Decimal(5, 2)
    else:
        (finding,) = report.findings
        assert finding.key == "d__dtype_range"
        assert finding.count == len(data)
        assert "magnitude under 1000" in finding.message


def test_an_unholdable_value_is_found_inside_lists_and_structs():
    spec = TableSpec(
        "T",
        {
            "l": ColSpec(pl.List(pl.Int8)),
            "a": ColSpec(pl.Array(pl.UInt8, 2)),
            "s": ColSpec(pl.Struct({"x": pl.Int16})),
            "ll": ColSpec(pl.List(pl.List(pl.Int8))),
        },
    )
    frame = pl.DataFrame(
        {
            "l": [[1, 1000], [2]],
            "a": pl.Series([[1, -1], [2, 3]], dtype=pl.Array(pl.Int64, 2)),
            "s": [{"x": 40_000}, {"x": 1}],
            "ll": [[[1], [500]], [[2]]],
        }
    )
    report = inspect(spec, frame)
    assert sorted(_dtype_findings(report)) == [
        "a__dtype_range",
        "l__dtype_range",
        "ll[]__dtype_range",
        "s.x__dtype_range",
    ]
    assert all(f.count == 1 for f in report.findings)
    # A field's samples are the field's values, as for any claim on a field.
    assert [f.samples for f in report.findings if f.key == "s.x__dtype_range"] == [
        (40_000,)
    ]


@pytest.mark.skipif(not hasattr(pl, "Map"), reason="Map is a Polars 2 dtype")
def test_an_unholdable_value_is_found_in_a_map():
    spec = TableSpec("T", {"m": ColSpec(pl.Map(pl.String, pl.Int8))})
    frame = pl.DataFrame(
        {
            "m": pl.Series([[{"key": "a", "value": 1000}]]).cast(
                pl.Map(pl.String, pl.Int64)
            )
        }
    )
    assert _dtype_findings(inspect(spec, frame)) == ["m.value__dtype_range"]


def test_an_unholdable_value_is_reported_with_the_bounds_checks_off():
    """The range is the dtype's claim, not the declared bounds'."""
    spec = TableSpec("T", {"n": ColSpec(pl.Int8, bounds=(0, 10))})
    frame = pl.DataFrame({"n": [1000]})
    assert sorted(f.code for f in inspect(spec, frame).findings) == ["bounds", "dtype"]
    assert [f.code for f in inspect(spec, frame, validate_bounds=False).findings] == [
        "dtype"
    ]


def test_an_unholdable_value_locates_its_rows():
    spec = TableSpec("T", {"n": ColSpec(pl.Int8)})
    frame = pl.DataFrame({"n": [1, 1000, 2, -500]})
    report = inspect(spec, frame)
    assert report.failing_rows().collect()["n"].to_list() == [1000, -500]
    assert report.passing_rows().collect()["n"].to_list() == [1, 2]


# ---------------------------------------------------------------------------
# A `Null` column holds nothing, whatever dtype it arrives as
# ---------------------------------------------------------------------------


def test_a_null_column_accepts_any_dtype_holding_only_nulls():
    """A text format hands an empty column back as `String`; a value in it
    is a row-level `dtype` finding, as a value an `Int8` cannot hold is."""
    spec = TableSpec("T", {"z": ColSpec(pl.Null, nullable=True)})
    empty = pl.DataFrame({"z": pl.Series([None, None], dtype=pl.String)})
    assert not inspect(spec, empty)
    assert validate(spec, empty, cast=True)["z"].dtype == pl.Null

    held = pl.DataFrame({"z": ["x", None, "y"]})
    (finding,) = inspect(spec, held).findings
    assert (finding.key, finding.count, finding.samples) == (
        "z__dtype_range",
        2,
        ("x", "y"),
    )
    assert "Null cannot hold (only nulls)" in finding.message
    assert [f.key for f in inspect(spec, empty, strict_dtypes=True).findings] == [
        "z__dtype"
    ]


def test_a_null_column_is_declared_nullable_with_every_value_null():
    assert ColSpec(pl.Null, nullable=True).null_probability == 1.0
    assert ColSpec(pl.Null, nullable=True, null_probability=1.0).null_probability == 1.0
    with pytest.raises(SpecError, match="must be declared nullable=True"):
        ColSpec(pl.Null)
    with pytest.raises(SpecError, match="every value of a Null column is null"):
        ColSpec(pl.Null, nullable=True, null_probability=0.5)
