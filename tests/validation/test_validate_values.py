"""`validate()` and what a value may be: choices, bounds (and turning them
off), string lengths, rules, and the samples a finding carries.
"""

from __future__ import annotations

import polars as pl
import pytest
from polspec import (
    Bound,
    ColRule,
    ColSpec,
    FrameSpec,
    ValidationError,
    col,
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
