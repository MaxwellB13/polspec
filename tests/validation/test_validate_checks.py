"""`validate()` and expressions: a spec's `__checks__` and a column's
`validators`, their names, descriptions and null handling.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest
from polspec import (
    Bound,
    Check,
    ColSpec,
    FrameSpec,
    ValidationError,
)

# =====================================================================
# Multi-Column & Cross-Field Constraints (Check / __checks__)
# =====================================================================


class OrderSpec(FrameSpec):
    created_at = ColSpec(pl.Datetime, nullable=False, tags="temporal")
    shipped_at = ColSpec(pl.Datetime, nullable=True, tags="temporal")
    subtotal = ColSpec(pl.Float64, bounds=Bound(0.0, 1_000_000.0), nullable=False)
    tax = ColSpec(pl.Float64, bounds=Bound(0.0, 100_000.0), nullable=False)
    total = ColSpec(pl.Float64, bounds=Bound(0.0, 1_100_000.0), nullable=False)

    __checks__ = [
        Check(
            pl.col("shipped_at") >= pl.col("created_at"),
            name="shipped_after_created",
            description="Shipped date must be on or after creation date.",
        ),
        Check(
            pl.col("total") >= pl.col("subtotal"),
            name="total_gte_subtotal",
        ),
    ]


def test_validation_check_success():
    df = pl.DataFrame(
        {
            "created_at": [
                datetime(2025, 1, 1, 10, 0),
                datetime(2025, 1, 2, 10, 0),
            ],
            "shipped_at": [
                datetime(2025, 1, 1, 12, 0),
                None,  # Nullable: should pass ignore_nulls=True
            ],
            "subtotal": [100.0, 50.0],
            "tax": [10.0, 5.0],
            "total": [110.0, 55.0],
        }
    )

    result = OrderSpec.validate(df)
    assert isinstance(result, pl.DataFrame)
    assert result.height == 2


def test_validation_check_failures():
    df = pl.DataFrame(
        {
            "created_at": [
                datetime(2025, 1, 5, 10, 0),
                datetime(2025, 1, 2, 10, 0),
            ],
            "shipped_at": [
                datetime(2025, 1, 1, 10, 0),  # Violates shipped >= created (1/1 < 1/5)
                datetime(2025, 1, 3, 10, 0),
            ],
            "subtotal": [100.0, 50.0],
            "tax": [10.0, 5.0],
            "total": [80.0, 55.0],  # Violates total >= subtotal (80 < 100)
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        OrderSpec.validate(df)

    err = exc_info.value
    err_str = str(err)
    assert len(err.errors) >= 2
    assert "shipped_after_created" in err_str
    assert "Shipped date must be on or after creation date" in err_str
    assert "total_gte_subtotal" in err_str
    assert "found 1 row(s) violating condition" in err_str


def test_validation_check_ignore_nulls_behavior():
    class NullCheckSpec(FrameSpec):
        a = ColSpec(pl.Int64, nullable=True)
        b = ColSpec(pl.Int64, nullable=True)

        __checks__ = [
            Check(
                pl.col("a") > pl.col("b"), name="strict_no_nulls", ignore_nulls=False
            ),
            Check(
                pl.col("a") > pl.col("b"), name="lenient_with_nulls", ignore_nulls=True
            ),
        ]

    # Row 1: 10 > 5 (True)
    # Row 2: None > 5 (Null) -> lenient passes, strict fails
    df = pl.DataFrame(
        {
            "a": [10, None],
            "b": [5, 5],
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        NullCheckSpec.validate(df)

    err_str = str(exc_info.value)
    assert "strict_no_nulls" in err_str
    assert "lenient_with_nulls" not in err_str


def test_validation_check_inheritance_and_bypass():
    class BaseOrderSpec(FrameSpec):
        a = ColSpec(pl.Int64)
        b = ColSpec(pl.Int64)
        __checks__ = [Check(pl.col("a") > 0, name="a_positive")]

    class ExtendedOrderSpec(BaseOrderSpec):
        c = ColSpec(pl.Int64)
        __checks__ = [Check(pl.col("c") > pl.col("b"), name="c_gt_b")]

    assert len(ExtendedOrderSpec.checks()) == 2
    check_names = [c.name for c in ExtendedOrderSpec.checks()]
    assert check_names == ["a_positive", "c_gt_b"]

    df_bad = pl.DataFrame({"a": [-1, 2], "b": [5, 5], "c": [3, 10]})

    # Both fail (a <= 0 in row 0, c <= b in row 0)
    with pytest.raises(ValidationError) as exc_info:
        ExtendedOrderSpec.validate(df_bad)
    assert len(exc_info.value.errors) == 2

    # validate_checks=False disables check execution
    df_valid_types = pl.DataFrame({"a": [-1, 2], "b": [5, 5], "c": [3, 10]})
    res = ExtendedOrderSpec.validate(df_valid_types, validate_checks=False)
    assert res.height == 2


# =====================================================================
# Column-level validators (ColSpec.validators)
# =====================================================================


class ProductValidatorSpec(FrameSpec):
    price = ColSpec(pl.Float64, validators=[pl.col("price") * 100 % 5 == 0])
    quantity = ColSpec(pl.Int64, validators=[pl.col("quantity") % 2 == 0])


def test_validation_column_validators_success():
    df = pl.DataFrame({"price": [1.00, 1.05, 2.50], "quantity": [2, 4, 100]})
    result = ProductValidatorSpec.validate(df)
    assert result.height == 3


def test_validation_column_validators_failure():
    df = pl.DataFrame({"price": [1.00, 1.03], "quantity": [2, 3]})
    with pytest.raises(ValidationError) as exc_info:
        ProductValidatorSpec.validate(df)

    err = exc_info.value
    assert len(err.errors) == 2
    err_str = str(err)
    assert "Column 'price': validator" in err_str
    assert "Column 'quantity': validator" in err_str
    assert "found 1 row(s) violating condition" in err_str


def test_validation_column_validators_uses_check_name_and_description():
    class ScoreSpec(FrameSpec):
        score = ColSpec(
            pl.Float64,
            nullable=True,
            validators=[
                Check(
                    pl.col("score") >= 0,
                    name="score_non_negative",
                    description="Scores can't be negative",
                )
            ],
        )

    df = pl.DataFrame({"score": [1.0, -5.0, None]})
    with pytest.raises(ValidationError) as exc_info:
        ScoreSpec.validate(df)

    err_str = str(exc_info.value)
    assert "validator 'score_non_negative' failed" in err_str
    assert "Scores can't be negative" in err_str


def test_validation_column_validators_ignore_nulls_behavior():
    class NullValidatorSpec(FrameSpec):
        a = ColSpec(pl.Int64, nullable=True)

    class StrictNullValidatorSpec(FrameSpec):
        a = ColSpec(
            pl.Int64,
            nullable=True,
            validators=[Check(pl.col("a") > 0, name="strict", ignore_nulls=False)],
        )

    class LenientNullValidatorSpec(FrameSpec):
        a = ColSpec(
            pl.Int64,
            nullable=True,
            validators=[Check(pl.col("a") > 0, name="lenient", ignore_nulls=True)],
        )

    df = pl.DataFrame({"a": [1, None]})
    # Sanity: the base spec (no validators) accepts the null just fine.
    assert NullValidatorSpec.validate(df).height == 2

    # ignore_nulls=True (default Check semantics): a null passes.
    assert LenientNullValidatorSpec.validate(df).height == 2

    # ignore_nulls=False: a null is treated as a violation.
    with pytest.raises(ValidationError, match="strict"):
        StrictNullValidatorSpec.validate(df)


def test_validation_column_validators_false_bypasses_check():
    df = pl.DataFrame({"price": [1.00, 1.03], "quantity": [2, 3]})
    result = ProductValidatorSpec.validate(df, validate_validators=False)
    assert result.height == 2


def test_validation_column_validators_combine_with_other_errors():
    class ComboSpec(FrameSpec):
        price = ColSpec(
            pl.Float64, bounds=Bound(0.0, 100.0), validators=[pl.col("price") > 0]
        )

    df = pl.DataFrame({"price": [-1.0, 200.0]})
    with pytest.raises(ValidationError) as exc_info:
        ComboSpec.validate(df)

    err = exc_info.value
    assert len(err.errors) == 2
    err_str = str(err)
    assert "out of bounds" in err_str
    assert "validator" in err_str
