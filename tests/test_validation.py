from datetime import datetime

import polars as pl
import pytest
from polspec import (
    Bound,
    Check,
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSchema,
    FrameSpec,
    ValidationError,
)


class ProduceInventory(FrameSchema):
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


def test_validation_colrule():
    class RuleSpec(FrameSpec):
        status = ColSpec(dtype=pl.Enum(["active", "inactive"]), nullable=False)
        discount = ColSpec(
            dtype=pl.Float64,
            bounds=Bound(0.0, 1.0),
            nullable=False,
            rules=(
                ColRule(
                    when={"column": "status", "equals": "inactive"},
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
                ColRule(when={"column": "tier", "equals": "gold"}, choices=[100]),
                ColRule(
                    when={"column": "tier", "in": ["gold", "silver"]}, choices=[50]
                ),
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
# Composite Keys (__unique_together__) & Single Unique
# =====================================================================


class SessionEventSpec(FrameSpec):
    user_id = ColSpec(pl.Int64, nullable=False, tags="index")
    session_id = ColSpec(pl.Int64, nullable=False, tags="index")
    event_id = ColSpec(pl.String, unique=True, nullable=False)
    event_time = ColSpec(pl.Datetime, nullable=False)
    payload = ColSpec(pl.String, nullable=True)

    __unique_together__ = [("user_id", "session_id", "event_time")]


def test_validation_single_and_composite_uniqueness_success():
    df = pl.DataFrame(
        {
            "user_id": [1, 1, 2],
            "session_id": [100, 101, 100],
            "event_id": ["E1", "E2", "E3"],
            "event_time": [
                datetime(2025, 1, 1, 0, 0),
                datetime(2025, 1, 1, 0, 0),
                datetime(2025, 1, 1, 0, 0),
            ],
            "payload": ["click", "view", "click"],
        }
    )
    result = SessionEventSpec.validate(df)
    assert result.height == 3


def test_validation_single_column_uniqueness_failure():
    df = pl.DataFrame(
        {
            "user_id": [1, 2],
            "session_id": [100, 200],
            "event_id": ["E1", "E1"],  # Duplicate in unique column
            "event_time": [datetime(2025, 1, 1, 0, 0), datetime(2025, 1, 1, 0, 1)],
            "payload": ["a", "b"],
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        SessionEventSpec.validate(df)

    err_str = str(exc_info.value)
    assert "Column 'event_id': unique column contains 2 duplicate value(s)" in err_str
    assert "E1" in err_str


def test_validation_composite_uniqueness_failure():
    df = pl.DataFrame(
        {
            "user_id": [1, 1, 2],
            "session_id": [100, 100, 100],
            "event_id": ["E1", "E2", "E3"],
            "event_time": [
                datetime(2025, 1, 1, 0, 0),
                datetime(
                    2025, 1, 1, 0, 0
                ),  # Duplicate (user_id=1, session_id=100, event_time)
                datetime(2025, 1, 1, 0, 0),
            ],
            "payload": ["a", "b", "c"],
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        SessionEventSpec.validate(df)

    err_str = str(exc_info.value)
    assert (
        "Composite unique key ['user_id', 'session_id', 'event_time'] violated"
        in err_str
    )
    assert "found 2 duplicate row(s)" in err_str


def test_validation_nullable_unique_column_allows_multiple_nulls():
    class AnonSpec(FrameSpec):
        id_col = ColSpec(pl.Int64, unique=True, nullable=True)
        val = ColSpec(pl.Int64)

        __checks__ = [Check(pl.col("val") > 0)]

    # Nullable unique column with multiple nulls and distinct non-nulls should pass
    df_nulls_unique = pl.DataFrame(
        {
            "id_col": [1, None, None, 2],
            "val": [10, 20, 30, 40],
        }
    )
    res = AnonSpec.validate(df_nulls_unique)
    assert res.height == 4

    # Nullable unique column with duplicate non-nulls should fail
    df_dup = pl.DataFrame(
        {
            "id_col": [1, 1, None, 2],
            "val": [10, 20, 30, 40],
        }
    )
    with pytest.raises(
        ValidationError, match="unique column contains 2 duplicate value"
    ):
        AnonSpec.validate(df_dup)


def test_validation_lazyframe_and_streaming_checks_and_uniqueness():
    class TestSpec(FrameSpec):
        id_col = ColSpec(pl.Int64, unique=True)
        val_1 = ColSpec(pl.Int64)
        val_2 = ColSpec(pl.Int64)

        __unique_together__ = [("val_1", "val_2")]
        __checks__ = [Check(pl.col("val_2") >= pl.col("val_1"), name="v2_gte_v1")]

    df = pl.DataFrame(
        {
            "id_col": [1, 2, 3],
            "val_1": [10, 20, 30],
            "val_2": [15, 25, 35],
        }
    )

    # Lazy validation
    lf = df.lazy()
    validated_lf = TestSpec.validate(lf)
    assert isinstance(validated_lf, pl.LazyFrame)
    assert validated_lf.collect().height == 3

    # Streaming lazy validation
    streaming_res = TestSpec.validate(lf, streaming=True)
    assert isinstance(streaming_res, pl.LazyFrame)
    assert streaming_res.collect().height == 3


# =====================================================================
# Referential Integrity (ForeignKey / __foreign_keys__)
# =====================================================================


class CustomerSpec(FrameSpec):
    id = ColSpec(pl.Int64, unique=True)
    name = ColSpec(pl.String)


class OrderSpec2(FrameSpec):
    order_id = ColSpec(pl.Int64, unique=True)
    customer_id = ColSpec(pl.Int64, nullable=True)

    __foreign_keys__ = [
        ForeignKey("customer_id", references=CustomerSpec, ref_columns="id"),
    ]


CUSTOMERS_DF = pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})


def test_validation_foreign_key_success_and_null_exempt():
    # A null customer_id is exempt from the referential check.
    df = pl.DataFrame({"order_id": [1, 2, 3], "customer_id": [1, 2, None]})
    result = OrderSpec2.validate(df, references={CustomerSpec: CUSTOMERS_DF})
    assert result.height == 3


def test_validation_foreign_key_failure():
    df = pl.DataFrame({"order_id": [1, 2], "customer_id": [1, 99]})
    with pytest.raises(ValidationError) as exc_info:
        OrderSpec2.validate(df, references={CustomerSpec: CUSTOMERS_DF})

    err_str = str(exc_info.value)
    assert "ForeignKey 'fk_customer_id__CustomerSpec' violated" in err_str
    assert "found 1 row(s) with no matching parent record" in err_str
    assert "99" in err_str


def test_validation_foreign_key_accepts_lazyframe_reference():
    df = pl.DataFrame({"order_id": [1, 2], "customer_id": [1, 2]})
    result = OrderSpec2.validate(df, references={CustomerSpec: CUSTOMERS_DF.lazy()})
    assert result.height == 2


def test_validation_foreign_key_missing_references_raises_value_error():
    df = pl.DataFrame({"order_id": [1], "customer_id": [1]})
    with pytest.raises(ValueError, match="no DataFrame for it was supplied"):
        OrderSpec2.validate(df)


def test_validation_foreign_key_validate_foreign_keys_false_bypasses_check():
    df = pl.DataFrame({"order_id": [1, 2], "customer_id": [1, 99]})
    # No references supplied, but the check is disabled so it never looks for one.
    result = OrderSpec2.validate(df, validate_foreign_keys=False)
    assert result.height == 2


def test_validation_foreign_key_self_reference():
    class EmployeeSpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        manager_id = ColSpec(pl.Int64, nullable=True)
        __foreign_keys__ = [
            ForeignKey("manager_id", references="self", ref_columns="id")
        ]

    df_valid = pl.DataFrame({"id": [1, 2, 3], "manager_id": [None, 1, 1]})
    assert EmployeeSpec.validate(df_valid).height == 3

    df_invalid = pl.DataFrame({"id": [1, 2], "manager_id": [1, 999]})
    with pytest.raises(ValidationError, match="fk_manager_id__self"):
        EmployeeSpec.validate(df_invalid)


def test_validation_foreign_key_composite():
    class RegionSpec(FrameSpec):
        tenant = ColSpec(pl.Int64)
        region_id = ColSpec(pl.Int64)
        __unique_together__ = [("tenant", "region_id")]

    class StoreSpec(FrameSpec):
        tenant = ColSpec(pl.Int64)
        region_id = ColSpec(pl.Int64)
        __foreign_keys__ = [
            ForeignKey(
                ["tenant", "region_id"],
                references=RegionSpec,
                ref_columns=["tenant", "region_id"],
            )
        ]

    regions_df = pl.DataFrame({"tenant": [1, 1, 2], "region_id": [10, 20, 10]})

    stores_ok = pl.DataFrame({"tenant": [1, 2], "region_id": [10, 10]})
    assert (
        StoreSpec.validate(stores_ok, references={RegionSpec: regions_df}).height == 2
    )

    stores_bad = pl.DataFrame({"tenant": [1], "region_id": [99]})
    with pytest.raises(ValidationError, match="ForeignKey"):
        StoreSpec.validate(stores_bad, references={RegionSpec: regions_df})


def test_validation_foreign_key_missing_ref_columns_in_parent_raises():
    bad_parent = pl.DataFrame({"other_col": [1, 2]})
    df = pl.DataFrame({"order_id": [1], "customer_id": [1]})
    with pytest.raises(ValueError, match="not present in the referenced DataFrame"):
        OrderSpec2.validate(df, references={CustomerSpec: bad_parent})
