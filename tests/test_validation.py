from datetime import date
import polars as pl
import pytest
from polspec import Bound, ColRule, ColSpec, DfSchema, DfSpec, ValidationError


class ProduceInventory(DfSchema):
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
    class RuleSpec(DfSpec):
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
