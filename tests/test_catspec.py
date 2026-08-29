from __future__ import annotations

import tempfile
from pathlib import Path
import polars as pl
import pytest

from polspec import Bound, CatSpec, ColSpec, FrameSpec


def test_catspec_basic_access():
    cats = CatSpec(
        enums={"ORDER_STATUS": ["PENDING", "PROCESSING", "SHIPPED", "DELIVERED"]},
        categoricals={
            "CURRENCY": pl.Categories("CURRENCY", physical=pl.UInt8),
            "COUNTRY": {"name": "COUNTRY", "physical": "UInt16", "namespace": "geo"},
            "SIMPLE": "UInt8",
        },
    )

    # Attribute lookup
    assert cats.ORDER_STATUS == ["PENDING", "PROCESSING", "SHIPPED", "DELIVERED"]
    assert isinstance(cats.CURRENCY, pl.Categories)
    assert cats.CURRENCY.name() == "CURRENCY"
    assert cats.CURRENCY.physical() == pl.UInt8

    assert cats.COUNTRY.name() == "COUNTRY"
    assert cats.COUNTRY.physical() == pl.UInt16
    assert cats.COUNTRY.namespace() == "geo"

    assert cats.SIMPLE.physical() == pl.UInt8

    # Dict subscripting
    assert cats["ORDER_STATUS"] == ["PENDING", "PROCESSING", "SHIPPED", "DELIVERED"]
    assert cats["CURRENCY"].name() == "CURRENCY"

    # Container ops
    assert "ORDER_STATUS" in cats
    assert "CURRENCY" in cats
    assert "NONEXISTENT" not in cats
    assert len(cats) == 4
    assert set(iter(cats)) == {"ORDER_STATUS", "CURRENCY", "COUNTRY", "SIMPLE"}

    # Property dictionaries
    assert "ORDER_STATUS" in cats.enums
    assert "CURRENCY" in cats.categoricals

    # Error handling
    with pytest.raises(AttributeError, match="has no Enum or Categorical"):
        _ = cats.UNKNOWN_KEY

    with pytest.raises(KeyError, match="has no Enum or Categorical"):
        _ = cats["UNKNOWN_KEY"]


def test_catspec_dtype_accessors():
    cats = CatSpec(
        enums={"STATUS": ["A", "B", "C"]},
        categoricals={"CURRENCY": pl.Categories("CURRENCY", physical=pl.UInt8)},
    )

    # Callable access
    enum_dt = cats.enum("STATUS")
    assert isinstance(enum_dt, pl.Enum)
    assert enum_dt.categories.to_list() == ["A", "B", "C"]

    cat_dt = cats.categorical("CURRENCY")
    assert isinstance(cat_dt, pl.Categorical)
    assert cat_dt.categories.name() == "CURRENCY"

    # Attribute property access
    assert cats.enum.STATUS == enum_dt
    assert cats.categorical.CURRENCY == cat_dt

    # Subscript access
    assert cats.enum["STATUS"] == enum_dt
    assert cats.categorical["CURRENCY"] == cat_dt


def test_catspec_framespec_generation_and_joins():
    CATEGORIES = CatSpec(
        enums={"STATUS": ["OPEN", "CLOSED"]},
        categoricals={
            "CURRENCY": pl.Categories("CURRENCY", physical=pl.UInt8),
            "FRUITS": pl.Categories("FRUITS", physical=pl.UInt8),
        },
    )

    class Items(FrameSpec):
        status = ColSpec(dtype=pl.Enum(CATEGORIES.STATUS), nullable=False)
        currency = ColSpec(dtype=pl.Categorical(CATEGORIES.CURRENCY), nullable=False)
        fruit = ColSpec(dtype=CATEGORIES.categorical.FRUITS, nullable=True)

    class Rates(FrameSpec):
        status = ColSpec(dtype=CATEGORIES.enum.STATUS, nullable=False)
        currency = ColSpec(dtype=CATEGORIES.categorical.CURRENCY, nullable=False)
        rate = ColSpec(dtype=pl.Float64, bounds=Bound(1.0, 10.0))

    items = Items.generate(n=100, seed=42)
    rates = Rates.generate(n=50, seed=42)

    assert items.schema["status"] == pl.Enum(["OPEN", "CLOSED"])
    assert items.schema["currency"] == pl.Categorical(CATEGORIES.CURRENCY)

    # Physical join on currency must succeed cleanly
    joined = items.join(rates, on="currency", how="inner")
    assert "rate" in joined.columns
    assert joined.height > 0


def test_catspec_yaml_roundtrip():
    cats = CatSpec(
        enums={"STATUS": ["OPEN", "IN_PROGRESS", "CLOSED"]},
        categoricals={
            "CURRENCY": {"name": "CURRENCY", "physical": "UInt8", "namespace": "finance", "categories": ["USD", "EUR"]},
            "PRODUCT": {"name": "PRODUCT", "physical": "UInt16"},
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "categories.yaml"
        cats.to_yaml(yaml_path)

        loaded = CatSpec.from_yaml(yaml_path)
        assert loaded.STATUS == ["OPEN", "IN_PROGRESS", "CLOSED"]
        assert loaded.CURRENCY.name() == "CURRENCY"
        assert loaded.CURRENCY.physical() == pl.UInt8
        assert loaded.CURRENCY.namespace() == "finance"
        assert loaded.PRODUCT.physical() == pl.UInt16
        assert loaded.get_choices("CURRENCY") == ["USD", "EUR"]


def test_catspec_flat_format():
    flat_data = {
        "ORDER_STATUS": ["PENDING", "SHIPPED"],
        "CURRENCY": {"physical": "UInt8"},
        "SIMPLE_CAT": "UInt16",
    }
    loaded = CatSpec.from_dict(flat_data)
    assert loaded.ORDER_STATUS == ["PENDING", "SHIPPED"]
    assert loaded.CURRENCY.physical() == pl.UInt8
    assert loaded.SIMPLE_CAT.physical() == pl.UInt16


def test_framespec_from_yaml_with_catspec():
    with tempfile.TemporaryDirectory() as tmpdir:
        cat_file = Path(tmpdir) / "cats.yaml"
        cat_spec = CatSpec(
            enums={"TIER": ["BRONZE", "SILVER", "GOLD"]},
            categoricals={"CURRENCY": pl.Categories("CURRENCY", physical=pl.UInt8)},
        )
        cat_spec.to_yaml(cat_file)

        spec_yaml = Path(tmpdir) / "spec.yaml"
        spec_yaml.write_text("""
name: CustomerOrder
categories: ./cats.yaml
columns:
  tier:
    dtype:
      Enum: $categories.TIER
  currency:
    dtype:
      Categorical: $categories.CURRENCY
  amount:
    dtype: Float64
""")

        # 1. Automatic categories file resolution from YAML
        LoadedSpec = FrameSpec.from_yaml(spec_yaml)
        df = LoadedSpec.generate(10, seed=1)
        assert df.schema["tier"] == pl.Enum(["BRONZE", "SILVER", "GOLD"])
        assert df.schema["currency"] == pl.Categorical(cat_spec.CURRENCY)

        # 2. Explicit categories passed as parameter
        LoadedSpec2 = FrameSpec.from_yaml(spec_yaml, categories=cat_spec)
        df2 = LoadedSpec2.generate(10, seed=1)
        assert df2.schema["tier"] == pl.Enum(["BRONZE", "SILVER", "GOLD"])


def test_catspec_from_dataframe_and_framespec():
    df = pl.DataFrame({
        "status": pl.Series(["OPEN", "CLOSED", "OPEN"], dtype=pl.Enum(["OPEN", "CLOSED"])),
        "currency": pl.Series(["USD", "EUR", "USD"], dtype=pl.Categorical(pl.Categories("CURRENCY", physical=pl.UInt8))),
        "val": [1, 2, 3],
    })

    # from_dataframe
    cats_from_df = CatSpec.from_dataframe(df)
    assert cats_from_df.status == ["OPEN", "CLOSED"]
    assert cats_from_df.currency.name() == "CURRENCY"
    assert cats_from_df.currency.physical() == pl.UInt8
    assert "val" not in cats_from_df

    # LazyFrame support
    cats_from_lazy = CatSpec.from_dataframe(df.lazy())
    assert cats_from_lazy.status == ["OPEN", "CLOSED"]

    # from_framespec & FrameSpec.catspec / generate_catspec / write_catspec
    class MySpec(FrameSpec):
        status = ColSpec(dtype=pl.Enum(["PENDING", "COMPLETED"]))
        currency = ColSpec(dtype=pl.Categorical(pl.Categories("CURRENCY", physical=pl.UInt16)), choices=["USD", "EUR"])
        amount = ColSpec(dtype=pl.Float64)

    cats_from_spec = MySpec.catspec()
    assert cats_from_spec.status == ["PENDING", "COMPLETED"]
    assert cats_from_spec.currency.name() == "CURRENCY"
    assert cats_from_spec.currency.physical() == pl.UInt16
    assert cats_from_spec.get_choices("currency") == ["USD", "EUR"]

    # Alias check
    assert MySpec.generate_catspec().enums == cats_from_spec.enums

    # write_catspec check
    with tempfile.TemporaryDirectory() as tmpdir:
        out_yaml = Path(tmpdir) / "extracted.yaml"
        MySpec.write_catspec(out_yaml)
        loaded = CatSpec.from_yaml(out_yaml)
        assert loaded.status == ["PENDING", "COMPLETED"]
        assert loaded.currency.physical() == pl.UInt16


def test_catspec_heuristic_inference():
    # DataFrame with:
    # 1. low cardinality string (< 30) -> should infer as Enum
    # 2. medium cardinality string (> 30, ratio < 0.20) -> should infer as Categorical
    # 3. ID column -> excluded by default
    # 4. integer column -> ignored
    n = 1000
    statuses = ["PENDING", "PROCESSING", "COMPLETED", "FAILED"] * 250
    countries = [f"COUNTRY_{i % 50}" for i in range(n)]
    order_ids = [f"ORD_{i:06d}" for i in range(n)]
    amounts = [float(i) for i in range(n)]

    df = pl.DataFrame({
        "status": statuses,
        "country": countries,
        "order_id": order_ids,
        "amount": amounts,
    })

    cats = CatSpec.infer_from_dataframe(df)

    # status has 4 unique values <= 30 -> Enum
    assert "status" in cats.enums
    assert set(cats.status) == {"PENDING", "PROCESSING", "COMPLETED", "FAILED"}

    # country has 50 unique values (ratio 50/1000 = 0.05 <= 0.20) -> Categorical (UInt8 because 50 < 256)
    assert "country" in cats.categoricals
    assert cats.country.physical() == pl.UInt8

    # order_id matches exclude_patterns (r".*_id$") -> skipped
    assert "order_id" not in cats

    # amount is float -> skipped
    assert "amount" not in cats

    # include_columns override
    cats_override = CatSpec.infer(df, include_columns=["order_id"])
    assert "order_id" in cats_override.enums or "order_id" in cats_override.categoricals


def test_framespec_infer_catspec_and_transformation():
    class RawOrders(FrameSpec):
        status = ColSpec(dtype=pl.String, choices=["NEW", "PAID", "CANCELLED"])
        tag = ColSpec(dtype=pl.String, choices=[f"TAG_{i}" for i in range(40)])
        user_id = ColSpec(dtype=pl.String)
        long_notes = ColSpec(dtype=pl.String, string_length=Bound(0, 1000))
        plain_str = ColSpec(dtype=pl.String)

    # 1. Infer CatSpec from FrameSpec schema
    cats = RawOrders.infer_catspec(max_enum_cardinality=10)
    assert "status" in cats.enums
    assert cats.status == ["NEW", "PAID", "CANCELLED"]

    assert "tag" in cats.categoricals
    assert cats.tag.physical() == pl.UInt8
    assert len(cats.get_choices("tag")) == 40

    # user_id matches exclude_patterns (r".*_id$") -> skipped
    assert "user_id" not in cats
    # long_notes max > 255 -> skipped
    assert "long_notes" not in cats

    # 2. Transform FrameSpec using with_inferred_catspec
    OptimizedOrders = RawOrders.with_inferred_catspec(catspec=cats)
    assert OptimizedOrders.schema()["status"] == pl.Enum(["NEW", "PAID", "CANCELLED"])
    assert isinstance(OptimizedOrders.schema()["tag"], pl.Categorical)
    assert OptimizedOrders.schema()["user_id"] == pl.String
    assert OptimizedOrders.schema()["long_notes"] == pl.String

    # 3. Generating data from Optimized FrameSpec
    gen_df = OptimizedOrders.generate(50, seed=42)
    assert gen_df.schema["status"] == pl.Enum(["NEW", "PAID", "CANCELLED"])
    assert gen_df["status"].is_in(["NEW", "PAID", "CANCELLED"]).all()


def test_catspec_case_insensitivity_and_resolution():
    cats = CatSpec(
        enums={"order_status": ["A", "B"]},
        categoricals={"CURRENCY": pl.Categories("CURRENCY", physical=pl.UInt8)},
    )

    # Mixed case lookups
    assert cats.get_enum("ORDER_STATUS") == ["A", "B"]
    assert cats.get_enum("order_status") == ["A", "B"]
    assert cats.enum("ORDER_STATUS") == pl.Enum(["A", "B"])
    assert cats.ORDER_STATUS == ["A", "B"]
    assert cats.order_status == ["A", "B"]
    assert "ORDER_STATUS" in cats
    assert "order_status" in cats

    assert cats.get_categorical("currency").physical() == pl.UInt8
    assert cats.get_categorical("CURRENCY").physical() == pl.UInt8
    assert cats.categorical("currency").categories.name() == "CURRENCY"
    assert cats.currency.physical() == pl.UInt8
    assert cats.CURRENCY.physical() == pl.UInt8
    assert "currency" in cats
    assert "CURRENCY" in cats


def test_framespec_infer_with_live_dataframe():
    df = pl.DataFrame({
        "status": ["A", "B", "A", "B"],
        "region": ["US", "EU", "APAC", "LATAM"],
        "uuid": ["123", "456", "789", "012"],
    })

    class SchemaSpec(FrameSpec):
        status = ColSpec(dtype=pl.String)
        region = ColSpec(dtype=pl.String)
        uuid = ColSpec(dtype=pl.String)

    # Infer using DataFrame data through FrameSpec.infer_catspec(data=df)
    cats = SchemaSpec.infer_catspec(data=df, max_enum_cardinality=3)
    assert "status" in cats.enums
    assert "region" in cats.categoricals  # 4 unique > max_enum_cardinality (3)
    assert "uuid" not in cats

    # with_inferred_catspec with live data
    Optimized = SchemaSpec.with_inferred_catspec(data=df, max_enum_cardinality=3)
    assert isinstance(Optimized.schema()["status"], pl.Enum)
    assert isinstance(Optimized.schema()["region"], pl.Categorical)
    assert Optimized.schema()["uuid"] == pl.String
