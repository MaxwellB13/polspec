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
