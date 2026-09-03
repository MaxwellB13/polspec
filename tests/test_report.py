"""`to_markdown` and `to_mermaid`: the generated data dictionary and ER diagram."""

import polars as pl
from polspec import (
    Bound,
    Check,
    ColSpec,
    ForeignKey,
    FrameSpec,
)


def test_framespec_to_markdown_and_to_mermaid(tmp_path):
    class CustomerSpec(FrameSpec):
        customer_id = ColSpec(
            pl.Int64, unique=True, bounds=Bound(1, 1_000_000), tags="index"
        )
        tier = ColSpec(
            pl.Enum(["BRONZE", "SILVER", "GOLD"]), nullable=False, tags="segment"
        )
        score = ColSpec(pl.Float64, bounds=Bound(0.0, 100.0), nullable=True)
        country = ColSpec(pl.String, choices=["US", "UK", "DE", "FR"], tags="geo")
        created_date = ColSpec(pl.Date, nullable=False, tags="temporal")

        __unique_together__ = [("customer_id", "country")]
        __checks__ = [
            Check(
                pl.col("score") >= 0.0,
                name="score_non_negative",
                description="Credit score must be non-negative if present",
            )
        ]

    # 1. to_markdown() without path
    md_str = CustomerSpec.to_markdown(title="Customer Data Dictionary")
    assert "# Customer Data Dictionary" in md_str
    assert "## Overview" in md_str
    assert "| `customer_id` |" in md_str
    assert "| `tier` |" in md_str
    assert "score_non_negative" in md_str
    assert "['customer_id', 'country']" in md_str
    assert "`index`" in md_str
    assert "`segment`" in md_str

    # 2. to_markdown() with file path
    md_file = tmp_path / "customer_dict.md"
    written_md = CustomerSpec.to_markdown(md_file)
    assert md_file.exists()
    assert md_file.read_text(encoding="utf-8") == written_md

    # 3. to_mermaid() without path
    mermaid_str = CustomerSpec.to_mermaid()
    assert "erDiagram" in mermaid_str
    assert "CustomerSpec {" in mermaid_str
    assert "Int64 customer_id PK" in mermaid_str
    assert "Enum tier" in mermaid_str
    assert "tags: [segment]" in mermaid_str
    assert "bounds: [1, 1000000]" in mermaid_str

    # 4. to_mermaid() with file path
    mermaid_file = tmp_path / "customer_erd.mmd"
    written_mermaid = CustomerSpec.to_mermaid(mermaid_file)
    assert mermaid_file.exists()
    assert mermaid_file.read_text(encoding="utf-8") == written_mermaid

    # 5. to_mermaid() with quotes in choices / tags / fk names
    class QuotedSpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        flag = ColSpec(pl.String, choices=['a"1', 'b"2'], tags=['geo"zone'])
        __foreign_keys__ = [ForeignKey("id", references="self", name='quoted"fk')]

    quoted_mmd = QuotedSpec.to_mermaid()
    # Ensure double quotes inside attributes are replaced to avoid breaking mermaid ER syntax
    assert "choices: [a'1, b'2]" in quoted_mmd
    assert "tags: [geo'zone]" in quoted_mmd
    assert "quoted'fk" in quoted_mmd


class CustomerFkSpec(FrameSpec):
    id = ColSpec(pl.Int64, unique=True)
    code = ColSpec(pl.String, unique=True)


def test_framespec_to_markdown_and_to_mermaid_with_foreign_keys():
    class OrderFkSpec(FrameSpec):
        order_id = ColSpec(pl.Int64, unique=True)
        customer_id = ColSpec(pl.Int64, nullable=True)
        __foreign_keys__ = [
            ForeignKey("customer_id", references=CustomerFkSpec, ref_columns="id")
        ]

    md = OrderFkSpec.to_markdown()
    assert "**Foreign Keys:** 1 key(s)" in md
    assert "### Foreign Keys" in md
    assert "fk_customer_id__CustomerFkSpec" in md
    assert "['customer_id']" in md
    assert "CustomerFkSpec.['id']" in md

    mmd = OrderFkSpec.to_mermaid()
    assert "Int64 customer_id FK" in mmd
    assert 'CustomerFkSpec ||--o{ OrderFkSpec : "fk_customer_id__CustomerFkSpec"' in mmd

    class SelfFkSpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        parent_id = ColSpec(pl.Int64, nullable=True)
        __foreign_keys__ = [
            ForeignKey("parent_id", references="self", ref_columns="id")
        ]

    self_mmd = SelfFkSpec.to_mermaid()
    assert 'SelfFkSpec ||--o{ SelfFkSpec : "fk_parent_id__self"' in self_mmd


def test_framespec_to_markdown_lists_column_validators():
    class ValidatedSpec(FrameSpec):
        price = ColSpec(
            pl.Float64,
            validators=[
                Check(
                    pl.col("price") > 0,
                    name="price_positive",
                    description="Price must be positive",
                )
            ],
        )

    md = ValidatedSpec.to_markdown()
    assert "### Column Validators" in md
    assert "Column `price`" in md
    assert "price_positive" in md
    assert "Price must be positive" in md
