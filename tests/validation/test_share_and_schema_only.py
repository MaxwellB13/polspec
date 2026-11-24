"""A finding's share of the rows, and a check of the schema alone -- both
since 1.4."""

from __future__ import annotations

import json

import polars as pl
import pytest
from polspec import (
    ColSpec,
    ForeignKey,
    FrameSpec,
    TableSpec,
    ValidationOptions,
    assert_valid,
    inspect,
    validate,
)

SPEC = TableSpec(
    "Orders",
    {
        "id": ColSpec(pl.Int64, unique=True),
        "total": ColSpec(pl.Float64, bounds=(0, 100)),
    },
)
FRAME = pl.DataFrame({"id": [1, 2, 3, 4, 4], "total": [10.0, 250.0, 30.0, 40.0, -1.0]})


def test_a_counted_finding_carries_its_share_of_the_rows():
    report = inspect(SPEC, FRAME)
    by_key = {f.key: f for f in report}
    assert (by_key["total__bounds"].count, by_key["total__bounds"].share) == (2, 0.4)
    assert (by_key["id__unique"].count, by_key["id__unique"].share) == (2, 0.4)


def test_a_structural_finding_has_no_share():
    report = inspect(SPEC, FRAME.with_columns(pl.lit(1).alias("extra")))
    extra = report.by_code("extra_columns")[0]
    assert (extra.count, extra.share) == (None, None)


def test_a_foreign_key_finding_has_a_share_too():
    class Customers(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)

    class Purchases(FrameSpec):
        customer_id = ColSpec(pl.Int64)
        __foreign_keys__ = [
            ForeignKey("customer_id", references=Customers, ref_columns="id")
        ]

    report = Purchases.inspect(
        pl.DataFrame({"customer_id": [1, 2, 9, 9]}),
        references={Customers: pl.DataFrame({"id": [1, 2]})},
    )
    (orphans,) = report.by_code("foreign_key")
    assert orphans.share == orphans.count / 4


def test_the_share_shows_in_every_rendering():
    report = inspect(SPEC, FRAME)
    assert "(40.0% of rows)" in str(report)
    assert "| 2 (40.0%) |" in report._repr_markdown_()
    found = json.loads(report.to_json())["findings"]
    assert {f["key"]: f["share"] for f in found}["total__bounds"] == 0.4


def test_schema_only_reads_the_schema_and_never_the_rows():
    def explode(_: pl.Series) -> pl.Series:
        raise AssertionError("schema_only collected the frame")

    lazy = (
        FRAME.lazy()
        .with_columns(pl.col("total").map_batches(explode, return_dtype=pl.Float64))
        .with_columns(pl.lit("x").alias("note"))
    )
    report = inspect(SPEC, lazy, schema_only=True)
    assert [f.code for f in report] == ["extra_columns"]
    assert inspect(SPEC, lazy.drop("note"), schema_only=True).passed


def test_schema_only_reports_a_wrong_or_missing_column():
    lazy = FRAME.lazy().with_columns(pl.col("total").cast(pl.String)).drop("id")
    report = inspect(SPEC, lazy, schema_only=True)
    assert sorted(f.code for f in report) == ["dtype", "missing_columns"]


def test_schema_only_runs_through_validate_and_assert_valid():
    bad_values = FRAME.lazy()  # the values fail; the schema does not
    assert isinstance(validate(SPEC, bad_values, schema_only=True), pl.LazyFrame)
    assert_valid(SPEC, bad_values, schema_only=True)
    with pytest.raises(AssertionError):
        assert_valid(SPEC, bad_values)


def test_schema_only_does_not_combine_with_parse_text():
    with pytest.raises(ValueError, match="schema_only"):
        ValidationOptions(schema_only=True, parse_text=True)
