"""`validate()` and referential integrity: a foreign key's parent frame,
self-references, composite keys and the nulls a key exempts.
"""

from __future__ import annotations

import polars as pl
import pytest
from polspec import (
    ColSpec,
    ForeignKey,
    FrameSpec,
    ValidationError,
)

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
