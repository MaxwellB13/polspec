"""`validate()` and distinctness: `unique=True`, `__unique_together__`,
and the nulls both exempt.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest
from polspec import (
    Check,
    ColSpec,
    FrameSpec,
    ValidationError,
)

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


def test_validation_composite_uniqueness_with_nulls():
    class CompNullSpec(FrameSpec):
        tenant_id = ColSpec(pl.Int64, nullable=True)
        user_id = ColSpec(pl.Int64, nullable=True)
        val = ColSpec(pl.String)

        __unique_together__ = [("tenant_id", "user_id")]

    # Multiple rows with nulls in composite unique columns should not trigger unique violation
    df_nulls = pl.DataFrame(
        {
            "tenant_id": [1, 1, None, None, 2],
            "user_id": [10, None, 10, None, 20],
            "val": ["a", "b", "c", "d", "e"],
        }
    )
    res = CompNullSpec.validate(df_nulls)
    assert res.height == 5

    # But non-null duplicate pairs still fail
    df_dups = pl.DataFrame(
        {
            "tenant_id": [1, 1, None],
            "user_id": [10, 10, None],
            "val": ["a", "b", "c"],
        }
    )
    with pytest.raises(ValidationError, match="Composite unique key"):
        CompNullSpec.validate(df_dups)
