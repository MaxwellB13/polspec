"""What several command-line test files read."""

import polars as pl
import pytest


@pytest.fixture
def sample_data(tmp_path):
    df = pl.DataFrame(
        {
            "order_id": list(range(1, 201)),
            "status": ["NEW", "PAID", "SHIPPED"] * 66 + ["NEW", "NEW"],
            "total": [round(10.0 + i * 0.5, 2) for i in range(200)],
        }
    )
    path = tmp_path / "orders.parquet"
    df.write_parquet(path)
    return path
