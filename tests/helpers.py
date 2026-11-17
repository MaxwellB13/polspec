"""What several test files need, said once.

On `pythonpath` (see `[tool.pytest.ini_options]`), so a test imports it as
`from helpers import spec_for`.
"""

from __future__ import annotations

import textwrap

import polars as pl
from polspec import ColSpec, FrameSpec
from polspec.cli import main


def spec_for(column: ColSpec, name: str = "Single") -> type[FrameSpec]:
    """A FrameSpec of one column, `c`, named `name`."""
    return type(name, (FrameSpec,), {"__columns__": {"c": column}})


def run_cli(*args: object) -> int:
    """`polspec` with `args`, each as text, as the command line runs it:
    the exit status."""
    return main([str(a) for a in args])


def write_orders_specs(tmp_path):
    """A `.py` source of two specs, `Customers` and `Orders`, with a key
    between them: what the command-line tests read."""
    source = tmp_path / "specs.py"
    source.write_text(
        textwrap.dedent(
            """
            import polars as pl
            from polspec import ColSpec, ForeignKey, FrameSpec

            class Customers(FrameSpec):
                id = ColSpec(pl.Int64, unique=True)

            class Orders(FrameSpec):
                order_id = ColSpec(pl.Int64, unique=True)
                customer_id = ColSpec(pl.Int64)
                total = ColSpec(pl.Float64, bounds=(0.0, 100.0))
                __foreign_keys__ = [
                    ForeignKey("customer_id", references=Customers, ref_columns="id")
                ]
            """
        )
    )
    customers = tmp_path / "customers.parquet"
    pl.DataFrame({"id": [1, 2, 3]}).write_parquet(customers)
    return source, customers


def write_yaml_spec(tmp_path, name="Orders"):
    """An `Orders` spec as a YAML file, named after `name`."""

    class Orders(FrameSpec):
        order_id = ColSpec(pl.Int64, bounds=(1, None), unique=True)
        status = ColSpec(pl.Enum(["NEW", "PAID"]))
        total = ColSpec(pl.Float64, bounds=(0.0, 100.0))
        placed = ColSpec(pl.Date, nullable=True, null_probability=0.2)

    path = tmp_path / f"{name.lower()}.yaml"
    Orders.to_yaml(path)
    return path
