"""The Python / Rust boundary.

A column crosses into the engine as a `ColumnPlan` -- typed, validated at
construction -- and comes back as its physical form: an integer for temporal
dtypes, indices for anything with a finite domain. These tests pin what that
boundary promises: exact integer bounds, typed choices, per-column seeds, one
source of truth for distribution parameters, and a stub that matches the
module.
"""

import ast
import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest
from polspec import ColRule, ColSpec, FrameSpec, GenerationError, _ffi, col
from polspec.distributions import DISTRIBUTIONS
from polspec.engine import _plan_column

# ---------------------------------------------------------------------------
# ColumnPlan
# ---------------------------------------------------------------------------


def test_plan_is_validated_at_construction_and_names_the_column():
    plan = _ffi.column_plan("amount", "int64", min=-5, max=10)
    assert plan.kind == "int64" and (plan.min, plan.max) == (-5, 10)
    assert "amount" in repr(plan) and "int64" in repr(plan)

    with pytest.raises(GenerationError, match="'varchar' for column 'amount'"):
        _ffi.column_plan("amount", "varchar")
    with pytest.raises(GenerationError, match="null_probability for column 'amount'"):
        _ffi.column_plan("amount", "int64", nullable=True, null_probability=1.5)
    with pytest.raises(GenerationError, match="must match number of categories"):
        _ffi.column_plan("amount", "index", n_categories=3, weights=[1.0, 1.0])
    with pytest.raises(GenerationError, match="must be finite"):
        _ffi.column_plan("amount", "float64", min=float("-inf"))


def test_integer_limits_keep_every_bit():
    plan = _ffi.column_plan("big", "uint64", min=2**64 - 16, max=2**64 - 1)
    assert plan.min == 2**64 - 16 and plan.max == 2**64 - 1
    plan = _ffi.column_plan("neg", "int64", min=-(2**63), max=2**53 + 1)
    assert plan.min == -(2**63) and plan.max == 2**53 + 1
    df = _ffi.generate_dataframe([plan], 2_000, 1)
    assert df["neg"].dtype == pl.Int64


def test_the_stub_matches_the_module():
    from polspec import _polspec

    stub = Path(_polspec.__file__).with_suffix(".pyi")
    if not stub.exists():  # an installed wheel places the stub beside the package
        stub = Path(__file__).parents[1] / "python" / "polspec" / "_polspec.pyi"
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    declared = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    exported = {name for name in dir(_polspec) if not name.startswith("_")}
    assert declared == exported

    plan_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    stub_props = {
        n.name
        for n in plan_cls.body
        if isinstance(n, ast.FunctionDef) and n.name != "__init__"
    }
    real_props = {name for name in dir(_polspec.ColumnPlan) if not name.startswith("_")}
    assert stub_props == real_props


def test_python_and_rust_agree_on_distribution_parameters():
    assert set(_ffi.distributions()) == set(DISTRIBUTIONS)
    for name, params in DISTRIBUTIONS.items():
        assert _ffi.distribution_params(name) == [
            (p.name, p.default, p.positive) for p in params
        ], name


def test_the_package_imports_without_the_extension(monkeypatch):
    import importlib

    import polspec

    # Make the extension unimportable: no module in sys.modules, no attribute
    # on the package for `from polspec import _polspec` to find.
    monkeypatch.setitem(sys.modules, "polspec._polspec", None)
    monkeypatch.delattr(polspec, "_polspec", raising=False)
    monkeypatch.delitem(sys.modules, "polspec.validation", raising=False)

    importlib.import_module("polspec.validation")  # no extension needed
    with pytest.raises(ImportError, match="maturin develop"):
        _ffi.generate_dataframe([], 1, 1)


# ---------------------------------------------------------------------------
# What a column becomes on the way in and back
# ---------------------------------------------------------------------------


def test_finite_domains_cross_as_indices_and_come_back_typed():
    plan, domain = _plan_column(
        "when",
        ColSpec(
            pl.Datetime("us"),
            choices=[dt.datetime(2024, 1, 1), dt.datetime(2025, 1, 1)],
        ),
    )
    assert plan.kind == "index" and plan.n_categories == 2
    assert domain.dtype == pl.Datetime("us")

    plan, domain = _plan_column(
        "status", ColSpec(pl.Enum(["NEW", "PAID"]), weights=[1.0, 3.0])
    )
    assert plan.kind == "index" and plan.weights == [1.0, 3.0]
    assert domain.dtype == pl.Enum(["NEW", "PAID"])

    plan, domain = _plan_column("flag", ColSpec(pl.Boolean, weights=[0.25, 0.75]))
    assert plan.kind == "bool" and domain is None and plan.p_true == 0.75

    plan, domain = _plan_column(
        "day", ColSpec(pl.Date, bounds=(dt.date(2024, 1, 1), dt.date(2024, 1, 31)))
    )
    assert plan.kind == "int32" and (plan.min, plan.max) == (19723, 19753)


def test_typed_choices_survive_generation_and_validation():
    class Spec(FrameSpec):
        when = ColSpec(
            pl.Datetime("us"),
            choices=[dt.datetime(2024, 1, 1, 12), dt.datetime(2025, 6, 30)],
        )
        blob = ColSpec(pl.Binary, choices=[b"\x00\x01", b"\xff"])
        flag = ColSpec(pl.Boolean, choices=[True])
        code = ColSpec(pl.String, choices=[True, "True"])  # distinct in String
        n = ColSpec(pl.Int64, choices=[1, 2, 3], nullable=True, null_probability=0.5)

    df = Spec.generate(500, seed=1)
    assert set(df["when"].to_list()) == {
        dt.datetime(2024, 1, 1, 12),
        dt.datetime(2025, 6, 30),
    }
    assert set(df["blob"].to_list()) == {b"\x00\x01", b"\xff"}
    assert df["flag"].all()
    assert set(df["code"].to_list()) == {"true", "True"}
    assert df["n"].null_count() > 100 and set(df["n"].drop_nulls().to_list()) <= {
        1,
        2,
        3,
    }
    Spec.validate(df)


def test_seeds_follow_column_names_not_positions():
    class Narrow(FrameSpec):
        a = ColSpec(pl.Int64)
        c = ColSpec(pl.Float64)

    class Wide(FrameSpec):
        a = ColSpec(pl.Int64)
        b = ColSpec(pl.String)  # inserted in the middle
        c = ColSpec(pl.Float64)

    narrow = Narrow.generate(1_000, seed=5)
    wide = Wide.generate(1_000, seed=5)
    assert narrow["a"].equals(wide["a"])
    assert narrow["c"].equals(wide["c"])


def test_rules_only_sample_what_they_scatter():
    class Spec(FrameSpec):
        region = ColSpec(pl.Enum(["UK", "US"]), weights=[1.0, 99.0])
        carrier = ColSpec(
            pl.Enum(["RM", "UPS", "DHL"]),
            rules=[
                ColRule(when=col("region") == "UK", choices=["RM"]),
                ColRule(
                    when=col("region") == "US",
                    choices={"UPS": 1.0, "DHL": 1.0},
                ),
            ],
        )
        n = ColSpec(
            pl.Int64,
            bounds=(0, 10),
            nullable=True,
            rules=[ColRule(when=col("region") == "US", choices=[7])],
        )

    df = Spec.generate(2_000, seed=2)
    assert set(df.filter(pl.col("region") == "UK")["carrier"].to_list()) <= {"RM"}
    assert set(df.filter(pl.col("region") == "US")["carrier"].to_list()) == {
        "UPS",
        "DHL",
    }
    assert df["carrier"].dtype == pl.Enum(["RM", "UPS", "DHL"])
    assert (df.filter(pl.col("region") == "US")["n"] == 7).all()
    Spec.validate(df)
