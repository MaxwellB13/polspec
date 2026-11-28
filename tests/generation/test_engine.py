"""The Python / Rust boundary.

A column crosses into the engine as a `ColumnPlan` and comes back as its
physical form. What that means for generated data is tested through
`generate` (`test_generate.py`), and the plan's own checks by `cargo test`
(`src/plan.rs`). What is left here has no public face: the stub matches the
module, both sides agree on the distributions, the package imports without
the extension, the engine's complaints arrive as `GenerationError`, and the
builders the engine exports for lists, decimals and permutations, and the
drawn order, which has no public face until a spec can ask for one.
"""

import ast
import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest
from polspec import ColSpec, FrameSpec, GenerationError, _ffi
from polspec.distributions import DISTRIBUTIONS
from polspec.generation.order import TABLE_ROWS, drawn_values, why_not_drawn

# ---------------------------------------------------------------------------
# ColumnPlan
# ---------------------------------------------------------------------------


def test_the_stub_matches_the_module():
    from polspec import _polspec

    stub = Path(_polspec.__file__).with_suffix(".pyi")
    if not stub.exists():  # an installed wheel places the stub beside the package
        stub = Path(__file__).parents[2] / "python" / "polspec" / "_polspec.pyi"
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
    import polspec.validation  # loaded lazily; this file may run first

    # Make the extension unimportable: no module in sys.modules, no attribute
    # on the package for `from polspec import _polspec` to find.
    monkeypatch.setitem(sys.modules, "polspec._polspec", None)
    monkeypatch.delattr(polspec, "_polspec", raising=False)
    # Importing the module again rebinds the package's `validation`
    # attribute; restoring the original keeps every later test on the one
    # module the rest of the package imports, its classes included.
    monkeypatch.setattr(polspec, "validation", polspec.validation, raising=False)
    monkeypatch.delitem(sys.modules, "polspec.validation", raising=False)

    importlib.import_module("polspec.validation")  # no extension needed
    with pytest.raises(ImportError, match="maturin develop"):
        _ffi.generate_dataframe([], 1, 1)


# ---------------------------------------------------------------------------
# What a column becomes on the way in and back
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# permuted_indices: the permutation, reachable from Python
# ---------------------------------------------------------------------------


def test_permuted_indices_are_a_permutation_and_a_window_is_its_slice():
    whole = _ffi.permuted_indices(10_000, 7, 0, 10_000)
    assert whole.dtype == pl.UInt64
    assert whole.sort().to_list() == list(range(10_000))
    for start, n in [(0, 5), (123, 1_000), (9_990, 10)]:
        assert _ffi.permuted_indices(10_000, 7, start, n).equals(whole.slice(start, n))
    assert not whole.equals(_ffi.permuted_indices(10_000, 8, 0, 10_000))


def test_permuted_indices_refuse_a_window_past_the_domain():
    with pytest.raises(GenerationError, match="110 distinct indices"):
        _ffi.permuted_indices(100, 1, 90, 20)
    assert _ffi.permuted_indices(0, 1, 0, 0).len() == 0


def test_rust_engine_complaints_surface_as_generation_error():
    class Spec(FrameSpec):
        a = ColSpec(
            pl.Float64,
            distribution="normal",
            distribution_params={"mean": 0.0, "std": 1.0},
        )

    # A valid spec; then reach the engine with a parameter it rejects by
    # bypassing ColSpec's own check, which is what a future bug would do.
    with pytest.raises(GenerationError) as info:
        _ffi.column_plan(
            "a", "float64", distribution="normal", params={"mean": 0.0, "std": -1.0}
        )
    assert isinstance(info.value.__cause__, ValueError)
    assert "'a'" in str(info.value) and "must be positive" in str(info.value)


def test_a_list_is_built_from_its_lengths_and_refuses_a_mismatch_by_name():
    lengths = pl.Series([2, None, 0, 1], dtype=pl.Int64)
    elements = pl.Series([dt.date(2024, 1, d) for d in (1, 2, 3)])
    cells = _ffi.list_from_lengths("when", lengths, elements)
    assert cells.name == "when" and cells.dtype == pl.List(pl.Date)
    assert cells.to_list() == [
        [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
        None,
        [],
        [dt.date(2024, 1, 3)],
    ]
    with pytest.raises(GenerationError, match="'when'"):
        _ffi.list_from_lengths("when", pl.Series([5], dtype=pl.Int64), elements)


def test_a_decimal_is_built_from_its_physical_integers():
    raw = pl.Series("amount", [12345, None, -5], dtype=pl.Int64)
    decimal = _ffi.decimal_from_physical(raw, 12, 2)
    assert decimal.dtype == pl.Decimal(12, 2)
    assert decimal.cast(pl.String).to_list() == ["123.45", None, "-0.05"]
    with pytest.raises(GenerationError, match="precision"):
        _ffi.decimal_from_physical(raw, 3, 2)


# ---------------------------------------------------------------------------
# A drawn order (2.2): a column in order, each row from its position
# ---------------------------------------------------------------------------

_DRAWN = {
    "int": ColSpec(pl.Int32, bounds=(0, 100)),
    "normal": ColSpec(pl.Float64, distribution="normal"),
    "gamma32": ColSpec(pl.Float32, distribution="gamma"),
    "when": ColSpec(
        pl.Datetime("us"),
        bounds=(dt.datetime(2024, 1, 1), dt.datetime(2025, 1, 1)),
        nullable=True,
        null_probability=0.05,
    ),
    "day": ColSpec(pl.Date),
    "wait": ColSpec(pl.Duration("ms")),
    "clock": ColSpec(pl.Time),
    "amount": ColSpec(pl.Decimal(12, 2), bounds=(0, 1000)),
    "wide": ColSpec(pl.UInt64, bounds=(0, 2**64 - 1)),
    "flag": ColSpec(pl.Boolean, nullable=True),
    "status": ColSpec(pl.String, choices=["open", "closed", "held"]),
    "level": ColSpec(pl.Enum(["high", "mid", "low"])),
    "id": ColSpec(pl.Int64, unique=True, bounds=(0, 10**15)),
    "id128": ColSpec(pl.Int128, unique=True),
    "stamp": ColSpec(pl.Datetime("ns"), unique=True),
    "share": ColSpec(pl.Float64, unique=True, bounds=(0, 1)),
}


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("n", [1_000, 70_000 * 2])  # within the table, and past it
@pytest.mark.parametrize("name", list(_DRAWN))
def test_a_drawn_column_is_in_order_and_any_window_is_the_frames(name, n, descending):
    spec = _DRAWN[name]
    whole = drawn_values(
        name, spec, 7, frame_rows=n, row_offset=0, rows=n, descending=descending
    )
    assert whole.dtype == spec.dtype and whole.len() == n and whole.name == name
    windows = pl.concat(
        drawn_values(
            name,
            spec,
            7,
            frame_rows=n,
            row_offset=offset,
            rows=min(4_999, n - offset),
            descending=descending,
        )
        for offset in range(0, n, 4_999)
    )
    assert windows.equals(whole)
    values = whole.drop_nulls()
    # Nulls last, as a sort puts them, whichever way the values run.
    assert whole.is_null().arg_true().to_list() == list(range(values.len(), n))
    assert values.equals(values.sort(descending=descending))
    if spec.unique:
        assert values.n_unique() == values.len()


def test_a_frame_within_the_table_draws_its_own_values_in_order():
    """Up to `TABLE_ROWS` valid rows, the column is the values the engine
    draws for it, sorted: a sample, not a stratified one."""
    from polspec.generation.engine import generate_random

    spec = ColSpec(pl.Float64, distribution="lognormal")
    drawn = drawn_values(
        "x", spec, 3, frame_rows=TABLE_ROWS, row_offset=0, rows=TABLE_ROWS
    )
    assert drawn.equals(generate_random({"x": spec}, TABLE_ROWS, 3)["x"].sort())


@pytest.mark.parametrize(
    ("distribution", "params"),
    [
        ("normal", None),
        ("lognormal", None),
        ("exponential", None),
        ("gamma", {"shape": 2.0}),
        ("beta", {"alpha": 2.0, "beta": 5.0}),
    ],
)
def test_a_drawn_column_holds_its_distribution(distribution, params):
    """Past the table, every percentile is within 1% of the range of the
    sorted column's."""
    from polspec.generation.engine import generate_random

    n = 400_000
    spec = ColSpec(pl.Float64, distribution=distribution, distribution_params=params)
    reference = generate_random({"x": spec}, n, 5)["x"]
    drawn = drawn_values("x", spec, 5, frame_rows=n, row_offset=0, rows=n)
    span = reference.max() - reference.min()
    for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        assert abs(reference.quantile(q) - drawn.quantile(q)) < 0.01 * span, q


def test_a_drawn_finite_column_holds_its_weights():
    spec = ColSpec(pl.String, choices={"a": 0.6, "b": 0.3, "c": 0.1})
    n = 300_000
    drawn = drawn_values("x", spec, 2, frame_rows=n, row_offset=0, rows=n)
    shares = dict(drawn.value_counts(normalize=True).iter_rows())
    for value, weight in zip("abc", (0.6, 0.3, 0.1), strict=True):
        assert abs(shares[value] - weight) < 0.01


@pytest.mark.parametrize(
    ("spec", "reason"),
    [
        (ColSpec(pl.String), "free-text"),
        (ColSpec(pl.List(pl.Int64)), "no order of its own"),
        (ColSpec(pl.Float64, nan_probability=0.1), "nan_probability"),
        (ColSpec(pl.Boolean, unique=True, nullable=True), "unique column"),
        (ColSpec(pl.String, choices=["a", "b"], unique=True), "unique column"),
    ],
)
def test_a_column_a_drawn_order_cannot_hold_is_named(spec, reason):
    assert reason in why_not_drawn(spec)
    with pytest.raises(GenerationError, match="'x' cannot be drawn in order"):
        drawn_values("x", spec, 1, frame_rows=10, row_offset=0, rows=10)


def test_a_unique_column_too_small_for_the_frame_is_refused():
    spec = ColSpec(pl.Int32, unique=True, bounds=(0, 99_999))
    with pytest.raises(GenerationError, match="fewer than"):
        drawn_values("x", spec, 1, frame_rows=100_001, row_offset=0, rows=10)
    assert (
        drawn_values("x", spec, 1, frame_rows=100_000, row_offset=0, rows=10).len()
        == 10
    )
