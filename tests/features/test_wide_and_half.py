"""`Int128`, `UInt128` and `Float16`: drawn through a type the engine has.

A 128-bit integer is drawn through 64 bits and a half through a single, then
cast -- the way a `Decimal` is drawn as its physical integer. Validation
reads the full type either way; generation refuses bounds past the draw, and
keeps a half's rounding from carrying a value over a bound.
"""

from __future__ import annotations

import polars as pl
import pytest
from helpers import spec_for
from polspec import ColSpec, FrameSpec, GenerationError, SpecError, TableSpec
from polspec.dtypes import float16_inside

# ---------------------------------------------------------------------------
# 128-bit integers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [pl.Int128, pl.UInt128])
def test_a_128_bit_integer_generates_within_its_bounds(dtype):
    spec_cls = spec_for(ColSpec(dtype, bounds=(5, 50), unique=True))
    df = spec_cls.generate(40, seed=1)
    assert df.schema["c"] == dtype
    assert df["c"].min() >= 5 and df["c"].max() <= 50
    assert df["c"].n_unique() == 40
    spec_cls.validate(df)


@pytest.mark.parametrize(
    ("dtype", "bounds"),
    [(pl.Int128, (0, 2**100)), (pl.UInt128, (2**70, 2**71))],
)
def test_bounds_past_the_64_bit_draw_validate_but_do_not_generate(dtype, bounds):
    spec_cls = spec_for(ColSpec(dtype, bounds=bounds))
    wide = pl.DataFrame({"c": pl.Series([bounds[1]], dtype=dtype)})
    spec_cls.validate(wide)
    with pytest.raises(GenerationError, match="range generation draws in"):
        spec_cls.generate(10, seed=1)


def test_an_open_end_widens_to_the_draw_not_the_dtype():
    """`(2_000_000, None)` lies past the default range; the open end widens
    to what 64 bits hold, not to 2**127, which could not be drawn."""
    df = spec_for(ColSpec(pl.Int128, bounds=(2_000_000, None))).generate(50, seed=1)
    assert df["c"].min() >= 2_000_000


# ---------------------------------------------------------------------------
# Half-precision floats
# ---------------------------------------------------------------------------


def test_a_half_generates_finite_values_by_default():
    for column in (
        ColSpec(pl.Float16),
        ColSpec(pl.Float16, distribution="normal", distribution_params={"std": 1e9}),
    ):
        df = spec_for(column).generate(2_000, seed=1)
        assert df.schema["c"] == pl.Float16
        assert df["c"].is_finite().all()


@pytest.mark.parametrize("bounds", [(0.0, 0.1001), (-1.3, -1.299), (1e-3, 2e-3)])
def test_rounding_to_a_half_never_carries_a_value_past_a_bound(bounds):
    """The nearest half to 0.1001 is above it; a value drawn at the bound
    would round out of it. Drawn between the halves inside, none does."""
    spec_cls = spec_for(ColSpec(pl.Float16, bounds=bounds))
    spec_cls.validate(spec_cls.generate(20_000, seed=3))


def test_the_half_inside_a_bound_is_a_neighbour_not_a_leap():
    assert float16_inside(0.1001, up=False) < 0.1001 < float16_inside(0.1001, up=True)
    assert float16_inside(0.5, up=True) == 0.5  # an exact half stays put
    assert float16_inside(0.0, up=True) == 0.0
    # A value that rounds to zero steps to the smallest subnormal past it.
    assert float16_inside(1e-10, up=True) > 1e-10 > float16_inside(1e-10, up=False)
    assert float16_inside(-1.3, up=True) > -1.3 > float16_inside(-1.3, up=False)


def test_bounds_that_hold_no_half_are_refused_where_they_are_written():
    """No half lies in `(-1.3, -1.2999)`: nothing could satisfy it."""
    with pytest.raises(SpecError, match="hold no Float16 value"):
        ColSpec(pl.Float16, bounds=(-1.3, -1.2999))


def test_a_unique_half_is_drawn_from_the_halves_themselves():
    """Distinct singles can round to the same half, so a unique half draws
    from the finite set of halves between its bounds."""
    spec_cls = spec_for(ColSpec(pl.Float16, bounds=(0, 1), unique=True))
    df = spec_cls.generate(5_000, seed=1)
    assert df["c"].n_unique() == 5_000
    spec_cls.validate(df)


def test_a_bound_a_half_cannot_hold_is_refused_where_it_is_written():
    with pytest.raises(SpecError, match="outside the range"):
        ColSpec(pl.Float16, bounds=(0, 100_000))


# ---------------------------------------------------------------------------
# Everywhere else a dtype has to be known
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [pl.Int128, pl.UInt128, pl.Float16])
def test_each_round_trips_through_yaml_and_python(dtype, tmp_path):
    spec_cls = spec_for(ColSpec(dtype, bounds=(1, 9), nullable=True))
    spec_cls.to_yaml(tmp_path / "s.yaml")
    assert FrameSpec.from_yaml(tmp_path / "s.yaml").spec == spec_cls.spec
    spec_cls.to_python(tmp_path / "s.py")
    namespace: dict = {}
    exec((tmp_path / "s.py").read_text(encoding="utf-8"), namespace)
    assert namespace[spec_cls.__name__].spec == spec_cls.spec


@pytest.mark.parametrize(
    ("dtype", "width"), [(pl.Int128, 16), (pl.UInt128, 16), (pl.Float16, 2)]
)
def test_each_is_sized_at_its_own_width(dtype, width):
    spec = TableSpec("S", {"c": ColSpec(dtype)})
    assert spec.estimated_size(1_000) == width * 1_000
    df = spec_for(ColSpec(dtype)).generate(1_000, seed=1)
    assert df.estimated_size() == width * 1_000


@pytest.mark.parametrize("dtype", [pl.Int128, pl.UInt128, pl.Float16])
def test_from_dataframe_redeclares_each(dtype):
    df = pl.DataFrame({"c": pl.Series([1, 2, 3], dtype=dtype)})
    profiled = FrameSpec.from_dataframe(df)
    assert profiled.spec.columns["c"].dtype == dtype
    profiled.validate(df)
