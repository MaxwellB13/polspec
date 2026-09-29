"""A fake dataset from a real one: `polspec.profile` and `polspec.synthesize`.

`profile()` describes a source -- a frame, a lazy frame, a data file -- with
the opt-in steps `from_dataframe()` leaves off: the distribution each
numeric column follows (`polspec.shape`), whether a column is a key, and
which columns' values not to carry over (`replace=`). `synthesize()`
generates from that description. These tests hold it to what the how-to
page promises: the fake frame has the source's schema, follows its
columns' shapes, frequencies and null rates, keeps its keys distinct,
carries no value of a replaced column, validates against its own spec and
does not drift from it -- and `from_dataframe()` with no options infers
exactly what it did before any of this existed.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from helpers import spec_for
from polspec import (
    ColSpec,
    FrameSpec,
    TableSpec,
    generate,
    profile,
    synthesize,
    validate,
)
from polspec.bound import Bound
from polspec.drift import drift
from polspec.serialization import to_dict
from polspec.shape import fit
from polspec.synthesis import synthesized

FIXTURES = Path(__file__).parent / "fixtures"
ROWS = 20_000


def golden_frame() -> pl.DataFrame:
    """A frame of every kind of column the profiler reads, built without a
    random number: the input of the golden check that default profiling
    has not moved."""
    n = 400
    return pl.DataFrame(
        {
            "id": list(range(n)),
            "amount": [round((i * 37 % 101) * 1.25 + 0.5, 2) for i in range(n)],
            "status": [("NEW", "PAID", "SHIPPED")[i % 3] for i in range(n)],
            "note": [f"note {i}" for i in range(n)],
            "flag": [i % 4 == 0 for i in range(n)],
            "day": [dt.date(2024, 1, 1) + dt.timedelta(days=i % 90) for i in range(n)],
            "tags": [list(range(i % 3)) for i in range(n)],
            "maybe": [None if i % 5 == 0 else i for i in range(n)],
            "contact": [f"user{i % 50}@example.com" for i in range(n)],
        }
    )


@pytest.fixture(scope="module")
def real() -> pl.DataFrame:
    """A realistic source: a key, a skewed amount, a clipped bell curve, a
    weighted status, a few null ratings, and thirty real names."""
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "id": np.arange(1, ROWS + 1),
            "amount": rng.lognormal(3, 0.8, ROWS),
            "age": np.clip(rng.normal(40, 12, ROWS), 18, 90).round().astype(np.int64),
            "status": rng.choice(["NEW", "PAID", "SHIPPED"], ROWS, p=[0.7, 0.2, 0.1]),
            "rating": [None if r < 0.2 else float(r) for r in rng.random(ROWS)],
            "name": [f"person_{i}" for i in rng.integers(0, 30, ROWS)],
        }
    )


# ---------------------------------------------------------------------------
# Defaults do not move
# ---------------------------------------------------------------------------


def test_from_dataframe_with_no_options_infers_what_it_always_did():
    """Profiling gained three opt-in steps in 0.13.0; none of them runs unless
    asked, so a spec `polspec schema infer` wrote before still comes out the
    same. 0.14.0 changed the default on purpose, twice: an Enum needs its
    values to repeat, and a text column names its format -- `contact` here.
    Regenerate the fixture only on purpose."""
    golden = yaml.safe_load(
        (FIXTURES / "profiled_by_default.yaml").read_text(encoding="utf-8")
    )
    spec = FrameSpec.from_dataframe(golden_frame(), name="Profiled").spec
    assert to_dict(spec) == golden


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".ndjson", ".arrow"])
def test_every_source_form_gives_the_same_profile(real, tmp_path, suffix):
    frame = real.drop("rating")  # a null float reads back from CSV as it is
    path = tmp_path / f"real{suffix}"
    {
        ".csv": frame.write_csv,
        ".parquet": frame.write_parquet,
        ".ndjson": frame.write_ndjson,
        ".arrow": frame.write_ipc,
    }[suffix](path)
    expected = profile(frame)
    assert profile(path) == expected
    assert profile(str(path)) == expected
    assert profile(frame.lazy()) == expected


def test_a_sample_profiles_that_many_rows(real):
    spec = profile(real, sample=1_000, seed=3)
    assert spec == profile(real, sample=1_000, seed=3)
    # A key needs a hundred distinct values; a sample of fifty has too few.
    assert not profile(real, sample=50)["id"].unique
    with pytest.raises(ValueError, match="positive row count"):
        profile(real, sample=0)


def test_a_source_that_is_not_data_is_refused(real):
    with pytest.raises(TypeError, match="DataFrame, a LazyFrame or the path"):
        profile(42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no rows"):
        profile(real.clear())


# ---------------------------------------------------------------------------
# What profiling learns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        (np.random.default_rng(1).lognormal(3, 0.8, 50_000), pl.Float64, "lognormal"),
        (np.random.default_rng(1).normal(40, 5, 50_000), pl.Float64, "normal"),
        (np.random.default_rng(1).poisson(3, 50_000), pl.Int64, "poisson"),
        (np.random.default_rng(1).beta(2, 5, 50_000), pl.Float64, "beta"),
        (np.random.default_rng(1).uniform(0, 100, 50_000), pl.Float64, None),
    ],
    ids=["lognormal", "normal", "poisson", "beta", "uniform"],
)
def test_the_shape_fitted_is_the_shape_drawn_from(values, dtype, expected):
    series = pl.Series(values).cast(dtype)
    fitted = fit(series, Bound(series.min(), series.max()), dtype, seed=0)
    assert (fitted.distribution if fitted else None) == expected


def test_a_key_is_all_distinct_and_integer_or_text(real):
    spec = profile(real)
    assert spec["id"].unique
    assert not spec["amount"].unique  # all distinct, and a float: not a key
    assert not spec["age"].unique
    assert spec["id"].distribution is None  # a key is drawn without replacement


# ---------------------------------------------------------------------------
# The fake frame
# ---------------------------------------------------------------------------


def test_the_fake_frame_looks_like_the_real_one(real):
    fake = synthesize(real, seed=1)
    assert fake.schema == real.schema
    assert fake.height == real.height
    for column in ("amount", "age"):
        for q in (0.1, 0.5, 0.9):
            assert fake[column].quantile(q) == pytest.approx(
                real[column].quantile(q), rel=0.08
            ), (column, q)
    shares = fake["status"].value_counts(normalize=True).sort("status")
    assert shares["proportion"].to_list() == pytest.approx([0.7, 0.2, 0.1], abs=0.02)
    assert fake["rating"].null_count() / ROWS == pytest.approx(0.2, abs=0.02)
    assert fake["id"].is_unique().all()


def test_the_fake_frame_is_not_the_real_one(real):
    fake = synthesize(real, seed=1)
    assert fake.join(real, on=list(real.columns), how="semi").is_empty()


def test_a_replaced_column_carries_none_of_its_values(real):
    spec = profile(real, replace=["name"])
    assert spec["name"].dtype == pl.String  # not an Enum of the real names
    assert spec["name"].choices is None and spec["name"].weights is None
    fake = synthesize(real, seed=1, replace=["name"])
    assert not set(fake["name"]) & set(real["name"])
    # Without it, the thirty names are an Enum, and they are carried over.
    assert set(synthesize(real, seed=1)["name"]) <= set(real["name"])


def test_replace_names_only_columns_the_source_has(real):
    with pytest.raises(ValueError, match=r"replace= names \['nope'\]"):
        profile(real, replace=["nope"])
    enum = real.with_columns(pl.col("status").cast(pl.Enum(["NEW", "PAID", "SHIPPED"])))
    with pytest.raises(ValueError, match="an Enum: its categories are its dtype"):
        profile(enum, replace=["status"])


def test_the_fake_frame_validates_and_does_not_drift(real):
    """The library's own comparison says the fake looks like the real."""
    spec = profile(real, replace=["name"])
    fake = synthesize(real, seed=1, replace=["name"])
    validate(spec, fake)
    assert drift(spec, fake).breaking == ()


def test_the_same_source_and_seed_give_the_same_frame(real):
    assert synthesize(real, seed=7).equals(synthesize(real, seed=7))
    assert not synthesize(real, seed=7).equals(synthesize(real, seed=8))


def test_a_key_widens_to_hold_more_rows_than_the_source(real):
    fake = synthesize(real, 3 * ROWS, seed=1)
    assert fake.height == 3 * ROWS
    assert fake["id"].is_unique().all()
    assert fake["id"].min() >= real["id"].min()


def test_a_key_that_cannot_widen_enough_is_refused():
    small = pl.DataFrame({"k": pl.Series(range(200), dtype=pl.UInt8)})
    with pytest.raises(ValueError, match="cannot hold 1,000 distinct values"):
        synthesize(small, 1_000, seed=1)


def test_nested_and_temporal_columns_synthesize():
    """Every kind of column, and a nullable key. A unique column's range has
    to hold every row, nulls included, so `maybe` -- 320 distinct values
    over 400 rows -- has its range widened by one to hold them: the fake
    frame validates against the spec it was generated from, which says so."""
    frame = golden_frame()
    fake, spec = synthesized(frame, seed=1)
    assert fake.schema == frame.schema
    assert spec["maybe"].unique and spec["maybe"].bounds.max == 400
    validate(spec, fake.with_columns(pl.col("status").cast(spec["status"].dtype)))


def test_a_profiled_spec_survives_a_file(real, tmp_path):
    spec = profile(real, replace=["name"])
    path = tmp_path / "profiled.yaml"
    FrameSpec.from_spec(spec).to_yaml(path)
    reloaded = FrameSpec.from_yaml(path)
    assert reloaded.spec == spec.with_name(reloaded.spec.name)
    assert generate(reloaded.spec, 100, seed=1).equals(generate(spec, 100, seed=1))


def test_spec_for_helper_still_profiles_a_single_column():
    """`from_dataframe` keywords compose with a one-column spec."""
    column = FrameSpec.from_dataframe(
        spec_for(ColSpec(pl.Int64, bounds=(1, 500))).generate(500, seed=1),
        shape=True,
        detect_unique=True,
    ).spec["c"]
    assert column.bounds is not None


# ---------------------------------------------------------------------------
# Sources with what a spec cannot say, or barely can
# ---------------------------------------------------------------------------


def test_a_source_holding_infinities_and_nans_synthesizes():
    """It used to fail -- a profiled bound must be finite. The fake frame
    keeps to the finite values' range, holds no infinity, and holds NaN on
    the share the source did."""
    source = pl.DataFrame({"f": [1.5, float("inf"), float("nan"), 3.0, 2.0] * 20})
    with pytest.warns(UserWarning, match="holds 20 infinite"):
        fake = synthesize(source, 5_000, seed=1)
    numbers = fake["f"].filter(fake["f"].is_not_nan())
    assert numbers.is_finite().all()
    assert numbers.min() >= 1.5
    assert numbers.max() <= 3.0
    # 20 NaN of the 100 values: a fifth of the fake ones, too.
    assert fake["f"].is_nan().mean() == pytest.approx(0.2, abs=0.02)


def test_a_null_column_and_an_empty_name_synthesize():
    """An empty column -- `Null`, as Parquet keeps one -- could be profiled
    and not generated; a column named "" came back from the engine renamed
    `column_0`."""
    source = pl.DataFrame(
        {
            "": ["x", "y", "x", "z", "y", "z"],
            "empty": pl.Series([None] * 6, dtype=pl.Null),
            "n": [1, 2, 3, 4, 5, 6],
        }
    )
    fake = synthesize(source, 30, seed=1)
    assert fake.schema == source.schema
    assert fake["empty"].null_count() == 30
    assert set(fake[""]) <= {"x", "y", "z"}
    validate(profile(source), fake)


def test_a_synthesized_column_has_its_sources_format():
    """Email-shaped fake emails: 0.13.0 made text of an email's length."""
    emails = generate(
        TableSpec("S", {"e": ColSpec(pl.String, format="email")}), 500, seed=4
    )["e"]
    source = pl.DataFrame({"email": emails, "name": emails.str.split("@").list.first()})
    fake = synthesize(source, 1_000, seed=2, replace=["email"])
    assert fake["email"].str.contains("@").all()
    assert not set(fake["email"]) & set(source["email"])
    spec = profile(source, replace=["email"])
    assert spec["email"].format == "email"
    validate(spec, fake)
