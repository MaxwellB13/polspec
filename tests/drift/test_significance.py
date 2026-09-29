"""Drift that knows how many rows it saw.

A rate, a set of frequencies or a distribution is reported as moved only
when the move is both significant at the sample size and at least its
tolerance. Two properties hold that together: a spec's own seeded output
never drifts from it, at any size; and a real move, large enough to matter
over enough rows to see it, always does. The statistics are pure Python, and
held to SciPy here.
"""

from __future__ import annotations

import datetime as dt
import math
import random

import numpy as np
import polars as pl
import pytest
from cases import COLUMN_CASES
from polspec import (
    ColRule,
    ColSpec,
    ForeignKey,
    GenerationError,
    Hierarchy,
    TableSpec,
    col,
    generate,
)
from polspec.drift import DriftOptions, drift, stats

scipy_stats = pytest.importorskip("scipy.stats")


# ---------------------------------------------------------------------------
# The statistics, against SciPy
# ---------------------------------------------------------------------------


def test_binomial_p_is_scipys_binomtest():
    rng = random.Random(0)
    for _ in range(1_500):
        n = rng.choice([1, 5, 20, 50, 300, 5_000, 200_000, 3_000_000])
        p = rng.choice([0.001, 0.01, 0.1, 0.3, 0.5, 0.9, 0.999])
        spread = 3 * math.sqrt(n * p * (1 - p)) + 2
        k = max(0, min(n, round(rng.gauss(n * p, spread))))
        expected = scipy_stats.binomtest(k, n, p).pvalue
        assert stats.binomial_p(k, n, p) == pytest.approx(expected, rel=1e-6, abs=1e-12)


def test_chi_square_p_is_scipys_chisquare():
    rng = random.Random(1)
    for _ in range(500):
        k = rng.randint(2, 8)
        weights = [rng.random() for _ in range(k)]
        shares = [w / sum(weights) for w in weights]
        n = rng.choice([10, 100, 10_000])
        counts = [0] * k
        for index in rng.choices(range(k), shares, k=n):
            counts[index] += 1
        expected = scipy_stats.chisquare(counts, [s * n for s in shares]).pvalue
        assert stats.chi_square_p(counts, shares) == pytest.approx(
            expected, rel=1e-9, abs=1e-12
        )


def test_ks_p_is_the_kolmogorov_survival_function():
    rng = random.Random(2)
    for _ in range(500):
        z = rng.uniform(0.2, 3.0)
        n, m = rng.randint(10, 10**6), rng.randint(10, 10**6)
        distance = z / math.sqrt(n * m / (n + m))
        expected = scipy_stats.kstwobign.sf(z)
        assert stats.ks_p(distance, n, m) == pytest.approx(expected, abs=1e-6)


def test_the_edges_of_each_statistic():
    assert stats.binomial_p(0, 0, 0.5) == 1.0
    assert stats.binomial_p(0, 10, 0.0) == 1.0
    assert stats.binomial_p(1, 10, 0.0) == 0.0
    assert stats.binomial_p(10, 10, 1.0) == 1.0
    assert stats.chi_square_p([5, 1], [1.0, 0.0]) == 0.0  # seen, never expected
    assert stats.chi_square_p([0, 0], [0.5, 0.5]) == 1.0
    assert stats.total_variation([1, 1], [0.5, 0.5]) == 0.0
    assert stats.total_variation([2, 0], [0.0, 1.0]) == 1.0
    assert stats.ks_p(0.0, 10, 10) == 1.0
    assert stats.unseen_p(0.01, 50) == pytest.approx(0.99**50)
    assert stats.unseen_p(1.0, 3) == 0.0


# ---------------------------------------------------------------------------
# A spec's own output never drifts from it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [50, 5_000])
@pytest.mark.parametrize("case", list(COLUMN_CASES))
def test_a_spec_does_not_drift_from_its_own_output(case, rows):
    """Fifty rows used to be enough for a false alarm: a null rate of 22%
    against a declared 30%, or a category of weight 1% never drawn."""
    spec = TableSpec("Own", {"c": COLUMN_CASES[case]})
    try:
        frame = generate(spec, rows, seed=17)
    except GenerationError as refused:
        pytest.skip(f"a unique column this small: {refused}")
    report = drift(spec, frame)
    assert report.unchanged, report.findings


_REWRITTEN = {
    "rules": TableSpec(
        "R",
        {
            "k": ColSpec(pl.Enum(["a", "b", "c"]), weights=[5, 3, 2]),
            "v": ColSpec(
                pl.Int64,
                choices=[1, 2, 3],
                weights=[1, 1, 1],
                rules=[ColRule(when=col("k") == "a", choices=[1])],
            ),
        },
    ),
    "foreign_key": TableSpec(
        "N",
        {
            "ref": ColSpec(pl.String, unique=True, string_length=(8, 8)),
            "parent": ColSpec(pl.String, nullable=True, null_probability=0.2),
        },
        foreign_keys=[ForeignKey("parent", references="self", ref_columns="ref")],
    ),
    "hierarchy": TableSpec(
        "H",
        {"child": ColSpec(pl.String), "parent": ColSpec(pl.String)},
        hierarchy=Hierarchy(child="child", parent="parent", max_depth=4),
    ),
    "unique_together": TableSpec(
        "U",
        {
            "a": ColSpec(
                pl.Int32,
                bounds=(0, 200_000),
                distribution="normal",
                distribution_params={"mean": 100_000, "std": 30_000},
            ),
            "b": ColSpec(pl.Enum(["x", "y"]), weights=[3, 1]),
        },
        unique_together=[("a", "b")],
    ),
}


@pytest.mark.parametrize("rows", [50, 5_000])
@pytest.mark.parametrize("name", list(_REWRITTEN))
def test_a_column_a_pass_rewrites_is_not_held_to_its_weights(name, rows):
    """A rule narrows `v` to 1 wherever `k` is "a", so `v`'s frequencies are
    not its weights -- and the same holds of a key a foreign key fills, a
    hierarchy's links, a composite key's repaired rows."""
    spec = _REWRITTEN[name]
    report = drift(spec, generate(spec, rows, seed=23))
    assert report.unchanged, report.findings


_WIDE = TableSpec(
    "Wide",
    {
        "status": ColSpec(pl.Enum(["NEW", "PAID", "SHIPPED"]), weights=[0.7, 0.2, 0.1]),
        "flag": ColSpec(pl.Boolean, weights=[0.8, 0.2]),
        "amount": ColSpec(
            pl.Float64,
            bounds=(0, 1_000),
            distribution="lognormal",
            distribution_params={"mean": 3, "std": 0.8},
            nullable=True,
            null_probability=0.1,
            nan_probability=0.02,
        ),
        "age": ColSpec(
            pl.Int64,
            bounds=(18, 90),
            distribution="normal",
            distribution_params={"mean": 40, "std": 12},
        ),
        "day": ColSpec(
            pl.Date,
            bounds=(dt.date(2020, 1, 1), dt.date(2024, 1, 1)),
            distribution="normal",
            distribution_params={"mean": 19_000, "std": 200},
        ),
        "tags": ColSpec(
            pl.List(pl.Float64), element_null_probability=0.1, list_length=(1, 3)
        ),
    },
)


@pytest.mark.parametrize("rows", [50, 5_000, 5_000_000])
def test_a_wide_spec_does_not_drift_from_its_own_output_at_any_size(rows):
    """At five million rows every difference is significant; the tolerance
    is what keeps a rounding error from being a finding."""
    report = drift(_WIDE, generate(_WIDE, rows, seed=99))
    assert report.unchanged, report.findings


# ---------------------------------------------------------------------------
# A real move is reported, with what moved
# ---------------------------------------------------------------------------


def _shifted(rows: int) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    frame = generate(_WIDE, rows, seed=5)
    return frame.with_columns(
        status=pl.Series(
            rng.choice(["NEW", "PAID", "SHIPPED"], rows, p=[0.5, 0.3, 0.2])
        ).cast(_WIDE["status"].dtype),
        flag=pl.Series(rng.random(rows) < 0.4),
        amount=pl.Series(rng.uniform(0, 1_000, rows)),
        age=pl.Series(
            np.clip(rng.normal(50, 12, rows).round(), 18, 90).astype(np.int64)
        ),
    )


def test_moved_frequencies_and_distributions_are_reported():
    report = drift(_WIDE, _shifted(5_000))
    found = {(f.columns[0], f.code) for f in report.findings}
    assert found == {
        ("status", "frequencies_moved"),
        ("flag", "frequencies_moved"),
        ("amount", "null_rate_moved"),  # the uniform replacement holds no nulls
        ("amount", "distribution_moved"),
        ("age", "distribution_moved"),
    }
    assert all(not f.breaking for f in report.findings)
    (status,) = [f for f in report.findings if f.key == "status__frequencies"]
    assert status.details["declared"] == {
        "NEW": pytest.approx(0.7),
        "PAID": pytest.approx(0.2),
        "SHIPPED": pytest.approx(0.1),
    }
    assert status.details["observed"]["NEW"] == pytest.approx(0.5, abs=0.02)
    assert status.details["distance"] == pytest.approx(0.2, abs=0.02)
    assert status.details["p_value"] < 1e-100
    assert "'NEW' 50." in status.message and "(declared 70.0%)" in status.message
    (age,) = [f for f in report.findings if f.key == "age__distribution"]
    assert age.details["distribution"] == "normal"
    assert age.details["distance"] > 0.25


def test_the_same_moves_over_fifty_rows_are_less_evidence():
    """Every column above moved; over fifty rows, only the plainest moves
    are unlikely enough to be more than chance -- each finding still clears
    the significance it was held to."""
    small = drift(_WIDE, _shifted(50))
    large = drift(_WIDE, _shifted(5_000))
    assert len(small.findings) < len(large.findings)
    assert all(f.details["p_value"] < 0.001 for f in small.findings)


def test_a_small_move_over_many_rows_is_within_tolerance():
    """Significant at a million rows, but a point off the declared 70%: not
    worth reporting until the tolerance says so."""
    rows = 1_000_000
    rng = np.random.default_rng(1)
    spec = TableSpec("T", {"s": ColSpec(pl.Enum(["a", "b"]), weights=[0.7, 0.3])})
    frame = pl.DataFrame(
        {"s": pl.Series(rng.choice(["a", "b"], rows, p=[0.69, 0.31]))}
    ).cast({"s": spec["s"].dtype})
    assert drift(spec, frame).unchanged
    (finding,) = drift(spec, frame, frequency_tolerance=0.005).findings
    assert finding.code == "frequencies_moved"


def test_significance_and_tolerances_are_options():
    frame = _shifted(5_000)
    assert drift(_WIDE, frame, distribution_tolerance=0.9).findings
    assert not any(
        f.code == "distribution_moved"
        for f in drift(_WIDE, frame, distribution_tolerance=0.95).findings
    )
    lax = drift(_WIDE, _shifted(50), significance=0.5)
    assert len(lax.findings) >= len(drift(_WIDE, _shifted(50)).findings)


@pytest.mark.parametrize(
    "option, value",
    [
        ("significance", 0.0),
        ("significance", 1.0),
        ("frequency_tolerance", 1.5),
        ("distribution_tolerance", -0.1),
    ],
)
def test_an_option_outside_its_range_is_refused(option, value):
    with pytest.raises(ValueError, match=option):
        DriftOptions(**{option: value})


def test_a_distribution_on_list_elements_is_drawn_and_compared_as_elements():
    spec = TableSpec(
        "T",
        {
            "xs": ColSpec(
                pl.List(pl.Float64),
                bounds=(0, 1),
                distribution="beta",
                distribution_params={"alpha": 2, "beta": 5},
                list_length=(2, 4),
            )
        },
    )
    own = generate(spec, 5_000, seed=3)
    assert drift(spec, own).unchanged
    flat = own.with_columns(pl.col("xs").list.eval(pl.element() * 0 + 0.9))
    (finding,) = drift(spec, flat).findings
    assert finding.code == "distribution_moved"


def test_a_value_a_rule_leaves_out_of_every_row_is_not_unseen():
    """The rule gives every row 1, so 2 and 3 never appear in the spec's own
    output -- which used to be `cardinality_moved`, weighed as though each
    were drawn a third of the time."""
    spec = TableSpec(
        "R",
        {
            "k": ColSpec(pl.Int64, bounds=(0, 9)),
            "v": ColSpec(
                pl.Int64,
                choices=[1, 2, 3],
                rules=[ColRule(when=col("k") >= 0, choices=[1])],
            ),
        },
    )
    assert drift(spec, generate(spec, 5_000, seed=1)).unchanged


def test_a_distribution_over_values_of_the_wrong_dtype_is_left_to_dtype():
    spec = TableSpec(
        "T",
        {
            "x": ColSpec(
                pl.Float64,
                bounds=(0, 1),
                distribution="beta",
                distribution_params={"alpha": 2, "beta": 2},
            )
        },
    )
    frame = pl.DataFrame({"x": ["0.5", "0.2"]})
    assert [f.code for f in drift(spec, frame).findings] == ["dtype_changed"]


def test_a_file_that_validates_has_not_drifted_on_dtype():
    """What a CSV hands back -- a String for an Enum, a Datetime for a
    Date -- validates as it is, so it is no finding unless dtypes are held
    strictly."""
    frame = generate(_WIDE, 2_000, seed=8).with_columns(
        pl.col("status").cast(pl.String), pl.col("day").cast(pl.Datetime)
    )
    assert drift(_WIDE, frame).unchanged
    strict = drift(_WIDE, frame, strict_dtypes=True)
    assert sorted(f.key for f in strict.findings) == ["day__dtype", "status__dtype"]
