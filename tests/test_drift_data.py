"""`drift(spec, df)`: how data has moved relative to its declaration.

Two contracts hold this together. First, the round trip extended to drift:
what `generate()` produces never drifts *breakingly* from the spec that
produced it. Second, the severity rule: every breaking finding corresponds
to a validation finding on the same column, so "breaking" means exactly
"validation fails here" and nothing looser.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from polspec import ColSpec, DriftOptions, FrameSpec, TableSpec, generate, inspect
from polspec.drift import drift

ROWS = 2_000
SEED = 3


class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, None), unique=True)
    status = ColSpec(pl.Enum(["NEW", "PAID", "SHIPPED"]))
    region = ColSpec(pl.String, choices=["UK", "US", "EU"])
    total = ColSpec(pl.Float64, bounds=(0.0, 1_000.0))
    note = ColSpec(
        pl.String, string_length=(0, 40), nullable=True, null_probability=0.3
    )
    placed = ColSpec(pl.Date, bounds=(dt.date(2024, 1, 1), dt.date(2025, 1, 1)))
    email = ColSpec(pl.String, format="email")
    country = ColSpec(pl.String, format="iso_country")
    flag = ColSpec(pl.Boolean)
    blob = ColSpec(pl.Binary, string_length=(4, 8))


ROUND_TRIP_SPECS = {
    "orders": Orders.spec,
    "temporal": TableSpec(
        "Temporal",
        {
            "d": ColSpec(pl.Date, bounds=(dt.date(2020, 1, 1), None)),
            "t": ColSpec(pl.Datetime("ms"), nullable=True),
            "u": ColSpec(pl.Duration("us"), bounds=(0, 1_000_000)),
        },
    ),
    "narrow_ints": TableSpec(
        "Narrow", {"i8": ColSpec(pl.Int8), "u16": ColSpec(pl.UInt16, bounds=(5, 9))}
    ),
}


@pytest.mark.parametrize("name", sorted(ROUND_TRIP_SPECS))
def test_generated_data_never_drifts_breakingly(name):
    spec = ROUND_TRIP_SPECS[name]
    report = drift(spec, generate(spec, ROWS, seed=SEED))
    assert report.breaking == (), str(report)


def test_generated_orders_do_not_drift_at_all():
    """With enough rows every declared value appears and the null rate lands
    inside the tolerance, so the report is empty, not merely non-breaking."""
    assert Orders.drift(Orders.generate(ROWS, seed=SEED)).unchanged


def test_every_breaking_finding_is_a_validation_failure():
    """The severity rule, checked against the thing it is defined by."""
    df = Orders.generate(ROWS, seed=SEED)
    broken = df.with_columns(
        total=pl.col("total") * 10,
        region=pl.lit("MARS"),
        placed=pl.col("placed") + pl.duration(days=400),
        email=pl.lit("not-an-address"),
        note=pl.lit("x" * 50),
        flag=pl.lit(None, dtype=pl.Boolean),
        blob=pl.lit(b"\x00" * 20),
    )
    report = Orders.drift(broken)
    validation = inspect(Orders.spec, broken).by_column()
    assert report.breaking, "the mutations should break something"
    for finding in report.breaking:
        for column in finding.columns:
            assert column in validation, (finding.code, column)


# ---------------------------------------------------------------------------
# Each code, on data built to trip it
# ---------------------------------------------------------------------------


def _drifted(column: str, spec: ColSpec, values: list, **options) -> tuple:
    """The findings on `column` after drifting `values` against `spec`."""
    table = TableSpec("T", {column: spec})
    report = drift(table, pl.DataFrame({column: values}), **options)
    return report.by_column().get(column, ())


def test_bounds_exceeded_says_by_how_much_in_the_columns_own_units():
    spec = ColSpec(pl.Date, bounds=(dt.date(2024, 1, 1), dt.date(2024, 12, 31)))
    (finding,) = _drifted(
        "d", spec, [dt.date(2024, 6, 1), dt.date(2025, 2, 12), dt.date(2023, 12, 31)]
    )
    assert finding.code == "bounds_exceeded" and finding.breaking
    assert finding.details["max_found"] == dt.date(2025, 2, 12)
    assert finding.details["above_by"] == dt.timedelta(days=43)
    assert finding.details["below_by"] == dt.timedelta(days=1)
    assert "max found 2025-02-12 by 43 days above" in finding.message
    assert "min found 2023-12-31 by 1 day below" in finding.message


def test_bounds_exceeded_on_numbers():
    (finding,) = _drifted("n", ColSpec(pl.Int64, bounds=(0, 10)), [3, 15, 7])
    assert finding.details["above_by"] == 5 and "below_by" not in finding.details
    assert finding.details["field"] == "bounds"


def test_an_open_end_is_not_a_bound():
    findings = _drifted("n", ColSpec(pl.Int64, bounds=(0, None)), [3, 10**9])
    assert findings == ()


def test_string_length_exceeded_is_bounds_exceeded_on_length():
    (finding,) = _drifted(
        "s", ColSpec(pl.String, string_length=(2, 4)), ["abc", "toolong", "a"]
    )
    assert finding.code == "bounds_exceeded"
    assert finding.details["field"] == "string_length"
    assert finding.details == {
        "above_by": 3,
        "below_by": 1,
        "field": "string_length",
        "min_found": 1,
        "max_found": 7,
    }


def test_new_values_are_named_and_counted():
    spec = ColSpec(pl.String, choices=["a", "b"])
    findings = _drifted("c", spec, ["a", "z", "z", "y", "b"])
    (new,) = [f for f in findings if f.code == "new_values"]
    assert new.breaking
    assert new.details == {"count": 3, "values": ["z", "y"]}
    assert "3 row(s) hold values outside one of ['a', 'b']: ['z', 'y']" in new.message


def test_new_values_on_an_enum_column_arriving_as_strings():
    """The way a CSV hands an Enum back: compared as the strings they hold."""
    findings = _drifted("c", ColSpec(pl.Enum(["a", "b"])), ["a", "c"])
    codes = sorted(f.code for f in findings)
    assert codes == ["cardinality_moved", "dtype_changed", "new_values"]
    assert not next(f for f in findings if f.code == "dtype_changed").breaking


def test_values_outside_a_finite_format_are_new_values():
    (finding,) = _drifted(
        "c",
        ColSpec(pl.String, format="iso_currency"),
        ["USD", "usd"],
        unseen_values=False,
    )
    assert finding.code == "new_values" and finding.details["values"] == ["usd"]


def test_format_violated_counts_and_samples():
    (finding,) = _drifted(
        "e", ColSpec(pl.String, format="email"), ["a@b.co", "nope", "nope", "x@y"]
    )
    assert finding.code == "format_violated" and finding.breaking
    assert finding.details == {
        "format": "email",
        "count": 3,
        "samples": ["nope", "x@y"],
    }
    assert "3 row(s) are not email (" in finding.message


def test_samples_are_capped_by_max_samples():
    (finding,) = _drifted(
        "c",
        ColSpec(pl.String, choices=["ok"]),
        [f"v{i}" for i in range(20)],
        max_samples=3,
        unseen_values=False,
    )
    assert finding.details["count"] == 20 and len(finding.details["values"]) == 3


def test_cardinality_moved_lists_the_values_never_seen():
    spec = ColSpec(pl.Enum(["a", "b", "c", "d"]))
    values = pl.Series(["a", "a", "b"], dtype=spec.dtype)
    report = drift(TableSpec("T", {"c": spec}), pl.DataFrame({"c": values}))
    (finding,) = report.findings
    assert finding.code == "cardinality_moved" and not finding.breaking
    assert finding.details == {"unseen": ["c", "d"], "declared": 4, "observed": 2}
    assert drift(
        TableSpec("T", {"c": spec}), pl.DataFrame({"c": values}), unseen_values=False
    ).unchanged


@pytest.mark.parametrize(
    "nulls, tolerance, moved",
    [(3, 0.05, True), (3, 0.25, False), (1, 0.05, False)],
)
def test_null_rate_moved_is_measured_against_the_tolerance(nulls, tolerance, moved):
    """Ten rows, declared null_probability=0.1: three nulls is 0.3."""
    spec = ColSpec(pl.Int64, nullable=True, null_probability=0.1)
    values = [None] * nulls + list(range(10 - nulls))
    findings = _drifted("n", spec, values, null_rate_tolerance=tolerance)
    assert bool(findings) is moved
    if moved:
        (finding,) = findings
        assert finding.code == "null_rate_moved" and not finding.breaking
        assert finding.details == {
            "declared": 0.1,
            "observed": 0.3,
            "tolerance": tolerance,
        }


def test_nulls_in_a_non_nullable_column_are_breaking():
    (finding,) = _drifted("n", ColSpec(pl.Int64), [1, None, None])
    assert finding.code == "nullability_changed" and finding.breaking
    assert finding.details == {"null_count": 2}
    assert "Declare nullable=True" in finding.message


def test_dtype_severity_follows_validation():
    (finding,) = _drifted("n", ColSpec(pl.Int64), pl.Series([1, 2], dtype=pl.Int32))
    assert finding.code == "dtype_changed" and not finding.breaking
    (finding,) = _drifted(
        "n", ColSpec(pl.Int64), pl.Series([1, 2], dtype=pl.Int32), strict_dtypes=True
    )
    assert finding.breaking
    (finding,) = _drifted("n", ColSpec(pl.Int64), ["1", "2"])
    assert finding.breaking and finding.details == {"old": "String", "new": "Int64"}


def test_an_empty_column_has_nothing_to_say():
    spec = ColSpec(pl.Int64, bounds=(0, 1), nullable=True, null_probability=0.5)
    findings = _drifted("n", spec, pl.Series([], dtype=pl.Int64))
    assert findings == ()
    findings = _drifted("n", spec, pl.Series([None, None], dtype=pl.Int64))
    assert [f.code for f in findings] == ["null_rate_moved"]


# ---------------------------------------------------------------------------
# Columns and plumbing
# ---------------------------------------------------------------------------


def test_extra_and_missing_columns_are_breaking():
    spec = TableSpec("T", {"a": ColSpec(pl.Int64), "b": ColSpec(pl.Int64)})
    report = drift(spec, pl.DataFrame({"a": [1], "c": [2]}))
    assert {(f.code, f.columns) for f in report} == {
        ("column_removed", ("b",)),
        ("column_added", ("c",)),
    }
    assert all(f.breaking for f in report)


def test_a_lazyframe_is_collected():
    df = Orders.generate(200, seed=1).lazy()
    report = Orders.drift(df)
    assert report.new == "LazyFrame" and report.old == "Orders"


def test_options_are_one_value_or_keywords_not_both():
    df = Orders.generate(50, seed=1)
    opts = DriftOptions(unseen_values=False)
    assert Orders.drift(df, options=opts).options is opts
    with pytest.raises(TypeError, match="not both"):
        Orders.drift(df, options=opts, unseen_values=False)
    with pytest.raises(TypeError, match="did you mean 'unseen_values'"):
        drift(Orders.spec, df, unseen_value=False)
    with pytest.raises(TypeError, match="must be a DriftOptions"):
        drift(Orders.spec, df, options={"unseen_values": False})


def test_the_facade_forwards_only_what_was_given():
    df = Orders.generate(50, seed=1)
    report = Orders.drift(df, null_rate_tolerance=0.5)
    assert report.options == DriftOptions(null_rate_tolerance=0.5)
