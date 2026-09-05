"""`inspect()` and `ValidationReport`: validation results as data.

`validate()` raising is the right shape for a person reading a traceback.
`inspect()` returns the same findings as structured `Finding` records, with
the offending rows reachable lazily, for anything that wants to act on them.
"""

import datetime as dt
import json

import polars as pl
import pytest
from polspec import (
    Check,
    ColRule,
    ColSpec,
    Finding,
    ForeignKey,
    FrameSpec,
    ValidationError,
    ValidationReport,
    col,
)
from polspec.validation import FINDING_COLUMN


class Customers(FrameSpec):
    id = ColSpec(pl.Int64, unique=True)


class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, unique=True)
    customer_id = ColSpec(pl.Int64)
    status = ColSpec(pl.Enum(["NEW", "PAID"]))
    carrier = ColSpec(
        pl.Enum(["RM", "UPS"]),
        rules=[ColRule(when=col("status") == "PAID", choices=["UPS"])],
    )
    total = ColSpec(pl.Float64, bounds=(0.0, 100.0), validators=[col("total") != 13.0])
    note = ColSpec(pl.String, string_length=(1, 5), nullable=True)
    placed = ColSpec(pl.Date, bounds=(dt.date(2024, 1, 1), None))
    __unique_together__ = [["customer_id", "order_id"]]
    __checks__ = [Check(col("total") >= 1.0, name="at_least_one")]
    __foreign_keys__ = [
        ForeignKey("customer_id", references=Customers, ref_columns="id")
    ]


CUSTOMERS = pl.DataFrame({"id": [1, 2, 3]})

BAD = pl.DataFrame(
    {
        "order_id": [1, 1, 3, 4],  # duplicate -> unique
        "customer_id": [1, 1, 9, 2],  # 9 -> foreign key
        "status": ["NEW", "PAID", "PAID", "NEW"],
        "carrier": ["RM", "RM", "UPS", "RM"],  # row 2: PAID must be UPS -> rule
        "total": [
            50.0,
            13.0,
            200.0,
            0.5,
        ],  # 13 -> validator; 200 -> bounds; 0.5 -> check
        "note": ["ok", None, "too long", "x"],  # -> string_length
        "placed": [dt.date(2024, 5, 1), dt.date(2023, 1, 1), None, dt.date(2024, 1, 1)],
    },
    schema_overrides={
        "status": pl.Enum(["NEW", "PAID"]),
        "carrier": pl.Enum(["RM", "UPS"]),
    },
)


def test_inspect_returns_a_report_and_never_raises():
    report = Orders.inspect(BAD, references={Customers: CUSTOMERS})
    assert isinstance(report, ValidationReport)
    assert not report.passed and not report
    assert report.spec_name == "Orders"
    codes = sorted(f.code for f in report)
    assert codes == sorted(
        [
            "unique",
            "unique_together",
            "foreign_key",
            "rule",
            "validator",
            "bounds",
            "bounds",
            "check",
            "string_length",
            "nullability",
        ]
    )


def test_findings_carry_structure_not_just_prose():
    report = Orders.inspect(BAD, references={Customers: CUSTOMERS})
    bounds = next(f for f in report.by_code("bounds") if "total" in f.columns)
    assert isinstance(bounds, Finding)
    assert bounds.key == "total__bounds"
    assert bounds.columns == ("total",)
    assert bounds.count == 1
    assert bounds.samples == (200.0,)
    assert bounds.details["bounds"] == [0.0, 100.0]
    assert bounds.details["max_found"] == 200.0
    assert "out of bounds" in bounds.message

    check = report.by_code("check")[0]
    assert check.key == "check:at_least_one"
    assert check.columns == ("total",)
    assert check.count == 1
    assert check.samples  # checks now sample the values they looked at
    assert check.details["check"] == "at_least_one"

    fk = report.by_code("foreign_key")[0]
    assert fk.details == {"target": "Customers", "ref_columns": ["id"]}
    assert fk.samples == (9,)


def test_rows_are_lazy_and_exact():
    report = Orders.inspect(BAD, references={Customers: CUSTOMERS})
    rule = report.by_code("rule")[0]
    rows = report.rows(rule)
    assert isinstance(rows, pl.LazyFrame)
    assert rows.collect()["order_id"].to_list() == [1]  # the PAID row carried RM

    fk = report.by_code("foreign_key")[0]
    assert report.rows(fk).collect()["customer_id"].to_list() == [9]

    failing = report.failing_rows().collect()
    assert FINDING_COLUMN in failing.columns
    assert set(failing[FINDING_COLUMN].to_list()) == {
        f.key for f in report if f.row_level
    }
    # A row violating several claims appears once per claim.
    assert failing.filter(pl.col("order_id") == 1).height >= 2


def test_structural_findings_have_no_rows():
    df = BAD.drop("note").with_columns(pl.lit(1).alias("extra"))
    report = Orders.inspect(df, references={Customers: CUSTOMERS})
    structural = {f.code for f in report if not f.row_level}
    assert {"extra_columns", "missing_columns"} <= structural
    missing = report.by_code("missing_columns")[0]
    assert missing.columns == ("note",)
    with pytest.raises(ValueError, match="structural and has no rows"):
        report.rows(missing)
    # Structural findings still group under the columns they name.
    grouped = report.by_column()
    assert {f.code for f in grouped["note"]} == {"missing_columns"}
    assert {f.code for f in grouped["extra"]} == {"extra_columns"}


def test_by_column_groups_every_involved_column():
    report = Orders.inspect(BAD, references={Customers: CUSTOMERS})
    grouped = report.by_column()
    assert {f.code for f in grouped["total"]} == {"bounds", "validator", "check"}
    assert {f.code for f in grouped["customer_id"]} >= {
        "foreign_key",
        "unique_together",
    }


def test_report_serialises_to_json():
    report = Orders.inspect(BAD, references={Customers: CUSTOMERS})
    data = json.loads(report.to_json())
    assert data["spec"] == "Orders" and data["passed"] is False
    placed = next(f for f in data["findings"] if f["key"] == "placed__bounds")
    assert placed["samples"] == ["2023-01-01"]  # dates become ISO strings
    assert placed["details"]["bounds"] == ["2024-01-01", None]
    assert set(data["findings"][0]) == {
        "code",
        "key",
        "message",
        "columns",
        "count",
        "samples",
        "details",
    }


def test_passing_report_is_truthy_and_empty():
    good = Orders.generate(50, seed=1, references={Customers: CUSTOMERS})
    report = Orders.inspect(
        good,
        references={Customers: CUSTOMERS},
        validate_unique=False,
        validate_validators=False,
        validate_checks=False,
    )
    assert report.passed and bool(report) and len(report) == 0
    assert report.failing_rows().collect().height == 0
    assert str(report) == "Validation passed for DataFrame against 'Orders'"
    report.raise_if_failed()  # no-op


def test_validate_raises_with_the_report_attached():
    with pytest.raises(ValidationError) as info:
        Orders.validate(BAD, references={Customers: CUSTOMERS})
    err = info.value
    assert isinstance(err.report, ValidationReport)
    assert err.errors == [f.message for f in err.report.findings]
    assert str(err) == str(err.report)
    assert str(err).startswith("Validation failed for DataFrame against 'Orders' (")


def test_unresolved_foreign_key_is_a_finding_not_an_exception():
    report = Orders.inspect(BAD)
    unresolved = report.by_code("foreign_key_unresolved")
    assert len(unresolved) == 1
    assert unresolved[0].details == {"target": "Customers"}
    assert not unresolved[0].row_level
    with pytest.raises(ValidationError, match="no DataFrame for it was supplied"):
        Orders.validate(BAD)


def test_parent_missing_the_referenced_column_is_a_finding():
    report = Orders.inspect(BAD, references={Customers: pl.DataFrame({"other": [1]})})
    fk = report.by_code("foreign_key")[0]
    assert fk.details["missing_ref_columns"] == ["id"]
    assert fk.count is None


def test_lazyframe_in_report_stays_lazy():
    report = Orders.inspect(BAD.lazy(), references={Customers: CUSTOMERS.lazy()})
    assert isinstance(report.frame, pl.LazyFrame)
    assert report.by_code("foreign_key")[0].count == 1
