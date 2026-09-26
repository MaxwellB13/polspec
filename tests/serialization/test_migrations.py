"""Old spec files, read by today's polspec.

A file records the format version that wrote it, and `migrations.py` brings
an older one forward before it is decoded. That is a promise about files
already on disk, so it is pinned against files: `fixtures/` holds one of
every kind -- spec, category registry, registry of specs -- at every version
that changed its shape, each compared here to the spec it has to become.
A fixture is never regenerated; it is what an old polspec wrote.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest
import yaml
from polspec import (
    CatSpec,
    Check,
    ColRule,
    ColSpec,
    ForeignKey,
    Registry,
    SerializationError,
    TableSpec,
    col,
)
from polspec.serialization import (
    FORMAT_VERSION,
    catspec_from_yaml,
    from_dict,
    from_yaml,
    registry_from_yaml,
    to_dict,
)
from polspec.serialization.migrations import MIGRATIONS, migrate

FIXTURES = Path(__file__).parent / "fixtures"

# One rule per version 1 condition operator, in the order the file lists
# them, as the predicate each has to become.
V1_RULES = [
    (col("g") == "a", "eq"),
    (col("g") != "b", "ne"),
    (col("g").is_in(["a", "b"]), "in"),
    (~col("g").is_in(["c"]), "not_in"),
    (col("n") < 10, "lt"),
    (col("n") <= 20, "le"),
    (col("n") > 90, "gt"),
    (col("n") >= 80, "ge"),
    (col("n").is_between(30, 40), "between"),
    (col("n").is_null(), "null"),
    (col("n").is_not_null(), "present"),
    (col("n").is_not_null(), "not_null"),
    (col("n").is_null(), "absent"),
]

V1_SPEC = TableSpec(
    "Legacy",
    {
        "g": ColSpec(pl.Enum(["a", "b", "c"]), tags=["geo", "dim"]),
        "n": ColSpec(pl.Int64, nullable=True, bounds=(0, 100), tags="num"),
        "x": ColSpec(
            pl.Float64,
            bounds=(0.0, 50.0),
            distribution="exponential",
            distribution_params={"rate": 2.0},
        ),
        "y": ColSpec(
            pl.Float64,
            bounds=(0.0, 10.0),
            distribution="normal",
            distribution_params={"mean": 5.0, "std": 1.0},
        ),
        "v": ColSpec(
            pl.String,
            rules=[ColRule(when=when, choices=[value]) for when, value in V1_RULES],
        ),
    },
    foreign_keys=[ForeignKey("n", references="self")],
)

V2_SPEC = TableSpec(
    "Orders",
    {
        "id": ColSpec(pl.Int64, unique=True, bounds=(1, 1_000_000)),
        "customer_id": ColSpec(pl.Int64, tags=["key"]),
        "total": ColSpec(
            pl.Decimal(10, 2),
            bounds=(Decimal("0.00"), Decimal("999.99")),
            validators=[Check(col("total") >= 0, name="total_non_negative")],
        ),
        "status": ColSpec(
            pl.Enum(["NEW", "PAID"]), choices=["NEW", "PAID"], weights=[3.0, 1.0]
        ),
    },
    foreign_keys=[ForeignKey("customer_id", references="Customers", ref_columns="id")],
    checks=[
        Check(
            (col("status") == "NEW") | (col("total") > 0),
            name="paid_orders_cost_something",
        )
    ],
    unique_together=[["customer_id", "status"]],
)


def _raw(name: str) -> dict:
    return yaml.safe_load((FIXTURES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Every fixture reads as the spec it has to become
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [("spec_v1.yaml", V1_SPEC), ("spec_v2.yaml", V2_SPEC)],
)
def test_an_old_spec_file_reads_as_the_spec_it_declared(fixture, expected):
    spec = from_yaml(FIXTURES / fixture)
    assert spec == expected
    # Its rules as predicates, compared structurally: `==` on a Pred builds one.
    for name, column in expected.columns.items():
        for got, want in zip(spec[name].rules, column.rules, strict=True):
            assert got.when.equals(want.when), (name, want.when)


@pytest.mark.parametrize("fixture", ["spec_v1.yaml", "spec_v2.yaml"])
def test_an_old_spec_file_writes_back_as_the_current_version(fixture, tmp_path):
    """Read, written, read again: the file is now the current version, and
    says the same thing."""
    spec = from_yaml(FIXTURES / fixture)
    written = to_dict(spec)
    assert written["version"] == FORMAT_VERSION
    assert from_dict(written) == spec


@pytest.mark.parametrize("fixture", ["catspec_v1.yaml", "catspec_v2.yaml"])
def test_an_old_category_registry_reads_as_it_declared(fixture):
    cats = catspec_from_yaml(FIXTURES / fixture)
    assert cats.status == pl.Enum(["A", "B"])
    assert cats.get_categorical("currency").physical() == pl.UInt8
    if fixture == "catspec_v2.yaml":
        assert cats.get_choices("CURRENCY") == ["GBP", "USD"]
        assert cats.get_choices("plain") == ["x", "y"]
    assert CatSpec.from_dict(cats.to_dict()) == cats


def test_an_old_registry_file_reads_as_it_declared():
    registry = registry_from_yaml(FIXTURES / "registry_v2.yaml")
    assert isinstance(registry, Registry)
    assert registry.names == ("Customers", "Orders")
    orders = registry["Orders"]
    assert orders["status"].dtype == pl.Enum(["A", "B"])
    assert orders.foreign_keys[0].references == "Customers"
    assert registry.resolve().order() == ("Customers", "Orders")


# ---------------------------------------------------------------------------
# The migration steps themselves
# ---------------------------------------------------------------------------


def test_every_kind_has_a_step_from_every_old_version():
    """A version bump that forgets a kind would leave its files unreadable."""
    for kind, steps in MIGRATIONS.items():
        assert sorted(steps) == list(range(1, FORMAT_VERSION)), kind


def test_a_legacy_tag_spelling_gives_way_to_tags():
    """`tags` wins where a version 1 file spelt it both ways."""
    migrated = migrate(
        {"columns": {"c": {"dtype": "Int64", "category": "old", "tags": "new"}}},
        "spec",
        "legacy.yaml",
    )
    assert migrated["columns"]["c"] == {"dtype": "Int64", "tags": "new"}


def test_an_unknown_version_1_distribution_is_left_for_the_decoder():
    migrated = migrate(
        {"columns": {"c": {"dtype": "Float64", "distribution": "cauchy"}}},
        "spec",
        "legacy.yaml",
    )
    assert migrated["columns"]["c"]["distribution"] == "cauchy"
    with pytest.raises(ValueError, match="cauchy"):
        from_dict(migrated)


@pytest.mark.parametrize(
    ("condition", "complaint"),
    [
        # A condition naming no column is not the version 1 form at all, so
        # it is read -- and refused -- as a predicate.
        ({"equals": "a"}, "unknown operation 'equals'"),
        ({"column": "g"}, "exactly one of"),
        ({"column": "g", "equals": "a", "in": ["a"]}, "exactly one of"),
        ({"column": "n", "between": [5, 1]}, "min <= max"),
        ({"column": "n", "between": [1]}, "2-element"),
        ({"column": "g", "in": "a"}, "requires a collection"),
    ],
)
def test_a_malformed_version_1_condition_is_refused_by_name(condition, complaint):
    data = {
        "name": "Legacy",
        "columns": {
            "g": {"dtype": "String"},
            "n": {"dtype": "Int64"},
            "v": {"dtype": "String", "rules": [{"when": condition, "choices": ["x"]}]},
        },
    }
    with pytest.raises(ValueError, match=complaint):
        from_dict(data)


def test_a_flat_version_1_registry_refuses_what_it_cannot_read():
    with pytest.raises(SerializationError, match="Cannot read registry entry"):
        CatSpec.from_dict({"STATUS": 3})


@pytest.mark.parametrize("version", [0, -1, True, 1.5])
def test_a_version_that_is_not_a_positive_integer_is_refused(version):
    with pytest.raises(SerializationError, match="positive integer"):
        migrate({"version": version}, "spec", "odd.yaml")


def test_a_file_that_is_not_a_mapping_is_refused():
    with pytest.raises(SerializationError, match="expected a mapping"):
        migrate(["not", "a", "mapping"], "spec", "odd.yaml")
