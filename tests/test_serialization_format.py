"""The spec-file format itself: the field registry, versions, and strictness.

`test_serialization.py` proves specs round-trip. This file pins the
machinery that makes that true: every dataclass field has a registry entry,
files carry a version, older versions migrate, unknown keys are refused.
"""

import dataclasses
import warnings

import polars as pl
import pytest
import yaml
from polspec import (
    CatSpec,
    Check,
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSpec,
    SerializationError,
    TableSpec,
    col,
)
from polspec.serialization import (
    CHECK_FIELDS,
    COLRULE_FIELDS,
    COLSPEC_FIELDS,
    FK_FIELDS,
    FORMAT_VERSION,
    TABLESPEC_FIELDS,
    from_dict,
    to_dict,
)
from polspec.serialization.migrations import migrate

# ---------------------------------------------------------------------------
# One registry entry per dataclass field
# ---------------------------------------------------------------------------

# Fields the registry deliberately covers under another key, or derives.
_DERIVED = {
    Check: {"expr"}
}  # `expr` is derived from `pred`; the registry writes `pred` as `expr`


@pytest.mark.parametrize(
    "cls, fields",
    [
        (ColSpec, COLSPEC_FIELDS),
        (ColRule, COLRULE_FIELDS),
        (Check, CHECK_FIELDS),
        (ForeignKey, FK_FIELDS),
        (TableSpec, TABLESPEC_FIELDS),
    ],
    ids=lambda x: getattr(x, "__name__", ""),
)
def test_every_dataclass_field_has_a_registry_entry(cls, fields):
    declared = {f.name for f in dataclasses.fields(cls)} - _DERIVED.get(cls, set())
    registered = {f.attribute for f in fields}
    assert registered == declared, (
        f"{cls.__name__}: registry and dataclass disagree. "
        f"Missing from registry: {declared - registered}; "
        f"registry-only: {registered - declared}"
    )


# ---------------------------------------------------------------------------
# Version key
# ---------------------------------------------------------------------------


class Sample(FrameSpec):
    a = ColSpec(pl.Int64, bounds=(0, 10))
    b = ColSpec(pl.Enum(["x", "y"]))


def test_files_record_the_format_version(tmp_path):
    path = tmp_path / "sample.yaml"
    Sample.to_yaml(path)
    data = yaml.safe_load(path.read_text())
    assert next(iter(data)) == "version"
    assert data["version"] == FORMAT_VERSION == 2
    assert to_dict(Sample.spec)["version"] == FORMAT_VERSION

    cat_path = tmp_path / "cats.yaml"
    CatSpec(enums={"STATUS": ["A", "B"]}).to_yaml(cat_path)
    assert yaml.safe_load(cat_path.read_text())["version"] == FORMAT_VERSION


def test_a_newer_file_is_refused_plainly(tmp_path):
    path = tmp_path / "future.yaml"
    path.write_text(
        yaml.safe_dump(
            {"version": 99, "name": "X", "columns": {"a": {"dtype": "Int64"}}}
        )
    )
    with pytest.raises(SerializationError, match="written by a newer polspec"):
        FrameSpec.from_yaml(path)
    with pytest.raises(SerializationError, match="must be a positive integer"):
        from_dict({"version": "two", "columns": {"a": {"dtype": "Int64"}}})


# ---------------------------------------------------------------------------
# Migrating a version-1 file
# ---------------------------------------------------------------------------

V1_FILE = {
    "name": "Legacy",
    "columns": {
        "region": {"dtype": {"Enum": ["UK", "US"]}, "category": "geo"},
        "amount": {
            "dtype": "Float64",
            "distribution": "Normal",
            "distribution_params": {"mu": 5, "sigma": 2},
        },
        "carrier": {
            "dtype": {"Enum": ["RM", "UPS"]},
            "rules": [
                {"when": {"column": "region", "equals": "UK"}, "choices": ["RM"]}
            ],
        },
    },
    "foreign_keys": [{"columns": ["region"], "references": "self"}],
}


def test_version_1_files_migrate_to_the_current_form():
    migrated = migrate(V1_FILE, "spec", "legacy.yaml")
    assert migrated["version"] == FORMAT_VERSION
    region = migrated["columns"]["region"]
    assert region["tags"] == "geo" and "category" not in region
    amount = migrated["columns"]["amount"]
    assert amount["distribution"] == "normal"
    assert amount["distribution_params"] == {"mean": 5, "std": 2}
    assert migrated["columns"]["carrier"]["rules"][0]["when"] == {
        "eq": [{"col": "region"}, "UK"]
    }

    spec = from_dict(V1_FILE)
    assert spec["region"].tags == ("geo",)
    assert spec["amount"].distribution_params == {"mean": 5.0, "std": 2.0}
    assert spec["carrier"].rules[0].when.equals(col("region") == "UK")
    # Writing it back produces a current-format file that reads identically.
    assert from_dict(to_dict(spec)) == spec


def test_declarations_store_canonical_distribution_forms():
    spec = ColSpec(pl.Float64, distribution="Exp", distribution_params={"lambda_": 3})
    assert spec.distribution == "exponential"
    assert spec.distribution_params == {"rate": 3.0}
    assert ColSpec(
        pl.Float64, distribution="normal", distribution_params={"loc": 1, "scale": 2}
    ).distribution_params == {
        "mean": 1.0,
        "std": 2.0,
    }
    # An alias the engine would never read is kept rather than dropped.
    assert ColSpec(
        pl.Float64, distribution="uniform", distribution_params={"whatever": 1}
    ).distribution_params == {"whatever": 1.0}


# ---------------------------------------------------------------------------
# Unknown keys
# ---------------------------------------------------------------------------


def test_unknown_keys_are_an_error_naming_the_closest_known_key(tmp_path):
    data = {
        "version": 2,
        "name": "X",
        "columns": {"a": {"dtype": "Int64", "nullible": True}},
    }
    with pytest.raises(
        SerializationError,
        match=r"'columns\.a\.nullible' \(did you mean 'nullable'\?\)",
    ):
        from_dict(data)
    with pytest.raises(
        SerializationError, match="'colums' \\(did you mean 'columns'\\?\\)"
    ):
        from_dict(
            {
                "version": 2,
                "name": "X",
                "colums": {},
                "columns": {"a": {"dtype": "Int64"}},
            }
        )


def test_strict_false_downgrades_unknown_keys_to_a_warning(tmp_path):
    path = tmp_path / "loose.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "name": "X",
                "columns": {"a": {"dtype": "Int64", "colour": "blue"}},
            }
        )
    )
    with pytest.warns(UserWarning, match="Unknown key"):
        loaded = FrameSpec.from_yaml(path, strict=False)
    assert list(loaded.spec) == ["a"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(SerializationError):
            FrameSpec.from_yaml(path)


def test_registry_files_reject_unknown_keys_and_physical_dtypes(tmp_path):
    with pytest.raises(SerializationError, match="Unknown key"):
        CatSpec.from_dict({"version": 2, "enums": {"A": ["x"]}, "enumz": {}})
    with pytest.raises(SerializationError, match="Unknown physical dtype 'UInt7'"):
        CatSpec(categoricals={"C": {"physical": "UInt7"}})


# ---------------------------------------------------------------------------
# CatSpec files
# ---------------------------------------------------------------------------


def test_registry_round_trips_including_loose_choices(tmp_path):
    cats = CatSpec(
        enums={"STATUS": ["NEW", "PAID"]},
        categoricals={"CURRENCY": pl.Categories("CURRENCY", physical=pl.UInt8)},
        choices={"CURRENCY": ["GBP", "USD"], "plain": ["a", "b"]},
    )
    path = tmp_path / "cats.yaml"
    cats.to_yaml(path)
    loaded = CatSpec.from_yaml(path)
    assert loaded.enums == {"STATUS": ["NEW", "PAID"]}
    assert loaded.currency.physical() == pl.UInt8
    assert loaded.get_choices("CURRENCY") == ["GBP", "USD"]
    assert loaded.get_choices("plain") == ["a", "b"]
    assert loaded.to_dict() == cats.to_dict()


def test_flat_version_1_registry_still_loads():
    cats = CatSpec.from_dict({"STATUS": ["A", "B"], "CURRENCY": {"physical": "UInt8"}})
    assert cats.status == ["A", "B"]
    assert cats.currency.physical() == pl.UInt8
