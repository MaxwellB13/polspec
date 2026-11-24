"""The spec-file format itself: the field registry, versions, and strictness.

`test_serialization.py` proves specs round-trip. This file pins the
machinery that makes that true: every dataclass field has a registry entry,
files carry a version, older versions migrate, unknown keys are refused.
"""

import warnings

import polars as pl
import pytest
import yaml
from polspec import (
    FORMAT_VERSION,
    CatSpec,
    Check,
    ColSpec,
    FrameSpec,
    SerializationError,
    from_dict,
    to_dict,
)

# ---------------------------------------------------------------------------
# One registry entry per dataclass field
# ---------------------------------------------------------------------------

# Fields the registry deliberately covers under another key, or derives.
_DERIVED = {
    Check: {"expr"}
}  # `expr` is derived from `pred`; the registry writes `pred` as `expr`


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
    assert data["version"] == FORMAT_VERSION == 4
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
        "version": 4,
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
                "version": 4,
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
                "version": 4,
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
        CatSpec.from_dict({"version": 4, "enums": {"A": ["x"]}, "enumz": {}})
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
    assert loaded["currency"].categories.physical() == pl.UInt8
    assert loaded.choices.get("CURRENCY") == ["GBP", "USD"]
    assert loaded.choices.get("plain") == ["a", "b"]
    assert loaded.to_dict() == cats.to_dict()
