"""The dtype codec: how a dtype is written to a spec file and read back.

Every dtype polspec generates is written as data and as Python, and has to
come back as itself from both; a named `Categories`, a zoned `Datetime` and
a registry reference are the forms with the most to lose. Each malformed
form is refused naming the form it should have taken.
"""

from __future__ import annotations

import polars as pl
import pytest
import yaml
from cases import EVERY_DTYPE
from polspec import (
    CatSpec,
    ColSpec,
    SerializationError,
    TableSpec,
    from_dict,
    profile,
    to_dict,
    to_python,
    to_yaml,
)

NAMED = [
    pl.Categorical(pl.Categories("colour")),
    pl.Categorical(pl.Categories("colour", namespace="shop", physical=pl.UInt8)),
    pl.Categorical(pl.Categories("size", physical=pl.UInt16)),
    pl.Datetime("ns", "Europe/London"),
    pl.Duration("ns"),
    pl.List(pl.Categorical(pl.Categories("colour"))),
    # Holds no data, but a profiled file can have a column of it.
    pl.Null(),
    pl.Struct({"empty": pl.Null, "x": pl.Int64}),
]


def _normalised(dtype):
    return dtype() if isinstance(dtype, type) else dtype


def _column(dtype: pl.DataType) -> TableSpec:
    return TableSpec("T", {"c": ColSpec(dtype, nullable=True)})


def _read(dtype_data, categories: CatSpec | None = None) -> pl.DataType:
    """The dtype a spec file holding `dtype_data` declares."""
    data = {"name": "T", "columns": {"c": {"dtype": dtype_data}}}
    return from_dict(data, categories=categories)["c"].dtype


@pytest.mark.parametrize("dtype", EVERY_DTYPE + NAMED, ids=str)
def test_every_dtype_survives_data_and_source(dtype):
    dtype = _normalised(dtype)
    spec = _column(dtype)
    data = yaml.safe_load(to_yaml(spec))
    assert from_dict(data)["c"].dtype == dtype
    namespace: dict = {}
    exec(to_python(spec), namespace)
    assert namespace["T"].spec["c"].dtype == dtype


def test_a_categoricals_physical_dtype_is_named_and_read_back():
    cats = CatSpec(categoricals={"C": {"physical": "UInt16"}})
    written = cats.to_dict()
    assert written["categoricals"]["C"]["physical"] == "UInt16"
    assert CatSpec.from_dict(written)["C"] == cats["C"]
    with pytest.raises(SerializationError, match="Unknown physical dtype"):
        CatSpec(categoricals={"C": {"physical": "Float64"}})


def test_a_registry_reference_resolves_against_the_registry():
    cats = CatSpec(
        enums={"STATUS": ["A", "B"]}, categoricals={"COLOUR": {"physical": "UInt8"}}
    )
    assert _read("$categories.STATUS", cats) == pl.Enum(["A", "B"])
    assert _read("categories.STATUS", cats) == pl.Enum(["A", "B"])
    assert _read("STATUS", cats) == pl.Enum(["A", "B"])
    assert _read({"Enum": "STATUS"}, cats) == pl.Enum(["A", "B"])
    colour = _read({"Categorical": "COLOUR"}, cats)
    assert colour == cats.get("COLOUR")
    assert _read({"Categorical": {"name": "COLOUR"}}, cats) == colour
    # A name the registry does not hold is a fresh, named Categories.
    assert _read({"Categorical": "fresh"}) == pl.Categorical(pl.Categories("fresh"))
    assert _read({"Categorical": "Categorical"}) == pl.Categorical()


@pytest.mark.parametrize(
    ("data", "complaint"),
    [
        ({"Decimal": [10, 2]}, r"\{Decimal: \{precision: P, scale: S\}\}"),
        ({"Struct": ["a"]}, r"\{Struct: \{name: <dtype>"),
        ({"Array": {"inner": "Int64"}}, r"\{Array: \{inner: <dtype>, width: N\}\}"),
        ({"Map": {"key": "Int64"}}, r"\{Map: \{key: <dtype>, value: <dtype>\}\}"),
        ({"Categorical": 3}, "must be a name or a mapping"),
        ({"Enum": "STATUS"}, "not found in provided CatSpec"),
        ("$categories.STATUS", "not found in provided CatSpec"),
        ({"Nope": 1}, "Unrecognized dtype mapping"),
        ({"List": "Int64", "Array": "Int64"}, "Unrecognized dtype mapping"),
        ("Nope", "Unrecognized dtype name"),
        (3, "must be a name or a mapping"),
    ],
)
def test_a_malformed_dtype_is_refused_naming_its_form(data, complaint):
    with pytest.raises(SerializationError, match=complaint):
        _read(data)


def test_a_dtype_with_no_file_form_is_refused():
    """An `Object` column is profiled faithfully, but has no file form."""
    spec = profile(pl.DataFrame({"c": pl.Series([object()], dtype=pl.Object)}))
    with pytest.raises(SerializationError, match="cannot write dtype"):
        to_dict(spec)
    with pytest.raises(SerializationError, match="cannot write dtype"):
        to_python(spec)
