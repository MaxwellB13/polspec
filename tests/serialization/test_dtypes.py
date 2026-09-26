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
from polspec import CatSpec, SerializationError
from polspec.serialization.dtypes import (
    dtype_from_data,
    dtype_to_data,
    dtype_to_source,
    physical_from_name,
    physical_name,
)

NAMED = [
    pl.Categorical(pl.Categories("colour")),
    pl.Categorical(pl.Categories("colour", namespace="shop", physical=pl.UInt8)),
    pl.Categorical(pl.Categories("size", physical=pl.UInt16)),
    pl.Datetime("ns", "Europe/London"),
    pl.Duration("ns"),
    pl.List(pl.Categorical(pl.Categories("colour"))),
]


def _normalised(dtype):
    return dtype() if isinstance(dtype, type) else dtype


@pytest.mark.parametrize("dtype", EVERY_DTYPE + NAMED, ids=str)
def test_every_dtype_survives_data_and_source(dtype):
    dtype = _normalised(dtype)
    data = yaml.safe_load(yaml.safe_dump(dtype_to_data(dtype)))
    assert dtype_from_data(data) == dtype
    assert eval(dtype_to_source(dtype), {"pl": pl}) == dtype  # noqa: S307


def test_a_physical_dtype_is_named_and_read_back():
    assert physical_name(pl.UInt8) == "UInt8"
    assert physical_from_name("UInt16") == pl.UInt16
    assert physical_from_name(pl.UInt8) == pl.UInt8
    with pytest.raises(SerializationError, match="is not a dtype a Categorical"):
        physical_name(pl.Object)
    with pytest.raises(SerializationError, match="Unknown physical dtype"):
        physical_from_name("Float64")


def test_a_registry_reference_resolves_against_the_registry():
    cats = CatSpec(
        enums={"STATUS": ["A", "B"]}, categoricals={"COLOUR": {"physical": "UInt8"}}
    )
    assert dtype_from_data("$categories.STATUS", cats) == pl.Enum(["A", "B"])
    assert dtype_from_data("categories.STATUS", cats) == pl.Enum(["A", "B"])
    assert dtype_from_data("STATUS", cats) == pl.Enum(["A", "B"])
    assert dtype_from_data({"Enum": "STATUS"}, cats) == pl.Enum(["A", "B"])
    colour = dtype_from_data({"Categorical": "COLOUR"}, cats)
    assert colour == cats.dtype_of("COLOUR")
    assert dtype_from_data({"Categorical": {"name": "COLOUR"}}, cats) == colour
    # A name the registry does not hold is a fresh, named Categories.
    assert dtype_from_data({"Categorical": "fresh"}) == pl.Categorical(
        pl.Categories("fresh")
    )
    assert dtype_from_data({"Categorical": "Categorical"}) == pl.Categorical()


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
        dtype_from_data(data)


@pytest.mark.parametrize("dtype", [pl.Object(), pl.Null()], ids=str)
def test_a_dtype_with_no_file_form_is_refused(dtype):
    with pytest.raises(SerializationError, match="cannot write dtype"):
        dtype_to_data(dtype)
    with pytest.raises(SerializationError, match="cannot write dtype"):
        dtype_to_source(dtype)
