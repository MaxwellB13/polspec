"""`Map(key, value)`: the dtype Polars 2 introduced, named before it is
generated.

A `Map` column declares and validates by dtype, profiles, drifts, and
round-trips through a spec file; `generate()` refuses it by name until its
values can be drawn. Polars 1 has no `Map`, so most of this runs only where
it exists, and one test pins what Polars 1 says on meeting one in a file.
"""

from __future__ import annotations

import polars as pl
import pytest
from polspec import ColSpec, FrameSpec, SpecError, TableSpec, generate, inspect
from polspec.drift import drift
from polspec.errors import SerializationError
from polspec.serialization import dtypes as dtype_codec

HAS_MAP = hasattr(pl, "Map")
needs_map = pytest.mark.skipif(not HAS_MAP, reason="Map is a Polars 2 dtype")


def _map(key, value):
    return pl.Map(key, value)  # type: ignore[attr-defined]


def _frame(dtype):
    return pl.DataFrame({"m": pl.Series([{"a": 1}, {"b": 2}, None], dtype=dtype)})


@needs_map
def test_a_map_declares_validates_and_drifts_by_dtype():
    dtype = _map(pl.String, pl.Int64)
    spec = TableSpec("T", {"m": ColSpec(dtype, nullable=True)})
    df = _frame(dtype)
    assert inspect(spec, df).passed
    assert drift(spec, df).breaking == ()
    wrong = df.with_columns(pl.lit(1).alias("m"))
    assert [f.code for f in inspect(spec, wrong)] == ["dtype"]


@needs_map
def test_generate_refuses_a_map_by_name():
    spec = TableSpec("T", {"m": ColSpec(_map(pl.String, pl.Int64))})
    with pytest.raises(SpecError, match=r"cannot generate data for dtype Map"):
        generate(spec, 5, seed=1)


@needs_map
def test_a_map_round_trips_through_a_spec_file(tmp_path):
    dtype = _map(pl.String, pl.List(pl.Int64))
    spec_cls = FrameSpec.from_spec(TableSpec("T", {"m": ColSpec(dtype, nullable=True)}))
    spec_cls.to_yaml(tmp_path / "s.yaml")
    text = (tmp_path / "s.yaml").read_text(encoding="utf-8")
    assert "Map:" in text and "key: String" in text
    assert FrameSpec.from_yaml(tmp_path / "s.yaml").spec == spec_cls.spec
    spec_cls.to_python(tmp_path / "s.py")
    namespace: dict = {}
    exec((tmp_path / "s.py").read_text(encoding="utf-8"), namespace)
    assert namespace[spec_cls.__name__].spec == spec_cls.spec


@needs_map
def test_a_profiled_map_column_can_be_saved_and_checked(tmp_path):
    """`polspec schema infer` on a file with a map column: profiled, written,
    read back, and the data it came from validates."""
    df = _frame(_map(pl.String, pl.Int64))
    profiled = FrameSpec.from_dataframe(df)
    profiled.to_yaml(tmp_path / "p.yaml")
    FrameSpec.from_yaml(tmp_path / "p.yaml").validate(df)


def test_polars_1_names_the_version_a_map_needs(monkeypatch):
    monkeypatch.setattr(dtype_codec, "_MAP", None)
    with pytest.raises(SerializationError, match="a dtype Polars 2 introduced"):
        dtype_codec.dtype_from_data({"Map": {"key": "String", "value": "Int64"}})
    with pytest.raises(SerializationError, match=r"written as \{Map"):
        dtype_codec.dtype_from_data({"Map": ["String", "Int64"]})
