"""`Map(key, value)`: the dtype Polars 2 introduced.

A `Map` is the list of `{key, value}` entries it is, so it is declared and
generated as one -- `list_length`, `fields` -- with a map's own two rules on
top: a key is never null, and no key repeats within a map. It validates,
drifts, profiles and is sized as that list too, with a finding about a key
or a value named `m.key` or `m.value`, and round-trips through a spec file.
Polars 1 has no `Map`, so most of this runs only where it exists, and one
test pins what Polars 1 says on meeting one in a file.
"""

from __future__ import annotations

import polars as pl
import pytest
from polspec import (
    ColSpec,
    FrameSpec,
    SpecError,
    TableSpec,
    generate,
    generate_batches,
    inspect,
)
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


def _distinct_keys(column: pl.Series) -> bool:
    """Every present map in `column` holds as many keys as distinct ones."""
    maps = column.drop_nulls()
    return bool((maps.map.len() == maps.map.keys().list.n_unique()).all())


@needs_map
def test_a_map_generates_with_distinct_keys_and_its_declared_claims():
    spec = TableSpec(
        "T",
        {
            "m": ColSpec(
                _map(pl.String, pl.Int64),
                list_length=(1, 4),
                fields={
                    "key": ColSpec(pl.String, format="hostname"),
                    "value": ColSpec(pl.Int64, bounds=(0, 9), nullable=True),
                },
            )
        },
    )
    df = generate(spec, 2_000, seed=1)
    m = df["m"]
    assert m.dtype == _map(pl.String, pl.Int64)
    assert m.null_count() == 0
    assert m.map.len().is_between(1, 4).all()
    assert _distinct_keys(m)
    keys = m.map.keys().explode()
    assert keys.null_count() == 0
    assert keys.str.contains(r"^[a-z0-9.-]+$").all()
    values = m.map.values().explode()
    assert values.null_count() > 0
    assert values.drop_nulls().is_between(0, 9).all()
    assert generate(spec, 2_000, seed=1).equals(df)
    assert inspect(spec, df).passed
    assert drift(spec, df).breaking == ()


@needs_map
def test_keys_too_few_for_a_long_map_are_separated_not_folded():
    """Eight entries from ten keys: nearly every map draws a repeat, and
    Polars' cast would fold one into the other and leave a map short."""
    spec = TableSpec(
        "T",
        {
            "m": ColSpec(
                _map(pl.Int8, pl.List(pl.Int8)),
                list_length=(8, 8),
                fields={"key": ColSpec(pl.Int8, bounds=(0, 9))},
            )
        },
    )
    m = generate(spec, 500, seed=3)["m"]
    assert (m.map.len() == 8).all()
    assert _distinct_keys(m)
    assert m.map.keys().explode().is_between(0, 9).all()


@needs_map
def test_an_undeclared_length_stops_where_the_keys_run_out():
    spec = TableSpec(
        "T",
        {
            "flags": ColSpec(_map(pl.Boolean, pl.String), nullable=True),
            "grades": ColSpec(_map(pl.Enum(["a", "b", "c"]), pl.Float64)),
        },
    )
    df = generate(spec, 1_000, seed=2)
    assert df["flags"].map.len().max() == 2
    assert df["flags"].null_count() > 0
    assert df["grades"].map.len().max() == 3
    assert _distinct_keys(df["flags"]) and _distinct_keys(df["grades"])


@needs_map
def test_a_map_nests_and_is_nested():
    inner = _map(pl.String, pl.Int8)
    spec = TableSpec(
        "T",
        {
            "of_maps": ColSpec(pl.List(inner), list_length=(1, 3)),
            "in_struct": ColSpec(
                pl.Struct({"m": inner}),
                fields={"m": ColSpec(inner, list_length=(2, 2))},
            ),
            "map_of_maps": ColSpec(_map(pl.Int64, inner)),
        },
    )
    df = generate(spec, 300, seed=4)
    assert df.schema == spec.schema()
    assert _distinct_keys(df["of_maps"].explode())
    assert (df["in_struct"].struct.field("m").map.len() == 2).all()
    assert _distinct_keys(df["map_of_maps"].map.values().explode())


@needs_map
def test_a_batched_map_column_has_the_shape_of_a_whole_one():
    spec = TableSpec("T", {"m": ColSpec(_map(pl.Int16, pl.Int16), list_length=(0, 6))})
    batched = pl.concat(generate_batches(spec, 3_000, batch_size=700, seed=5))
    assert batched.height == 3_000
    assert batched["m"].map.len().is_between(0, 6).all()
    assert _distinct_keys(batched["m"])
    assert generate(spec, 3_000, seed=5)["m"].map.len().equals(batched["m"].map.len())


@needs_map
def test_no_batch_repeats_anothers_entries():
    """Entries are drawn per batch, each from a seed of its own -- not from
    the first batch's, over and over."""
    spec = TableSpec("T", {"m": ColSpec(_map(pl.Int64, pl.Int64), list_length=(3, 3))})
    batches = [
        b["m"].to_list() for b in generate_batches(spec, 30, batch_size=10, seed=5)
    ]
    assert batches[0] != batches[1] != batches[2] != batches[0]


@needs_map
@pytest.mark.parametrize(
    ("column", "complaint"),
    [
        (
            lambda m: ColSpec(m, fields={"key": ColSpec(pl.String, nullable=True)}),
            r"fields\['key'\] is nullable, but a key .* is never null",
        ),
        (
            lambda m: ColSpec(m, fields={"size": ColSpec(pl.Int64)}),
            r"names 'size'.*Its fields are key, value",
        ),
        (
            lambda m: ColSpec(m, element_null_probability=0.1),
            r"never a null one",
        ),
        (lambda m: ColSpec(m, choices=[{"key": "a", "value": 1}]), r"choices has no"),
        (lambda m: ColSpec(m, unique=True), r"unique=True"),
        (
            lambda m: ColSpec(
                m,
                list_length=(1, 5),
                fields={"key": ColSpec(pl.String, choices=["a", "b", "c"])},
            ),
            r"up to 5 entries, which needs 5 distinct keys, but its keys can take 3",
        ),
    ],
    ids=["nullable-key", "unknown-field", "null-entry", "choices", "unique", "keys"],
)
def test_what_a_map_cannot_be_declared_as(column, complaint):
    with pytest.raises(SpecError, match=complaint):
        column(_map(pl.String, pl.Int64))


@needs_map
def test_a_declared_length_the_keys_can_reach_is_accepted():
    """Exactly as many keys as the longest map needs is enough."""
    spec = TableSpec(
        "T",
        {
            "m": ColSpec(
                _map(pl.UInt8, pl.Boolean),
                list_length=(4, 4),
                fields={"key": ColSpec(pl.UInt8, bounds=(1, 4))},
            )
        },
    )
    m = generate(spec, 200, seed=6)["m"]
    assert m.map.keys().list.sort().to_list() == [[1, 2, 3, 4]] * 200


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


# ---------------------------------------------------------------------------
# Validation, drift, profiling, sizing and the report read a map as its list
# ---------------------------------------------------------------------------


def _metrics():
    return ColSpec(
        _map(pl.String, pl.Int64),
        nullable=True,
        list_length=(1, 3),
        fields={
            "key": ColSpec(pl.String, choices=["cpu", "mem", "disk"]),
            "value": ColSpec(pl.Int64, bounds=(0, 100)),
        },
    )


def _metrics_frame(*maps):
    return pl.DataFrame({"m": pl.Series(list(maps), dtype=_map(pl.String, pl.Int64))})


@needs_map
def test_a_bad_key_or_value_is_found_under_its_name():
    spec = TableSpec("T", {"m": _metrics()})
    df = _metrics_frame(
        {"cpu": 5},
        {"gpu": 1},
        {"mem": 500},
        {"cpu": 1, "mem": 2, "disk": 3},
        {},
        None,
    )
    report = inspect(spec, df)
    assert {f.key: f.count for f in report.findings} == {
        "m.key__choices": 1,
        "m.value__bounds": 1,
        "m__list_len": 1,
    }
    assert report.rows(report.by_code("bounds")[0]).collect()["m"].to_list() == [
        {"mem": 500}
    ]


@needs_map
def test_a_map_read_from_arrow_can_repeat_a_key_and_is_found():
    """Polars folds a repeated key away when it casts to a `Map`, but a map
    that arrives from Arrow keeps both entries."""
    import pyarrow as pa

    maps = pa.array(
        [[("cpu", 1), ("cpu", 2)], [("mem", 3)]],
        type=pa.map_(pa.string(), pa.int64()),
    )
    df = pl.DataFrame({"m": pl.from_arrow(maps)})
    assert df["m"].map.len().to_list() == [2, 1]
    (finding,) = inspect(TableSpec("T", {"m": _metrics()}), df).findings
    assert (finding.key, finding.code, finding.count) == ("m.key__unique", "unique", 1)


@needs_map
def test_a_map_of_a_wider_key_or_value_is_compatible_unless_strict():
    spec = TableSpec("T", {"m": ColSpec(_map(pl.String, pl.Int32))})
    wider = pl.DataFrame({"m": pl.Series([{"a": 1}], dtype=_map(pl.String, pl.Int64))})
    assert inspect(spec, wider).passed
    assert [f.code for f in inspect(spec, wider, strict_dtypes=True)] == ["dtype"]
    as_list = wider.with_columns(
        pl.col("m")
        .cast(_map(pl.String, pl.Int64))
        .cast(pl.List(pl.Struct({"key": pl.String, "value": pl.Int64})))
    )
    assert [f.code for f in inspect(spec, as_list)] == ["dtype"]


@needs_map
def test_a_map_in_a_list_is_named_apart_from_the_list():
    inner = _map(pl.String, pl.Int8)
    spec = TableSpec(
        "T",
        {
            "c": ColSpec(
                pl.List(inner),
                list_length=(1, 2),
            )
        },
    )
    df = generate(spec, 200, seed=1)
    assert inspect(spec, df).passed
    too_long = pl.DataFrame({"c": pl.Series([[{"a": 1}] * 3], dtype=pl.List(inner))})
    assert [f.key for f in inspect(spec, too_long).findings] == ["c__list_len"]


@needs_map
def test_drift_reads_a_maps_key_value_and_length():
    spec = TableSpec("T", {"m": _metrics()})
    df = _metrics_frame(
        {"gpu": 1}, {"mem": 500}, {"cpu": 1, "mem": 2, "disk": 3, "x": 4}
    )
    by_key = {f.key: f for f in drift(spec, df).breaking}
    assert set(by_key) >= {"m.key__values", "m.value__bounds"}
    assert any(key.startswith("m__list") for key in by_key), sorted(by_key)
    assert by_key["m.value__bounds"].details["max_found"] == 500
    assert all(f.columns == ("m",) for f in by_key.values())


@needs_map
def test_from_dataframe_re_declares_a_map_by_its_key_and_value():
    source = TableSpec("T", {"m": _metrics()})
    df = generate(source, 2_000, seed=7)
    profiled = FrameSpec.from_dataframe(df).spec.columns["m"]
    assert profiled.list_length is not None
    assert (profiled.list_length.min, profiled.list_length.max) == (1, 3)
    assert profiled.fields is not None
    assert isinstance(profiled.fields["key"].dtype, pl.Enum)
    assert profiled.dtype == _map(profiled.fields["key"].dtype, pl.Int64)
    assert profiled.element_null_probability == 0.0
    assert inspect(TableSpec("P", {"m": profiled}), df).passed


@needs_map
def test_a_map_is_sized_as_the_list_it_is():
    as_map = ColSpec(
        _map(pl.Int64, pl.Float64),
        list_length=(4, 4),
        fields={"key": ColSpec(pl.Int64, bounds=(0, 10_000))},
    )
    spec = TableSpec("S", {"c": as_map})
    df = generate(spec, 100_000, seed=1)
    assert spec.estimated_size(100_000) == pytest.approx(df.estimated_size(), rel=0.02)
    # An undeclared length is as long as the map draws: two, for Boolean keys.
    flags = TableSpec("F", {"c": ColSpec(_map(pl.Boolean, pl.Int64))})
    assert flags.estimated_size(100_000) == pytest.approx(
        generate(flags, 100_000, seed=1).estimated_size(), rel=0.05
    )


@needs_map
def test_the_report_describes_a_maps_entries_and_their_key_and_value():
    spec_cls = FrameSpec.from_spec(TableSpec("Metrics", {"m": _metrics()}))
    text = spec_cls.to_markdown()
    assert "1..3 entries" in text
    assert "`m.key`" in text and "`m.value`" in text
    assert "Map m" in spec_cls.to_mermaid()


@needs_map
def test_a_declared_map_round_trips_through_a_spec_file(tmp_path):
    spec_cls = FrameSpec.from_spec(TableSpec("T", {"m": _metrics()}))
    spec_cls.to_yaml(tmp_path / "s.yaml")
    assert FrameSpec.from_yaml(tmp_path / "s.yaml").spec == spec_cls.spec
    spec_cls.to_python(tmp_path / "s.py")
    namespace: dict = {}
    exec((tmp_path / "s.py").read_text(encoding="utf-8"), namespace)
    assert namespace[spec_cls.__name__].spec == spec_cls.spec
