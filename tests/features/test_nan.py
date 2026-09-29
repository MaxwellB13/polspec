"""`ColSpec.nan_probability`: NaN in a float value, declared.

A float column says how often a present value is NaN -- 0 by default, which
claims there are none. Generation draws NaN on that share of the present
rows, validation reports an undeclared one as a `nan` finding, and `bounds`
and `choices` never judge a NaN: it sits in no range and equals nothing.
"""

from __future__ import annotations

import math

import polars as pl
import pytest
from polspec import (
    ColSpec,
    ForeignKey,
    FrameSpec,
    Registry,
    RegistryError,
    SpecError,
    TableSpec,
    generate,
    generate_batches,
    inspect,
)
from polspec.drift import diff, drift
from polspec.serialization import from_dict, to_dict

NAN = float("nan")


def _spec(**columns: ColSpec) -> TableSpec:
    return TableSpec("T", columns)


def _as_text(frame: pl.DataFrame) -> pl.DataFrame:
    """`frame` with NaN comparable: `equals` holds no NaN equal to another."""
    return frame.select(pl.all().cast(pl.String))


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column",
    [
        ColSpec(pl.Float64, nan_probability=0.2),
        ColSpec(pl.Float16, nan_probability=1.0),
        ColSpec(pl.List(pl.Float32), nan_probability=0.3),
        ColSpec(pl.Array(pl.Float64, 2), nan_probability=0.3),
        ColSpec(
            pl.Struct({"x": pl.Float64}),
            fields={"x": ColSpec(pl.Float64, nan_probability=0.1)},
        ),
    ],
    ids=["float64", "float16", "list", "array", "struct_field"],
)
def test_a_float_value_declares_a_nan_share(column):
    declared = column.nan_probability or column.fields["x"].nan_probability
    assert 0 < declared <= 1


@pytest.mark.parametrize(
    "kwargs, complaint",
    [
        (
            {"dtype": pl.Int64, "nan_probability": 0.1},
            "only supported for a float value",
        ),
        (
            {"dtype": pl.Decimal(10, 2), "nan_probability": 0.1},
            "only supported for a float",
        ),
        (
            {"dtype": pl.List(pl.Int8), "nan_probability": 0.1},
            "only supported for a float",
        ),
        ({"dtype": pl.Float64, "nan_probability": 1.5}, "between 0 and 1"),
        ({"dtype": pl.Float64, "nan_probability": -0.1}, "between 0 and 1"),
        (
            {"dtype": pl.Float64, "nan_probability": 0.1, "unique": True},
            "no NaN equals another",
        ),
    ],
)
def test_a_nan_share_is_refused_where_it_means_nothing(kwargs, complaint):
    with pytest.raises(SpecError, match=complaint):
        ColSpec(**kwargs)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_nan_is_drawn_on_its_share_of_the_present_rows():
    spec = _spec(
        f=ColSpec(
            pl.Float64,
            bounds=(0, 1),
            nullable=True,
            null_probability=0.2,
            nan_probability=0.25,
        )
    )
    column = generate(spec, 40_000, seed=1)["f"]
    assert column.null_count() / len(column) == pytest.approx(0.2, abs=0.01)
    present = column.drop_nulls()
    assert present.is_nan().mean() == pytest.approx(0.25, abs=0.01)
    numbers = present.filter(present.is_not_nan())
    assert numbers.min() >= 0
    assert numbers.max() <= 1


@pytest.mark.parametrize("dtype", [pl.Float16, pl.Float32, pl.Float64], ids=str)
def test_nan_keeps_the_declared_dtype(dtype):
    column = generate(_spec(f=ColSpec(dtype, nan_probability=0.5)), 200, seed=3)["f"]
    assert column.dtype == dtype
    assert 0 < column.is_nan().sum() < 200


def test_nan_overlays_the_values_a_spec_without_it_draws():
    """The share is drawn from a column of its own, so the numbers under the
    NaN are the ones the column always drew -- and a column that declares no
    NaN, beside one that does, draws exactly what it did."""
    plain = _spec(f=ColSpec(pl.Float64), g=ColSpec(pl.Float64))
    with_nan = _spec(f=ColSpec(pl.Float64, nan_probability=0.3), g=ColSpec(pl.Float64))
    before, after = generate(plain, 2_000, seed=7), generate(with_nan, 2_000, seed=7)
    assert after["g"].equals(before["g"])
    kept = after["f"].is_not_nan()
    assert after["f"].filter(kept).equals(before["f"].filter(kept))


def test_a_batch_holds_the_nan_the_whole_frame_holds_there():
    spec = _spec(
        f=ColSpec(pl.Float64, nan_probability=0.3, nullable=True),
        s=ColSpec(
            pl.Struct({"x": pl.Float32}),
            fields={"x": ColSpec(pl.Float32, nan_probability=0.5)},
        ),
    )
    whole = generate(spec, 1_000, seed=11)
    batched = pl.concat(generate_batches(spec, 1_000, batch_size=137, seed=11))
    assert _as_text(batched).equals(_as_text(whole))


def test_list_elements_and_struct_fields_draw_their_share():
    spec = _spec(
        l=ColSpec(pl.List(pl.Float64), nan_probability=0.4, list_length=(3, 3)),
        s=ColSpec(
            pl.Struct({"x": pl.Float64}),
            fields={"x": ColSpec(pl.Float64, nan_probability=0.6)},
        ),
    )
    frame = generate(spec, 10_000, seed=5)
    assert frame["l"].explode(empty_as_null=False).is_nan().mean() == pytest.approx(
        0.4, abs=0.02
    )
    assert frame["s"].struct.field("x").is_nan().mean() == pytest.approx(0.6, abs=0.02)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_an_undeclared_nan_is_a_nan_finding_with_its_rows():
    spec = _spec(
        f=ColSpec(pl.Float64),
        l=ColSpec(pl.List(pl.Float64)),
        s=ColSpec(pl.Struct({"x": pl.Float64})),
    )
    frame = pl.DataFrame(
        {
            "f": [1.0, NAN, 2.0, NAN],
            "l": [[1.0], [NAN, 1.0], [2.0], [3.0]],
            "s": [{"x": 1.0}, {"x": 1.0}, {"x": NAN}, {"x": 1.0}],
        }
    )
    report = inspect(spec, frame)
    assert [(f.key, f.code, f.count) for f in report.findings] == [
        ("f__nan", "nan", 2),
        ("l__nan", "nan", 1),
        ("s.x__nan", "nan", 1),
    ]
    # One row holds no NaN; failing_rows() lists a row once per finding.
    assert report.passing_rows().collect()["f"].to_list() == [1.0]
    assert "declare nan_probability" in report.findings[0].message


def test_a_declared_nan_is_no_finding():
    spec = _spec(f=ColSpec(pl.Float64, nan_probability=0.1))
    assert not inspect(spec, pl.DataFrame({"f": [1.0, NAN]}))


def test_bounds_and_choices_never_judge_a_nan():
    """A NaN used to be out of bounds -- it compares greater than every
    number -- and outside any choices. It is judged once, as a NaN."""
    spec = _spec(
        b=ColSpec(pl.Float64, bounds=(0, 1)),
        c=ColSpec(pl.Float64, choices=[0.5]),
    )
    frame = pl.DataFrame({"b": [0.5, NAN], "c": [0.5, NAN]})
    assert [f.key for f in inspect(spec, frame).findings] == ["b__nan", "c__nan"]
    declared = _spec(
        b=ColSpec(pl.Float64, bounds=(0, 1), nan_probability=0.5),
        c=ColSpec(pl.Float64, choices=[0.5], nan_probability=0.5),
    )
    assert not inspect(declared, frame)
    outside = pl.DataFrame({"b": [2.0, NAN], "c": [0.7, NAN]})
    assert [f.key for f in inspect(declared, outside).findings] == [
        "b__bounds",
        "c__choices",
    ]


def test_a_decimal_holds_no_nan_whatever_it_declares():
    """A `Decimal` cannot declare NaN; a float NaN arriving for one is a
    value the Decimal cannot hold, as before."""
    spec = _spec(d=ColSpec(pl.Decimal(10, 2)))
    report = inspect(spec, pl.DataFrame({"d": [1.0, NAN]}))
    assert [f.key for f in report.findings] == ["d__dtype_range"]


def test_integers_standing_in_for_a_float_are_not_asked_about_nan():
    assert not inspect(_spec(f=ColSpec(pl.Float64)), pl.DataFrame({"f": [1, 2]}))


# ---------------------------------------------------------------------------
# Spec files and drift
# ---------------------------------------------------------------------------


def test_a_nan_share_is_written_to_a_spec_file_only_when_declared():
    spec = _spec(
        f=ColSpec(pl.Float64, nan_probability=0.25),
        g=ColSpec(pl.Float64),
    )
    data = to_dict(spec)
    assert data["columns"]["f"]["nan_probability"] == 0.25
    assert "nan_probability" not in data["columns"]["g"]
    assert from_dict(data) == spec


def test_diff_says_allowing_nan_is_compatible_and_forbidding_it_breaks():
    none, some = (
        _spec(f=ColSpec(pl.Float64)),
        _spec(f=ColSpec(pl.Float64, nan_probability=0.1)),
    )
    widened = diff(none, some)
    assert [(f.code, f.severity) for f in widened.findings] == [
        ("domain_widened", "compatible")
    ]
    narrowed = diff(some, none)
    assert [(f.code, f.severity) for f in narrowed.findings] == [
        ("domain_narrowed", "breaking")
    ]
    moved = diff(some, _spec(f=ColSpec(pl.Float64, nan_probability=0.2)))
    assert [f.code for f in moved.findings] == ["field_changed"]


def test_drift_reports_an_undeclared_nan_and_a_moved_nan_rate():
    values = [1.0] * 70 + [NAN] * 30
    undeclared = drift(_spec(f=ColSpec(pl.Float64)), pl.DataFrame({"f": values}))
    assert [(f.code, f.severity) for f in undeclared.findings] == [
        ("new_values", "breaking")
    ]
    assert undeclared.findings[0].details["nan_count"] == 30

    declared = _spec(f=ColSpec(pl.Float64, nan_probability=0.3))
    assert not drift(declared, pl.DataFrame({"f": values})).findings
    moved = drift(declared, pl.DataFrame({"f": [1.0] * 95 + [NAN] * 5}))
    assert [f.code for f in moved.findings] == ["nan_rate_moved"]
    assert moved.findings[0].details["observed"] == pytest.approx(0.05)


def test_drift_measures_extremes_over_the_numbers_alone():
    spec = _spec(f=ColSpec(pl.Float64, bounds=(0, 1), nan_probability=0.5))
    frame = pl.DataFrame({"f": [0.2, NAN, 0.8, NAN]})
    assert not drift(spec, frame).findings


def test_a_spec_validates_what_it_generates_with_nan():
    spec = _spec(reading=ColSpec(pl.Float64, bounds=(0, 100), nan_probability=0.05))
    frame = generate(spec, 10_000, seed=0)
    assert not inspect(spec, frame)
    assert math.isclose(frame["reading"].is_nan().mean(), 0.05, abs_tol=0.01)


# ---------------------------------------------------------------------------
# Rendering and foreign keys
# ---------------------------------------------------------------------------


def test_the_data_dictionary_and_the_diagram_show_the_nan_share():
    spec = FrameSpec.from_spec(
        _spec(
            level=ColSpec(pl.Float64, nullable=True, nan_probability=0.05),
            series=ColSpec(
                pl.List(pl.Float32),
                element_null_probability=0.1,
                nan_probability=0.2,
            ),
            point=ColSpec(
                pl.Struct({"x": pl.Float64}),
                fields={"x": ColSpec(pl.Float64, nan_probability=0.3)},
            ),
        )
    )
    markdown = spec.to_markdown()
    assert "| `level` | `Float64` | Yes; NaN 5% |" in markdown
    assert "No (elements 10%); NaN 20%" in markdown
    assert "| `point.x` | `Float64` | No; NaN 30% |" in markdown
    assert 'Float64 level "nullable, NaN: 5%"' in spec.to_mermaid()


def test_a_key_filled_from_a_nan_parent_must_declare_nan():
    """A parent holding NaN hands its NaN keys to the child, which then
    failed its own `nan` claim: generated data failing its spec. Refused
    where it is declared, and in a registry that resolves it by name."""
    parent = _spec(k=ColSpec(pl.Float64, nan_probability=0.3))
    with pytest.raises(SpecError, match="declares no NaN, but the key fills it"):
        TableSpec(
            "C",
            {"k": ColSpec(pl.Float64)},
            foreign_keys=[ForeignKey("k", references=parent)],
        )
    by_name = TableSpec(
        "C", {"k": ColSpec(pl.Float64)}, foreign_keys=[ForeignKey("k", references="T")]
    )
    with pytest.raises(RegistryError, match="declares no NaN"):
        Registry(parent, by_name).resolve()

    child = TableSpec(
        "C",
        {"k": ColSpec(pl.Float64, nan_probability=0.1)},
        foreign_keys=[ForeignKey("k", references=parent)],
    )
    parents = generate(parent, 20, seed=1)
    children = generate(child, 2_000, seed=2, references={parent: parents})
    assert children["k"].is_nan().any()
    assert not inspect(child, children, references={parent: parents})
