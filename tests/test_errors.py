"""The exception hierarchy: one base class to catch, precise subclasses to match.

Every error polspec raises on its own behalf derives from `PolspecError`, and
each subclass keeps the built-in type it replaced so `except ValueError` and
`except TypeError` written against earlier versions still catch it.
"""

import polars as pl
import pytest
import yaml
from polspec import (
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSpec,
    GenerationError,
    PolspecError,
    RegistryError,
    SerializationError,
    SpecError,
    ValidationError,
    col,
)
from polspec.errors import CliError


def test_every_error_is_a_polspec_error():
    for cls in (
        SpecError,
        ValidationError,
        GenerationError,
        SerializationError,
        RegistryError,
        CliError,
    ):
        assert issubclass(cls, PolspecError)


def test_subclasses_keep_the_builtin_types_they_replaced():
    assert issubclass(SpecError, ValueError)
    assert issubclass(SpecError, TypeError)
    assert issubclass(ValidationError, ValueError)
    assert issubclass(GenerationError, ValueError)
    assert issubclass(SerializationError, ValueError)
    assert issubclass(RegistryError, LookupError)


def test_bad_declaration_is_a_spec_error():
    with pytest.raises(SpecError, match="outside the range Int8 can represent"):
        ColSpec(pl.Int8, bounds=(0, 1_000))
    with pytest.raises(SpecError, match="must be a predicate"):
        ColRule(when="not a dict", choices=["X"])
    with pytest.raises(SpecError, match="must have the same length"):
        ForeignKey(["a", "b"], references="self", ref_columns="a")
    with pytest.raises(SpecError, match="references unknown column"):

        class Broken(FrameSpec):
            a = ColSpec(
                pl.Int64,
                rules=[ColRule(when=col("zzz") == 1, choices=[1])],
            )


def test_argument_misuse_stays_a_plain_value_error():
    class Spec(FrameSpec):
        a = ColSpec(pl.Int64)

    with pytest.raises(ValueError, match="n must be >= 0") as info:
        Spec.generate(-1)
    assert not isinstance(info.value, PolspecError)


def test_generation_failure_is_a_generation_error():
    class NoCoverage(FrameSpec):
        a = ColSpec(pl.String)

    with pytest.raises(GenerationError, match="needs at least one"):
        NoCoverage.generate(10, method="cartesian")


def test_rust_engine_complaints_surface_as_generation_error():
    class Spec(FrameSpec):
        a = ColSpec(
            pl.Float64,
            distribution="normal",
            distribution_params={"mean": 0.0, "std": 1.0},
        )

    # A valid spec; then reach the engine with a parameter it rejects by
    # bypassing ColSpec's own check, which is what a future bug would do.
    from polspec._ffi import column_plan

    with pytest.raises(GenerationError) as info:
        column_plan(
            "a", "float64", distribution="normal", params={"mean": 0.0, "std": -1.0}
        )
    assert isinstance(info.value.__cause__, ValueError)
    assert "'a'" in str(info.value) and "must be positive" in str(info.value)


def test_unreadable_file_is_a_serialization_error(tmp_path):
    path = tmp_path / "spec.yaml"
    path.write_text(
        yaml.safe_dump({"name": "X", "columns": {"a": {"dtype": "NotADtype"}}})
    )
    with pytest.raises(SerializationError, match="Unrecognized dtype name"):
        FrameSpec.from_yaml(path)


def test_validation_error_carries_every_finding():
    class Spec(FrameSpec):
        a = ColSpec(pl.Int64, bounds=(0, 10))
        b = ColSpec(pl.Int64, bounds=(0, 10))

    df = pl.DataFrame({"a": [11], "b": [12]})
    with pytest.raises(ValidationError) as info:
        Spec.validate(df)
    assert isinstance(info.value, PolspecError)
    assert len(info.value.errors) == 2


def test_one_clause_catches_them_all():
    caught = []
    for action in (
        lambda: ColSpec(pl.Int8, bounds=(0, 1_000)),
        lambda: FrameSpec.from_yaml("/definitely/not/here.yaml"),
    ):
        try:
            action()
        except PolspecError as exc:
            caught.append(type(exc).__name__)
        except FileNotFoundError:
            caught.append("FileNotFoundError")
    assert caught == ["SpecError", "FileNotFoundError"]
