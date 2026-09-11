"""The exception hierarchy: one base class to catch, precise subclasses to match.

Every error polspec raises on its own behalf derives from `PolspecError`, and
each subclass keeps the built-in type it replaced so `except ValueError` and
`except TypeError` written against earlier versions still catch it.
"""

import warnings

import polars as pl
import pytest
import yaml
from polspec import (
    CliError,
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSpec,
    GenerationError,
    PolspecError,
    Registry,
    RegistryError,
    SerializationError,
    SpecError,
    ValidationError,
    ValidationOptions,
    col,
    inspect,
    validate,
)


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


def test_every_error_is_reachable_from_the_package():
    """One import for the hierarchy, including the one the CLI raises.

    `CliError` used to live only in `polspec.errors`, so `except
    polspec.CliError` failed while its five siblings worked.
    """
    import polspec

    for cls in (
        PolspecError,
        SpecError,
        ValidationError,
        GenerationError,
        SerializationError,
        RegistryError,
        CliError,
    ):
        assert getattr(polspec, cls.__name__) is cls
        assert cls.__name__ in polspec.__all__


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


# ---------------------------------------------------------------------------
# Declarations and arguments that used to fail silently or unhelpfully
#
# Each of these is a case where polspec accepted something that could not mean
# what it said, or reported it in a way that named nothing the caller could
# act on. The message is asserted, not just the type: the message is the whole
# value of these.
# ---------------------------------------------------------------------------


def test_references_must_be_a_mapping_not_a_sequence():
    """A list of parents reached `.items()` and raised a bare AttributeError."""

    class Parent(FrameSpec):
        pid = ColSpec(pl.Int64, bounds=(1, 50), unique=True)

    class Child(FrameSpec):
        pid = ColSpec(pl.Int64, bounds=(1, 50))
        __foreign_keys__ = [ForeignKey("pid", references="Parent", ref_columns="pid")]

    parent = Parent.generate(10, seed=1)
    with pytest.raises(SpecError, match="references= must be a mapping"):
        Child.generate(5, references=[parent])
    # And it is a PolspecError, which an AttributeError was not.
    with pytest.raises(PolspecError):
        Child.generate(5, references=[parent])


def test_a_misspelled_reference_key_warns_rather_than_passing_silently():
    """The typo generated freely and said nothing; validate() then complained."""

    class Customers(FrameSpec):
        cid = ColSpec(pl.Int64, bounds=(1, 50), unique=True)

    class Orders(FrameSpec):
        cid = ColSpec(pl.Int64, bounds=(1, 50))
        __foreign_keys__ = [
            ForeignKey("cid", references="Customers", ref_columns="cid")
        ]

    customers = Customers.generate(20, seed=1)
    with pytest.warns(UserWarning, match="Custmers") as caught:
        Orders.generate(10, seed=1, references={"Custmers": customers})
    message = str(caught[0].message)
    # It names the key that went unfilled, and suggests the one supplied.
    assert "'Customers'" in message
    assert "supplied 'Custmers'?" in message


def test_a_correct_reference_key_warns_about_nothing():
    class Customers(FrameSpec):
        cid = ColSpec(pl.Int64, bounds=(1, 50), unique=True)

    class Orders(FrameSpec):
        cid = ColSpec(pl.Int64, bounds=(1, 50))
        __foreign_keys__ = [
            ForeignKey("cid", references="Customers", ref_columns="cid")
        ]

    customers = Customers.generate(20, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Orders.generate(10, seed=1, references={"Customers": customers})


def test_supplying_no_references_at_all_warns_about_nothing():
    """An unfilled key on its own is documented behaviour, not a mistake."""

    class Orders(FrameSpec):
        cid = ColSpec(pl.Int64, bounds=(1, 50))
        __foreign_keys__ = [
            ForeignKey("cid", references="Customers", ref_columns="cid")
        ]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Orders.generate(10, seed=1)


def test_a_registry_handing_over_every_frame_warns_about_nothing():
    """Registry passes each spec the whole set, so unused parents are normal."""

    class Customers(FrameSpec):
        cid = ColSpec(pl.Int64, bounds=(1, 50), unique=True)

    class Products(FrameSpec):
        sku = ColSpec(pl.Int64, bounds=(1, 50), unique=True)

    class Orders(FrameSpec):
        cid = ColSpec(pl.Int64, bounds=(1, 50))
        __foreign_keys__ = [
            ForeignKey("cid", references="Customers", ref_columns="cid")
        ]

    registry = Registry(Customers, Products, Orders)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        registry.generate_all(10, seed=1)


def test_batched_generation_warns_once_not_once_per_batch():
    class Customers(FrameSpec):
        cid = ColSpec(pl.Int64, bounds=(1, 50), unique=True)

    class Orders(FrameSpec):
        cid = ColSpec(pl.Int64, bounds=(1, 50))
        __foreign_keys__ = [
            ForeignKey("cid", references="Customers", ref_columns="cid")
        ]

    customers = Customers.generate(20, seed=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batches = list(
            Orders.generate_batches(
                50, batch_size=10, seed=1, references={"Custmers": customers}
            )
        )
    assert len(batches) == 5
    assert sum("Custmers" in str(w.message) for w in caught) == 1


def test_null_probability_without_nullable_warns():
    with pytest.warns(UserWarning, match="nullable=False"):
        ColSpec(pl.Int64, null_probability=0.9)


def test_a_null_rate_left_behind_by_turning_nullability_off_stays_quiet():
    """The default and an explicit zero already agree with nullable=False."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ColSpec(pl.Int64)
        ColSpec(pl.Int64, null_probability=0.0)
        ColSpec(pl.Int64, nullable=True, null_probability=0.9)


def test_an_unknown_validation_option_names_the_one_you_meant():
    """The functional API takes its options as keywords, so a typo is caught
    there by name rather than by the dataclass it feeds.
    """

    class Rows(FrameSpec):
        a = ColSpec(pl.Int64, bounds=(1, 10))

    df = Rows.generate(5, seed=1)
    with pytest.raises(TypeError) as excinfo:
        inspect(Rows.spec, df, validate_uniqe=True)
    message = str(excinfo.value)
    assert "validate_uniqe" in message
    assert "did you mean 'validate_unique'?" in message
    # The private options dataclass is no longer what gets named.
    assert "ValidationOptions" not in message


def test_the_facade_refuses_a_typo_from_its_own_signature():
    """`FrameSpec.validate` spells every option out, so a typo never reaches
    the function it forwards to -- and an editor sees it before Python does.
    """

    class Rows(FrameSpec):
        a = ColSpec(pl.Int64, bounds=(1, 10))

    df = Rows.generate(5, seed=1)
    for verb in (Rows.inspect, Rows.validate):
        with pytest.raises(
            TypeError, match="unexpected keyword argument 'validate_uniqe'"
        ):
            verb(df, validate_uniqe=True)


def test_the_facade_names_exactly_the_options_the_function_accepts():
    """One list of options, on `ValidationOptions`; the facade's explicit
    signature is a copy, and this is what keeps the copy honest.
    """
    from inspect import signature

    from polspec.validation import _ACCEPTED_OPTIONS

    for verb in (FrameSpec.inspect, FrameSpec.validate):
        params = signature(verb).parameters
        named = {name for name in params if name not in ("df", "options", "references")}
        assert named == set(_ACCEPTED_OPTIONS), verb.__name__
        # `None` is "not given": the defaults stay on the dataclass alone.
        assert all(params[name].default is None for name in named), verb.__name__


def test_the_option_switches_have_one_spelling():
    """`inspect` used to accept a second, bare spelling that `validate` did not."""

    class Rows(FrameSpec):
        a = ColSpec(pl.Int64, unique=True, bounds=(1, 1000))

    dupes = pl.DataFrame({"a": [1, 1, 2]})
    assert Rows.inspect(dupes, validate_unique=False).passed
    assert Rows.validate(dupes, validate_unique=False).height == 3
    for verb in (inspect, validate):
        with pytest.raises(TypeError, match="Unknown validation option"):
            verb(Rows.spec, dupes, unique=False)
    for verb in (Rows.inspect, Rows.validate):
        with pytest.raises(TypeError, match="unexpected keyword argument 'unique'"):
            verb(dupes, unique=False)


def test_options_can_be_passed_as_one_value():
    class Rows(FrameSpec):
        a = ColSpec(pl.Int64, unique=True, bounds=(1, 1000))

    dupes = pl.DataFrame({"a": [1, 1, 2]})
    opts = ValidationOptions(unique=False)
    assert Rows.inspect(dupes, options=opts).passed
    assert Rows.validate(dupes, options=opts).height == 3

    with pytest.raises(TypeError, match="not both"):
        Rows.validate(dupes, options=opts, validate_unique=False)
    with pytest.raises(TypeError, match="must be a ValidationOptions"):
        Rows.validate(dupes, options={"unique": False})
