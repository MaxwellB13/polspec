"""Every field has its entry everywhere a field must be known.

A field added to `ColSpec` (or `TableSpec`, `Check`, `ForeignKey`, `ColRule`)
has to reach the serialization registry, the drift comparators, the typed
`__init__` type checkers read, and -- for an option -- the facade's spelled-out
signatures. Each test here fails, naming what is missing, until it has.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap

import polars as pl
import pytest
from polspec import (
    Check,
    ColRule,
    ColSpec,
    ForeignKey,
    FrameSpec,
    TableSpec,
)
from polspec.drift.fields import FIELD_COMPARATORS, TABLE_COMPARATORS
from polspec.serialization import (
    CHECK_FIELDS,
    COLRULE_FIELDS,
    COLSPEC_FIELDS,
    FK_FIELDS,
    TABLESPEC_FIELDS,
)

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
# Completeness
# ---------------------------------------------------------------------------


def test_every_colspec_field_has_a_comparator():
    assert set(FIELD_COMPARATORS) == {f.name for f in dataclasses.fields(ColSpec)}


def test_every_tablespec_field_has_a_comparator():
    assert set(TABLE_COMPARATORS) == {f.name for f in dataclasses.fields(TableSpec)}


# ---------------------------------------------------------------------------
# What the constructor accepts vs what the instance holds
# ---------------------------------------------------------------------------


def _type_checking_init(cls: type) -> ast.FunctionDef:
    """The `__init__` a class spells out under `if TYPE_CHECKING:`."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                    return stmt
    raise AssertionError(f"{cls.__name__} declares no TYPE_CHECKING __init__")


@pytest.mark.parametrize("cls", [ColSpec, Check, ForeignKey], ids=lambda c: c.__name__)
def test_the_declared_constructor_matches_the_fields(cls):
    """`ColSpec`, `Check` and `ForeignKey` annotate their fields with what an
    instance *holds* and spell out what the constructor *accepts* in an
    `__init__` that exists only for type checkers. The dataclass generates
    the real one from the fields, so the two must name the same parameters
    in the same order, with a default on the same ones -- or a field added
    to one is silently missing from the other.
    """
    stub = _type_checking_init(cls)
    params = [a.arg for a in stub.args.args if a.arg != "self"]
    fields = [f.name for f in dataclasses.fields(cls)]
    assert params == fields
    defaulted = params[len(params) - len(stub.args.defaults) :]
    assert defaulted == [
        f.name
        for f in dataclasses.fields(cls)
        if f.default is not dataclasses.MISSING
        or f.default_factory is not dataclasses.MISSING
    ]


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


# ---------------------------------------------------------------------------
# The facade forwards to functions over `cls.spec`, with the same signature.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verb",
    [
        "generate",
        "generate_batches",
        "sink_parquet",
        "sink_csv",
        "sink_ipc",
        "sink_ndjson",
        "read",
    ],
)
def test_the_facade_spells_out_the_signature_it_forwards_to(verb):
    """`generation.sinks` spells its options out so a typo fails by name;
    a `**kwargs` facade in front of it would defeat that where most calls
    are made. So the classmethod carries the function's parameters, minus
    `spec`, and this keeps the two from drifting.
    """
    from inspect import signature

    import polspec

    facade = signature(getattr(FrameSpec, verb)).parameters
    function = signature(getattr(polspec, verb)).parameters
    assert next(iter(function)) == "spec"
    expected = {name: p for name, p in function.items() if name != "spec"}
    assert {n: p.default for n, p in facade.items()} == {
        n: p.default for n, p in expected.items()
    }, verb


# ---------------------------------------------------------------------------
# Every function taking a spec takes the class, as the facade's own
# classmethods and `drift`, `read` and the renderers always have.
# ---------------------------------------------------------------------------


class _Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, unique=True, bounds=(1, 1_000))
    status = ColSpec(pl.Enum(["NEW", "PAID"]))


_FRAME = pl.DataFrame(
    {"order_id": [1, 2, 2], "status": ["NEW", "PAID", "LOST"]}
).with_columns(pl.col("status").cast(pl.String))

_SINKS = ("sink_parquet", "sink_csv", "sink_ipc", "sink_ndjson")
_READERS = {
    "sink_parquet": pl.read_parquet,
    "sink_csv": pl.read_csv,
    "sink_ipc": pl.read_ipc,
    "sink_ndjson": pl.read_ndjson,
}


def _call(verb: str, spec, tmp_path) -> object:
    import polspec
    from polspec.sizing import estimated_size

    if verb == "generate":
        return polspec.generate(spec, 20, seed=1)
    if verb == "generate_batches":
        return pl.concat(polspec.generate_batches(spec, 20, batch_size=7, seed=1))
    if verb == "scan":
        return polspec.scan(spec, 20, seed=1).collect()
    if verb == "estimated_size":
        return estimated_size(spec, 1_000)
    if verb == "inspect":
        return polspec.inspect(spec, _FRAME).findings
    if verb == "validate":
        with pytest.raises(polspec.ValidationError) as caught:
            polspec.validate(spec, _FRAME)
        return caught.value.report.findings
    assert verb in _SINKS
    path = tmp_path / f"{verb}_{type(spec).__name__}.out"
    getattr(polspec, verb)(spec, path, 20, seed=1)
    return _READERS[verb](path)


@pytest.mark.parametrize(
    "verb",
    [
        "generate",
        "generate_batches",
        "scan",
        "estimated_size",
        "inspect",
        "validate",
        *_SINKS,
    ],
)
def test_every_verb_takes_a_framespec_class_as_its_spec(verb, tmp_path):
    by_class = _call(verb, _Orders, tmp_path)
    by_spec = _call(verb, _Orders.spec, tmp_path)
    if isinstance(by_spec, pl.DataFrame):
        assert isinstance(by_class, pl.DataFrame)
        assert by_class.equals(by_spec)
    else:
        assert by_class == by_spec


def test_a_class_that_is_not_a_spec_is_refused_by_name():
    import polspec

    with pytest.raises(TypeError, match="FrameSpec subclass, got the class dict"):
        polspec.generate(dict, 5)  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# `polspec drift` sets every DriftOptions field
# ---------------------------------------------------------------------------

# Each field, the flags that set it, and the value they set it to -- never the
# default, so the test sees the flag reach the option.
_DRIFT_FLAGS = {
    "significance": (["--significance", "0.05"], 0.05),
    "null_rate_tolerance": (["--null-rate-tolerance", "0.2"], 0.2),
    "frequency_tolerance": (["--frequency-tolerance", "0.2"], 0.2),
    "distribution_tolerance": (["--distribution-tolerance", "0.2"], 0.2),
    "unseen_values": (["--no-unseen"], False),
    "strict_dtypes": (["--strict-dtypes"], True),
    "max_samples": (["--max-samples", "3"], 3),
}


def test_every_drift_option_has_a_flag():
    """0.14.0 added three drift options and no flags for them; this is what
    holds the command to the options from now on."""
    from polspec.cli import _build_parser
    from polspec.cli._drift import _drift_options
    from polspec.drift import DriftOptions

    assert set(_DRIFT_FLAGS) == {f.name for f in dataclasses.fields(DriftOptions)}
    parser = _build_parser()
    defaults = _drift_options(parser.parse_args(["drift", "spec.yaml", "data.csv"]))
    assert defaults == DriftOptions()
    for name, (flags, value) in _DRIFT_FLAGS.items():
        assert value != getattr(DriftOptions(), name), name
        args = parser.parse_args(["drift", "spec.yaml", "data.csv", *flags])
        assert getattr(_drift_options(args), name) == value, name


def test_a_drift_option_out_of_range_is_a_cli_error(tmp_path, capsys):
    from polspec.cli import main

    spec, data = tmp_path / "s.yaml", tmp_path / "d.csv"
    spec.write_text(
        "version: 4\nname: S\ncolumns:\n  a: {dtype: Int64}\n", encoding="utf-8"
    )
    data.write_text("a\n1\n", encoding="utf-8")
    assert main(["drift", str(spec), str(data), "--significance", "2"]) == 1
    assert "significance must be between 0 and 1" in capsys.readouterr().err
