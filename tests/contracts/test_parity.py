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
    ],
)
def test_the_facade_spells_out_the_signature_it_forwards_to(verb):
    """`generation.sinks` spells its options out so a typo fails by name;
    a `**kwargs` facade in front of it would defeat that where most calls
    are made. So the classmethod carries the function's parameters, minus
    `spec`, and this keeps the two from drifting.
    """
    from inspect import signature

    from polspec import generation

    facade = signature(getattr(FrameSpec, verb)).parameters
    function = signature(getattr(generation, verb)).parameters
    assert next(iter(function)) == "spec"
    expected = {name: p for name, p in function.items() if name != "spec"}
    assert {n: p.default for n, p in facade.items()} == {
        n: p.default for n, p in expected.items()
    }, verb
