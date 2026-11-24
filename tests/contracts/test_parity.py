"""Every field and option reaches everywhere it must, held by what it does.

A field added to `ColSpec` (or `TableSpec`, `Check`, `ColRule`, `ForeignKey`,
`Hierarchy`) has to survive a spec file and be reported by `diff` when it
changes, and the typed `__init__` type checkers read has to name it. An
option added to validation, drift or profiling has to reach every entry point
that takes it, and the command line. Each test here fails, naming what is
missing, until it has -- and each asks the public interface, not the tables
behind it.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import re
import textwrap

import numpy as np
import polars as pl
import polspec
import pytest
import yaml
from field_examples import COLUMN_FIELDS, TABLE_FIELDS, with_column
from polspec import (
    Check,
    ColSpec,
    DriftOptions,
    ForeignKey,
    FrameSpec,
    ProfileOptions,
    TableSpec,
    ValidationOptions,
)
from polspec.cli import main

# ---------------------------------------------------------------------------
# Every field survives a spec file, and `diff` reports a change to it
# ---------------------------------------------------------------------------

# A spec's name is what it is called, not what it says: `diff` compares
# what two specs declare, whatever each is named.
_NOT_A_DIFFERENCE = {(TableSpec, "name")}


def _constructed(cls: type) -> set[str]:
    """The fields a caller gives: one filled in from another is not."""
    # `_weights` is held as `weights`, under that key in a file, and
    # declared inside `choices` (2.0).
    return {f.name.removeprefix("_") for f in dataclasses.fields(cls) if f.init}


def _through_a_file(spec: TableSpec, tmp_path) -> TableSpec:
    path = tmp_path / f"{spec.name}.yaml"
    polspec.to_yaml(spec, path)
    return polspec.from_yaml(path)


def test_every_column_field_has_an_example():
    assert set(COLUMN_FIELDS) | {"col_name"} == _constructed(ColSpec)


@pytest.mark.parametrize("cls", list(TABLE_FIELDS), ids=lambda c: c.__name__)
def test_every_table_field_has_an_example(cls):
    assert set(TABLE_FIELDS[cls]) == _constructed(cls)


@pytest.mark.parametrize("field", sorted(COLUMN_FIELDS))
def test_every_column_field_survives_a_file_and_is_diffed(field, tmp_path):
    without, with_it = (with_column(c) for c in COLUMN_FIELDS[field])
    assert _through_a_file(with_it, tmp_path) == with_it
    assert polspec.diff(without, with_it).findings, field


@pytest.mark.parametrize(
    ("cls", "field"),
    [(cls, field) for cls, fields in TABLE_FIELDS.items() for field in sorted(fields)],
    ids=lambda x: getattr(x, "__name__", x),
)
def test_every_table_field_survives_a_file_and_is_diffed(cls, field, tmp_path):
    without, with_it = TABLE_FIELDS[cls][field]
    assert _through_a_file(with_it, tmp_path) == with_it
    if (cls, field) not in _NOT_A_DIFFERENCE:
        assert polspec.diff(without, with_it).findings, (cls.__name__, field)


def test_a_column_name_unlike_its_attribute_survives_a_file_and_is_diffed(tmp_path):
    """`col_name` is a class body's: the attribute a column is reached by,
    beside the name it has in the data."""

    class Priced(FrameSpec):
        unit_price = ColSpec(pl.Float64, col_name="Unit Price")

    class Renamed(FrameSpec):
        unit_price = ColSpec(pl.Float64, col_name="Price")

    # A file holds the column by the name it has in the data, which is
    # what `col_name` says; the attribute is the class body's alone.
    assert list(Priced.spec.columns) == ["Unit Price"]
    assert list(_through_a_file(Priced.spec, tmp_path).columns) == ["Unit Price"]
    assert polspec.diff(Priced, Renamed).findings


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


@pytest.mark.parametrize(
    "cls", [ColSpec, Check, ForeignKey, TableSpec], ids=lambda c: c.__name__
)
def test_the_declared_constructor_matches_the_fields(cls):
    """`ColSpec`, `Check`, `ForeignKey` and `TableSpec` annotate their fields with what an
    instance *holds* and spell out what the constructor *accepts* in an
    `__init__` that exists only for type checkers. The dataclass generates
    the real one from the fields, so the two must name the same parameters
    in the same order, with a default on the same ones -- or a field added
    to one is silently missing from the other.
    """
    stub = _type_checking_init(cls)
    positional = [a.arg for a in stub.args.args if a.arg != "self"]
    keyword_only = [a.arg for a in stub.args.kwonlyargs]
    # What the constructor takes; a field filled in for it (`init=False`) is not.
    # A private field (`_weights`) is declared through another (`choices`).
    fields = [
        f for f in dataclasses.fields(cls) if f.init and not f.name.startswith("_")
    ]
    assert positional + keyword_only == [f.name for f in fields]
    # Keyword-only from the same field on (0.23): what identifies the thing
    # is positional, the rest is named.
    assert keyword_only == [f.name for f in fields if f.kw_only]
    defaulted = positional[len(positional) - len(stub.args.defaults) :] + [
        a.arg
        for a, default in zip(stub.args.kwonlyargs, stub.args.kw_defaults, strict=True)
        if default is not None
    ]
    assert defaulted == [
        f.name
        for f in fields
        if f.default is not dataclasses.MISSING
        or f.default_factory is not dataclasses.MISSING
    ]


# ---------------------------------------------------------------------------
# Every option, as a typed keyword, wherever it is taken
# ---------------------------------------------------------------------------


def _keywords(function) -> set[str]:
    """The names `function` takes as `**keywords`, read from its
    `Unpack[...]` annotation: the typed copy of an options class. The name
    is looked up where the public verbs of each kind are defined, since a
    facade may import it for type checkers alone."""
    last = list(inspect.signature(function).parameters.values())[-1]
    assert last.kind is inspect.Parameter.VAR_KEYWORD, function.__qualname__
    found = re.search(r"Unpack\[(\w+)\]", str(last.annotation))
    assert found, (function.__qualname__, last.annotation)
    homes = (function, polspec.inspect, polspec.drift, polspec.profile)
    typed = next(h.__globals__[found[1]] for h in homes if found[1] in h.__globals__)
    return set(typed.__annotations__)


def _fields(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


@pytest.mark.parametrize(
    ("options", "functions"),
    [
        (
            ValidationOptions,
            (
                polspec.validate,
                polspec.inspect,
                FrameSpec.validate,
                FrameSpec.inspect,
                polspec.Registry.inspect_all,
                polspec.Registry.validate_all,
            ),
        ),
        (
            DriftOptions,
            (polspec.drift, polspec.diff, FrameSpec.drift, FrameSpec.diff),
        ),
        (ProfileOptions, (polspec.profile, polspec.profile_all)),
    ],
    ids=lambda x: getattr(x, "__name__", ""),
)
def test_every_entry_point_takes_every_option_as_a_typed_keyword(options, functions):
    """Until 0.23 `FrameSpec.drift` named four of the seven drift options."""
    for function in functions:
        assert "options" in inspect.signature(function).parameters, (
            function.__qualname__
        )
        assert _keywords(function) == _fields(options), function.__qualname__


def test_synthesize_takes_every_profiling_option_but_the_seed_it_takes_itself():
    for function in (polspec.synthesize, polspec.synthesize_all):
        assert _keywords(function) == _fields(ProfileOptions) - {"seed"}
        assert "seed" in inspect.signature(function).parameters


def test_the_validation_facades_name_only_the_frame_options_and_parents():
    """Every other option arrives as a typed keyword, so it is named once."""
    for verb in (FrameSpec.inspect, FrameSpec.validate):
        params = inspect.signature(verb).parameters
        named = {
            n for n, p in params.items() if p.kind is not inspect.Parameter.VAR_KEYWORD
        }
        assert named == {"df", "options", "references"}, verb.__name__


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

    if verb == "generate":
        return polspec.generate(spec, 20, seed=1)
    if verb == "generate_batches":
        return pl.concat(polspec.generate_batches(spec, 20, batch_size=7, seed=1))
    if verb == "scan":
        return polspec.scan(spec, 20, seed=1).collect()
    if verb == "estimated_size":
        return polspec.estimated_size(spec, 1_000)
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
# `polspec drift` sets every DriftOptions field: each flag changes the report
# ---------------------------------------------------------------------------

_ROWS = 2_000


def _drifting(tmp_path):
    """A spec, and a file that has moved from it every way drift can see."""
    rng = np.random.default_rng(3)
    spec = TableSpec(
        "Feed",
        {
            "rate": ColSpec(pl.Int64, nullable=True, null_probability=0.1),
            "mild": ColSpec(pl.Int64, nullable=True, null_probability=0.1),
            "tier": ColSpec(pl.String, choices={"a": 1, "b": 1, "c": 1}),
            "amount": ColSpec(
                pl.Float64,
                bounds=(-10, 10),
                distribution="normal",
                distribution_params={"mean": 0.0, "std": 1.0},
            ),
            "code": ColSpec(pl.String, choices=["x", "y"]),
            "small": ColSpec(pl.Int64),
        },
    )
    frame = pl.DataFrame(
        {
            # 30% null for a declared 10%; and 12%, which only a looser
            # significance reports.
            "rate": pl.Series(
                [None if i % 10 < 3 else 1 for i in range(_ROWS)], dtype=pl.Int64
            ),
            "mild": pl.Series(
                [None if i % 25 < 3 else 1 for i in range(_ROWS)], dtype=pl.Int64
            ),
            "tier": ["a" if i % 5 else "b" for i in range(_ROWS)],  # never "c"
            "amount": rng.normal(1.0, 1.0, _ROWS),
            "code": [f"n{i % 12}" if i % 2 else "x" for i in range(_ROWS)],
            "small": pl.Series([1] * _ROWS, dtype=pl.Int32),
        }
    )
    spec_path, data = tmp_path / "feed.yaml", tmp_path / "feed.parquet"
    polspec.to_yaml(spec, spec_path)
    frame.write_parquet(data)
    return str(spec_path), str(data)


def _found(report: dict) -> set[tuple[str, str]]:
    return {(f["code"], f["columns"][0]) for f in report["findings"]}


# Each DriftOptions field, the flags that set it, and what they change in the
# report of `_drifting`'s file: a field with no entry fails the test.
_DRIFT_FLAGS = {
    "significance": (
        ["--significance", "0.05", "--null-rate-tolerance", "0"],
        lambda r: ("null_rate_moved", "mild") in _found(r),
    ),
    "null_rate_tolerance": (
        ["--null-rate-tolerance", "0.5"],
        lambda r: ("null_rate_moved", "rate") not in _found(r),
    ),
    "frequency_tolerance": (
        ["--frequency-tolerance", "0.9"],
        lambda r: ("frequencies_moved", "tier") not in _found(r),
    ),
    "distribution_tolerance": (
        ["--distribution-tolerance", "0.9"],
        lambda r: ("distribution_moved", "amount") not in _found(r),
    ),
    "unseen_values": (
        ["--no-unseen"],
        lambda r: ("cardinality_moved", "tier") not in _found(r),
    ),
    "strict_dtypes": (
        ["--strict-dtypes"],
        lambda r: ("dtype_changed", "small") in _found(r),
    ),
    "max_samples": (
        ["--max-samples", "2"],
        lambda r: (
            [
                len(f["details"]["values"])
                for f in r["findings"]
                if f["code"] == "new_values"
            ]
            == [2]
        ),
    ),
}


def _drift_report(capsys, files, *flags: str) -> dict:
    main(["drift", *files, "--json", *flags])
    return json.loads(capsys.readouterr().out)


def test_every_drift_option_has_a_flag():
    """0.14.0 added three drift options and no flags for them."""
    assert set(_DRIFT_FLAGS) == _fields(DriftOptions)


@pytest.mark.parametrize("option", sorted(_DRIFT_FLAGS))
def test_each_drift_flag_changes_the_report(option, tmp_path, capsys):
    files = _drifting(tmp_path)
    flags, changed = _DRIFT_FLAGS[option]
    assert not changed(_drift_report(capsys, files)), "the default already shows it"
    assert changed(_drift_report(capsys, files, *flags)), option


def test_a_drift_option_out_of_range_is_a_cli_error(tmp_path, capsys):
    spec, data = tmp_path / "s.yaml", tmp_path / "d.csv"
    spec.write_text(
        "version: 4\nname: S\ncolumns:\n  a: {dtype: Int64}\n", encoding="utf-8"
    )
    data.write_text("a\n1\n", encoding="utf-8")
    assert main(["drift", str(spec), str(data), "--significance", "2"]) == 1
    assert "significance must be between 0 and 1" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# `schema infer` and `synthesize` set the profiling options: each flag
# changes the spec written
# ---------------------------------------------------------------------------


def _profiled_source(tmp_path) -> str:
    """A file with a column to show each profiling option's effect."""
    rng = np.random.default_rng(4)
    n = 200
    frame = pl.DataFrame(
        {
            "id": range(1, n + 1),
            "status": ["NEW" if i % 3 else "PAID" for i in range(n)],
            "email": [f"user{i}@example.com" for i in range(n)],
            "ref": [f"ORD-{i:06d}" for i in range(n)],
            "amount": [
                None if i % 8 == 0 else float(v)
                for i, v in enumerate(rng.lognormal(3, 0.5, n))
            ],
            "text": [f"{i}.5" for i in range(n)],
        }
    )
    path = tmp_path / "source.parquet"
    frame.write_parquet(path)
    return str(path)


def _status_is_an_enum(spec: dict) -> bool:
    return not isinstance(spec["columns"]["status"]["dtype"], str)


# Each ProfileOptions field but `seed`, the flags that set it, and what they
# change in the spec written for `_profiled_source`'s file.
_PROFILE_FLAGS = {
    "weights": (["--no-weights"], lambda s: "weights" not in s["columns"]["status"]),
    "max_unique_enum": (
        ["--max-unique-enum", "1"],
        lambda s: not _status_is_an_enum(s),
    ),
    "formats": (["--no-formats"], lambda s: "format" not in s["columns"]["email"]),
    "patterns": (["--no-patterns"], lambda s: "pattern" not in s["columns"]["ref"]),
    "parse_text": (
        ["--parse-text"],
        lambda s: s["columns"]["text"]["dtype"] == "Float64",
    ),
    "calculate_bounds": (["--no-bounds"], lambda s: "bounds" not in s["columns"]["id"]),
    "sample": (["--sample", "20"], lambda s: s["columns"]["id"]["bounds"][1] < 200),
    "shape": (["--no-shape"], lambda s: "distribution" not in s["columns"]["amount"]),
    "detect_unique": (["--no-keys"], lambda s: "unique" not in s["columns"]["id"]),
    "detect_order": (["--no-order"], lambda s: "order_by" not in s),
    "round_bounds": (
        ["--round-bounds", "auto"],
        lambda s: s["columns"]["id"]["bounds"][0] != 1,
    ),
    "round_rates": (
        ["--round-rates", "auto"],
        lambda s: s["columns"]["amount"]["null_probability"] != 0.125,
    ),
    "replace": (["--replace", "status"], lambda s: not _status_is_an_enum(s)),
}

# What `synthesize` profiles with: the source imitated as it stands, so no
# option that would describe it less than whole.
_SYNTHESIZE_FLAGS = (
    "max_unique_enum",
    "formats",
    "patterns",
    "parse_text",
    "sample",
    "replace",
)


def _written_spec(tmp_path, command: str, source: str, *flags: str) -> dict:
    out = tmp_path / "spec.yaml"
    if command == "schema infer":
        args = ["schema", "infer", source, "-o", str(out)]
    else:
        fake = tmp_path / "fake.parquet"
        args = [
            "synthesize",
            source,
            "-o",
            str(fake),
            "-n",
            "50",
            "--seed",
            "1",
            "--spec",
            str(out),
        ]
    assert main([*args, *flags]) == 0
    return yaml.safe_load(out.read_text(encoding="utf-8"))


def test_every_profiling_option_has_a_flag():
    """Until 1.7 the same option list was written out four times across the
    two commands."""
    assert set(_PROFILE_FLAGS) == _fields(ProfileOptions) - {"seed"}


@pytest.mark.parametrize(
    ("command", "option"),
    [("schema infer", option) for option in sorted(_PROFILE_FLAGS)]
    + [("synthesize", option) for option in _SYNTHESIZE_FLAGS],
)
def test_each_profiling_flag_changes_the_spec_written(command, option, tmp_path):
    source = _profiled_source(tmp_path)
    flags, changed = _PROFILE_FLAGS[option]
    assert not changed(_written_spec(tmp_path, command, source)), (
        "the default already shows it"
    )
    assert changed(_written_spec(tmp_path, command, source, *flags)), (command, option)


# What `--describe` leaves out, and the flag that asks for it back.
_DESCRIBED = {
    "weights": "--weights",
    "shape": "--shape",
    "detect_unique": "--keys",
    "detect_order": "--order",
}


@pytest.mark.parametrize("option", sorted(_DESCRIBED))
def test_describe_leaves_out_what_its_flag_asks_back(option, tmp_path):
    """2.0's one set of defaults records everything; `--describe` is 1.x's
    lean `schema infer`, each part of it a flag away."""
    source = _profiled_source(tmp_path)
    _, left_out = _PROFILE_FLAGS[option]
    assert left_out(_written_spec(tmp_path, "schema infer", source, "--describe"))
    asked = _written_spec(
        tmp_path, "schema infer", source, "--describe", _DESCRIBED[option]
    )
    assert not left_out(asked)


@pytest.mark.parametrize(
    "option", [*sorted(set(_PROFILE_FLAGS) - set(_SYNTHESIZE_FLAGS)), "describe"]
)
def test_synthesize_refuses_the_flags_that_only_describe(option, capsys):
    """They describe a source less than whole, which a stand-in for it must
    not be."""
    flags = ["--describe"] if option == "describe" else _PROFILE_FLAGS[option][0]
    with pytest.raises(SystemExit):
        main(["synthesize", "x.csv", "-o", "y.csv", *flags])
    assert "unrecognized arguments" in capsys.readouterr().err
