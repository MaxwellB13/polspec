"""The round-trip property over specs nobody wrote down.

`test_roundtrip.py` and `test_streaming.py` hold a hand-picked catalogue to
the contract. This holds *generated* specs to it: Hypothesis draws a
`TableSpec` -- dtypes, nesting, nullability, bounds, domains, formats,
weights, distributions, uniqueness, in combinations the catalogue never
listed -- and every one must

- generate a frame its own `validate()` accepts, whole and in batches;
- survive a spec file, as YAML and as Python, unchanged -- and generate the
  same frame for the same seed afterwards.

A failure prints the smallest spec Hypothesis could shrink it to, which is a
case for the catalogue once it is fixed.

CI runs each property as many times as it affords. A deep run sets
`POLSPEC_DEEP_EXAMPLES` -- say 3000 -- and runs every one that many times,
on whichever Polars is installed; see CONTRIBUTING.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import importlib.util
import os
import re
import string
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from polspec import (
    Check,
    ColRule,
    ColSpec,
    GenerationError,
    SerializationError,
    SpecError,
    TableSpec,
    col,
    frames,
    generate,
    generate_batches,
    inspect,
    validate,
)
from polspec.drift import drift
from polspec.formats import FORMATS
from polspec.render import framespec_to_markdown, framespec_to_mermaid
from polspec.serialization import from_dict, to_dict, to_python

MAX_ROWS = 120

# A deep run's count for every property, or 0 for each one's own.
DEEP_EXAMPLES = int(os.environ.get("POLSPEC_DEEP_EXAMPLES", "0"))


def _examples(usual: int) -> int:
    return DEEP_EXAMPLES or usual


# Generating and validating a frame is milliseconds, not microseconds, and a
# drawn spec is large; neither is a sign of a slow test. A failure prints a
# blob that replays it with `@reproduce_failure`.
SETTINGS = settings(
    max_examples=_examples(150),
    print_blob=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

_TEXT = st.text(alphabet=string.ascii_letters, min_size=1, max_size=6)
_INT_DTYPES = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
]
_INT_LIMITS = {
    pl.Int8: (-(2**7), 2**7 - 1),
    pl.Int16: (-(2**15), 2**15 - 1),
    pl.Int32: (-(2**31), 2**31 - 1),
    pl.Int64: (-(2**63), 2**63 - 1),
    pl.UInt8: (0, 2**8 - 1),
    pl.UInt16: (0, 2**16 - 1),
    pl.UInt32: (0, 2**32 - 1),
    pl.UInt64: (0, 2**64 - 1),
}


def _ordered(pair: tuple[Any, Any]) -> tuple[Any, Any]:
    return (pair[0], pair[1]) if pair[0] <= pair[1] else (pair[1], pair[0])


def _maybe_open(draw: st.DrawFn, pair: tuple[Any, Any]) -> tuple[Any, Any]:
    """Either end may be left open, as `bounds` allows."""
    lo, hi = pair
    return (
        None if draw(st.booleans()) and draw(st.booleans()) else lo,
        None if draw(st.booleans()) and draw(st.booleans()) else hi,
    )


def _maybe_choices(
    draw: st.DrawFn, fields: dict[str, Any], value: st.SearchStrategy[Any]
) -> dict[str, Any]:
    """Sometimes, choices from the dtype's own value space -- in place of
    its bounds and distribution, which choices would have to fit inside --
    with weights, sometimes."""
    if not draw(st.booleans()):
        return fields
    fields.pop("bounds", None)
    fields.pop("distribution", None)
    choices = draw(st.lists(value, min_size=1, max_size=5, unique=True))
    fields["choices"] = choices
    if draw(st.booleans()):
        fields["weights"] = draw(_weights(len(choices)))
    return fields


def _weights(n: int) -> st.SearchStrategy[list[float]]:
    return st.lists(st.floats(0.1, 10.0, allow_nan=False), min_size=n, max_size=n)


# ---------------------------------------------------------------------------
# What one value is: a dtype and the fields that describe it
# ---------------------------------------------------------------------------


@st.composite
def _integer(draw: st.DrawFn) -> dict[str, Any]:
    dtype = draw(st.sampled_from(_INT_DTYPES))
    lo, hi = _INT_LIMITS[dtype]
    fields: dict[str, Any] = {"dtype": dtype}
    if draw(st.booleans()):
        pair = _ordered((draw(st.integers(lo, hi)), draw(st.integers(lo, hi))))
        fields["bounds"] = _maybe_open(draw, pair)
    if draw(st.booleans()):
        fields["distribution"] = draw(st.sampled_from(["uniform", "normal"]))
    return _maybe_choices(draw, fields, st.integers(lo, hi))


@st.composite
def _float(draw: st.DrawFn) -> dict[str, Any]:
    dtype = draw(st.sampled_from([pl.Float32, pl.Float64]))
    value = st.floats(-1e6, 1e6, allow_nan=False, width=32)
    fields: dict[str, Any] = {"dtype": dtype}
    if draw(st.booleans()):
        fields["bounds"] = _maybe_open(draw, _ordered((draw(value), draw(value))))
    if draw(st.booleans()):
        fields["distribution"] = draw(st.sampled_from(["uniform", "normal"]))
    if draw(st.booleans()):
        fields["nan_probability"] = draw(st.floats(0.0, 1.0))
    return _maybe_choices(draw, fields, value)


@st.composite
def _decimal(draw: st.DrawFn) -> dict[str, Any]:
    value = st.decimals(-(10**6), 10**6, places=2, allow_nan=False)
    fields: dict[str, Any] = {"dtype": pl.Decimal(10, 2)}
    if draw(st.booleans()):
        fields["bounds"] = _ordered((draw(value), draw(value)))
    return _maybe_choices(draw, fields, value)


@st.composite
def _temporal(draw: st.DrawFn) -> dict[str, Any]:
    kind = draw(st.sampled_from(["date", "datetime", "time", "duration"]))
    if kind == "date":
        value = st.dates(dt.date(1970, 1, 1), dt.date(2100, 1, 1))
        fields: dict[str, Any] = {"dtype": pl.Date}
    elif kind == "datetime":
        value = st.datetimes(dt.datetime(1970, 1, 1), dt.datetime(2100, 1, 1))
        fields = {"dtype": pl.Datetime("us")}
    elif kind == "time":
        value = st.times()
        fields = {"dtype": pl.Time}
    else:
        # Whole milliseconds: a `Duration("ms")` column cannot hold a finer
        # one, and a choice it cannot hold is refused.
        value = st.integers(-(10**12), 10**12).map(
            lambda ms: dt.timedelta(milliseconds=ms)
        )
        fields = {"dtype": pl.Duration("ms")}
    if draw(st.booleans()):
        fields["bounds"] = _ordered((draw(value), draw(value)))
    return _maybe_choices(draw, fields, value)


@st.composite
def _boolean(draw: st.DrawFn) -> dict[str, Any]:
    fields: dict[str, Any] = {"dtype": pl.Boolean}
    if draw(st.booleans()):
        fields["weights"] = draw(_weights(2))
    return fields


@st.composite
def _text(draw: st.DrawFn) -> dict[str, Any]:
    dtype = draw(st.sampled_from([pl.String, pl.Binary]))
    fields: dict[str, Any] = {"dtype": dtype}
    shape = draw(st.sampled_from(["length", "choices", "format", "free"]))
    if dtype == pl.Binary and shape in ("choices", "format"):
        shape = "length"
    if shape == "length":
        lo = draw(st.integers(0, 12))
        fields["string_length"] = (lo, draw(st.integers(lo, 20)))
    elif shape == "choices":
        choices = draw(st.lists(_TEXT, min_size=1, max_size=5, unique=True))
        fields["choices"] = choices
        if draw(st.booleans()):
            fields["weights"] = draw(_weights(len(choices)))
    elif shape == "format":
        fields["format"] = draw(st.sampled_from(sorted(FORMATS)))
        if draw(st.booleans()):
            # `~` is in no format, so an extra is never one of its values.
            extras = draw(
                st.lists(
                    _TEXT.map(lambda s: f"~{s}"), min_size=1, max_size=3, unique=True
                )
            )
            fields["extra_values"] = (
                extras
                if draw(st.booleans())
                else {e: draw(st.floats(0.01, 0.3)) for e in extras}
            )
    return fields


@st.composite
def _categories(draw: st.DrawFn) -> dict[str, Any]:
    categories = draw(st.lists(_TEXT, min_size=1, max_size=5, unique=True))
    if draw(st.booleans()):
        fields: dict[str, Any] = {"dtype": pl.Enum(categories)}
        if draw(st.booleans()):
            subset = draw(
                st.lists(st.sampled_from(categories), min_size=1, unique=True)
            )
            fields["choices"] = subset
            if draw(st.booleans()):
                fields["weights"] = draw(_weights(len(subset)))
        elif draw(st.booleans()):
            fields["weights"] = draw(_weights(len(categories)))
        return fields
    fields = {"dtype": pl.Categorical}
    if draw(st.booleans()):
        fields["choices"] = categories
    return fields


_VALUE = st.one_of(
    _integer(), _float(), _decimal(), _temporal(), _boolean(), _text(), _categories()
)


# ---------------------------------------------------------------------------
# A column: a value, perhaps nested, with its nullability and uniqueness
# ---------------------------------------------------------------------------


def _uniquely_drawable(fields: dict[str, Any]) -> bool:
    """Whether a unique column has room for every row, and nothing a unique
    column refuses."""
    dtype = fields["dtype"]
    if "weights" in fields or fields.get("distribution") not in (None, "uniform"):
        return False
    if "choices" in fields:
        return False  # five choices cannot fill a unique column of MAX_ROWS
    if dtype in _INT_DTYPES:
        lo, hi = _INT_LIMITS[dtype]
        low, high = fields.get("bounds") or (None, None)
        low = lo if low is None else low
        high = hi if high is None else high
        # An open end generates within the default range, not the dtype's.
        if fields.get("bounds") is not None and None in fields["bounds"]:
            return False
        return high - low + 1 >= MAX_ROWS
    if dtype == pl.String:
        return "choices" not in fields and (
            "string_length" in fields and fields["string_length"][1] >= 3
        )
    return False


@st.composite
def columns(draw: st.DrawFn) -> ColSpec:
    fields = draw(_VALUE)
    wrap = draw(st.sampled_from(["none", "none", "list", "array"]))
    if wrap == "list":
        fields["dtype"] = pl.List(fields["dtype"])
        lo = draw(st.integers(0, 3))
        fields["list_length"] = (lo, draw(st.integers(lo, 4)))
    elif wrap == "array":
        fields["dtype"] = pl.Array(fields["dtype"], draw(st.integers(1, 3)))
    if wrap != "none" and draw(st.booleans()):
        fields["element_null_probability"] = draw(st.floats(0.0, 1.0))
    if draw(st.booleans()):
        fields["nullable"] = True
        fields["null_probability"] = draw(st.floats(0.0, 1.0))
    if wrap == "none" and _uniquely_drawable(fields) and draw(st.booleans()):
        fields["unique"] = True
    return ColSpec(**fields)


def _check_that_holds(name: str, column: ColSpec) -> Check | None:
    """A check whose literal has the column's own type, and which the
    column's declaration makes true of every generated row: `>=` its lower
    bound, or `is_in` its choices. Validation-only, so it has to hold by
    construction -- and it carries a date, a time or a duration literal
    through the spec file."""
    value = column.value_dtype
    if column.dtype != value or value.is_float() or value.is_decimal():
        return None  # a float literal is not the column's float; no Decimal literal
    if column.choices is not None:
        return Check(col(name).is_in(list(column.choices)), name=f"{name}_in_choices")
    bounds = column.bounds
    if bounds is not None and bounds.min is not None and bounds.max is not None:
        # Names the column twice, as a range check does -- which validation
        # used to turn into a sample struct with two fields of one name.
        return Check(
            (col(name) >= bounds.min) & (col(name) <= bounds.max),
            name=f"{name}_in_range",
        )
    if bounds is not None and bounds.min is not None:
        return Check(col(name) >= bounds.min, name=f"{name}_at_least_min")
    return None


# Names a renderer or a spec file has to take care over: Markdown's and
# Mermaid's punctuation, whitespace, a keyword, a key word, a leading digit,
# letters outside ASCII, the empty string -- and arbitrary text besides.
_AWKWARD_NAMES = [
    "a|b",
    "with space",
    "",
    "1st",
    "class",
    "PK",
    "pk",
    "größe",
    "名前",
    "a-b",
    "a_b",
    "a b",
    'quote"d',
    "back`tick",
    "new\nline",
    "slash\\|pipe",
    "%%",
    # Names Polars reads as patterns: every column, and regular expressions.
    "*",
    "^c0$",
    "^.*$",
    # Names polspec or Polars use for helper columns beside the data's own.
    "__row",
    "__idx",
    "__pool",
    "__polspec_row",
    "__polspec_key",
    "__polspec_value",
    "__polspec_count",
    "count",
    "len",
]
NAMES = st.one_of(
    st.sampled_from(_AWKWARD_NAMES),
    st.text(st.characters(exclude_categories=("Cs",)), max_size=8),
).filter(
    # `flag` is the rule's key, below; a NUL is refused by name.
    lambda name: name != "flag" and "\x00" not in name
)


@st.composite
def specs(
    draw: st.DrawFn,
    names: st.SearchStrategy[str] | None = None,
    spec_name: st.SearchStrategy[str] | None = None,
) -> TableSpec:
    """A drawn spec, its columns named `c0`, `c1`... unless `names` draws
    them, and itself named `Drawn` unless `spec_name` does."""
    drawn = draw(st.lists(columns(), min_size=1, max_size=4))
    keys = (
        draw(st.lists(names, min_size=len(drawn), max_size=len(drawn), unique=True))
        if names is not None
        else [f"c{i}" for i in range(len(drawn))]
    )
    cols = dict(zip(keys, drawn, strict=True))
    if draw(st.booleans()):
        # A rule, keyed on a flag, rewriting a column to some of its own
        # choices: which fit its domain and its dtype by construction.
        ruleable = [
            n
            for n, c in cols.items()
            if c.choices is not None and not c.unique and c.dtype == c.value_dtype
        ]
        if ruleable:
            target = draw(st.sampled_from(ruleable))
            chosen = draw(
                st.lists(
                    st.sampled_from(cols[target].choices),
                    min_size=1,
                    max_size=3,
                    unique=True,
                )
            )
            cols["flag"] = ColSpec(pl.Boolean)
            cols[target] = dataclasses.replace(
                cols[target], rules=(ColRule(when=col("flag"), choices=chosen),)
            )
    checks = [
        check
        for name, column in cols.items()
        if name  # col() takes no empty name
        and draw(st.booleans())
        and (check := _check_that_holds(name, column)) is not None
    ]
    name = draw(spec_name) if spec_name is not None else "Drawn"
    return TableSpec(name, cols, checks=checks)


_ROWS = st.integers(0, MAX_ROWS)
_SEEDS = st.integers(0, 2**32)


# ---------------------------------------------------------------------------
# The properties
# ---------------------------------------------------------------------------


@SETTINGS
@given(spec=specs(), n=_ROWS, seed=_SEEDS)
def test_what_a_drawn_spec_generates_it_validates(spec, n, seed):
    df = generate(spec, n, seed=seed)
    assert df.height == n
    assert df.schema == spec.schema()
    df.to_dicts()
    validate(spec, df)


@SETTINGS
@given(spec=specs(), n=_ROWS, seed=_SEEDS, batch_size=st.integers(1, 50))
def test_what_a_drawn_spec_streams_it_validates(spec, n, seed, batch_size):
    parts = list(generate_batches(spec, n, batch_size=batch_size, seed=seed))
    streamed = pl.concat(parts) if parts else generate(spec, 0, seed=seed)
    assert streamed.height == n
    assert streamed.schema == spec.schema()
    validate(spec, streamed)


@SETTINGS
@given(spec=specs(), seed=_SEEDS)
def test_a_drawn_spec_survives_a_yaml_file(spec, seed):
    text = yaml.safe_dump(to_dict(spec), sort_keys=False)
    reloaded = from_dict(yaml.safe_load(text))
    assert reloaded == spec
    assert generate(reloaded, 50, seed=seed).equals(generate(spec, 50, seed=seed))


@settings(SETTINGS, max_examples=_examples(25))
@given(spec=specs(), seed=_SEEDS)
def test_a_drawn_spec_survives_a_python_file(spec, seed):
    reloaded = _through_python(spec)
    assert reloaded == spec
    assert generate(reloaded, 50, seed=seed).equals(generate(spec, 50, seed=seed))


def _through_python(spec: TableSpec) -> TableSpec:
    """`spec` written by `to_python`, imported, and read back -- or the
    `to_python` error, with nothing left on disk."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "drawn_spec.py"
        try:
            to_python(spec, path)
        except SerializationError:
            assert not path.exists()
            raise
        module_spec = importlib.util.spec_from_file_location("_drawn_spec", path)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        sys.modules["_drawn_spec"] = module
        try:
            module_spec.loader.exec_module(module)
        finally:
            del sys.modules["_drawn_spec"]
    return vars(module)[spec.name].spec  # `__dict__` is a name too


def _python_declares(name: str) -> bool:
    """Whether `class <name>` declares a class under that very name -- asked
    of Python itself, and only of identifiers, so nothing else is run."""
    if not name.isidentifier():
        return False
    namespace: dict[str, Any] = {}
    try:
        exec(f"class {name}: pass", namespace)  # an identifier: nothing else runs
    except SyntaxError:
        return False
    return name in namespace


# Spec names `to_python` writes, or refuses: drawn text, names the file's own
# imports also use, soft keywords, keywords, and names Python normalises.
_CLASS_NAMES = st.one_of(
    NAMES.filter(bool),
    st.sampled_from(
        [
            *("Drawn", "pl", "col", "ColSpec", "FrameSpec", "match", "_"),
            *("größe", "class", "None", "__debug__"),
            "\N{LATIN SMALL LIGATURE FI}le",  # Python reads it as `file`
            "\N{ROMAN NUMERAL NINE}",  # ... and this as `IX`
        ]
    ),
)


@settings(SETTINGS, max_examples=_examples(60))
@given(spec=specs(names=NAMES, spec_name=_CLASS_NAMES), seed=_SEEDS)
def test_any_name_is_written_as_python_or_refused_by_name(spec, seed):
    """Before 0.15.1 a spec named `Odd Name` or `class` wrote a module that
    did not parse."""
    try:
        reloaded = _through_python(spec)
    except SerializationError as error:
        assert not _python_declares(spec.name)
        assert repr(spec.name) in str(error)
        return
    assert _python_declares(spec.name)
    assert reloaded == spec
    assert generate(reloaded, 20, seed=seed).equals(generate(spec, 20, seed=seed))


@settings(SETTINGS, max_examples=_examples(30))
@given(spec=specs(), seed=_SEEDS)
def test_a_drawn_spec_does_not_drift_from_its_own_output(spec, seed):
    """0.14.0's promise, over specs nobody wrote down: data a spec generated
    -- enough of it for the statistics to speak -- is not drift from it."""
    try:
        frame = generate(spec, 3_000, seed=seed)
    except GenerationError:
        assume(False)  # a unique domain smaller than 3,000 rows, refused by name
    report = drift(spec, frame)
    assert report.unchanged, report.findings


_NAMED = specs(names=NAMES, spec_name=NAMES.filter(bool))

# A `|` splits a GFM table cell -- inside backticks too -- unless an odd run
# of backslashes escapes it.
_CELL_BREAK = re.compile(r"(?<!\\)(?:\\\\)*\|")


@SETTINGS
@given(spec=_NAMED, n=_ROWS, seed=_SEEDS)
def test_any_name_generates_what_it_validates(spec, n, seed):
    """Before 0.15.1 a list column named "" could not generate."""
    df = generate(spec, n, seed=seed)
    assert df.columns == list(spec.columns)
    validate(spec, df)


@SETTINGS
@given(spec=_NAMED, n=st.integers(1, MAX_ROWS), seed=_SEEDS)
def test_any_name_splits_a_frame_into_passing_and_failing_exactly(spec, n, seed):
    """Every third row loses its first column's value, which the spec does
    not allow: the rows `passing_rows()` keeps and the rows `--failing`
    writes are the frame, each row once, whatever the columns are named."""
    first = next(iter(spec.columns))
    assume(not spec.columns[first].nullable)
    df = generate(spec, n, seed=seed)
    bad = df.with_columns(
        pl.when(pl.int_range(pl.len()) % 3 == 0)
        .then(pl.lit(None, dtype=df.schema[first]))
        .otherwise(frames.column(first))
        .alias(first)
    )
    report = inspect(spec, bad)
    passing = report.passing_rows().collect()
    failing = report._failing_once().collect()
    assert passing.height + failing.height == n
    assert failing.height >= (n + 2) // 3
    assert passing.columns == bad.columns


@SETTINGS
@given(spec=_NAMED, seed=_SEEDS)
def test_any_name_survives_a_yaml_file(spec, seed):
    text = yaml.safe_dump(to_dict(spec), sort_keys=False)
    reloaded = from_dict(yaml.safe_load(text))
    assert reloaded == spec
    assert generate(reloaded, 20, seed=seed).equals(generate(spec, 20, seed=seed))


@SETTINGS
@given(spec=_NAMED)
def test_any_name_keeps_the_data_dictionary_a_table(spec):
    """Every row of every table has as many cells as its header, whatever
    the names, choices and categories hold."""
    tables: list[list[str]] = [[]]
    for line in framespec_to_markdown(spec).splitlines():
        if line.startswith("|"):
            tables[-1].append(line)
        elif tables[-1]:
            tables.append([])
    for rows in filter(None, tables):
        assert len({len(_CELL_BREAK.findall(row)) for row in rows}) == 1, rows


@SETTINGS
@given(spec=_NAMED)
def test_any_name_draws_an_er_diagram_mermaid_reads(spec):
    """An entity is one word; each attribute a different one, never read as
    a key; each comment one quoted string on the attribute's own line."""
    lines = framespec_to_mermaid(spec).splitlines()
    assert lines[0] == "erDiagram"
    assert re.fullmatch(r"    [A-Za-z_][A-Za-z0-9_]* \{", lines[1]), lines[1]
    assert lines[-1] == "    }"
    attribute = re.compile(
        r'        \S+ ([A-Za-z_][A-Za-z0-9_-]*)( (PK|UK|FK))?( "[^"]*")?'
    )
    names = []
    for line in lines[2:-1]:
        match = attribute.fullmatch(line)
        assert match, line
        assert "~" not in line, line  # hangs Mermaid 10's lexer
        names.append(match[1])
    assert len(names) == len(set(names)) == len(spec.columns)
    assert not {n.upper() for n in names} & {"PK", "FK", "UK"}


def test_the_strategy_reaches_every_kind_of_column():
    """A strategy that quietly stopped drawing a whole family of dtypes would
    leave these properties passing on less than they claim."""
    seen: set[str] = set()

    # Derandomized: the same draws every run, so a rare field (a format is
    # one column in dozens) is either reached or not -- never flaky.
    @settings(max_examples=1_000, deadline=None, database=None, derandomize=True)
    @given(column=columns())
    def record(column: ColSpec) -> None:
        dtype = column.dtype
        if isinstance(dtype, (pl.List, pl.Array)):
            seen.add(type(dtype).__name__)
            dtype = column.value_dtype
        seen.add(
            "integer"
            if dtype.is_integer()
            else "float"
            if dtype.is_float()
            else type(dtype).__name__
        )
        seen.update(
            name
            for name in (
                "unique",
                "nullable",
                "format",
                "extra_values",
                "choices",
                "weights",
                "nan_probability",
            )
            if getattr(column, name)
        )
        family = "integer" if dtype.is_integer() else type(dtype).__name__
        if column.choices is not None and dtype != pl.String:
            seen.add(f"{family} choices")
        if column.bounds is not None and family in ("Time", "Duration"):
            seen.add(f"{family} bounds")

    record()
    assert {
        "integer",
        "float",
        "Decimal",
        "Date",
        "Boolean",
        "String",
        "Enum",
        "Categorical",
        "List",
        "Array",
        "unique",
        "nullable",
        "format",
        "extra_values",
        "choices",
        "weights",
        "nan_probability",
        "integer choices",
        "Float64 choices",
        "Decimal choices",
        "Date choices",
        "Datetime choices",
        "Time choices",
        "Duration choices",
        "Time bounds",
        "Duration bounds",
    } <= seen, sorted(seen)


def test_the_spec_strategy_reaches_rules_and_checks():
    """Each defect the 0.13.0 review found sat in a gap of the strategy --
    no rules, no checks, no typed literals -- so those are held to it too."""
    seen: set[str] = set()

    @settings(max_examples=1_000, deadline=None, database=None, derandomize=True)
    @given(spec=specs())
    def record(spec: TableSpec) -> None:
        if any(c.rules for c in spec.columns.values()):
            seen.add("rules")
        for check in spec.checks:
            seen.add("checks")
            literals = check.pred.literals() if check.pred is not None else []
            seen.update(type(v).__name__ for v in literals)

    record()
    assert {"rules", "checks", "date", "datetime", "time", "timedelta"} <= seen, sorted(
        seen
    )


# ---------------------------------------------------------------------------
# What must be refused, and how
# ---------------------------------------------------------------------------


@st.composite
def _invalid_declarations(draw: st.DrawFn) -> tuple[str, Any]:
    """A declaration that cannot mean anything, as a thunk that builds it."""
    kind = draw(
        st.sampled_from(
            [
                "integer out of range",
                "integer with a fraction",
                "text on a temporal column",
                "decimal past its scale",
                "rule choice out of range",
                "check on an undeclared column",
            ]
        )
    )
    if kind == "integer out of range":
        dtype = draw(st.sampled_from([pl.Int8, pl.Int16, pl.UInt8, pl.UInt16]))
        lo, hi = _INT_LIMITS[dtype]
        bad = draw(
            st.one_of(st.integers(hi + 1, hi * 4), st.integers(lo - hi * 4, lo - 1))
        )
        return kind, lambda: ColSpec(dtype, choices=[bad])
    if kind == "integer with a fraction":
        bad = draw(st.floats(-100, 100).filter(lambda f: not f.is_integer()))
        return kind, lambda: ColSpec(pl.Int32, choices=[bad])
    if kind == "text on a temporal column":
        dtype = draw(st.sampled_from([pl.Date, pl.Datetime("us")]))
        text = draw(st.text(alphabet=string.ascii_letters, min_size=1, max_size=8))
        return kind, lambda: ColSpec(dtype, choices=[text])
    if kind == "decimal past its scale":
        bad = draw(
            st.decimals(-100, 100, places=4).filter(
                lambda d: d != d.quantize(Decimal("0.01"))
            )
        )
        return kind, lambda: ColSpec(pl.Decimal(10, 2), choices=[bad])
    if kind == "rule choice out of range":
        return kind, lambda: ColSpec(
            pl.Int8,
            rules=[ColRule(when=col("g"), choices=[draw(st.integers(128, 10_000))])],
        )
    unknown = draw(st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=6))
    return kind, lambda: TableSpec(
        "T",
        {"a": ColSpec(pl.Int64)},
        checks=[Check(col(f"x_{unknown}") > 0, name="unknown")],
    )


@SETTINGS
@given(declaration=_invalid_declarations())
def test_an_invalid_declaration_is_refused_as_a_spec_error(declaration):
    """Refused where it is written, as polspec's own error -- never accepted
    and then failing inside Polars at generation or validation, which is
    what each of these used to do."""
    kind, build = declaration
    try:
        build()
    except SpecError:
        return
    pytest.fail(f"{kind}: accepted")


# ---------------------------------------------------------------------------
# Data that arrives as some other dtype
# ---------------------------------------------------------------------------

# What a column of each kind can arrive as: the dtype another pipeline, a
# text format or a wider type hands back. Some stand in for the declaration
# and some do not; `inspect()` has to answer for all of them.
_WIDE_INTS = [*_INT_DTYPES, pl.Int128]


def _transports(value: pl.DataType) -> list[pl.DataType]:
    if value.is_integer():
        return [*_WIDE_INTS, pl.Float64, pl.String]
    if value.is_float():
        return [pl.Float32, pl.Float64, pl.Int64, pl.String]
    if value.is_decimal():
        return [
            pl.Decimal(38, 4),
            pl.Decimal(12, 2),
            pl.Decimal(6, 1),
            pl.Float64,
            pl.Int64,
        ]
    if value == pl.Date:
        return [
            pl.Date,
            pl.Datetime("ms"),
            pl.Datetime("us", "UTC"),
            pl.String,
            pl.Int32,
        ]
    if isinstance(value, pl.Datetime):
        return [
            pl.Date,
            pl.Datetime("ns"),
            pl.Datetime("us", "Asia/Tokyo"),
            pl.Time,
            pl.Int64,
        ]
    if value == pl.Time:
        return [pl.Time, pl.Duration("us"), pl.String, pl.Int64]
    if isinstance(value, pl.Duration):
        return [pl.Duration("us"), pl.Duration("ns"), pl.Int64, pl.Time]
    if value == pl.Boolean:
        return [pl.Boolean, pl.Int8, pl.String]
    if isinstance(value, pl.Enum):
        wider = pl.Enum([*value.categories.to_list(), "~extra"])
        return [pl.String, pl.Categorical, wider]
    if value == pl.Categorical or isinstance(value, pl.Categorical):
        return [pl.String, pl.Categorical]
    return [pl.String, pl.Binary, pl.Categorical]


def _transport(draw: st.DrawFn, dtype: pl.DataType) -> pl.DataType:
    """A dtype `dtype` may arrive as, recursing into a list's elements."""
    if isinstance(dtype, pl.Array):
        inner = _transport(draw, dtype.inner)
        return draw(st.sampled_from([pl.Array(inner, dtype.size), pl.List(inner)]))
    if isinstance(dtype, pl.List):
        return pl.List(_transport(draw, dtype.inner))
    return draw(st.sampled_from(_transports(dtype)))


def _full_range(dtype: pl.DataType) -> st.SearchStrategy[Any] | None:
    """Any value a column of `dtype` holds, including ones a narrower
    declaration cannot: where a range check has to earn its keep."""
    if dtype.is_integer():
        lo, hi = (-(2**127), 2**127 - 1) if dtype == pl.Int128 else _INT_LIMITS[dtype]
        # Just past the narrower widths, where a range check has an edge.
        edges = [v for v in (lo, hi, -129, -1, 0, 128, 256, 70_000) if lo <= v <= hi]
        return st.one_of(st.integers(lo, hi), st.sampled_from(edges))
    if dtype.is_float():
        return st.floats(width=32 if dtype == pl.Float32 else 64)
    if isinstance(dtype, pl.Decimal):
        places = dtype.scale or 0
        # Capped at twenty integer digits -- past every declared Decimal here,
        # and short of the 38 Polars 2 will not build from Python at the edge.
        digits = min((dtype.precision or 38) - places, 20)
        limit = Decimal(10) ** digits
        return st.decimals(
            -limit + 1, limit - 1, places=places, allow_nan=False, allow_infinity=False
        )
    return None


@st.composite
def _lone_number(draw: st.DrawFn) -> TableSpec:
    """One integer or Decimal column: the declarations a wider dtype can
    carry values past, drawn often enough for the range check to be tried --
    and often unbounded, since a checked bound leaves the range check
    nothing to do on its side."""
    fields = draw(st.one_of(_integer(), _decimal()))
    if draw(st.booleans()):
        fields.pop("bounds", None)
    if draw(st.booleans()):
        fields["nullable"] = True
    return TableSpec("Drawn", {"c0": ColSpec(**fields)})


@st.composite
def _arrived(draw: st.DrawFn) -> tuple[TableSpec, pl.DataFrame]:
    """A drawn spec, and a frame that satisfied it before its columns were
    carried through other dtypes -- and, sometimes, had values from the
    whole range of those dtypes written over a few rows."""
    spec = draw(st.one_of(specs(), _lone_number()))
    n = draw(st.integers(0, 40))
    frame = generate(spec, n, seed=draw(_SEEDS))
    arrived = {}
    for name, column in spec.columns.items():
        series = frame[name]
        if draw(st.integers(0, 3)):  # three times in four
            target = _transport(draw, column.dtype)
            # No such cast in Polars: the column arrives as it was.
            with contextlib.suppress(pl.exceptions.PolarsError):
                series = series.cast(target, strict=False)
        values = _full_range(series.dtype)
        if values is not None and n and draw(st.integers(0, 3)):
            rows = draw(
                st.lists(st.integers(0, n - 1), min_size=1, max_size=3, unique=True)
            )
            written = pl.Series(
                draw(st.lists(values, min_size=len(rows), max_size=len(rows))),
                dtype=series.dtype,
            )
            series = series.scatter(rows, written)
        arrived[name] = series
    return spec, pl.DataFrame(arrived)


@SETTINGS
@given(case=_arrived())
def test_inspect_answers_for_data_of_any_dtype(case):
    """Whatever dtype a column arrives as, `inspect()` returns a report;
    and a report with nothing in it is a promise `cast=True` keeps -- the
    frame it hands back has the declared schema and is still clean. Before
    0.13.1 a `Time` on a bounded `Date` raised Polars' cast error, and an
    `Int8` holding 1000 passed and then failed its own cast."""
    spec, frame = case
    report = inspect(spec, frame)
    if not report:
        cast = validate(spec, frame, cast=True)
        assert cast.schema == spec.schema()
        assert not inspect(spec, cast)


def test_the_arrival_strategy_reaches_both_verdicts():
    """The cast property only says something for clean reports, and the
    range check only for the rows written past a declaration; both have to
    be drawn often enough to count."""
    seen: dict[str, int] = {"clean": 0, "dtype": 0, "dtype_range": 0, "transported": 0}

    # As many draws as the other reach tests take: at 400 the range findings
    # fell as low as 13 in a full-suite run, too close to the floor to trust.
    @settings(max_examples=1_000, deadline=None, database=None, derandomize=True)
    @given(case=_arrived())
    def record(case: tuple[TableSpec, pl.DataFrame]) -> None:
        spec, frame = case
        if frame.schema != spec.schema():
            seen["transported"] += 1
        report = inspect(spec, frame)
        if not report:
            seen["clean"] += 1
        for finding in report.findings:
            if finding.code == "dtype":
                seen["dtype_range" if finding.row_level else "dtype"] += 1

    record()
    assert all(count >= 20 for count in seen.values()), seen
    print(seen)
