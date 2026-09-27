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
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import importlib.util
import string
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from polspec import (
    Check,
    ColRule,
    ColSpec,
    SpecError,
    TableSpec,
    col,
    generate,
    generate_batches,
    validate,
)
from polspec.formats import FORMATS
from polspec.serialization import from_dict, to_dict, to_python

MAX_ROWS = 120

# Generating and validating a frame is milliseconds, not microseconds, and a
# drawn spec is large; neither is a sign of a slow test.
SETTINGS = settings(
    max_examples=150,
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


@st.composite
def specs(draw: st.DrawFn) -> TableSpec:
    drawn = draw(st.lists(columns(), min_size=1, max_size=4))
    cols = {f"c{i}": column for i, column in enumerate(drawn)}
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
        if draw(st.booleans())
        and (check := _check_that_holds(name, column)) is not None
    ]
    return TableSpec("Drawn", cols, checks=checks)


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


@settings(SETTINGS, max_examples=25)
@given(spec=specs(), seed=_SEEDS)
def test_a_drawn_spec_survives_a_python_file(spec, seed):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "drawn_spec.py"
        to_python(spec, path)
        module_spec = importlib.util.spec_from_file_location("_drawn_spec", path)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        sys.modules["_drawn_spec"] = module
        try:
            module_spec.loader.exec_module(module)
        finally:
            del sys.modules["_drawn_spec"]
    reloaded = module.Drawn.spec
    assert reloaded == spec
    assert generate(reloaded, 50, seed=seed).equals(generate(spec, 50, seed=seed))


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
            for name in ("unique", "nullable", "format", "choices", "weights")
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
        "choices",
        "weights",
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
