"""Deprecation, said one way (`polspec._deprecation`).

A deprecated name keeps working until its removal release, and every use
warns at the caller's own line, naming the replacement. Each deprecated
public name has a test of its own beside the feature it belongs to; these
hold the mechanism, and the public surface, to the policy.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import polars as pl
import polspec
import pytest
from polspec._deprecation import REMOVAL, deprecated, is_deprecated


@deprecated("example.old", use="example.new")
def _old(x: int) -> int:
    """Doubles x."""
    return 2 * x


def test_a_deprecated_function_warns_at_the_callers_line_and_still_works():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _old(2) == 4
    (warning,) = caught
    assert warning.category is DeprecationWarning
    assert str(warning.message) == (
        f"example.old is deprecated, and is removed in polspec {REMOVAL}: "
        "use example.new instead."
    )
    assert warning.filename == __file__, "the warning points at the caller"


def test_a_deprecated_function_says_so_first_in_its_docstring():
    assert is_deprecated(_old)
    assert _old.__doc__ is not None
    assert _old.__doc__.startswith(
        "Deprecated: removed in polspec 1.0. Use `example.new` instead."
    )
    assert "Doubles x." in _old.__doc__


def test_every_deprecated_public_name_is_marked_and_documented():
    """A public name that warns is marked, so its docstring and the API
    reference say so -- and the list is what this release deprecates."""
    marked = sorted(
        name for name in polspec.__all__ if is_deprecated(getattr(polspec, name))
    )
    assert marked == ["profile_dataframe"]


@pytest.mark.parametrize("name", ["profile_dataframe"])
def test_a_deprecated_name_is_labelled_in_the_api_reference(name):
    from pathlib import Path

    reference = Path(__file__).resolve().parents[2] / "docs" / "reference" / "api"
    pages = "\n".join(p.read_text(encoding="utf-8") for p in reference.glob("*.md"))
    assert f"## {name} (deprecated)" in pages
    assert f"`{name}`][polspec.{name}] (deprecated)" in pages


def test_the_public_names_0_18_adds_are_at_the_top_level():
    """The names `PLAN-1.0.0` found only in submodules -- each the object
    its submodule holds, so either path is the same function."""
    from polspec import expr, render, serialization, sizing

    assert polspec.to_yaml is serialization.to_yaml
    assert polspec.from_yaml is serialization.from_yaml
    assert polspec.to_python is serialization.to_python
    assert polspec.to_markdown is render.to_markdown
    assert polspec.to_mermaid is render.to_mermaid
    assert polspec.lit is expr.lit
    assert polspec.estimated_size is sizing.estimated_size


@pytest.mark.parametrize(
    "name",
    [
        "CHECK_FIELDS",
        "COLRULE_FIELDS",
        "COLSPEC_FIELDS",
        "FK_FIELDS",
        "TABLESPEC_FIELDS",
    ],
)
def test_the_field_tables_left_serialization_and_still_work(name):
    """`polspec.serialization` exported its field tables until 0.18; they are
    internal now, and the old path warns until 1.0."""
    from polspec import serialization
    from polspec.serialization import fields

    assert name not in serialization.__all__
    with pytest.warns(
        DeprecationWarning, match=f"polspec.serialization.{name}"
    ) as caught:
        table = getattr(serialization, name)
    assert table is getattr(fields, name)
    assert caught[0].filename == __file__, "the warning points at the caller"


# ---------------------------------------------------------------------------
# The inventory: what 1.0 removes, on the Stability page
# ---------------------------------------------------------------------------

STABILITY = (
    Path(__file__).resolve().parents[2] / "docs" / "explanation" / "stability.md"
)
PUBLIC = ("polspec", "polspec.drift", "polspec.validation", "polspec.serialization")


def _inventory() -> list[str]:
    """The names in the Stability page's *Removed at 1.0* table, as written."""
    page = STABILITY.read_text(encoding="utf-8")
    section = page.split("### Removed at 1.0\n", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"^\| `([^`]+)` \|", section, re.M)


def _name(entry: str) -> str:
    """`polspec.profile_dataframe(df)` -> `polspec.profile_dataframe`."""
    return entry.split("(", 1)[0]


def _deprecated_now() -> set[str]:
    """Everything that warns: marked public functions and methods, and the
    names a module's `__getattr__` keeps for the old path."""
    import importlib

    from polspec.serialization import _MOVED_FIELDS

    found = {f"polspec.serialization.{name}" for name in _MOVED_FIELDS}
    for module_name in PUBLIC:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            obj = getattr(module, name)
            if is_deprecated(obj):
                found.add(f"{module_name}.{name}")
            if isinstance(obj, type):
                found |= {
                    f"{name}.{attr}"
                    for attr in dir(obj)
                    if not attr.startswith("_") and is_deprecated(getattr(obj, attr))
                }
    return found


def test_the_inventory_is_everything_that_warns():
    """The Stability page lists what 1.0 removes; a deprecation it leaves
    out, or one it lists that no longer warns, fails here."""
    assert {_name(entry) for entry in _inventory()} == _deprecated_now()


class _Declared(polspec.FrameSpec):
    a = polspec.ColSpec(pl.Int64)


def _use(name: str) -> object:
    """The deprecated use `name` stands for."""
    if name == "polspec.profile_dataframe":
        return polspec.profile_dataframe(pl.DataFrame({"a": [1, 2, 3]}))
    if name.startswith("polspec.serialization."):
        from polspec import serialization

        return getattr(serialization, name.rsplit(".", 1)[1])
    if name.startswith("FrameSpec."):
        return getattr(_Declared, name.split(".", 1)[1])()
    raise AssertionError(f"say how to use {name} here")


@pytest.mark.parametrize("entry", _inventory())
def test_every_name_in_the_inventory_warns_and_still_works(entry):
    with pytest.warns(DeprecationWarning, match=f"removed in polspec {REMOVAL}"):
        _use(_name(entry))
