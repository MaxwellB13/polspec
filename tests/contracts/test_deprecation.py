"""Deprecation, said one way (`polspec._deprecation`).

A deprecated name keeps working until its removal release, and every use
warns at the caller's own line, naming the replacement. Each deprecated
public name has a test of its own beside the feature it belongs to; these
hold the mechanism, and the public surface, to the policy.
"""

from __future__ import annotations

import warnings

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
