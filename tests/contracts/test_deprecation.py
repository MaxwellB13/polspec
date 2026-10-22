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

import polspec
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
        f"Deprecated: removed in polspec {REMOVAL}. Use `example.new` instead."
    )
    assert "Doubles x." in _old.__doc__


# ---------------------------------------------------------------------------
# The inventory: what is deprecated, on the Stability page
# ---------------------------------------------------------------------------

STABILITY = (
    Path(__file__).resolve().parents[2] / "docs" / "explanation" / "stability.md"
)


def _inventory() -> list[str]:
    """The names in the Stability page's *Deprecated* table, as written."""
    page = STABILITY.read_text(encoding="utf-8")
    section = page.split("### Deprecated\n", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"^\| `([^`]+)` \|", section, re.M)


def _name(entry: str) -> str:
    """`polspec.old_name(df)` -> `polspec.old_name`."""
    return entry.split("(", 1)[0]


def _deprecated_now() -> set[str]:
    """Everything that warns: public functions, and methods of public
    classes, marked deprecated."""
    found = set()
    for name in polspec.__all__:
        obj = getattr(polspec, name)
        if is_deprecated(obj):
            found.add(f"polspec.{name}")
        if isinstance(obj, type):
            found |= {
                f"{name}.{attr}"
                for attr in dir(obj)
                if not attr.startswith("_") and is_deprecated(getattr(obj, attr))
            }
    return found


def test_the_inventory_is_everything_that_warns():
    """The Stability page lists what is deprecated; a deprecation it leaves
    out, or one it lists that no longer warns, fails here. 1.0 removed what
    0.x deprecated, so it starts empty."""
    assert {_name(entry) for entry in _inventory()} == _deprecated_now()


def test_a_deprecation_now_goes_at_the_next_major():
    assert REMOVAL == "2.0"
