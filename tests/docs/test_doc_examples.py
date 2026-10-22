"""Every Python example in `docs/` and the README is executed.

The rest of `test_docs.py` checks that the documentation *points* at things
that exist -- every exported name reachable, every link resolving, every page
in the nav. This checks that the code on those pages runs, which nothing did
before: the docs carried over a hundred Python blocks and not one of them was
ever executed, so an example could go stale for a whole release without
anything noticing. One already had.

A page's blocks run in order in one namespace, on top of a preamble of the
imports every page takes for granted -- and nothing else, so a page declares
every spec and frame its examples use, and a reader can copy them. A block
that cannot run that way says so in an HTML comment on the line above it,
which renders as nothing:

    <!-- docs: skip -->      an illustrative fragment, not a runnable example
    <!-- docs: raises -->    demonstrates an error, and must actually raise
    <!-- docs: warns -->     demonstrates a warning, and must actually warn
    <!-- docs: polars2 -->   uses a Polars 2 dtype; runs only where it exists

A marker is a claim about the block, so both are checked: `raises` fails the
suite if the block stops raising, the same way `xfail(strict=True)` pins the
known limitations.

What a page shows is checked too, since it goes stale as quietly as code:
a `raises` block's `# SomeError: message` comment names the class raised
and the start of its message, and an output block straight after a Python
block is what that block prints. A shown message or output may stop short
with `...`.
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import polars as pl
import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"

# A ```python fence, with whatever HTML comment sits on the line above it.
BLOCK = re.compile(
    r"(?:^[ \t]*<!--[ \t]*docs:[ \t]*(?P<marker>[a-z0-9]+)[ \t]*-->[ \t]*\n)?"
    r"^```python\n(?P<code>.*?)^```",
    re.S | re.M,
)

MARKERS = {"skip", "raises", "warns", "polars2"}

# `# SpecError: the message`, and the `# ...` lines that carry it on.
CLAIM = re.compile(
    r"^[ \t]*#[ \t]*(?P<name>\w+(?:Error|Exception)):[ \t]*(?P<rest>.*)$"
)
CARRIED = re.compile(r"^[ \t]*#[ \t]?(?P<rest>.*)$")

# An output block straight after a Python one: a ```text fence, or a bare one.
OUTPUT = re.compile(r"\s*```(?:text)?\n(?P<shown>.*?)^```", re.S | re.M)

# `Map` is the Polars 2 dtype a `polars2` block uses; on Polars 1 the block
# is skipped, and it runs -- and must pass -- everywhere else.
HAS_POLARS_2 = hasattr(pl, "Map")

# The imports pages take for granted. Specs and frames are not among them:
# a page declares those itself, where a reader can see them.
PREAMBLE = """
from datetime import date, datetime
from pathlib import Path

import polars as pl
import polspec
from polspec import (
    Bound,
    CatSpec,
    Check,
    ColRule,
    ColSpec,
    Finding,
    ForeignKey,
    FrameSpec,
    Hierarchy,
    MultiValidationError,
    Registry,
    TableSpec,
    ValidationError,
    ValidationReport,
    col,
)

"""


def _pages() -> list[Path]:
    return sorted(p for p in DOCS.rglob("*.md") if BLOCK.search(p.read_text("utf-8")))


def _blocks(page: Path) -> list[tuple[int, str | None, str]]:
    """Every python block on a page, as (line number, marker, source)."""
    return [(line, marker, code) for line, marker, code, _ in _shown_blocks(page)]


def _shown_blocks(page: Path) -> list[tuple[int, str | None, str, str | None]]:
    """Every python block on a page, with the output shown straight after it."""
    text = page.read_text(encoding="utf-8")
    out: list[tuple[int, str | None, str, str | None]] = []
    for match in BLOCK.finditer(text):
        marker = match.group("marker")
        if marker is not None and marker not in MARKERS:
            raise AssertionError(
                f"{page.name}: unknown marker <!-- docs: {marker} -->; "
                f"expected one of {sorted(MARKERS)}"
            )
        shown = OUTPUT.match(text, match.end())
        out.append(
            (
                text[: match.start()].count("\n") + 1,
                marker,
                match.group("code"),
                shown.group("shown") if shown else None,
            )
        )
    return out


def _claim(code: str) -> tuple[str, str] | None:
    """The error a `raises` block says it raises: its class and message."""
    lines = code.splitlines()
    for i, line in enumerate(lines):
        match = CLAIM.match(line)
        if match is None:
            continue
        message = [match.group("rest")]
        for more in lines[i + 1 :]:
            carried = CARRIED.match(more)
            if carried is None:
                break
            message.append(carried.group("rest"))
        return match.group("name"), " ".join(message)
    return None


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _shows(shown: str, actual: str) -> bool:
    """Whether `shown` is `actual`, where a `...` in it stands for any text."""
    pieces = [_plain(piece) for piece in _plain(shown).replace("…", "...").split("...")]
    pattern = r"\s*.*?\s*".join(re.escape(piece) for piece in pieces)
    return re.fullmatch(pattern, _plain(actual), re.S) is not None


@pytest.mark.parametrize("page", _pages(), ids=lambda p: p.relative_to(DOCS).as_posix())
def test_a_pages_examples_run(page: Path, tmp_path: Path, monkeypatch) -> None:
    # Pages write files (`Orders.to_yaml("orders.yaml")`), so give each its own
    # directory rather than the repository root.
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "docs_example"}
    exec(compile(PREAMBLE, "<preamble>", "exec"), namespace)

    for line, marker, code, shown in _shown_blocks(page):
        if marker == "skip" or (marker == "polars2" and not HAS_POLARS_2):
            continue
        where = f"{page.relative_to(DOCS).as_posix()}:{line}"
        compiled = compile(code, where, "exec")
        if marker == "raises":
            with pytest.raises(Exception) as raised:
                exec(compiled, namespace)
            claim = _claim(code)
            if claim is not None:
                name, message = claim
                raised_name = type(raised.value).__name__
                assert raised_name == name, f"{where} says {name}, raises {raised_name}"
                assert _shows(message, str(raised.value)), (
                    f"{where} shows: {_plain(message)}\n"
                    f"but raises: {_plain(str(raised.value))}"
                )
            continue
        if marker == "warns":
            # Warnings are errors in this suite, so a block that demonstrates
            # one says so, and must keep doing it.
            with pytest.warns(Warning):
                exec(compiled, namespace)
            continue
        printed = io.StringIO()
        try:
            with contextlib.redirect_stdout(printed):
                exec(compiled, namespace)
        except Exception as exc:  # noqa: BLE001 - reported as a test failure
            pytest.fail(
                f"{where} raised {type(exc).__name__}: {exc}\n\n"
                f"Fix the example, or mark the block:\n"
                f"    <!-- docs: skip -->     if it is an illustrative fragment\n"
                f"    <!-- docs: raises -->   if it demonstrates this error\n"
                f"    <!-- docs: warns -->    if it demonstrates this warning"
            )
        if shown is not None and printed.getvalue().strip():
            assert _shows(shown, printed.getvalue()), (
                f"{where} shows:\n{shown}\nbut prints:\n{printed.getvalue()}"
            )


def test_the_readmes_examples_run_on_their_own(tmp_path: Path, monkeypatch) -> None:
    """The first code anyone copies, run as they would run it: with nothing
    declared beforehand, no preamble, no markers."""
    readme = DOCS.parent / "README.md"
    blocks = _blocks(readme)
    assert blocks, "README.md has no python example"
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "readme_example"}
    for line, marker, code in blocks:
        assert marker is None, f"README.md:{line} needs no marker"
        exec(compile(code, f"README.md:{line}", "exec"), namespace)


def test_every_page_with_examples_is_covered() -> None:
    """The parametrisation is derived, so a new page cannot slip past it."""
    with_blocks = {p.relative_to(DOCS).as_posix() for p in _pages()}
    assert "tutorial/getting-started.md" in with_blocks
    assert len(with_blocks) >= 15
