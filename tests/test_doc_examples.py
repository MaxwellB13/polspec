"""Every Python example in `docs/` is executed.

The rest of `test_docs.py` checks that the documentation *points* at things
that exist -- every exported name reachable, every link resolving, every page
in the nav. This checks that the code on those pages runs, which nothing did
before: the docs carried over a hundred Python blocks and not one of them was
ever executed, so an example could go stale for a whole release without
anything noticing. One already had.

A page's blocks run in order in one namespace, on top of a shared preamble
holding the `Customers`/`Orders`/`OrderLines` specs most pages assume. A block
that cannot run that way says so in an HTML comment on the line above it,
which renders as nothing:

    <!-- docs: skip -->      an illustrative fragment, not a runnable example
    <!-- docs: raises -->    demonstrates an error, and must actually raise

A marker is a claim about the block, so both are checked: `raises` fails the
suite if the block stops raising, the same way `xfail(strict=True)` pins the
known limitations.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"

# A ```python fence, with whatever HTML comment sits on the line above it.
BLOCK = re.compile(
    r"(?:^[ \t]*<!--[ \t]*docs:[ \t]*(?P<marker>[a-z]+)[ \t]*-->[ \t]*\n)?"
    r"^```python\n(?P<code>.*?)^```",
    re.S | re.M,
)

MARKERS = {"skip", "raises"}

# The names pages take for granted, declared once. A page that needs more of
# its own declares it in a block of its own, which is also what a reader
# needs to see.
PREAMBLE = '''
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
    MultiValidationError,
    Registry,
    TableSpec,
    ValidationError,
    ValidationReport,
    col,
    profile_dataframe,
)


class Customers(FrameSpec):
    """The spec most pages reach for without declaring it."""

    id = ColSpec(pl.Int64, bounds=(1, 10_000), unique=True)
    name = ColSpec(pl.String, string_length=(3, 20), tags="pii")
    country = ColSpec(pl.Enum(["UK", "US", "DE"]))
    signed_up = ColSpec(pl.Date, bounds=(date(2020, 1, 1), date(2026, 1, 1)))


class Orders(FrameSpec):
    # No cross-spec key: pages use `Orders.validate(df)` on its own, and a key
    # with no parent frame is a `foreign_key_unresolved` finding by design.
    # `OrderLines` below carries the key the relationship pages need.
    order_id = ColSpec(pl.Int64, bounds=(1, 1_000_000), unique=True)
    customer_id = ColSpec(pl.Int64, bounds=(1, 10_000))
    status = ColSpec(pl.Enum(["NEW", "PAID", "SHIPPED"]))
    total = ColSpec(pl.Float64, bounds=(0.0, 5_000.0))
    internal_note = ColSpec(pl.String, string_length=(0, 40), nullable=True)


class OrderLines(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, 1_000_000))
    line_no = ColSpec(pl.Int32, bounds=(1, 50))
    quantity = ColSpec(pl.UInt16, bounds=(1, 500))
    __unique_together__ = [["order_id", "line_no"]]
    __foreign_keys__ = [ForeignKey("order_id", references=Orders)]


customers = Customers.generate(200, seed=1)
orders = Orders.generate(500, seed=2)
order_lines = OrderLines.generate(800, seed=3, references={Orders: orders})
df = orders
categories = CatSpec(
    enums={"STATUS": ["NEW", "PAID", "SHIPPED"]},
    categoricals={"CURRENCY": pl.Categories("CURRENCY", physical=pl.UInt8)},
)
'''


def _pages() -> list[Path]:
    return sorted(p for p in DOCS.rglob("*.md") if BLOCK.search(p.read_text("utf-8")))


def _blocks(page: Path) -> list[tuple[int, str | None, str]]:
    """Every python block on a page, as (line number, marker, source)."""
    text = page.read_text(encoding="utf-8")
    out: list[tuple[int, str | None, str]] = []
    for match in BLOCK.finditer(text):
        marker = match.group("marker")
        if marker is not None and marker not in MARKERS:
            raise AssertionError(
                f"{page.name}: unknown marker <!-- docs: {marker} -->; "
                f"expected one of {sorted(MARKERS)}"
            )
        out.append((text[: match.start()].count("\n") + 1, marker, match.group("code")))
    return out


@pytest.mark.parametrize("page", _pages(), ids=lambda p: p.relative_to(DOCS).as_posix())
def test_a_pages_examples_run(page: Path, tmp_path: Path, monkeypatch) -> None:
    # Pages write files (`Orders.to_yaml("orders.yaml")`), so give each its own
    # directory rather than the repository root.
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "docs_example"}
    exec(compile(PREAMBLE, "<preamble>", "exec"), namespace)

    for line, marker, code in _blocks(page):
        if marker == "skip":
            continue
        where = f"{page.relative_to(DOCS).as_posix()}:{line}"
        compiled = compile(code, where, "exec")
        if marker == "raises":
            with pytest.raises(Exception):  # noqa: B017 - the page names which
                exec(compiled, namespace)
            continue
        try:
            exec(compiled, namespace)
        except Exception as exc:  # noqa: BLE001 - reported as a test failure
            pytest.fail(
                f"{where} raised {type(exc).__name__}: {exc}\n\n"
                f"Fix the example, or mark the block:\n"
                f"    <!-- docs: skip -->     if it is an illustrative fragment\n"
                f"    <!-- docs: raises -->   if it demonstrates this error"
            )


def test_every_page_with_examples_is_covered() -> None:
    """The parametrisation is derived, so a new page cannot slip past it."""
    with_blocks = {p.relative_to(DOCS).as_posix() for p in _pages()}
    assert "tutorial/getting-started.md" in with_blocks
    assert len(with_blocks) >= 15
