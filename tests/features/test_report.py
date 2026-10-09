"""`to_markdown` and `to_mermaid`: the generated data dictionary and ER diagram."""

import re

import polars as pl
from polspec import (
    Bound,
    CatSpec,
    Check,
    ColSpec,
    ForeignKey,
    FrameSpec,
    TableSpec,
)
from polspec.drift import diff
from polspec.render import framespec_to_markdown, framespec_to_mermaid


def test_framespec_to_markdown_and_to_mermaid(tmp_path):
    class CustomerSpec(FrameSpec):
        customer_id = ColSpec(
            pl.Int64, unique=True, bounds=Bound(1, 1_000_000), tags="index"
        )
        tier = ColSpec(
            pl.Enum(["BRONZE", "SILVER", "GOLD"]), nullable=False, tags="segment"
        )
        score = ColSpec(pl.Float64, bounds=Bound(0.0, 100.0), nullable=True)
        country = ColSpec(pl.String, choices=["US", "UK", "DE", "FR"], tags="geo")
        created_date = ColSpec(pl.Date, nullable=False, tags="temporal")

        __unique_together__ = [("customer_id", "country")]
        __checks__ = [
            Check(
                pl.col("score") >= 0.0,
                name="score_non_negative",
                description="Credit score must be non-negative if present",
            )
        ]

    # 1. to_markdown() without path
    md_str = CustomerSpec.to_markdown(title="Customer Data Dictionary")
    assert "# Customer Data Dictionary" in md_str
    assert "## Overview" in md_str
    assert "| `customer_id` |" in md_str
    assert "| `tier` |" in md_str
    assert "score_non_negative" in md_str
    assert "['customer_id', 'country']" in md_str
    assert "`index`" in md_str
    assert "`segment`" in md_str

    # 2. to_markdown() with file path
    md_file = tmp_path / "customer_dict.md"
    written_md = CustomerSpec.to_markdown(md_file)
    assert md_file.exists()
    assert md_file.read_text(encoding="utf-8") == written_md

    # 3. to_mermaid() without path
    mermaid_str = CustomerSpec.to_mermaid()
    assert "erDiagram" in mermaid_str
    assert "CustomerSpec {" in mermaid_str
    assert "Int64 customer_id PK" in mermaid_str
    assert "Enum tier" in mermaid_str
    assert "tags: [segment]" in mermaid_str
    assert "bounds: [1, 1000000]" in mermaid_str

    # 4. to_mermaid() with file path
    mermaid_file = tmp_path / "customer_erd.mmd"
    written_mermaid = CustomerSpec.to_mermaid(mermaid_file)
    assert mermaid_file.exists()
    assert mermaid_file.read_text(encoding="utf-8") == written_mermaid

    # 5. to_mermaid() with quotes in choices / tags / fk names
    class QuotedSpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        flag = ColSpec(pl.String, choices=['a"1', 'b"2'], tags=['geo"zone'])
        __foreign_keys__ = [ForeignKey("id", references="self", name='quoted"fk')]

    quoted_mmd = QuotedSpec.to_mermaid()
    # Ensure double quotes inside attributes are replaced to avoid breaking mermaid ER syntax
    assert "choices: [a'1, b'2]" in quoted_mmd
    assert "tags: [geo'zone]" in quoted_mmd
    assert "quoted'fk" in quoted_mmd


class CustomerFkSpec(FrameSpec):
    id = ColSpec(pl.Int64, unique=True)
    code = ColSpec(pl.String, unique=True)


def test_framespec_to_markdown_and_to_mermaid_with_foreign_keys():
    class OrderFkSpec(FrameSpec):
        order_id = ColSpec(pl.Int64, unique=True)
        customer_id = ColSpec(pl.Int64, nullable=True)
        __foreign_keys__ = [
            ForeignKey("customer_id", references=CustomerFkSpec, ref_columns="id")
        ]

    md = OrderFkSpec.to_markdown()
    assert "**Foreign Keys:** 1 key(s)" in md
    assert "### Foreign Keys" in md
    assert "fk_customer_id__CustomerFkSpec" in md
    assert "['customer_id']" in md
    assert "CustomerFkSpec.['id']" in md

    mmd = OrderFkSpec.to_mermaid()
    assert "Int64 customer_id FK" in mmd
    assert 'CustomerFkSpec ||--o{ OrderFkSpec : "fk_customer_id__CustomerFkSpec"' in mmd

    class SelfFkSpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        parent_id = ColSpec(pl.Int64, nullable=True)
        __foreign_keys__ = [
            ForeignKey("parent_id", references="self", ref_columns="id")
        ]

    self_mmd = SelfFkSpec.to_mermaid()
    assert 'SelfFkSpec ||--o{ SelfFkSpec : "fk_parent_id__self"' in self_mmd


def test_framespec_to_markdown_lists_column_validators():
    class ValidatedSpec(FrameSpec):
        price = ColSpec(
            pl.Float64,
            validators=[
                Check(
                    pl.col("price") > 0,
                    name="price_positive",
                    description="Price must be positive",
                )
            ],
        )

    md = ValidatedSpec.to_markdown()
    assert "### Column Validators" in md
    assert "Column `price`" in md
    assert "price_positive" in md
    assert "Price must be positive" in md


def test_mermaid_marks_one_primary_key_and_otherwise_unique_keys():
    """A lone `unique=True` column is the entity's PK; several unique columns
    are each a UK, since an entity has one primary key."""

    class One(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        name = ColSpec(pl.String)

    class Two(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        code = ColSpec(pl.String, unique=True)
        pair_a = ColSpec(pl.Int64)
        pair_b = ColSpec(pl.Int64)
        __unique_together__ = [("pair_a", "pair_b")]

    one = One.to_mermaid()
    assert "Int64 id PK" in one and "UK" not in one
    two = Two.to_mermaid()
    assert "PK" not in two
    assert "Int64 id UK" in two and "String code UK" in two
    assert "Int64 pair_a UK" in two


# ---------------------------------------------------------------------------
# Names and values Markdown or Mermaid would otherwise misread
# ---------------------------------------------------------------------------

# A `|` splits a GFM table cell -- inside backticks too -- unless an odd run
# of backslashes escapes it.
_CELL_BREAK = re.compile(r"(?<!\\)(?:\\\\)*\|")

# An attribute name every Mermaid version reads: what render promises.
_ATTRIBUTE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")


def _cells_per_row(markdown: str) -> set[int]:
    return {
        len(_CELL_BREAK.findall(line))
        for line in markdown.splitlines()
        if line.startswith("|")
    }


def _attributes(mermaid: str) -> list[str]:
    """The attribute names in an ER diagram's entity blocks."""
    names, inside = [], False
    for line in mermaid.splitlines():
        stripped = line.strip()
        if stripped.endswith("{"):
            inside = True
        elif stripped == "}":
            inside = False
        elif inside:
            names.append(stripped.split()[1])
    return names


AWKWARD = TableSpec(
    "Odd Name",
    {
        "a|b": ColSpec(pl.Enum(["x|y", "new\nline"])),
        "with space": ColSpec(pl.String, choices=["p|q"], tags="t|u"),
        "": ColSpec(pl.Int64, bounds=(0, 5)),
        "1st": ColSpec(pl.Boolean),
        "größe": ColSpec(pl.Float64),
        "pk": ColSpec(pl.Int8, unique=True),
        "a_b": ColSpec(pl.Int8),
        "new\nline": ColSpec(pl.String, pattern=r"[a|b]{2}"),
        r"slash\|pipe": ColSpec(pl.Int8),
    },
)


def test_markdown_escapes_what_would_split_a_cell():
    md = framespec_to_markdown(AWKWARD)
    assert _cells_per_row(md) == {10}
    assert r"| `a\|b` | `Enum(['x\|y', 'new\nline'])` |" in md
    assert "| `new line` |" in md  # a newline would have ended the row
    assert r"`slash\\\|pipe`" in md  # the backslash cannot escape the escape


def test_drift_and_catspec_markdown_escape_their_cells():
    old = TableSpec("T", {"a|b": ColSpec(pl.Int64, bounds=(0, 10))})
    new = TableSpec("T", {"a|b": ColSpec(pl.Int64)})
    md = diff(old, new).to_markdown()
    assert r"`a\|b`" in md
    assert _cells_per_row(md) == {4}
    cats = CatSpec(
        enums={"A|B": ["x|y", "z"]},
        categoricals={"C|D": pl.Categories("C|D")},
        choices={"C|D": ["p|q"]},
    )
    md = cats.to_markdown()
    assert r"`A\|B`" in md and r"'x\|y'" in md and r"'p\|q'" in md
    rows = [line for line in md.splitlines() if line.startswith("|")]
    assert {len(_CELL_BREAK.findall(line)) for line in rows[:3]} == {4}
    assert {len(_CELL_BREAK.findall(line)) for line in rows[3:]} == {6}


def test_mermaid_attribute_names_are_words_mermaid_reads():
    mermaid = framespec_to_mermaid(AWKWARD)
    names = _attributes(mermaid)
    assert len(names) == len(AWKWARD.columns) == len(set(names))
    for name in names:
        assert _ATTRIBUTE.fullmatch(name), name
        assert name.upper() not in {"PK", "FK", "UK"}, name
    # A name that reads as it is stays; one renamed keeps its real name.
    assert "        Int8 a_b\n" in mermaid
    assert "Enum a_b_2 \"name: 'a|b'\"" in mermaid
    assert "Int8 pk_ PK \"name: 'pk'\"" in mermaid
    assert f'"name: {"new\nline"!r}, pattern: [a|b]{{2}}"' in mermaid
    assert mermaid.startswith("erDiagram\n    Odd_Name {")


def test_mermaid_notes_hold_no_tilde():
    """A `~` in a note, before a long enough tail, hung Mermaid 10's parser."""
    spec = TableSpec("T", {"x": ColSpec(pl.String, choices=["~a", "b~"])})
    mermaid = framespec_to_mermaid(spec)
    assert "~" not in mermaid
    assert "choices: [\N{TILDE OPERATOR}a, b\N{TILDE OPERATOR}]" in mermaid


def test_mermaid_entity_names_start_as_mermaid_reads_them():
    one = TableSpec("2023 Sales", {"x": ColSpec(pl.Int8)})
    two = TableSpec("Café", {"x": ColSpec(pl.Int8)})
    assert framespec_to_mermaid(one).startswith("erDiagram\n    _2023_Sales {")
    assert framespec_to_mermaid(two).startswith("erDiagram\n    Caf_ {")
    # A word Mermaid reserves outside a block -- `class` styles an entity.
    three = TableSpec("class", {"x": ColSpec(pl.Int8)})
    assert framespec_to_mermaid(three).startswith("erDiagram\n    class_ {")


def test_markdown_fences_a_name_holding_backticks():
    spec = TableSpec("T", {"back`tick": ColSpec(pl.Int8), "": ColSpec(pl.Int8)})
    md = framespec_to_markdown(spec)
    assert "| ``back`tick`` |" in md
    assert "| ` ` |" in md  # an empty name is still a code span, not "``"


# ---------------------------------------------------------------------------
# The top-level forms (0.18.0): each takes a TableSpec or a FrameSpec class
# ---------------------------------------------------------------------------


def test_the_top_level_renderers_match_the_framespec_methods():
    import polspec

    class Orders(FrameSpec):
        order_id = ColSpec(pl.Int64, unique=True)
        status = ColSpec(pl.Enum(["NEW", "PAID"]))

    for spec in (Orders, Orders.spec):
        assert polspec.to_markdown(spec) == Orders.to_markdown()
        assert polspec.to_mermaid(spec) == Orders.to_mermaid()
    assert polspec.to_markdown(Orders, title="Orders table").startswith(
        "# Orders table"
    )
