"""Writing and reading a spec by hand: a weighted Enum's categories from its
choices, a spec from a schema, `ColSpec.replace`, column names as
attributes, and the short `repr` a spec prints as."""

import pickle

import polars as pl
import polspec
import pytest
from polspec import ColRule, ColSpec, FrameSpec, SpecError, TableSpec, col

WEIGHTED = {"A": 0.5, "B": 0.3, "C": 0.2}

# ---------------------------------------------------------------------------
# An Enum's categories from its choices
# ---------------------------------------------------------------------------


def test_a_bare_enum_takes_its_categories_from_its_choices():
    column = ColSpec(pl.Enum, choices=WEIGHTED)
    assert column.dtype == pl.Enum(["A", "B", "C"])
    assert (column.choices, column.weights) == (None, (0.5, 0.3, 0.2))


def test_the_three_spellings_of_a_weighted_enum_are_one_spec():
    spellings = [
        ColSpec(pl.Enum, choices=WEIGHTED),
        ColSpec(pl.Enum(["A", "B", "C"]), weights=[0.5, 0.3, 0.2]),
        ColSpec(pl.Enum(["A", "B", "C"]), choices=WEIGHTED),
    ]
    assert spellings[0] == spellings[1] == spellings[2]
    frames = [
        polspec.generate(TableSpec("T", {"c": s}), 2_000, seed=4) for s in spellings
    ]
    assert frames[0].equals(frames[1]) and frames[0].equals(frames[2])


def test_a_list_of_choices_makes_an_unweighted_enum():
    assert ColSpec(pl.Enum, choices=["x", "y"]) == ColSpec(pl.Enum(["x", "y"]))


def test_choices_narrower_than_the_categories_stay_choices():
    column = ColSpec(pl.Enum(["A", "B", "C"]), choices=["A", "B"])
    assert column.choices == ("A", "B")


@pytest.mark.parametrize(
    ("given", "message"),
    [
        ({}, "A bare pl.Enum has no categories"),
        ({"choices": [1, 2]}, "an Enum's categories are text"),
        ({"choices": ["a", "a"]}, "the same once cast to String"),
    ],
)
def test_a_bare_enum_that_cannot_take_categories_says_why(given, message):
    with pytest.raises(SpecError, match=message):
        ColSpec(pl.Enum, **given)


def test_a_weighted_enum_is_written_as_python_beside_its_shares():
    class Loans(FrameSpec):
        grade = ColSpec(pl.Enum(["A", "B", "C"]), weights=[0.5, 0.3, 0.2])
        product = ColSpec(pl.Enum(["mortgage", "card"]))

    source = Loans.to_python()
    assert "'grade': ColSpec(pl.Enum, choices={'A': 0.5, 'B': 0.3, 'C': 0.2})" in source
    assert "'product': ColSpec(pl.Enum(['mortgage', 'card']))" in source
    namespace: dict = {}
    exec(source, namespace)
    assert namespace["Loans"].spec == Loans.spec


def test_a_spec_file_keeps_the_long_form_and_reads_the_short_one():
    spec = TableSpec("T", {"grade": ColSpec(pl.Enum, choices=WEIGHTED)})
    text = polspec.to_yaml(spec)
    assert "Enum:\n      - A" in text
    assert "weights:" in text
    short = polspec.from_dict(
        {"name": "T", "columns": {"grade": {"dtype": "Enum", "choices": WEIGHTED}}}
    )
    assert short == spec


# ---------------------------------------------------------------------------
# A spec from a schema
# ---------------------------------------------------------------------------

SCHEMA = pl.Schema({"order_id": pl.Int64, "placed": pl.Date})
EXPECTED = TableSpec("Orders", {"order_id": pl.Int64, "placed": pl.Date})


def test_from_schema_takes_a_polars_schema_a_mapping_or_an_arrow_schema():
    pyarrow = pytest.importorskip("pyarrow")
    assert TableSpec.from_schema(SCHEMA, name="Orders") == EXPECTED
    assert TableSpec.from_schema(dict(SCHEMA), name="Orders") == EXPECTED
    arrow = pyarrow.schema(
        [("order_id", pyarrow.int64()), ("placed", pyarrow.date32())]
    )
    assert TableSpec.from_schema(arrow, name="Orders") == EXPECTED
    assert TableSpec.from_schema(EXPECTED, name="Orders") == EXPECTED


def test_from_schema_describes_an_empty_frame():
    """`profile()` refuses a frame with no rows; its schema is still a spec."""
    empty = pl.DataFrame(schema=SCHEMA)
    assert TableSpec.from_schema(empty.schema, name="Orders") == EXPECTED


def test_from_schema_refuses_what_is_not_a_schema():
    with pytest.raises(SpecError, match=r"takes a pl.Schema"):
        TableSpec.from_schema(["order_id"])


# ---------------------------------------------------------------------------
# replace and names
# ---------------------------------------------------------------------------


def test_replace_makes_a_new_column_checked_as_one():
    column = ColSpec(pl.Float64, bounds=(0.0, 1.0))
    assert column.replace(bounds=None) == ColSpec(pl.Float64)
    with pytest.raises(SpecError):
        column.replace(bounds=(2.0, 1.0))


class Invoices(FrameSpec):
    unit_price = ColSpec(pl.Float64, col_name="Unit Price")
    quantity = pl.Int64
    __columns__ = {"Line Note": pl.String}


def test_names_give_each_column_by_its_attribute():
    assert Invoices.names.unit_price == "Unit Price"
    assert Invoices.names.quantity == "quantity"
    assert list(Invoices.names) == ["Unit Price", "quantity", "Line Note"]
    assert getattr(Invoices.names, "Line Note") == "Line Note"


def test_a_name_that_is_no_column_says_the_closest():
    with pytest.raises(AttributeError, match="did you mean 'quantity'"):
        _ = Invoices.names.quantiy


def test_a_tablespec_names_each_column_as_itself_and_names_are_read_only():
    names = Invoices.spec.names
    assert list(names) == ["Unit Price", "quantity", "Line Note"]
    assert names.quantity == "quantity"
    with pytest.raises(AttributeError, match="read-only"):
        names.quantity = "q"
    assert list(pickle.loads(pickle.dumps(names))) == list(names)  # noqa: S301


# ---------------------------------------------------------------------------
# The short repr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("column", "shown"),
    [
        (ColSpec(pl.Int64), "ColSpec(Int64)"),
        (ColSpec(pl.Int64, bounds=(0, 10)), "ColSpec(Int64, bounds=(0, 10))"),
        (
            ColSpec(pl.Int64, nullable=True, null_probability=0.3),
            "ColSpec(Int64, nullable=True, null_probability=0.3)",
        ),
        (
            ColSpec(pl.Enum, choices={"A": 0.5, "B": 0.5}),
            "ColSpec(Enum(categories=['A', 'B']), weights=(0.5, 0.5))",
        ),
        (
            ColSpec(pl.Int8, rules=[ColRule(when=col("x") > 1, choices=[1])]),
            "ColSpec(Int8, rules=(ColRule(when=col('x') > 1, choices=(1,), weights=None),))",
        ),
    ],
)
def test_a_column_prints_only_what_it_declares(column, shown):
    assert repr(column) == shown


def test_a_spec_prints_one_column_to_a_line_and_its_table_claims():
    spec = TableSpec(
        "Loans",
        {"id": ColSpec(pl.Int64, unique=True), "lgd": ColSpec(pl.Float64)},
        order_by="id",
    )
    assert repr(spec) == (
        "TableSpec('Loans', {\n"
        "    'id': ColSpec(Int64, unique=True),\n"
        "    'lgd': ColSpec(Float64),\n"
        "}, order_by=('id',))"
    )


def test_reports_show_as_markdown_in_a_notebook():
    spec = TableSpec("Orders", {"a": ColSpec(pl.Int64, bounds=(0, 1))})
    frame = pl.DataFrame({"a": [5, 0]})
    shown = polspec.inspect(spec, frame)._repr_markdown_()
    assert shown.startswith("**Orders**: 1 finding(s)")
    assert "| `bounds` | `a` | 1 (50.0%) |" in shown
    assert polspec.inspect(spec, frame.tail(1))._repr_markdown_() == (
        "**Orders**: no findings."
    )
    assert polspec.drift(spec, frame)._repr_markdown_().startswith("# Drift")
