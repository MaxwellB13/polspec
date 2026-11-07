"""`description=`: what a table or column means, for a reader. It is carried
everywhere a spec goes, and changes nothing a spec does."""

import polars as pl
import pytest
from polspec import (
    ColSpec,
    FrameSpec,
    SpecError,
    TableSpec,
    diff,
    from_dict,
    from_yaml,
    generate,
    to_dict,
    to_markdown,
    to_python,
    to_yaml,
    validate,
)

SPEC = TableSpec(
    "Orders",
    {
        "id": ColSpec(pl.Int64, unique=True, description="The order's number."),
        "note": ColSpec(pl.String, nullable=True),
        "point": ColSpec(
            pl.Struct({"lat": pl.Float64}),
            fields={"lat": ColSpec(pl.Float64, description="Degrees north.")},
        ),
    },
    description="One row per order.",
)


# ---------------------------------------------------------------------------
# Declaring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", 3])
def test_a_description_with_nothing_for_a_reader_is_refused(bad):
    with pytest.raises(SpecError, match=r"ColSpec\.description"):
        ColSpec(pl.Int64, description=bad)
    with pytest.raises(SpecError, match=r"TableSpec\.description"):
        TableSpec("T", {"a": ColSpec(pl.Int64)}, description=bad)


def test_a_framespec_declares_its_description_in_the_class_body():
    class Orders(FrameSpec):
        __description__ = "One row per order."
        id = ColSpec(pl.Int64)

    assert Orders.spec.description == "One row per order."


def test_a_docstring_is_not_a_description():
    class Orders(FrameSpec):
        """Kept in step with the billing export."""

        id = ColSpec(pl.Int64)

    assert Orders.spec.description is None


def test_a_subclass_describes_another_table_but_keeps_its_columns_words():
    class Orders(FrameSpec):
        __description__ = "One row per order."
        id = ColSpec(pl.Int64, description="The order's number.")

    class PaidOrders(Orders):
        pass

    assert PaidOrders.spec.description is None
    assert PaidOrders.spec["id"].description == "The order's number."


def test_with_description_replaces_or_clears_it():
    assert SPEC.with_description("Orders.").description == "Orders."
    assert SPEC.with_description(None).description is None


def test_reshaping_a_spec_keeps_what_it_says():
    renamed = SPEC.rename({"id": "order_id"})
    assert renamed.description == "One row per order."
    assert renamed["order_id"].description == "The order's number."
    assert SPEC.select("id")["id"].description == "The order's number."
    assert SPEC.drop("note").description == "One row per order."


# ---------------------------------------------------------------------------
# Writing it down
# ---------------------------------------------------------------------------


def test_spec_files_and_dicts_carry_it(tmp_path):
    path = tmp_path / "orders.yaml"
    to_yaml(SPEC, path)
    assert from_yaml(path) == SPEC
    assert from_dict(to_dict(SPEC)) == SPEC


def test_an_undescribed_spec_writes_no_key():
    data = to_dict(TableSpec("T", {"a": ColSpec(pl.Int64)}))
    assert "description" not in data
    assert "description" not in data["columns"]["a"]


def test_python_source_carries_it():
    namespace: dict = {}
    exec(to_python(SPEC), namespace)
    assert namespace["Orders"].spec == SPEC


def test_markdown_shows_the_table_and_each_column():
    document = to_markdown(SPEC)
    assert "One row per order." in document
    header = next(line for line in document.splitlines() if "| Column |" in line)
    assert header.endswith("| Description |")
    rows = {
        line.split("|")[1].strip(): line
        for line in document.splitlines()
        if line.startswith("|")
    }
    assert rows["`id`"].endswith("| The order's number. |")
    assert rows["`note`"].endswith("| - |")
    assert rows["`point.lat`"].endswith("| Degrees north. |")


def test_markdown_has_no_description_column_with_nothing_to_put_there():
    document = to_markdown(TableSpec("T", {"a": ColSpec(pl.Int64)}))
    assert "Description" not in document


def test_a_description_cannot_break_the_table():
    spec = TableSpec(
        "T", {"a": ColSpec(pl.Int64, description="either | or\nthen this")}
    )
    row = next(
        line for line in to_markdown(spec).splitlines() if line.startswith("| `a`")
    )
    assert row.endswith(r"| either \| or then this |")


# ---------------------------------------------------------------------------
# Changing nothing
# ---------------------------------------------------------------------------


def test_no_seed_moves():
    plain = TableSpec("Orders", {"id": ColSpec(pl.Int64, unique=True)})
    described = TableSpec(
        "Orders",
        {"id": ColSpec(pl.Int64, unique=True, description="The order's number.")},
        description="One row per order.",
    )
    assert generate(described, 50, seed=1).equals(generate(plain, 50, seed=1))


def test_validation_does_not_read_it():
    frame = generate(SPEC, 20, seed=1)
    assert validate(SPEC, frame).equals(frame)


def test_diff_reports_a_change_of_words_as_compatible():
    changed = SPEC.with_description("Orders, as placed.").with_columns(
        id=ColSpec(pl.Int64, unique=True, description="The number.")
    )
    report = diff(SPEC, changed)
    assert report.compatible
    assert {f.key for f in report.by_code("field_changed")} == {
        "description",
        "id__description",
    }
    assert all(f.severity == "compatible" for f in report.findings)
