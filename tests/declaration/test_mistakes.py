"""Declarations that cannot mean what they say, refused where they are
written: a dtype that is not a Polars one, a range that is not a pair, and
class-body slips -- a trailing comma, a misspelt dunder -- that would
otherwise build a spec saying less than its class body."""

import datetime as dt
import decimal
import re

import polars as pl
import polspec
import pytest
from polspec import Check, ColRule, ColSpec, ForeignKey, FrameSpec, SpecError, col

# ---------------------------------------------------------------------------
# A dtype is a Polars dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "spelling"),
    [
        (int, "pl.Int64"),
        (float, "pl.Float64"),
        (str, "pl.String"),
        (bool, "pl.Boolean"),
        (bytes, "pl.Binary"),
        (dt.date, "pl.Date"),
        (dt.datetime, "pl.Datetime"),
        (dt.time, "pl.Time"),
        (dt.timedelta, "pl.Duration"),
        (decimal.Decimal, "pl.Decimal"),
        ("Int64", "pl.Int64"),
        ("String", "pl.String"),
    ],
)
def test_a_python_type_or_a_name_is_refused_with_its_polars_spelling(given, spelling):
    with pytest.raises(SpecError, match=f"must be a Polars dtype.*use {spelling}"):
        ColSpec(given)


@pytest.mark.parametrize("given", ["Integer", 3, object()])
def test_anything_else_is_refused_too(given):
    with pytest.raises(SpecError, match="must be a Polars dtype"):
        ColSpec(given)


def test_no_dtype_says_one_is_needed():
    with pytest.raises(SpecError, match="needs a dtype"):
        ColSpec(None)


@pytest.mark.parametrize("dtype", [pl.Object, pl.Unknown])
def test_dtypes_that_hold_no_generatable_data_still_declare(dtype):
    """They validate today, so refusing them would break a working spec."""
    assert ColSpec(dtype).dtype == dtype


# ---------------------------------------------------------------------------
# A range is a pair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["bounds", "string_length", "list_length"])
@pytest.mark.parametrize("given", [(5,), (1, 2, 3), 5])
def test_a_range_that_is_not_a_pair_says_so(field, given):
    dtype = {"bounds": pl.Int64, "string_length": pl.String}.get(
        field, pl.List(pl.Int8)
    )
    with pytest.raises(
        SpecError, match=rf"ColSpec\.{field} must be a \(min, max\) pair"
    ):
        ColSpec(dtype, **{field: given})


def test_a_decimal_range_that_is_not_a_pair_says_so():
    with pytest.raises(SpecError, match=r"must be a \(min, max\) pair"):
        ColSpec(pl.Decimal(10, 2), bounds=(1,))


# ---------------------------------------------------------------------------
# Class-body slips
# ---------------------------------------------------------------------------


def _declare(**body):
    return type("Orders", (FrameSpec,), {"id": ColSpec(pl.Int64), **body})


def test_a_trailing_comma_after_a_colspec_is_refused():
    with pytest.raises(
        SpecError, match=r"Orders\.total is a tuple holding a ColSpec.*trailing comma"
    ):
        _declare(total=(ColSpec(pl.Float64),))


@pytest.mark.parametrize(
    ("given", "meant"),
    [
        ("__unique_togther__", "__unique_together__"),
        ("__check__", "__checks__"),
        ("__foreign_key__", "__foreign_keys__"),
        ("__row__", "__rows__"),
        ("__order_bye__", "__order_by__"),
        ("__colums__", "__columns__"),
        ("__descripton__", "__description__"),
    ],
)
def test_a_misspelt_dunder_is_refused_with_the_one_meant(given, meant):
    with pytest.raises(SpecError, match=f"{given}.*did you mean {meant}"):
        _declare(**{given: ["id"]})


@pytest.mark.parametrize(
    ("value", "home"),
    [
        ([Check(col("id") > 0)], "__checks__"),
        (Check(col("id") > 0), "__checks__"),
        ([ForeignKey("id", references="Customers")], "__foreign_keys__"),
        (ColRule(when=col("id") > 1, choices=[1]), "rules="),
    ],
)
def test_a_declaration_under_a_name_polspec_does_not_read_is_refused(value, home):
    with pytest.raises(SpecError, match=f"Orders.misc holds .* only from .*{home}"):
        _declare(misc=value)


def test_ordinary_class_attributes_are_left_alone():
    """Constants, methods, a docstring and a dunder of the class's own pass:
    only what looks like a declaration polspec would not read is refused."""

    class Orders(FrameSpec):
        """Orders placed."""

        id = ColSpec(pl.Int64)
        TABLE = "orders"
        KEYS = ["id"]
        __version__ = "3"

        @staticmethod
        def helper() -> int:
            return 1

    assert Orders.schema() == pl.Schema({"id": pl.Int64})
    assert (Orders.TABLE, Orders.KEYS, Orders.helper()) == ("orders", ["id"], 1)


def test_the_declaration_spellings_without_dunders_still_declare():
    Orders = _declare(checks=[Check(col("id") > 0, name="positive")])
    assert [c.name for c in Orders.spec.checks] == ["positive"]


def test_assigning_none_still_removes_an_inherited_column():
    class Base(FrameSpec):
        id = ColSpec(pl.Int64)
        total = ColSpec(pl.Float64)

    class WithoutTotal(Base):
        total = None

    assert WithoutTotal.schema() == pl.Schema({"id": pl.Int64})


# ---------------------------------------------------------------------------
# Spec files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("given", "meant"), [("Int46", "Int64"), ("string", "String")])
def test_an_unknown_dtype_name_in_a_spec_file_names_the_closest(given, meant):
    with pytest.raises(
        polspec.SerializationError,
        match=f"spec file: '{given}'. Did you mean '{meant}'",
    ):
        polspec.from_dict({"name": "T", "columns": {"a": {"dtype": given}}})


def test_an_unknown_dtype_name_with_nothing_close_lists_the_names():
    with pytest.raises(polspec.SerializationError, match=r"Known names: .*Int64"):
        polspec.from_dict({"name": "T", "columns": {"a": {"dtype": "Integer"}}})


def test_values_where_the_dtype_goes_say_how_to_declare_an_enum_of_them():
    """A mapping of value to share, or a list of values, in the dtype's place
    was an unhashable-type error; it names the Enum it was meant to be."""
    with pytest.raises(
        SpecError,
        match=re.escape("use ColSpec(pl.Enum, choices={'A': 0.7, 'B': 0.3})"),
    ):
        ColSpec({"A": 0.7, "B": 0.3})
    with pytest.raises(SpecError, match=re.escape("use ColSpec(pl.Enum(['A', 'B']))")):
        ColSpec(["A", "B"])
