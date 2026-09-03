"""`ForeignKey`: declaration-time checks, YAML persistence, and generation that
samples keys from a parent frame.

Validation of foreign keys against data lives in `test_validation.py`.
"""

import polars as pl
import pytest
from polspec import (
    ColSpec,
    ForeignKey,
    FrameSpec,
    ValidationError,
)

# ---------------------------------------------------------------------------
# declaration
# ---------------------------------------------------------------------------


class CustomerFkSpec(FrameSpec):
    id = ColSpec(pl.Int64, unique=True)
    code = ColSpec(pl.String, unique=True)


def test_foreign_key_declaration_variations():
    # Single column, ref_columns defaults to the same name
    class SameNameFk(FrameSpec):
        id = ColSpec(pl.Int64)
        __foreign_keys__ = [ForeignKey("id", references=CustomerFkSpec)]

    fk = SameNameFk.foreign_keys()[0]
    assert fk.columns == ("id",)
    assert fk.ref_columns == ("id",)
    assert fk.references == "CustomerFkSpec"
    assert fk.target is CustomerFkSpec.spec
    assert fk.name == "fk_id__CustomerFkSpec"

    # Explicit ref_columns and name
    class RenamedFk(FrameSpec):
        customer_id = ColSpec(pl.Int64)
        __foreign_keys__ = [
            ForeignKey(
                "customer_id",
                references=CustomerFkSpec,
                ref_columns="id",
                name="customer_fk",
            )
        ]

    fk2 = RenamedFk.foreign_keys()[0]
    assert fk2.columns == ("customer_id",)
    assert fk2.ref_columns == ("id",)
    assert fk2.name == "customer_fk"

    # Composite key
    class CompositeFk(FrameSpec):
        a = ColSpec(pl.Int64)
        b = ColSpec(pl.String)
        __foreign_keys__ = [
            ForeignKey(
                ["a", "b"], references=CustomerFkSpec, ref_columns=["id", "code"]
            )
        ]

    fk3 = CompositeFk.foreign_keys()[0]
    assert fk3.columns == ("a", "b")
    assert fk3.ref_columns == ("id", "code")

    # Self reference
    class SelfFk(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        parent_id = ColSpec(pl.Int64, nullable=True)
        __foreign_keys__ = [
            ForeignKey("parent_id", references="self", ref_columns="id")
        ]

    fk4 = SelfFk.foreign_keys()[0]
    assert fk4.references == "self"
    assert fk4.name == "fk_parent_id__self"


def test_foreign_key_mismatched_ref_columns_length_raises():
    with pytest.raises(ValueError, match="must have the same length"):
        ForeignKey(["a", "b"], references=CustomerFkSpec, ref_columns=["id"])


def test_foreign_key_rejects_invalid_references_type():
    with pytest.raises(TypeError, match="must be a FrameSpec subclass"):
        ForeignKey("a", references=123)  # type: ignore[arg-type]


def test_foreign_key_declaration_rejects_unknown_local_column():
    with pytest.raises(ValueError, match="unknown local column 'missing'"):

        class BadLocal(FrameSpec):
            a = ColSpec(pl.Int64)
            __foreign_keys__ = [ForeignKey("missing", references=CustomerFkSpec)]


def test_foreign_key_declaration_rejects_unknown_ref_column():
    with pytest.raises(
        ValueError, match="unknown column 'missing' on 'CustomerFkSpec'"
    ):

        class BadRef(FrameSpec):
            a = ColSpec(pl.Int64)
            __foreign_keys__ = [
                ForeignKey("a", references=CustomerFkSpec, ref_columns="missing")
            ]


def test_foreign_key_declaration_rejects_dtype_mismatch():
    with pytest.raises(ValueError, match="not dtype-compatible"):

        class BadDtype(FrameSpec):
            a = ColSpec(pl.String)
            __foreign_keys__ = [
                ForeignKey("a", references=CustomerFkSpec, ref_columns="id")
            ]


def test_foreign_key_declaration_allows_string_enum_categorical_bucket():
    # String/Enum/Categorical are treated as one compatible domain bucket.
    class EnumChild(FrameSpec):
        code = ColSpec(pl.Enum(["A", "B"]))
        __foreign_keys__ = [
            ForeignKey("code", references=CustomerFkSpec, ref_columns="code")
        ]

    assert EnumChild.foreign_keys()[0].columns == ("code",)


def test_foreign_key_declaration_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate ForeignKey name 'dup'"):

        class DupFk(FrameSpec):
            a = ColSpec(pl.Int64)
            b = ColSpec(pl.String)
            __foreign_keys__ = [
                ForeignKey(
                    "a", references=CustomerFkSpec, ref_columns="id", name="dup"
                ),
                ForeignKey(
                    "b", references=CustomerFkSpec, ref_columns="code", name="dup"
                ),
            ]


def test_foreign_key_inheritance_deduplicates_and_accumulates():
    class BaseFkSpec(FrameSpec):
        a = ColSpec(pl.Int64)
        __foreign_keys__ = [
            ForeignKey("a", references=CustomerFkSpec, ref_columns="id")
        ]

    class ExtendedFkSpec(BaseFkSpec):
        b = ColSpec(pl.String)
        __foreign_keys__ = [
            ForeignKey("b", references=CustomerFkSpec, ref_columns="code")
        ]

    assert len(ExtendedFkSpec.foreign_keys()) == 2
    fk_names = [fk.name for fk in ExtendedFkSpec.foreign_keys()]
    assert fk_names == ["fk_a__CustomerFkSpec", "fk_b__CustomerFkSpec"]


# ---------------------------------------------------------------------------
# YAML persistence
# ---------------------------------------------------------------------------


def test_foreign_key_yaml_roundtrip_self_reference(tmp_path):
    class HierarchySpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        parent_id = ColSpec(pl.Int64, nullable=True)
        __foreign_keys__ = [
            ForeignKey("parent_id", references="self", ref_columns="id")
        ]

    yaml_path = tmp_path / "hierarchy_spec.yaml"
    HierarchySpec.to_yaml(yaml_path)
    assert "foreign_keys" in yaml_path.read_text()

    Loaded = FrameSpec.from_yaml(yaml_path)
    assert Loaded.foreign_keys() == (
        ForeignKey("parent_id", references="self", ref_columns="id"),
    )

    df_valid = pl.DataFrame({"id": [1, 2, 3], "parent_id": [None, 1, 1]})
    assert Loaded.validate(df_valid).height == 3

    df_invalid = pl.DataFrame({"id": [1, 2], "parent_id": [1, 999]})
    with pytest.raises(ValidationError, match="ForeignKey"):
        Loaded.validate(df_invalid)


def test_to_yaml_warns_when_foreign_keys_to_other_specs_are_not_persisted(tmp_path):
    class ChildSpec(FrameSpec):
        customer_id = ColSpec(pl.Int64)
        __foreign_keys__ = [
            ForeignKey("customer_id", references=CustomerFkSpec, ref_columns="id")
        ]

    yaml_path = tmp_path / "child_spec.yaml"
    with pytest.warns(UserWarning, match="fk_customer_id__CustomerFkSpec"):
        ChildSpec.to_yaml(yaml_path)

    Loaded = FrameSpec.from_yaml(yaml_path)
    assert Loaded.foreign_keys() == ()


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


class GenCustomerSpec(FrameSpec):
    id = ColSpec(pl.Int64, unique=True)
    name = ColSpec(pl.String)


class GenOrderSpec(FrameSpec):
    order_id = ColSpec(pl.Int64, unique=True)
    customer_id = ColSpec(pl.Int64, nullable=True, null_probability=0.3)
    __foreign_keys__ = [
        ForeignKey("customer_id", references=GenCustomerSpec, ref_columns="id")
    ]


def test_generate_foreign_key_samples_from_parent_when_references_supplied():
    customers = GenCustomerSpec.generate(5, seed=1)
    customer_ids = set(customers["id"].to_list())

    orders = GenOrderSpec.generate(200, seed=2, references={GenCustomerSpec: customers})
    non_null = [v for v in orders["customer_id"].to_list() if v is not None]

    assert non_null  # sanity: at least some non-null rows given the sample size
    assert all(v in customer_ids for v in non_null)
    # No validation error against the very parent it was sampled from.
    GenOrderSpec.validate(orders, references={GenCustomerSpec: customers})


def test_generate_foreign_key_accepts_lazyframe_reference():
    customers = GenCustomerSpec.generate(5, seed=1)
    customer_ids = set(customers["id"].to_list())

    orders = GenOrderSpec.generate(
        50, seed=2, references={GenCustomerSpec: customers.lazy()}
    )
    non_null = [v for v in orders["customer_id"].to_list() if v is not None]
    assert non_null
    assert all(v in customer_ids for v in non_null)


def test_generate_foreign_key_without_references_uses_free_generation():
    # No `references` supplied: runs fine and behaves exactly as if the FK
    # weren't declared (deterministic given the same seed).
    orders_a = GenOrderSpec.generate(50, seed=2)
    orders_b = GenOrderSpec.generate(50, seed=2)
    assert orders_a.height == 50
    assert orders_a.equals(orders_b)


def test_generate_foreign_key_respects_null_probability():
    customers = GenCustomerSpec.generate(5, seed=1)
    orders = GenOrderSpec.generate(
        2_000, seed=7, references={GenCustomerSpec: customers}
    )
    null_frac = orders["customer_id"].null_count() / orders.height
    # null_probability=0.3 on a 2000-row sample; loose tolerance for randomness.
    assert 0.2 < null_frac < 0.4


def test_generate_foreign_key_self_reference():
    class EmployeeGenSpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        manager_id = ColSpec(pl.Int64, nullable=True, null_probability=0.3)
        __foreign_keys__ = [
            ForeignKey("manager_id", references="self", ref_columns="id")
        ]

    emps = EmployeeGenSpec.generate(100, seed=3)
    ids = set(emps["id"].to_list())
    non_null_mgrs = [v for v in emps["manager_id"].to_list() if v is not None]
    assert non_null_mgrs
    assert all(v in ids for v in non_null_mgrs)


def test_generate_foreign_key_composite_samples_jointly():
    class RegionGenSpec(FrameSpec):
        tenant = ColSpec(pl.Int64)
        region_id = ColSpec(pl.Int64)
        __unique_together__ = [("tenant", "region_id")]

    class StoreGenSpec(FrameSpec):
        tenant = ColSpec(pl.Int64)
        region_id = ColSpec(pl.Int64)
        __foreign_keys__ = [
            ForeignKey(
                ["tenant", "region_id"],
                references=RegionGenSpec,
                ref_columns=["tenant", "region_id"],
            )
        ]

    regions = pl.DataFrame({"tenant": [1, 1, 2], "region_id": [10, 20, 10]})
    valid_pairs = {(1, 10), (1, 20), (2, 10)}

    stores = StoreGenSpec.generate(100, seed=4, references={RegionGenSpec: regions})
    pairs = set(
        zip(stores["tenant"].to_list(), stores["region_id"].to_list(), strict=True)
    )
    assert pairs <= valid_pairs
    StoreGenSpec.validate(stores, references={RegionGenSpec: regions})


def test_generate_foreign_key_unique_column_samples_without_replacement():
    class ProfileGenSpec(FrameSpec):
        user_id = ColSpec(pl.Int64, unique=True)
        __foreign_keys__ = [
            ForeignKey("user_id", references=GenCustomerSpec, ref_columns="id")
        ]

    customers = GenCustomerSpec.generate(10, seed=1)
    customer_ids = set(customers["id"].to_list())

    # Parent has exactly as many rows as requested: a clean one-to-one mapping
    # should come out as a permutation of the parent ids, with no duplicates.
    profiles = ProfileGenSpec.generate(
        10, seed=5, references={GenCustomerSpec: customers}
    )
    uids = profiles["user_id"].to_list()
    assert len(set(uids)) == len(uids)
    assert set(uids) == customer_ids


def test_generate_foreign_key_empty_parent_raises():
    empty_customers = pl.DataFrame({"id": pl.Series([], dtype=pl.Int64)})
    with pytest.raises(ValueError, match="cannot generate values"):
        GenOrderSpec.generate(10, references={GenCustomerSpec: empty_customers})


def test_generate_foreign_key_parent_with_nulls_filters_them():
    parent_df = pl.DataFrame({"id": [1, None, 2, None, 3]})

    class ChildGenSpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        customer_id = ColSpec(pl.Int64, nullable=False)
        __foreign_keys__ = [
            ForeignKey("customer_id", references=GenCustomerSpec, ref_columns="id")
        ]

    df = ChildGenSpec.generate(20, seed=42, references={GenCustomerSpec: parent_df})
    assert df["customer_id"].null_count() == 0
    assert set(df["customer_id"].to_list()).issubset({1, 2, 3})


def test_generate_foreign_key_parent_all_nulls_raises():
    all_null_parent = pl.DataFrame({"id": [None, None]})

    class ChildGenSpec(FrameSpec):
        id = ColSpec(pl.Int64, unique=True)
        customer_id = ColSpec(pl.Int64, nullable=False)
        __foreign_keys__ = [
            ForeignKey("customer_id", references=GenCustomerSpec, ref_columns="id")
        ]

    with pytest.raises(ValueError, match="cannot generate values"):
        ChildGenSpec.generate(10, references={GenCustomerSpec: all_null_parent})


def test_generate_batches_and_sink_thread_foreign_key_references(tmp_path):
    customers = GenCustomerSpec.generate(5, seed=1)
    customer_ids = set(customers["id"].to_list())

    batches = list(
        GenOrderSpec.generate_batches(
            50, batch_size=10, seed=6, references={GenCustomerSpec: customers}
        )
    )
    all_vals = [v for b in batches for v in b["customer_id"].to_list() if v is not None]
    assert all_vals
    assert all(v in customer_ids for v in all_vals)

    out_path = tmp_path / "orders.csv"
    GenOrderSpec.sink_csv(out_path, 30, seed=6, references={GenCustomerSpec: customers})
    sunk = pl.read_csv(out_path)
    sunk_vals = [v for v in sunk["customer_id"].to_list() if v is not None]
    assert sunk_vals
    assert all(v in customer_ids for v in sunk_vals)
