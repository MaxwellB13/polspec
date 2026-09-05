"""`Registry`: several specs that belong together.

One spec knows the *name* of the spec its foreign key points at and nothing
else. The registry is what resolves that name, orders parents before
children, and generates or validates the whole set.
"""

import textwrap
import types

import polars as pl
import pytest
import yaml
from polspec import (
    CatSpec,
    ColSpec,
    ForeignKey,
    FrameSpec,
    Registry,
    RegistryError,
    TableSpec,
    ValidationError,
)


class Customers(FrameSpec):
    id = ColSpec(pl.Int64, bounds=(1, 1_000_000), unique=True)
    country = ColSpec(pl.Enum(["UK", "US"]))


class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, 1_000_000), unique=True)
    customer_id = ColSpec(pl.Int64, bounds=(1, 1_000_000))
    status = ColSpec(pl.Enum(["NEW", "PAID"]))
    __foreign_keys__ = [
        ForeignKey("customer_id", references=Customers, ref_columns="id")
    ]


class OrderLines(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, 1_000_000))
    line_no = ColSpec(pl.Int32, bounds=(1, 100))
    # Declared against a *name*: nothing checks it until a registry does.
    __foreign_keys__ = [
        ForeignKey("order_id", references="Orders", ref_columns="order_id")
    ]


class Employees(FrameSpec):
    # Generation does not enforce unique=True yet; a wide domain keeps the
    # odds of a collision in a small frame negligible.
    id = ColSpec(pl.Int64, bounds=(1, 1_000_000), unique=True)
    manager_id = ColSpec(pl.Int64, bounds=(1, 1_000_000), nullable=True)
    __foreign_keys__ = [ForeignKey("manager_id", references="self", ref_columns="id")]


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------


def test_registry_holds_specs_by_name_in_insertion_order():
    registry = Registry(OrderLines, Orders, Customers.spec)
    assert registry.names == ("OrderLines", "Orders", "Customers")
    assert registry["Orders"] is Orders.spec
    assert registry[Orders] is Orders.spec  # class, spec or name all look up
    assert Orders in registry and "Customers" in registry and "Nope" not in registry
    assert len(registry) == 3 and list(registry) == list(registry.names)
    assert repr(registry) == "Registry(OrderLines, Orders, Customers)"


def test_registry_refuses_two_different_specs_with_one_name():
    other = TableSpec("Orders", {"x": ColSpec(pl.Int64)})
    with pytest.raises(
        RegistryError, match="Two different specs are both named 'Orders'"
    ):
        Registry(Orders, other)
    # The same spec twice is fine; nothing is duplicated.
    assert len(Registry(Orders, Orders.spec)) == 1


def test_unknown_name_suggests_the_closest():
    registry = Registry(Customers, Orders)
    with pytest.raises(RegistryError, match=r"'Order' .*did you mean 'Orders'"):
        registry["Order"]


def test_add_chains_and_from_module_collects_specs():
    registry = Registry().add(Customers).add(Orders)
    assert registry.names == ("Customers", "Orders")

    module = types.ModuleType("fake_specs")
    module.Customers = Customers
    module.Orders = Orders
    module.Loose = TableSpec("Loose", {"a": ColSpec(pl.Int64)})
    module.FrameSpec = FrameSpec  # the base class is never a spec
    assert Registry.from_module(module).names == ("Customers", "Orders", "Loose")
    # own_only keeps only what the module itself defines: the TableSpec has no
    # __module__ and counts as the module's own.
    assert Registry.from_module(module, own_only=True).names == ("Loose",)


# ---------------------------------------------------------------------------
# Resolution and ordering
# ---------------------------------------------------------------------------


def test_resolve_binds_keys_declared_against_names():
    registry = Registry(Customers, Orders, OrderLines, Employees)
    assert OrderLines.spec.foreign_keys[0].target is None
    resolved = registry.resolve()
    fk = resolved["OrderLines"].foreign_keys[0]
    assert fk.target is Orders.spec and fk.references == "Orders"
    # The original registry is untouched.
    assert registry["OrderLines"].foreign_keys[0].target is None


def test_resolve_refuses_a_target_outside_the_registry():
    with pytest.raises(
        RegistryError, match="references 'Orders', which is not in the registry"
    ):
        Registry(Customers, OrderLines).resolve()


def test_resolve_runs_the_declaration_checks_a_bare_name_skipped():
    class Bad(FrameSpec):
        order_ref = ColSpec(pl.String)
        __foreign_keys__ = [
            ForeignKey("order_ref", references="Orders", ref_columns="order_id")
        ]

    with pytest.raises(RegistryError, match=r"not\s+dtype-compatible"):
        Registry(Customers, Orders, Bad).resolve()

    class Missing(FrameSpec):
        order_id = ColSpec(pl.Int64)
        __foreign_keys__ = [
            ForeignKey("order_id", references="Orders", ref_columns="nope")
        ]

    with pytest.raises(RegistryError, match="unknown column 'nope' on 'Orders'"):
        Registry(Customers, Orders, Missing).resolve()


def test_resolve_checks_the_parent_domain_fits_the_local_column():
    """A key naming its target as a string has no spec to check against until
    the registry binds one, so the domain check lands here rather than at
    declaration.
    """

    class Narrow(FrameSpec):
        # Orders.order_id is bounded 1..1_000_000, which does not fit here.
        order_id = ColSpec(pl.Int64, bounds=(1, 500))
        __foreign_keys__ = [
            ForeignKey("order_id", references="Orders", ref_columns="order_id")
        ]

    with pytest.raises(RegistryError, match="do not fit inside"):
        Registry(Customers, Orders, Narrow).resolve()


def test_order_puts_parents_first_and_keeps_declaration_order_otherwise():
    registry = Registry(OrderLines, Employees, Orders, Customers)
    assert registry.order() == ("Employees", "Customers", "Orders", "OrderLines")
    assert registry.parents(OrderLines) == ("Orders",)
    assert registry.parents(Employees) == ()  # self-references impose nothing
    assert registry.ancestors(OrderLines) == ("Customers", "Orders")


def test_a_cycle_is_an_error():
    a = TableSpec(
        "A",
        {"id": ColSpec(pl.Int64), "b_id": ColSpec(pl.Int64)},
        foreign_keys=[ForeignKey("b_id", references="B", ref_columns="id")],
    )
    b = TableSpec(
        "B",
        {"id": ColSpec(pl.Int64), "a_id": ColSpec(pl.Int64)},
        foreign_keys=[ForeignKey("a_id", references="A", ref_columns="id")],
    )
    with pytest.raises(RegistryError, match="cycle between specs: A, B"):
        Registry(a, b).order()


# ---------------------------------------------------------------------------
# Generating
# ---------------------------------------------------------------------------


def test_generate_all_threads_parents_into_children():
    registry = Registry(OrderLines, Orders, Customers, Employees)
    frames = registry.generate_all(
        {Customers: 50, "Orders": 200, OrderLines: 400, Employees: 20}, seed=1
    )
    assert list(frames) == ["Customers", "Employees", "Orders", "OrderLines"]
    assert frames["Orders"].height == 200
    assert set(frames["Orders"]["customer_id"]) <= set(frames["Customers"]["id"])
    assert set(frames["OrderLines"]["order_id"]) <= set(frames["Orders"]["order_id"])
    registry.validate_all(frames)  # every key satisfied by construction


def test_generate_all_seed_is_per_spec_so_adding_one_changes_nothing_else():
    small = Registry(Customers, Orders).generate_all(100, seed=7)
    bigger = Registry(Customers, Orders, Employees).generate_all(100, seed=7)
    assert small["Customers"].equals(bigger["Customers"])
    assert small["Orders"].equals(bigger["Orders"])
    # And a different seed is a different frame.
    assert not small["Orders"].equals(
        Registry(Customers, Orders).generate_all(100, seed=8)["Orders"]
    )


def test_generate_all_row_counts_must_cover_every_spec():
    registry = Registry(Customers, Orders)
    with pytest.raises(RegistryError, match=r"No row count given for \['Orders'\]"):
        registry.generate_all({Customers: 10})
    with pytest.raises(RegistryError, match="not in the registry"):
        registry.generate_all({Customers: 10, Orders: 10, "Ghost": 1})


def test_generate_all_uses_supplied_frames_instead_of_generating():
    registry = Registry(Customers, Orders)
    customers = pl.DataFrame(
        {
            "id": [7, 8],
            "country": pl.Series(["UK", "US"], dtype=Customers.schema()["country"]),
        }
    )
    frames = registry.generate_all(30, seed=1, references={Customers: customers})
    assert frames["Customers"] is customers
    assert set(frames["Orders"]["customer_id"]) <= {7, 8}


def test_generate_all_needs_every_parent_from_somewhere():
    with pytest.raises(RegistryError, match="neither in the registry nor supplied"):
        Registry(OrderLines).generate_all(10)
    orders = Orders.generate(
        5, seed=1, references={Customers: Customers.generate(5, seed=1)}
    )
    frames = Registry(OrderLines).generate_all(10, seed=1, references={Orders: orders})
    assert list(frames) == ["OrderLines"]
    assert set(frames["OrderLines"]["order_id"]) <= set(orders["order_id"])


def test_generate_related_stops_at_what_one_spec_needs():
    registry = Registry(Customers, Orders, OrderLines, Employees)
    frames = registry.generate_related(Orders, 40, seed=1)
    assert list(frames) == ["Customers", "Orders"]
    assert list(registry.generate_related("Employees", 10, seed=1)) == ["Employees"]
    with pytest.raises(RegistryError, match="No spec named 'Nope'"):
        registry.generate_related("Nope", 10)


# ---------------------------------------------------------------------------
# Validating
# ---------------------------------------------------------------------------


def test_validate_all_reports_every_spec_at_once():
    registry = Registry(Customers, Orders)
    frames = registry.generate_all(20, seed=3)
    bad_orders = frames["Orders"].with_columns(
        pl.when(pl.arange(0, 20) == 0)
        .then(pl.lit(999_999_999))
        .otherwise("customer_id")
        .alias("customer_id")
    )
    bad_customers = frames["Customers"].with_columns(pl.lit(-1).alias("id"))
    with pytest.raises(ValidationError) as info:
        registry.validate_all({Orders: bad_orders, Customers: bad_customers})
    message = str(info.value)
    assert "against 'Orders'" in message and "against 'Customers'" in message
    assert any("customer_id" in e for e in info.value.errors)

    reports = registry.inspect_all({Orders: bad_orders, Customers: bad_customers})
    assert list(reports) == ["Customers", "Orders"]
    assert reports["Orders"].by_code("foreign_key")
    assert reports["Customers"].by_code("bounds")


def test_validate_all_applies_transformations_and_keeps_laziness():
    registry = Registry(Customers, Orders)
    frames = registry.generate_all(10, seed=1)
    with_extra = frames["Orders"].with_columns(pl.lit(1).alias("extra")).lazy()
    out = registry.validate_all(
        {Orders: with_extra, Customers: frames["Customers"]}, extra_cols="drop"
    )
    assert isinstance(out["Orders"], pl.LazyFrame)
    assert out["Orders"].collect_schema().names() == list(Orders.spec)
    assert isinstance(out["Customers"], pl.DataFrame)


def test_frames_for_specs_outside_the_registry_are_refused():
    with pytest.raises(RegistryError, match="No spec named 'Employees'"):
        Registry(Customers).inspect_all({Employees: Employees.generate(3, seed=1)})


def test_a_child_validated_alone_still_needs_its_parent():
    registry = Registry(Customers, Orders)
    frames = registry.generate_all(10, seed=1)
    reports = registry.inspect_all({Orders: frames["Orders"]})
    assert reports["Orders"].by_code("foreign_key_unresolved")
    # ...unless it comes from outside.
    reports = registry.inspect_all(
        {Orders: frames["Orders"]}, references={"Customers": frames["Customers"]}
    )
    assert reports["Orders"].passed


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------


def test_catspec_merges_what_the_specs_declare():
    cats = Registry(Customers, Orders).catspec()
    assert cats.enums == {"country": ["UK", "US"], "status": ["NEW", "PAID"]}


def test_catspec_refuses_two_specs_that_disagree():
    class Products(FrameSpec):
        status = ColSpec(pl.Enum(["ACTIVE", "RETIRED"]))

    with pytest.raises(
        RegistryError, match="Products and Orders disagree about 'status'"
    ):
        Registry(Orders, Products).catspec()

    settled = CatSpec(enums={"STATUS": ["NEW", "PAID"]})
    assert Registry(Orders, Products, categories=settled).catspec() is settled


def test_resolve_checks_columns_against_declared_categories():
    shared = CatSpec(enums={"STATUS": ["NEW", "PAID", "SHIPPED"]})
    with pytest.raises(RegistryError, match=r"Orders\.status declares Enum"):
        Registry(Customers, Orders, categories=shared).resolve()
    agreeing = CatSpec(enums={"STATUS": ["NEW", "PAID"]})
    Registry(Customers, Orders, categories=agreeing).resolve()


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def test_registry_round_trips_through_one_yaml_file(tmp_path):
    shared = CatSpec(enums={"STATUS": ["NEW", "PAID"]})
    registry = Registry(Customers, Orders, OrderLines, categories=shared)
    path = tmp_path / "specs.yaml"
    registry.to_yaml(path)

    data = yaml.safe_load(path.read_text())
    assert list(data) == ["version", "categories", "specs"]
    assert list(data["specs"]) == ["Customers", "Orders", "OrderLines"]
    assert "name" not in data["specs"]["Orders"]  # the key is the name
    assert data["specs"]["Orders"]["foreign_keys"][0]["references"] == "Customers"

    loaded = Registry.from_yaml(path)
    assert loaded.names == registry.names
    assert loaded.categories.enums == {"STATUS": ["NEW", "PAID"]}
    for name in registry:
        assert loaded[name] == registry[name]
    resolved = loaded.resolve()
    assert resolved["Orders"].foreign_keys[0].target == Customers.spec
    assert Registry.from_dict(registry.to_dict()).names == registry.names


def test_registry_file_rejects_unknown_keys_and_disagreeing_names(tmp_path):
    with pytest.raises(Exception, match=r"'spec' \(did you mean 'specs'\?\)"):
        Registry.from_dict({"version": 2, "spec": {}})
    with pytest.raises(Exception, match="needs a non-empty 'specs' mapping"):
        Registry.from_dict({"version": 2})
    with pytest.raises(Exception, match=r"specs\.Orders carries name 'Other'"):
        Registry.from_dict(
            {
                "version": 2,
                "specs": {
                    "Orders": {"name": "Other", "columns": {"a": {"dtype": "Int64"}}}
                },
            }
        )


def test_registry_file_may_point_at_a_categories_file(tmp_path):
    CatSpec(enums={"STATUS": ["NEW", "PAID"]}).to_yaml(tmp_path / "cats.yaml")
    (tmp_path / "specs.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "categories": "cats.yaml",
                "specs": {
                    "Orders": {"columns": {"status": {"dtype": {"Enum": "STATUS"}}}}
                },
            }
        )
    )
    loaded = Registry.from_yaml(tmp_path / "specs.yaml")
    assert loaded["Orders"]["status"].dtype == pl.Enum(["NEW", "PAID"])
    assert loaded.categories is not None


def test_discover_walks_python_and_yaml(tmp_path):
    (tmp_path / "specs.py").write_text(
        textwrap.dedent(
            """
            import polars as pl
            from polspec import ColSpec, ForeignKey, FrameSpec

            class Customers(FrameSpec):
                id = ColSpec(pl.Int64, unique=True)

            class Orders(FrameSpec):
                order_id = ColSpec(pl.Int64, bounds=(1, 1_000_000), unique=True)
                customer_id = ColSpec(pl.Int64)
                __foreign_keys__ = [
                    ForeignKey("customer_id", references=Customers, ref_columns="id")
                ]
            """
        )
    )
    (tmp_path / "nested").mkdir()
    Employees.to_yaml(tmp_path / "nested" / "employees.yaml")
    Registry(OrderLines).to_yaml(tmp_path / "bundle.yaml")
    (tmp_path / "test_ignored.py").write_text("raise RuntimeError('never imported')\n")
    (tmp_path / "_private.py").write_text("raise RuntimeError('never imported')\n")
    (tmp_path / "notes.txt").write_text("not a spec")

    registry = Registry.discover(tmp_path)
    assert set(registry.names) == {"Customers", "Orders", "Employees", "OrderLines"}
    # Files are visited in sorted order: bundle.yaml, nested/, specs.py.
    assert registry.names == ("OrderLines", "Employees", "Customers", "Orders")
    assert registry.resolve().order() == (
        "Employees",
        "Customers",
        "Orders",
        "OrderLines",
    )

    single = Registry.discover(tmp_path / "nested" / "employees.yaml")
    assert single.names == ("Employees",)
    with pytest.raises(RegistryError, match="no such file or directory"):
        Registry.discover(tmp_path / "missing.yaml")
    with pytest.raises(
        RegistryError, match=r"don't know how to read specs from '\.txt'"
    ):
        Registry.discover(tmp_path / "notes.txt")


def test_discover_reports_a_broken_module(tmp_path):
    (tmp_path / "broken.py").write_text("raise ValueError('boom')\n")
    with pytest.raises(RegistryError, match=r"error importing .*broken\.py: boom"):
        Registry.discover(tmp_path / "broken.py")


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------


def test_to_mermaid_draws_every_entity_and_every_key(tmp_path):
    registry = Registry(Customers, Orders, OrderLines, Employees)
    mmd = registry.to_mermaid(tmp_path / "er.mmd")
    assert mmd.startswith("erDiagram\n")
    for name in registry:
        assert f"    {name} {{" in mmd
    assert 'Customers ||--o{ Orders : "fk_customer_id__Customers"' in mmd
    assert 'Orders ||--o{ OrderLines : "fk_order_id__Orders"' in mmd
    assert 'Employees ||--o{ Employees : "fk_manager_id__self"' in mmd
    assert (tmp_path / "er.mmd").read_text(encoding="utf-8") == mmd
    # Entities come first, then the relationships between them.
    assert mmd.index("||--o{") > mmd.rindex("    }")
