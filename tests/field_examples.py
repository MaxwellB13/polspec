"""One example per field: for each field of `ColSpec`, `TableSpec`, `Check`,
`ColRule`, `ForeignKey` and `Hierarchy`, a spec without it and a spec with it,
differing in that field alone.

`contracts/test_parity.py` holds each to a spec file and to `diff`;
`docs/test_docs.py` holds the reference pages to the keys they write. A field
with no example fails both, so a new one is written up before it ships.
"""

from __future__ import annotations

import dataclasses

import polars as pl
from polspec import Check, ColRule, ColSpec, ForeignKey, Hierarchy, TableSpec, col

_RULE = ColRule(when=col("x") > 5, choices=[1, 2])

# Each ColSpec field, and a column without it and one with it: the only
# difference between the two. A field with no entry fails the test below.
COLUMN_FIELDS: dict[str, tuple[ColSpec, ColSpec]] = {
    "dtype": (ColSpec(pl.Int64), ColSpec(pl.Int32)),
    "seed_name": (ColSpec(pl.Int64), ColSpec(pl.Int64, seed_name="older")),
    "nullable": (ColSpec(pl.Int64), ColSpec(pl.Int64, nullable=True)),
    "optional": (ColSpec(pl.Int64), ColSpec(pl.Int64, optional=True)),
    "default": (ColSpec(pl.Int64), ColSpec(pl.Int64, default=3)),
    "bounds": (ColSpec(pl.Int64), ColSpec(pl.Int64, bounds=(0, 9))),
    "tags": (ColSpec(pl.Int64), ColSpec(pl.Int64, tags="pii")),
    "description": (ColSpec(pl.Int64), ColSpec(pl.Int64, description="A number.")),
    "unique": (ColSpec(pl.Int64), ColSpec(pl.Int64, unique=True)),
    "null_probability": (
        ColSpec(pl.Int64, nullable=True, null_probability=0.1),
        ColSpec(pl.Int64, nullable=True, null_probability=0.3),
    ),
    "nan_probability": (ColSpec(pl.Float64), ColSpec(pl.Float64, nan_probability=0.1)),
    "string_length": (ColSpec(pl.String), ColSpec(pl.String, string_length=(1, 5))),
    "list_length": (
        ColSpec(pl.List(pl.Int64)),
        ColSpec(pl.List(pl.Int64), list_length=(1, 3)),
    ),
    "element_null_probability": (
        ColSpec(pl.List(pl.Int64)),
        ColSpec(pl.List(pl.Int64), element_null_probability=0.2),
    ),
    "fields": (
        ColSpec(pl.Struct({"w": pl.Int16})),
        ColSpec(
            pl.Struct({"w": pl.Int16}), fields={"w": ColSpec(pl.Int16, bounds=(1, 9))}
        ),
    ),
    "format": (ColSpec(pl.String), ColSpec(pl.String, format="email")),
    "extra_values": (
        ColSpec(pl.String, format="iso_country"),
        ColSpec(pl.String, format="iso_country", extra_values=["UK (ISO)"]),
    ),
    "pattern": (ColSpec(pl.String), ColSpec(pl.String, pattern=r"^[A-Z]{3}$")),
    "distribution": (
        ColSpec(pl.Float64, bounds=(0, 10)),
        ColSpec(pl.Float64, bounds=(0, 10), distribution="normal"),
    ),
    "distribution_params": (
        ColSpec(pl.Float64, bounds=(0, 10), distribution="normal"),
        ColSpec(
            pl.Float64,
            bounds=(0, 10),
            distribution="normal",
            distribution_params={"mean": 5.0, "std": 2.0},
        ),
    ),
    "choices": (ColSpec(pl.String), ColSpec(pl.String, choices=["a", "b"])),
    "weights": (
        ColSpec(pl.String, choices=["a", "b"]),
        ColSpec(pl.String, choices={"a": 1, "b": 3}),
    ),
    "rules": (
        ColSpec(pl.Int64, bounds=(0, 9)),
        ColSpec(pl.Int64, bounds=(0, 9), rules=[_RULE]),
    ),
    "validators": (ColSpec(pl.Int64), ColSpec(pl.Int64, validators=[col("c") > 0])),
    "derive": (ColSpec(pl.Int64), ColSpec(pl.Int64, derive=col("x") + 1)),
    "text_format": (ColSpec(pl.Date), ColSpec(pl.Date, text_format="%d/%m/%Y")),
}


def with_column(column: ColSpec) -> TableSpec:
    return TableSpec("T", {"x": ColSpec(pl.Int64, bounds=(0, 9)), "c": column})


_BASE = TableSpec(
    "T",
    {
        "id": ColSpec(pl.Int64, unique=True, bounds=(1, 100)),
        "id2": ColSpec(pl.Int64, unique=True, bounds=(1, 100)),
        "a": ColSpec(pl.Int64, bounds=(0, 9)),
        "b": ColSpec(pl.Int64, bounds=(0, 9)),
        "parent": ColSpec(pl.Int64, nullable=True),
        "parent2": ColSpec(pl.Int64, nullable=True),
    },
)
_CHECK = Check(col("a") <= col("b"), name="ordered")
_KEY = ForeignKey("parent", references="self", ref_columns="id")


def _ruled(**changes) -> TableSpec:
    rule = dataclasses.replace(_RULE, **{"when": col("a") > 5, **changes})
    return _BASE.with_columns({"r": ColSpec(pl.Int64, bounds=(0, 9), rules=[rule])})


def _tree(**changes) -> TableSpec:
    columns = {
        "node": ColSpec(pl.String),
        "up": ColSpec(pl.String),
        "other": ColSpec(pl.String),
    }
    hierarchy = {"child": "node", "parent": "up", "max_depth": 3, **changes}
    return TableSpec("Tree", columns, hierarchy=Hierarchy(**hierarchy))


def _checked(**changes) -> TableSpec:
    given = {"name": "ordered", **changes}
    return _BASE.with_checks(Check(col("a") <= col("b"), **given))


def _keyed(**changes) -> TableSpec:
    given = {"columns": "parent", "references": "self", "ref_columns": "id", **changes}
    return _BASE.with_foreign_keys(ForeignKey(**given))


# Each field of the table and of what it holds, as a spec without it and a
# spec with it.
TABLE_FIELDS: dict[type, dict[str, tuple[TableSpec, TableSpec]]] = {
    TableSpec: {
        "name": (_BASE, _BASE.with_name("U")),
        "columns": (_BASE, _BASE.with_columns({"z": ColSpec(pl.String)})),
        "description": (_BASE, _BASE.with_description("Rows.")),
        "rows": (_BASE, _BASE.with_rows((1, None))),
        "checks": (_BASE, _BASE.with_checks(_CHECK)),
        "unique_together": (_BASE, _BASE.with_unique_together(("a", "b"))),
        "foreign_keys": (_BASE, _BASE.with_foreign_keys(_KEY)),
        "hierarchy": (_tree().with_hierarchy(None), _tree()),
        "order_by": (_BASE, _BASE.with_order_by("a")),
        "descending": (
            _BASE.with_order_by("a"),
            _BASE.with_order_by("a", descending=True),
        ),
    },
    Check: {
        "expr": (
            _checked(),
            _BASE.with_checks(Check(col("a") < col("b"), name="ordered")),
        ),
        "name": (_checked(), _checked(name="other")),
        "description": (_checked(), _checked(description="Why.")),
        "ignore_nulls": (_checked(), _checked(ignore_nulls=False)),
    },
    ColRule: {
        "when": (_ruled(), _ruled(when=col("a") > 6)),
        "choices": (_ruled(), _ruled(choices=(1, 3))),
        "weights": (_ruled(), _ruled(choices={1: 1.0, 3: 3.0})),
    },
    ForeignKey: {
        "columns": (_keyed(), _keyed(columns="parent2")),
        "references": (_keyed(references="Owners"), _keyed(references="Parents")),
        "ref_columns": (_keyed(), _keyed(ref_columns="id2")),
        "name": (_keyed(), _keyed(name="up")),
    },
    Hierarchy: {
        "child": (_tree(), _tree(child="other")),
        "parent": (_tree(), _tree(parent="other")),
        "max_depth": (_tree(), _tree(max_depth=4)),
        "branching": (_tree(), _tree(branching=2.0)),
        "roots": (_tree(), _tree(roots=5)),
        "layout": (_tree(), _tree(layout="whole")),
    },
}
