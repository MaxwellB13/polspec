"""Specs as files: YAML in both directions, and Python source out.

Everything here is driven by the field registry in `fields.py`, so the YAML
a spec writes, the YAML it reads, and the Python it emits cannot disagree
about which fields exist. Files carry a `version:`; `migrations.py` brings an
older file forward before it is read.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import yaml

from polspec.errors import SerializationError, SpecError
from polspec.serialization.dtypes import (
    DTYPE_NAMES,
    NAME_TO_DTYPE,
    dtype_from_data,
    dtype_to_data,
    dtype_to_source,
    physical_name,
)
from polspec.serialization.fields import (
    CHECK_FIELDS,
    COLRULE_FIELDS,
    COLSPEC_FIELDS,
    FK_FIELDS,
    TABLESPEC_FIELDS,
    Ctx,
    check_to_source,
    check_unknown_keys,
    colspec_from_data,
    colspec_to_data,
    colspec_to_source,
    fk_to_source,
    needs_datetime_import,
    tablespec_from_data,
    tablespec_to_data,
)
from polspec.serialization.migrations import FORMAT_VERSION, migrate
from polspec.tablespec import TableSpec

if TYPE_CHECKING:
    from polspec.catspec import CatSpec
    from polspec.registry import Registry

__all__ = [
    "CHECK_FIELDS",
    "COLRULE_FIELDS",
    "COLSPEC_FIELDS",
    "FK_FIELDS",
    "FORMAT_VERSION",
    "TABLESPEC_FIELDS",
    "catspec_from_dict",
    "catspec_from_yaml",
    "catspec_to_dict",
    "catspec_to_yaml",
    "from_dict",
    "from_yaml",
    "registry_from_dict",
    "registry_from_yaml",
    "registry_to_dict",
    "registry_to_yaml",
    "to_dict",
    "to_python",
    "to_yaml",
]


# ---------------------------------------------------------------------------
# What a file cannot hold
# ---------------------------------------------------------------------------

_LOSS = {
    "yaml": (
        "YAML",
        "They will be lost on FrameSpec.from_yaml() unless re-declared on a "
        "subclass of the loaded spec.",
    ),
    "python": ("generated Python", "Re-declare them by hand on the generated class."),
}


def _warn_unserializable(spec: TableSpec, source: str | Path, kind: str) -> None:
    """Warns, naming exactly what a file cannot hold and will therefore lose.

    A `Check` or validator over a raw `polars.Expr` has no representation in
    a file; one written with `polspec.col()` does. Nothing else is lost.
    """
    medium, tail = _LOSS[kind]
    raw_checks = [c for c in spec.checks if c.pred is None]
    if raw_checks:
        names = ", ".join(repr(c.name) for c in raw_checks)
        warnings.warn(
            f"{spec.name} declares {len(raw_checks)} __checks__ ({names}) that "
            f"cannot be represented in {medium} (a Check over a raw polars.Expr; "
            f"write it with polspec.col() to persist it) and will NOT be written "
            f"to {source!s}. {tail}",
            stacklevel=3,
        )
    validators = [
        f"{col}.{v.name}"
        for col, cs in spec.columns.items()
        for v in cs.validators
        if v.pred is None
    ]
    if validators:
        names = ", ".join(repr(n) for n in validators)
        warnings.warn(
            f"{spec.name} declares {len(validators)} column-level validator(s) "
            f"({names}) that cannot be represented in {medium} (a validator over "
            f"a raw polars.Expr; write it with polspec.col() to persist it) and "
            f"will NOT be written to {source!s}. {tail}",
            stacklevel=3,
        )


def _require_columns(spec: TableSpec) -> None:
    if not spec.columns:
        raise SpecError(f"{spec.name} declares no ColSpec columns")


# ---------------------------------------------------------------------------
# TableSpec <-> data
# ---------------------------------------------------------------------------


def to_dict(spec: TableSpec) -> dict[str, Any]:
    """The YAML-ready data form of `spec`, `version` first."""
    return {"version": FORMAT_VERSION, **tablespec_to_data(spec)}


def from_dict(
    data: Mapping[str, Any] | None,
    *,
    categories: CatSpec | None = None,
    strict: bool = True,
    source: str = "spec data",
) -> TableSpec:
    """A `TableSpec` from data in any format version this polspec can read.

    An unknown key is an error, naming the closest known key, unless
    `strict=False`, which downgrades it to a warning.
    """
    current = migrate(data, "spec", source)
    return tablespec_from_data(current, Ctx(categories=categories, strict=strict))


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


def to_yaml(spec: TableSpec, source: str | Path) -> None:
    """Writes `spec` to a human-readable YAML file at `source`.

    Defaults are omitted so the file shows only what was declared. Checks and
    validators over raw expressions cannot be written and warn.
    """
    _require_columns(spec)
    _warn_unserializable(spec, source, "yaml")
    p = Path(source)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(to_dict(spec), sort_keys=False), encoding="utf-8")


def _resolve_categories(
    categories: CatSpec | str | Path | None, data: Mapping[str, Any], base: Path
) -> CatSpec | None:
    from polspec.catspec import CatSpec

    if categories is not None:
        if isinstance(categories, (str, Path)):
            return CatSpec.from_yaml(categories)
        return categories
    declared = data.get("categories")
    if declared is None:
        return None
    if isinstance(declared, (str, Path)):
        path = Path(declared)
        if not path.is_absolute():
            path = base / path
        return CatSpec.from_yaml(path)
    if isinstance(declared, Mapping):
        return catspec_from_dict(declared)
    raise SerializationError(
        f"'categories' must be a path or a registry mapping, got {declared!r}"
    )


def from_yaml(
    source: str | Path,
    *,
    categories: CatSpec | str | Path | None = None,
    strict: bool = True,
) -> TableSpec:
    """Reads a `TableSpec` from a YAML file written by `to_yaml`.

    `categories` is a CatSpec registry, or a path to one, used to resolve
    shared Enums and Categoricals; when omitted, a `categories:` key in the
    file is loaded automatically, relative to the file. A file from an older
    format version is migrated on the way in.
    """
    path = Path(source)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is not None and not isinstance(raw, Mapping):
        raise SerializationError(
            f"{source}: expected a mapping at the top level, got {type(raw).__name__}"
        )
    registry = _resolve_categories(categories, raw or {}, path.parent)
    return from_dict(raw, categories=registry, strict=strict, source=str(source))


# ---------------------------------------------------------------------------
# Python source
# ---------------------------------------------------------------------------


def to_python(spec: TableSpec, source: str | Path) -> None:
    """Writes `spec` as a Python module defining a `FrameSpec` subclass.

    Columns are declared through `__columns__`, since a name straight from
    data is not always a valid identifier. Checks and validators over raw
    expressions cannot be written and warn.
    """
    _require_columns(spec)
    _warn_unserializable(spec, source, "python")

    data = tablespec_to_data(spec)
    persistable_checks = [c for c in spec.checks if c.pred is not None]
    has_validators = any("validators" in c for c in data["columns"].values())
    has_rules = any("rules" in c for c in data["columns"].values())

    imports = []
    if persistable_checks or has_validators:
        imports.append("Check")
    if has_rules:
        imports.append("ColRule")
    imports += ["ColSpec", "FrameSpec"]
    if spec.foreign_keys:
        imports.append("ForeignKey")
    if has_rules or has_validators or persistable_checks:
        imports.append("col")

    lines = [f'"""Declares the {spec.name} schema."""', "", "import polars as pl"]
    if needs_datetime_import(data):
        lines.append("import datetime")
    lines.append(f"from polspec import {', '.join(imports)}")
    lines.extend(["", "", f"class {spec.name}(FrameSpec):", "    __columns__ = {"])
    for name, cs in spec.columns.items():
        lines.append(f"        {name!r}: {colspec_to_source(cs)},")
    lines.append("    }")
    if spec.unique_together:
        groups = ", ".join(repr(list(group)) for group in spec.unique_together)
        lines.append(f"    __unique_together__ = [{groups}]")
    if spec.foreign_keys:
        fks = ", ".join(fk_to_source(fk) for fk in spec.foreign_keys)
        lines.append(f"    __foreign_keys__ = [{fks}]")
    if persistable_checks:
        checks = ", ".join(check_to_source(c) for c in persistable_checks)
        lines.append(f"    __checks__ = [{checks}]")

    p = Path(source)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CatSpec
# ---------------------------------------------------------------------------

_CATSPEC_KEYS: tuple[str, ...] = ("version", "enums", "categoricals", "choices")


def catspec_to_dict(catspec: CatSpec) -> dict[str, Any]:
    """The data form of a registry: enums, categoricals, and any loose choices."""
    out: dict[str, Any] = {}
    enums = catspec.enums
    if enums:
        out["enums"] = {k: list(v) for k, v in enums.items()}
    categoricals = catspec.categoricals
    loose_choices: dict[str, list[Any]] = {}
    if categoricals:
        cats: dict[str, Any] = {}
        for key, cat in categoricals.items():
            info: dict[str, Any] = {"name": cat.name()}
            if cat.namespace():
                info["namespace"] = cat.namespace()
            if cat.physical() != pl.UInt32:
                info["physical"] = physical_name(cat.physical())
            choices = catspec.get_choices(key)
            if choices:
                info["categories"] = list(choices)
            cats[key] = info
        out["categoricals"] = cats
    for key, choices in catspec.choices.items():
        if key not in categoricals and choices:
            loose_choices[key] = list(choices)
    if loose_choices:
        out["choices"] = loose_choices
    return out


def catspec_from_dict(
    data: Mapping[str, Any] | None,
    *,
    strict: bool = True,
    source: str = "registry data",
) -> CatSpec:
    from polspec.catspec import CatSpec

    current = migrate(data, "catspec", source)
    check_unknown_keys(current, _CATSPEC_KEYS, Ctx(strict=strict), "")
    return CatSpec(
        enums=current.get("enums"),
        categoricals=current.get("categoricals"),
        choices=current.get("choices"),
    )


def catspec_to_yaml(catspec: CatSpec, source: str | Path | None = None) -> str | None:
    dumped = yaml.safe_dump(
        {"version": FORMAT_VERSION, **catspec_to_dict(catspec)}, sort_keys=False
    )
    if source is None:
        return dumped
    p = Path(source)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumped, encoding="utf-8")
    return None


def catspec_from_yaml(source: str | Path, *, strict: bool = True) -> CatSpec:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"CatSpec file not found: {source}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return catspec_from_dict(raw, strict=strict, source=str(source))


# ---------------------------------------------------------------------------
# Registry files: several specs, and their shared categories, in one file
# ---------------------------------------------------------------------------

_REGISTRY_KEYS: tuple[str, ...] = ("version", "categories", "specs")


def registry_to_dict(registry: Registry) -> dict[str, Any]:
    """`version`, the declared `categories` if any, and `specs` keyed by name."""
    out: dict[str, Any] = {"version": FORMAT_VERSION}
    if registry.categories is not None:
        out["categories"] = catspec_to_dict(registry.categories)
    specs: dict[str, Any] = {}
    for spec in registry.specs:
        body = tablespec_to_data(spec)
        body.pop("name", None)
        specs[spec.name] = body
    out["specs"] = specs
    return out


def registry_from_dict(
    data: Mapping[str, Any] | None,
    *,
    strict: bool = True,
    source: str = "registry data",
    base: Path | None = None,
) -> Registry:
    from polspec.registry import Registry

    current = migrate(data, "registry", source)
    ctx = Ctx(strict=strict)
    check_unknown_keys(current, _REGISTRY_KEYS, ctx, "")

    declared = current.get("categories")
    categories: CatSpec | None = None
    if isinstance(declared, Mapping):
        categories = catspec_from_dict(declared, strict=strict, source=source)
    elif isinstance(declared, (str, Path)):
        if base is None:
            raise SerializationError(
                f"{source}: 'categories' names a file ({declared!r}); read the "
                "registry with from_yaml so the path can be resolved"
            )
        path = Path(declared)
        categories = catspec_from_yaml(path if path.is_absolute() else base / path)
    elif declared is not None:
        raise SerializationError(
            f"{source}: 'categories' must be a mapping or a path, got {declared!r}"
        )

    specs = current.get("specs")
    if not isinstance(specs, Mapping) or not specs:
        raise SerializationError(
            f"{source}: a registry file needs a non-empty 'specs' mapping, keyed "
            "by spec name"
        )
    registry = Registry(categories=categories)
    for name, body in specs.items():
        if not isinstance(body, Mapping):
            raise SerializationError(
                f"{source}: specs.{name} must be a mapping, got {type(body).__name__}"
            )
        if "name" in body and body["name"] != name:
            raise SerializationError(
                f"{source}: specs.{name} carries name {body['name']!r}; the key "
                "is the name, so drop one or make them agree"
            )
        registry.add(
            tablespec_from_data(
                {"name": str(name), **body}, Ctx(categories=categories, strict=strict)
            )
        )
    return registry


def registry_to_yaml(registry: Registry, source: str | Path) -> None:
    for spec in registry.specs:
        _warn_unserializable(spec, source, "yaml")
    p = Path(source)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(registry_to_dict(registry), sort_keys=False), encoding="utf-8"
    )


def registry_from_yaml(source: str | Path, *, strict: bool = True) -> Registry:
    path = Path(source)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return registry_from_dict(raw, strict=strict, source=str(source), base=path.parent)


# ---------------------------------------------------------------------------
# Names kept for callers of the previous module layout
# ---------------------------------------------------------------------------

_YAML_DTYPES = DTYPE_NAMES
_YAML_NAME_TO_DTYPE = NAME_TO_DTYPE
_dtype_to_yaml = dtype_to_data
_dtype_from_yaml = dtype_from_data
_dtype_to_python = dtype_to_source
_colspec_to_yaml = colspec_to_data
_colspec_to_python = colspec_to_source


def _colspec_from_yaml(data: Mapping[str, Any], categories: CatSpec | None = None):
    return colspec_from_data(data, Ctx(categories=categories), "column")
