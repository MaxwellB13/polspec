"""Specs as files: YAML in both directions, and Python source out.

Everything here is driven by the field registry in `fields.py`, so the YAML
a spec writes, the YAML it reads, and the Python it emits cannot disagree
about which fields exist. Files carry a `version:`; `migrations.py` brings an
older file forward before it is read.

What is public here is exported from `polspec` itself.

Internal: not part of the public API.
"""

from __future__ import annotations

import keyword
import re
import unicodedata
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import polars as pl
import yaml
from yaml.representer import RepresenterError

from polspec.catspec import pool_of
from polspec.errors import SerializationError, SpecError
from polspec.serialization.dtypes import physical_name
from polspec.serialization.fields import (
    Ctx,
    check_to_source,
    check_unknown_keys,
    colspec_to_source,
    fk_to_source,
    hierarchy_to_source,
    tablespec_from_data,
    tablespec_to_data,
)
from polspec.serialization.migrations import FORMAT_VERSION, migrate
from polspec.tablespec import SpecLike, TableSpec, as_table_spec, require_columns

if TYPE_CHECKING:
    from polspec.catspec import CatSpec
    from polspec.registry import Registry

__all__ = [
    "FORMAT_VERSION",
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

# The field tables, exported here until 0.18 and internal since: each still
# works, and warns, until 1.0.
_MOVED_FIELDS = frozenset(
    {
        "CHECK_FIELDS",
        "COLRULE_FIELDS",
        "COLSPEC_FIELDS",
        "FK_FIELDS",
        "TABLESPEC_FIELDS",
    }
)


def __getattr__(name: str) -> Any:
    if name in _MOVED_FIELDS:
        from polspec import _deprecation
        from polspec.serialization import fields

        _deprecation.warn_deprecated(
            f"polspec.serialization.{name}",
            use="the keys the Spec files reference lists",
        )
        return getattr(fields, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


class _Located:
    """Prefixes `where` to a declaration or file error raised inside it, so
    an error found reading a file names the file -- keeping the error's own
    type, which a caller may already catch."""

    def __init__(self, where: str) -> None:
        self.where = where

    def __enter__(self) -> None:
        return None

    def __exit__(
        self, kind: type | None, exc: BaseException | None, tb: object
    ) -> Literal[False]:
        if not isinstance(exc, (SpecError, SerializationError)):
            return False
        if str(exc).startswith(self.where):
            return False
        raise type(exc)(f"{self.where}: {exc}") from exc


def _written(text: str, path: str | Path | None) -> str | None:
    """`text` written to `path` -- its folders made -- or, with no path,
    returned."""
    if path is None:
        return text
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return None


def _warn_unserializable(spec: TableSpec, where: str | Path, kind: str) -> None:
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
            f"to {where!s}. {tail}",
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
            f"will NOT be written to {where!s}. {tail}",
            stacklevel=3,
        )


def dump_yaml(data: Mapping[str, Any], where: str | Path) -> str:
    """`data` as YAML text, or a `SerializationError` naming what YAML
    cannot hold and where it sits.

    Dumped to a string before any file is opened, so a value that cannot be
    written leaves no half-written file behind. The values a spec holds
    that YAML has no form for -- a time, a duration -- are tagged before
    they get here (`polspec.scalars`); this is for anything that is not.
    """
    try:
        return yaml.safe_dump(dict(data), sort_keys=False)
    except RepresenterError as exc:
        value = exc.args[1] if len(exc.args) > 1 else None
        inside = _path_to(data, value) or "a value"
        raise SerializationError(
            f"{where}: {inside} holds {value!r} (of type {type(value).__name__}), "
            "which a YAML spec file cannot hold."
        ) from exc


def _path_to(data: Any, target: Any, path: str = "") -> str | None:
    """Where in `data` the object `target` sits, as `columns.c.bounds[0]`."""
    if data is target:
        return path or None
    if isinstance(data, Mapping):
        items = ((f"{path}.{k}" if path else str(k), v) for k, v in data.items())
    elif isinstance(data, (list, tuple)):
        items = ((f"{path}[{i}]", v) for i, v in enumerate(data))
    else:
        return None
    for where, value in items:
        found = _path_to(value, target, where)
        if found is not None:
            return found
    return None


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
    where: str = "spec data",
) -> TableSpec:
    """A `TableSpec` from data in any format version this polspec can read.

    An unknown key is an error, naming the closest known key, unless
    `strict=False`, which downgrades it to a warning. `where` names the data
    at the front of any error -- a file's path, say -- as `from_yaml` names
    the file it read.
    """
    with _Located(where):
        current = migrate(data, "spec", where)
        return tablespec_from_data(current, Ctx(categories=categories, strict=strict))


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


@overload
def to_yaml(spec: SpecLike, path: None = None) -> str: ...
@overload
def to_yaml(spec: SpecLike, path: str | Path) -> None: ...
def to_yaml(spec: SpecLike, path: str | Path | None = None) -> str | None:
    """`spec` -- a `TableSpec` or a `FrameSpec` class -- as human-readable
    YAML: written to the file `path`, or, with no `path`, returned as
    text.

    Defaults are omitted so the file shows only what was declared. Checks and
    validators over raw expressions cannot be written and warn.
    """
    spec = as_table_spec(spec)
    require_columns(spec)
    label = path if path is not None else f"spec {spec.name!r}"
    _warn_unserializable(spec, label, "yaml")
    return _written(dump_yaml(to_dict(spec), label), path)


def _resolve_categories(
    categories: CatSpec | type[CatSpec] | str | Path | None,
    data: Mapping[str, Any],
    base: Path,
) -> CatSpec | None:
    from polspec.catspec import CatSpec, as_catspec

    if categories is not None:
        if isinstance(categories, (str, Path)):
            return CatSpec.from_yaml(categories)
        return as_catspec(categories)
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
    categories: CatSpec | type[CatSpec] | str | Path | None = None,
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
    return from_dict(raw, categories=registry, strict=strict, where=str(source))


# ---------------------------------------------------------------------------
# Python source
# ---------------------------------------------------------------------------


def _class_name_problem(name: str) -> str | None:
    """Why `class <name>` would not declare a class of that very name."""
    if not name.isidentifier():
        return "is not a Python identifier"
    if keyword.iskeyword(name) or name == "__debug__":
        return "is reserved by Python"
    normalized = unicodedata.normalize("NFKC", name)
    if normalized != name:
        return f"would be declared as {normalized!r}, which Python reads it as"
    return None


def _require_class_name(name: str) -> None:
    """A spec's name is the class `to_python` writes, so it must be one
    Python declares under that same name -- or nothing is written."""
    why = _class_name_problem(name)
    if why is None:
        return
    words = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", name))
    suggestion = "".join(w[:1].upper() + w[1:] for w in words) or "Spec"
    if not suggestion[0].isalpha():
        suggestion = f"Spec{suggestion}"
    if _class_name_problem(suggestion) is not None:
        suggestion = "Spec"
    raise SerializationError(
        f"Cannot write spec {name!r} as Python: a spec's name is its class "
        f"name, and this one {why}. Rename it first -- "
        f"spec.with_name({suggestion!r}) -- or write it with to_yaml(), which "
        "keeps any name."
    )


@overload
def to_python(spec: SpecLike, path: None = None) -> str: ...
@overload
def to_python(spec: SpecLike, path: str | Path) -> None: ...
def to_python(spec: SpecLike, path: str | Path | None = None) -> str | None:
    """`spec` -- a `TableSpec` or a `FrameSpec` class -- as a Python module
    defining a `FrameSpec` subclass: written to the file `path`, or, with
    no `path`, returned as text.

    Columns are declared through `__columns__`, since a name straight from
    data is not always a valid identifier. Checks and validators over raw
    expressions cannot be written and warn.

    The spec's own name is the class name, so it must be one: a name that is
    not an identifier, or is a keyword, raises `SerializationError` before
    anything is written. Rename it with `spec.with_name(...)`, or write it
    with `to_yaml`, which keeps any name.
    """
    spec = as_table_spec(spec)
    require_columns(spec)
    _require_class_name(spec.name)
    _warn_unserializable(
        spec, path if path is not None else f"spec {spec.name!r}", "python"
    )

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
    if spec.hierarchy is not None:
        imports.append("Hierarchy")
    if has_rules or has_validators or persistable_checks:
        imports.append("col")

    body = [f"class {spec.name}(FrameSpec):", "    __columns__ = {"]
    for name, cs in spec.columns.items():
        body.append(f"        {name!r}: {colspec_to_source(cs)},")
    body.append("    }")
    if spec.unique_together:
        groups = ", ".join(repr(list(group)) for group in spec.unique_together)
        body.append(f"    __unique_together__ = [{groups}]")
    if spec.foreign_keys:
        fks = ", ".join(fk_to_source(fk) for fk in spec.foreign_keys)
        body.append(f"    __foreign_keys__ = [{fks}]")
    if spec.hierarchy is not None:
        body.append(f"    __hierarchy__ = {hierarchy_to_source(spec.hierarchy)}")
    if spec.order_by:
        body.append(f"    __order_by__ = {list(spec.order_by)!r}")
        if any(spec.descending):
            flags = spec.descending
            written = flags[0] if len(set(flags)) == 1 else list(flags)
            body.append(f"    __descending__ = {written!r}")
    if persistable_checks:
        checks = ", ".join(check_to_source(c) for c in persistable_checks)
        body.append(f"    __checks__ = [{checks}]")

    # The imports follow from the source written, not from a guess at which
    # values will need them: a Decimal choice or a time literal in a rule is
    # written wherever it is, and must be importable wherever that is. An
    # import the text only appears to need (inside a string) costs nothing.
    written = "\n".join(body)
    lines = [f'"""Declares the {spec.name} schema."""', "", "import polars as pl"]
    if "datetime." in written:
        lines.append("import datetime")
    if "Decimal(" in written:
        lines.append("from decimal import Decimal")
    lines.append(f"from polspec import {', '.join(imports)}")
    lines.extend(["", "", *body])
    return _written("\n".join(lines) + "\n", path)


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
            choices = pool_of(catspec, key)
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
    where: str = "registry data",
) -> CatSpec:
    """A `CatSpec` from its data form, the inverse of `catspec_to_dict`.

    An older format version is migrated first. `strict=False` warns about
    an unknown key rather than raising; `where` names the data in errors.
    """
    from polspec.catspec import CatSpec

    with _Located(where):
        current = migrate(data, "catspec", where)
        check_unknown_keys(current, _CATSPEC_KEYS, Ctx(strict=strict), "")
        return CatSpec(
            enums=current.get("enums"),
            categoricals=current.get("categoricals"),
            choices=current.get("choices"),
        )


def catspec_to_yaml(catspec: CatSpec, path: str | Path | None = None) -> str | None:
    """A `CatSpec` as YAML: written to `path` when one is given, else
    returned as text."""
    dumped = dump_yaml(
        {"version": FORMAT_VERSION, **catspec_to_dict(catspec)},
        path if path is not None else "a category registry",
    )
    if path is None:
        return dumped
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumped, encoding="utf-8")
    return None


def catspec_from_yaml(source: str | Path, *, strict: bool = True) -> CatSpec:
    """The `CatSpec` a YAML file holds, as `catspec_to_yaml` writes it."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"CatSpec file not found: {source}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return catspec_from_dict(raw, strict=strict, where=str(source))


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
    where: str = "registry data",
    base: Path | None = None,
) -> Registry:
    """A `Registry` from its data form, the inverse of `registry_to_dict`.

    `categories` may be the categories themselves or the path of a
    category file, resolved against `base` -- which `registry_from_yaml`
    sets to the registry file's directory. `strict` and `where` are as
    for `catspec_from_dict`.
    """
    from polspec.registry import Registry

    current = migrate(data, "registry", where)
    ctx = Ctx(strict=strict)
    check_unknown_keys(current, _REGISTRY_KEYS, ctx, "")

    declared = current.get("categories")
    categories: CatSpec | None = None
    if isinstance(declared, Mapping):
        categories = catspec_from_dict(declared, strict=strict, where=where)
    elif isinstance(declared, (str, Path)):
        if base is None:
            raise SerializationError(
                f"{where}: 'categories' names a file ({declared!r}); read the "
                "registry with registry_from_yaml so the path can be resolved"
            )
        path = Path(declared)
        categories = catspec_from_yaml(path if path.is_absolute() else base / path)
    elif declared is not None:
        raise SerializationError(
            f"{where}: 'categories' must be a mapping or a path, got {declared!r}"
        )

    specs = current.get("specs")
    if not isinstance(specs, Mapping) or not specs:
        raise SerializationError(
            f"{where}: a registry file needs a non-empty 'specs' mapping, keyed "
            "by spec name"
        )
    registry = Registry(categories=categories)
    for name, body in specs.items():
        if not isinstance(body, Mapping):
            raise SerializationError(
                f"{where}: specs.{name} must be a mapping, got {type(body).__name__}"
            )
        if "name" in body and body["name"] != name:
            raise SerializationError(
                f"{where}: specs.{name} carries name {body['name']!r}; the key "
                "is the name, so drop one or make them agree"
            )
        with _Located(f"{where}: specs.{name}"):
            spec = tablespec_from_data(
                {"name": str(name), **body}, Ctx(categories=categories, strict=strict)
            )
        registry.add(spec)
    return registry


@overload
def registry_to_yaml(registry: Registry, path: None = None) -> str: ...
@overload
def registry_to_yaml(registry: Registry, path: str | Path) -> None: ...
def registry_to_yaml(registry: Registry, path: str | Path | None = None) -> str | None:
    """A `Registry` as one YAML document -- its specs keyed by name, and its
    categories: written to the file `path`, or, with no `path`, returned
    as text."""
    label = path if path is not None else "a registry"
    for spec in registry.specs:
        _warn_unserializable(spec, label, "yaml")
    return _written(dump_yaml(registry_to_dict(registry), label), path)


def registry_from_yaml(source: str | Path, *, strict: bool = True) -> Registry:
    """The `Registry` a YAML file holds, as `registry_to_yaml` writes it. A
    `categories:` path is read relative to the file."""
    path = Path(source)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return registry_from_dict(raw, strict=strict, where=str(source), base=path.parent)
