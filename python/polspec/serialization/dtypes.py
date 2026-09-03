"""One codec per dtype: its YAML data form, its Python source, and back.

Plain dtypes are a name (`Int64`). Parametrized ones are a one-key mapping
naming the dtype (`{Enum: [...]}`, `{Datetime: {time_unit: us}}`), and a
`Categorical` backed by a named `pl.Categories` registry writes its name,
namespace and physical dtype so two specs sharing it still share it after a
round-trip. Registry references (`$categories.STATUS`) resolve against the
`CatSpec` handed to the reader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from polspec.errors import SerializationError
from polspec.spec import _is_categorical_dtype

if TYPE_CHECKING:
    from polspec.catspec import CatSpec

# Every dtype with a fixed, unparametrized identity.
DTYPE_NAMES: dict[pl.DataType, str] = {
    pl.String: "String",
    pl.Boolean: "Boolean",
    pl.Int8: "Int8",
    pl.Int16: "Int16",
    pl.Int32: "Int32",
    pl.Int64: "Int64",
    pl.UInt8: "UInt8",
    pl.UInt16: "UInt16",
    pl.UInt32: "UInt32",
    pl.UInt64: "UInt64",
    pl.Float32: "Float32",
    pl.Float64: "Float64",
    pl.Date: "Date",
    pl.Time: "Time",
    pl.Datetime: "Datetime",
    pl.Duration: "Duration",
    pl.Binary: "Binary",
}
NAME_TO_DTYPE: dict[str, pl.DataType] = {name: dt for dt, name in DTYPE_NAMES.items()}

_DEFAULT_PHYSICAL = pl.UInt32


def physical_name(dtype: pl.DataType) -> str:
    """The name of a Categorical's physical dtype."""
    name = DTYPE_NAMES.get(dtype)
    if name is None:
        raise SerializationError(f"{dtype!r} is not a dtype a Categorical can use")
    return name


def physical_from_name(name: object) -> pl.DataType:
    """The physical dtype a Categorical entry names; unknown names are an error."""
    if isinstance(name, pl.DataType):
        return name
    dtype = NAME_TO_DTYPE.get(str(name))
    if dtype is None or not (dtype.is_integer()):
        raise SerializationError(
            f"Unknown physical dtype {name!r} for a Categorical; expected one of "
            f"{sorted(n for d, n in DTYPE_NAMES.items() if d.is_integer())}"
        )
    return dtype


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _categories_info(cats: pl.Categories) -> dict[str, Any]:
    info: dict[str, Any] = {"name": cats.name()}
    namespace = cats.namespace()
    if namespace:
        info["namespace"] = namespace
    physical = cats.physical()
    if physical != _DEFAULT_PHYSICAL:
        info["physical"] = physical_name(physical)
    return info


def dtype_to_data(dtype: pl.DataType) -> str | dict[str, Any]:
    if isinstance(dtype, pl.Enum):
        return {"Enum": dtype.categories.to_list()}
    if isinstance(dtype, pl.Datetime):
        info: dict[str, Any] = {"time_unit": dtype.time_unit}
        if dtype.time_zone is not None:
            info["time_zone"] = dtype.time_zone
        return {"Datetime": info}
    if isinstance(dtype, pl.Duration):
        return {"Duration": {"time_unit": dtype.time_unit}}
    if _is_categorical_dtype(dtype):
        if isinstance(dtype, pl.Categorical) and dtype.categories.name():
            return {"Categorical": _categories_info(dtype.categories)}
        return "Categorical"
    name = DTYPE_NAMES.get(dtype)
    if name is None:
        raise SerializationError(f"polspec cannot write dtype {dtype!r} to a spec file")
    return name


def dtype_to_source(dtype: pl.DataType) -> str:
    if isinstance(dtype, pl.Enum):
        return f"pl.Enum({dtype.categories.to_list()!r})"
    if isinstance(dtype, pl.Datetime):
        parts = [f"time_unit={dtype.time_unit!r}"]
        if dtype.time_zone is not None:
            parts.append(f"time_zone={dtype.time_zone!r}")
        return f"pl.Datetime({', '.join(parts)})"
    if isinstance(dtype, pl.Duration):
        return f"pl.Duration(time_unit={dtype.time_unit!r})"
    if _is_categorical_dtype(dtype):
        if isinstance(dtype, pl.Categorical) and dtype.categories.name():
            cats = dtype.categories
            parts = [repr(cats.name())]
            if cats.namespace():
                parts.append(f"namespace={cats.namespace()!r}")
            if cats.physical() != _DEFAULT_PHYSICAL:
                parts.append(f"physical=pl.{physical_name(cats.physical())}")
            return f"pl.Categorical(pl.Categories({', '.join(parts)}))"
        return "pl.Categorical"
    name = DTYPE_NAMES.get(dtype)
    if name is None:
        raise SerializationError(f"polspec cannot write dtype {dtype!r} to Python")
    return f"pl.{name}"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _strip_registry_prefix(name: str) -> str:
    """`$categories.STATUS` and `categories.STATUS` both name `STATUS`."""
    return name.removeprefix("$categories.").removeprefix("categories.")


def _from_registry(name: str, categories: CatSpec | None) -> pl.DataType | None:
    """Resolves a name against a CatSpec, as an Enum or a Categorical."""
    if categories is None or name not in categories:
        return None
    resolved = categories[name]
    if isinstance(resolved, pl.Categories):
        return pl.Categorical(resolved)
    if isinstance(resolved, (list, tuple)):
        return pl.Enum(resolved)
    return None


def _enum_from_data(payload: Any, categories: CatSpec | None) -> pl.DataType:
    if not isinstance(payload, str):
        return pl.Enum(payload)
    name = _strip_registry_prefix(payload)
    if categories is not None and name in categories:
        return pl.Enum(categories.get_enum(name))
    raise SerializationError(
        f"Enum {payload!r} referenced in YAML but not found in provided CatSpec"
    )


def categories_from_data(
    payload: Any, categories: CatSpec | None, key: str = ""
) -> pl.Categories:
    """A `pl.Categories` from its data form: a registry name, or a mapping."""
    if isinstance(payload, str):
        name = _strip_registry_prefix(payload)
        if categories is not None and name in categories:
            return categories.get_categorical(name)
        return pl.Categories(name)
    if not isinstance(payload, dict):
        raise SerializationError(
            f"A Categorical entry must be a name or a mapping, got {payload!r}"
        )
    name = str(payload.get("name", key))
    if categories is not None and name and name in categories.categoricals:
        return categories.get_categorical(name)
    return pl.Categories(
        name,
        namespace=str(payload.get("namespace", "")),
        physical=physical_from_name(payload.get("physical", "UInt32")),
    )


def _categorical_from_data(payload: Any, categories: CatSpec | None) -> pl.DataType:
    if payload == "Categorical":
        return pl.Categorical()
    return pl.Categorical(categories_from_data(payload, categories))


_BUILDERS = {
    "Enum": _enum_from_data,
    "Categorical": _categorical_from_data,
    "Datetime": lambda payload, _: pl.Datetime(
        time_unit=payload.get("time_unit", "us"), time_zone=payload.get("time_zone")
    ),
    "Duration": lambda payload, _: pl.Duration(
        time_unit=payload.get("time_unit", "us")
    ),
}


def dtype_from_data(value: Any, categories: CatSpec | None = None) -> pl.DataType:
    """Reads a dtype back, resolving registry references as it goes."""
    if isinstance(value, dict):
        if len(value) == 1:
            ((key, payload),) = value.items()
            build = _BUILDERS.get(key)
            if build is not None:
                return build(payload, categories)
        raise SerializationError(f"Unrecognized dtype mapping in YAML: {value!r}")
    if not isinstance(value, str):
        raise SerializationError(f"A dtype must be a name or a mapping, got {value!r}")
    if value == "Categorical":
        return pl.Categorical()
    if value.startswith(("$categories.", "categories.")):
        dtype = _from_registry(_strip_registry_prefix(value), categories)
        if dtype is not None:
            return dtype
        raise SerializationError(
            f"Category reference {value!r} not found in provided CatSpec"
        )
    dtype = NAME_TO_DTYPE.get(value) or _from_registry(value, categories)
    if dtype is None:
        raise SerializationError(f"Unrecognized dtype name in YAML: {value!r}")
    return dtype
