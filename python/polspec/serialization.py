from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import polars as pl

from polspec.constants import _DEFAULT_NULL_PROBABILITY
from polspec.foreign_key import ForeignKey, _default_fk_name
from polspec.rules import ColRule
from polspec.spec import ColSpec, _is_categorical_dtype

if TYPE_CHECKING:
    from polspec.catspec import CatSpec

# Every dtype polspec can generate that has a fixed, unparametrized identity
# -- Enum, Datetime, Duration, and (parametrized) Categorical are handled separately
# since they carry their own metadata.
_YAML_DTYPES: dict[pl.DataType, str] = {
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
_YAML_NAME_TO_DTYPE = {name: dtype for dtype, name in _YAML_DTYPES.items()}


def _dtype_to_yaml(dtype: pl.DataType) -> str | dict:
    if isinstance(dtype, pl.Enum):
        return {"Enum": dtype.categories.to_list()}
    if isinstance(dtype, pl.Datetime):
        res: dict = {"time_unit": dtype.time_unit}
        if dtype.time_zone is not None:
            res["time_zone"] = dtype.time_zone
        return {"Datetime": res}
    if isinstance(dtype, pl.Duration):
        return {"Duration": {"time_unit": dtype.time_unit}}
    if _is_categorical_dtype(dtype):
        if isinstance(dtype, pl.Categorical):
            cats = dtype.categories
            cat_name = cats.name()
            if cat_name:
                # A named pl.Categories() registry: must round-trip by name
                # (and namespace/physical) so specs sharing one -- e.g. two
                # FrameSpecs whose columns are meant to be joined on physical
                # codes -- still share it after a YAML round-trip.
                info: dict = {"name": cat_name}
                namespace = cats.namespace()
                if namespace:
                    info["namespace"] = namespace
                physical = cats.physical()
                if physical != pl.UInt32:
                    info["physical"] = _YAML_DTYPES.get(physical, str(physical))
                return {"Categorical": info}
        return "Categorical"
    name = _YAML_DTYPES.get(dtype)
    if name is None:
        raise TypeError(f"polspec cannot write dtype {dtype!r} to YAML")
    return name


def _strip_registry_prefix(name: str) -> str:
    """`$categories.STATUS` and `categories.STATUS` both name `STATUS`."""
    return name.removeprefix("$categories.").removeprefix("categories.")


def _dtype_from_registry(name: str, categories: CatSpec | None) -> pl.DataType | None:
    """Resolves a name against a CatSpec, as an Enum or a Categorical.

    Returns None when the registry does not hold it, leaving the caller to
    decide whether that is an error or just a name to try elsewhere.
    """
    if categories is None or name not in categories:
        return None
    resolved = categories[name]
    if isinstance(resolved, pl.Categories):
        return pl.Categorical(resolved)
    if isinstance(resolved, (list, tuple)):
        return pl.Enum(resolved)
    return None


def _enum_from_yaml(payload, categories: CatSpec | None) -> pl.DataType:
    if not isinstance(payload, str):
        return pl.Enum(payload)
    name = _strip_registry_prefix(payload)
    if categories is not None and name in categories:
        return pl.Enum(categories.get_enum(name))
    raise ValueError(
        f"Enum {payload!r} referenced in YAML but not found in provided CatSpec"
    )


def _categorical_from_yaml(payload, categories: CatSpec | None) -> pl.DataType:
    if isinstance(payload, str):
        name = _strip_registry_prefix(payload)
        if categories is not None and name in categories:
            return pl.Categorical(categories.get_categorical(name))
        if name == "Categorical":
            return pl.Categorical()
        return pl.Categorical(pl.Categories(name))

    name = payload.get("name", "")
    if categories is not None and name and name in categories.categoricals:
        return pl.Categorical(categories.get_categorical(name))
    return pl.Categorical(
        pl.Categories(
            name,
            namespace=payload.get("namespace", ""),
            physical=_YAML_NAME_TO_DTYPE.get(
                payload.get("physical", "UInt32"), pl.UInt32
            ),
        )
    )


# Parametrized dtypes arrive as a single-key mapping naming the dtype.
_YAML_DTYPE_BUILDERS = {
    "Enum": _enum_from_yaml,
    "Categorical": _categorical_from_yaml,
    "Datetime": lambda payload, _: pl.Datetime(
        time_unit=payload.get("time_unit", "us"), time_zone=payload.get("time_zone")
    ),
    "Duration": lambda payload, _: pl.Duration(
        time_unit=payload.get("time_unit", "us")
    ),
}


def _dtype_from_yaml(
    value: str | dict, categories: CatSpec | None = None
) -> pl.DataType:
    """Reads a dtype back from YAML, resolving registry references as it goes."""
    if isinstance(value, dict):
        for key, build in _YAML_DTYPE_BUILDERS.items():
            if key in value:
                return build(value[key], categories)
        raise ValueError(f"Unrecognized dtype mapping in YAML: {value!r}")

    if value == "Categorical":
        return pl.Categorical()

    if value.startswith(("$categories.", "categories.")):
        dtype = _dtype_from_registry(_strip_registry_prefix(value), categories)
        if dtype is not None:
            return dtype
        raise ValueError(f"Category reference {value!r} not found in provided CatSpec")

    dtype = _YAML_NAME_TO_DTYPE.get(value) or _dtype_from_registry(value, categories)
    if dtype is None:
        raise ValueError(f"Unrecognized dtype name in YAML: {value!r}")
    return dtype


def _colspec_to_yaml(spec: ColSpec) -> dict:
    data: dict = {"dtype": _dtype_to_yaml(spec.dtype)}
    if spec.nullable:
        data["nullable"] = True
    if spec.bounds is not None:
        data["bounds"] = [spec.bounds.min, spec.bounds.max]
    if spec.tags:
        data["tags"] = list(spec.tags) if len(spec.tags) > 1 else spec.tags[0]
    if spec.unique:
        data["unique"] = True
    if spec.null_probability != _DEFAULT_NULL_PROBABILITY:
        data["null_probability"] = spec.null_probability
    if spec.string_length is not None:
        data["string_length"] = [spec.string_length.min, spec.string_length.max]
    if spec.distribution is not None:
        data["distribution"] = spec.distribution
    if spec.distribution_params:
        data["distribution_params"] = dict(spec.distribution_params)
    if spec.choices is not None:
        data["choices"] = list(spec.choices)
    if spec.weights is not None:
        data["weights"] = list(spec.weights)
    if spec.rules:
        rule_list = []
        for rule in spec.rules:
            rule_dict = {"when": dict(rule.when), "choices": list(rule.choices)}
            if rule.weights is not None:
                rule_dict["weights"] = list(rule.weights)
            rule_list.append(rule_dict)
        data["rules"] = rule_list
    return data


def _colspec_from_yaml(data: dict, categories: CatSpec | None = None) -> ColSpec:
    kwargs: dict = {}
    if "nullable" in data:
        kwargs["nullable"] = data["nullable"]
    if "bounds" in data:
        kwargs["bounds"] = tuple(data["bounds"])
    if "tags" in data:
        kwargs["tags"] = data["tags"]
    elif "category" in data:
        kwargs["tags"] = data["category"]
    elif "categories" in data:
        kwargs["tags"] = data["categories"]
    if "unique" in data:
        kwargs["unique"] = data["unique"]
    if "null_probability" in data:
        kwargs["null_probability"] = data["null_probability"]
    if "string_length" in data:
        kwargs["string_length"] = tuple(data["string_length"])
    if "distribution" in data:
        kwargs["distribution"] = data["distribution"]
    if "distribution_params" in data:
        kwargs["distribution_params"] = data["distribution_params"]
    if "choices" in data:
        kwargs["choices"] = data["choices"]
    if "weights" in data:
        kwargs["weights"] = data["weights"]
    if "rules" in data:
        kwargs["rules"] = tuple(
            ColRule(
                when=rule["when"],
                choices=rule["choices"],
                weights=rule.get("weights"),
            )
            for rule in data["rules"]
        )
    return ColSpec(
        dtype=_dtype_from_yaml(data["dtype"], categories=categories), **kwargs
    )


def _foreignkey_to_yaml(fk: ForeignKey) -> dict:
    """Serializes a self-referencing ForeignKey to YAML.

    Only `references="self"` keys are representable: a ForeignKey pointing at
    another FrameSpec class has no stable, round-trippable name for that
    class in a standalone YAML file, so callers must filter those out first
    (see FrameSpec.to_yaml's warning).
    """
    data: dict = {"columns": list(fk.columns), "references": "self"}
    if fk.ref_columns != fk.columns:
        data["ref_columns"] = list(fk.ref_columns)
    if fk.name != _default_fk_name(fk.columns, "self"):
        data["name"] = fk.name
    return data


def _foreignkey_from_yaml(data: dict) -> ForeignKey:
    return ForeignKey(
        columns=data["columns"],
        references=data.get("references", "self"),
        ref_columns=data.get("ref_columns"),
        name=data.get("name"),
    )


# ---------------------------------------------------------------------------
# Python source (FrameSpec.to_python)
# ---------------------------------------------------------------------------
#
# `repr()` alone already renders every value these fields can hold -- str,
# int, float, bool, None, and the tuples/lists/dicts wrapping them -- as
# valid Python source, recursively. `datetime.date`/`datetime`/`time`/
# `timedelta` are the one case where the value under `repr()` names a type
# the generated file must import; `_needs_datetime_import` is only here to
# decide whether that import line belongs in the header.


def _needs_datetime_import(value: object) -> bool:
    if isinstance(value, (datetime.date, datetime.time, datetime.timedelta)):
        return True
    if isinstance(value, dict):
        return any(_needs_datetime_import(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_needs_datetime_import(v) for v in value)
    return False


def _colspec_needs_datetime_import(spec: ColSpec) -> bool:
    values: list = []
    if spec.bounds is not None:
        values += [spec.bounds.min, spec.bounds.max]
    if spec.string_length is not None:
        values += [spec.string_length.min, spec.string_length.max]
    if spec.choices is not None:
        values += list(spec.choices)
    for rule in spec.rules:
        values += list(rule.choices)
        values += list(rule.when.values())
    return _needs_datetime_import(values)


def _dtype_to_python(dtype: pl.DataType) -> str:
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
        if isinstance(dtype, pl.Categorical):
            cats = dtype.categories
            cat_name = cats.name()
            if cat_name:
                # A named pl.Categories() registry -- see _dtype_to_yaml.
                parts = [repr(cat_name)]
                namespace = cats.namespace()
                if namespace:
                    parts.append(f"namespace={namespace!r}")
                physical = cats.physical()
                if physical != pl.UInt32:
                    phys_name = _YAML_DTYPES.get(physical, str(physical))
                    parts.append(f"physical=pl.{phys_name}")
                return f"pl.Categorical(pl.Categories({', '.join(parts)}))"
        return "pl.Categorical"
    name = _YAML_DTYPES.get(dtype)
    if name is None:
        raise TypeError(f"polspec cannot write dtype {dtype!r} to Python")
    return f"pl.{name}"


def _colrule_to_python(rule: ColRule) -> str:
    args = [f"when={dict(rule.when)!r}", f"choices={list(rule.choices)!r}"]
    if rule.weights is not None:
        args.append(f"weights={list(rule.weights)!r}")
    return f"ColRule({', '.join(args)})"


def _colspec_to_python(spec: ColSpec) -> str:
    """Python source for one column's `ColSpec(...)` call.

    Mirrors `_colspec_to_yaml` field-for-field, but the dtype and every
    argument are rendered as Python source rather than YAML-safe data, since
    the destination is a `.py` file, not a document `from_yaml` will parse.
    """
    args = [_dtype_to_python(spec.dtype)]
    if spec.nullable:
        args.append("nullable=True")
    if spec.bounds is not None:
        args.append(f"bounds=({spec.bounds.min!r}, {spec.bounds.max!r})")
    if spec.tags:
        args.append(
            f"tags={spec.tags[0]!r}"
            if len(spec.tags) == 1
            else f"tags={list(spec.tags)!r}"
        )
    if spec.unique:
        args.append("unique=True")
    if spec.null_probability != _DEFAULT_NULL_PROBABILITY:
        args.append(f"null_probability={spec.null_probability!r}")
    if spec.string_length is not None:
        args.append(
            f"string_length=({spec.string_length.min!r}, {spec.string_length.max!r})"
        )
    if spec.distribution is not None:
        args.append(f"distribution={spec.distribution!r}")
    if spec.distribution_params:
        args.append(f"distribution_params={dict(spec.distribution_params)!r}")
    if spec.choices is not None:
        args.append(f"choices={list(spec.choices)!r}")
    if spec.weights is not None:
        args.append(f"weights={list(spec.weights)!r}")
    if spec.rules:
        rules_src = ", ".join(_colrule_to_python(rule) for rule in spec.rules)
        args.append(f"rules=[{rules_src}]")
    return f"ColSpec({', '.join(args)})"


def _foreignkey_to_python(fk: ForeignKey) -> str:
    """Python source for a self-referencing ForeignKey -- see `_foreignkey_to_yaml`."""
    args = [f"columns={list(fk.columns)!r}", "references='self'"]
    if fk.ref_columns != fk.columns:
        args.append(f"ref_columns={list(fk.ref_columns)!r}")
    if fk.name != _default_fk_name(fk.columns, "self"):
        args.append(f"name={fk.name!r}")
    return f"ForeignKey({', '.join(args)})"
