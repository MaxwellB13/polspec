from __future__ import annotations

from typing import TYPE_CHECKING, Any
import polars as pl

from polspec.constants import _DEFAULT_NULL_PROBABILITY
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


def _dtype_from_yaml(
    value: str | dict, categories: CatSpec | None = None
) -> pl.DataType:
    if isinstance(value, dict):
        if "Enum" in value:
            enum_val = value["Enum"]
            if isinstance(enum_val, str):
                clean_name = enum_val.removeprefix("$categories.").removeprefix("categories.")
                if categories is not None and clean_name in categories:
                    return pl.Enum(categories.get_enum(clean_name))
                raise ValueError(
                    f"Enum {enum_val!r} referenced in YAML but not found in provided CatSpec"
                )
            return pl.Enum(enum_val)
        if "Datetime" in value:
            dt_info = value["Datetime"]
            return pl.Datetime(
                time_unit=dt_info.get("time_unit", "us"),
                time_zone=dt_info.get("time_zone"),
            )
        if "Duration" in value:
            dur_info = value["Duration"]
            return pl.Duration(time_unit=dur_info.get("time_unit", "us"))
        if "Categorical" in value:
            cat_info = value["Categorical"]
            if isinstance(cat_info, str):
                clean_name = cat_info.removeprefix("$categories.").removeprefix("categories.")
                if categories is not None and clean_name in categories:
                    return pl.Categorical(categories.get_categorical(clean_name))
                if clean_name == "Categorical":
                    return pl.Categorical()
                return pl.Categorical(pl.Categories(clean_name))
            if isinstance(cat_info, dict):
                cat_name = cat_info.get("name", "")
                if categories is not None and cat_name and cat_name in categories.categoricals:
                    return pl.Categorical(categories.get_categorical(cat_name))
                physical = _YAML_NAME_TO_DTYPE.get(cat_info.get("physical", "UInt32"), pl.UInt32)
                return pl.Categorical(
                    pl.Categories(
                        cat_name,
                        namespace=cat_info.get("namespace", ""),
                        physical=physical,
                    )
                )
        raise ValueError(f"Unrecognized dtype mapping in YAML: {value!r}")
    if value == "Categorical":
        return pl.Categorical()
    if value.startswith("$categories.") or value.startswith("categories."):
        clean_name = value.removeprefix("$categories.").removeprefix("categories.")
        if categories is not None and clean_name in categories:
            resolved = categories[clean_name]
            if isinstance(resolved, pl.Categories):
                return pl.Categorical(resolved)
            if isinstance(resolved, (list, tuple)):
                return pl.Enum(resolved)
        raise ValueError(f"Category reference {value!r} not found in provided CatSpec")
    dtype = _YAML_NAME_TO_DTYPE.get(value)
    if dtype is None:
        if categories is not None and value in categories:
            resolved = categories[value]
            if isinstance(resolved, pl.Categories):
                return pl.Categorical(resolved)
            if isinstance(resolved, (list, tuple)):
                return pl.Enum(resolved)
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


def _colspec_from_yaml(
    data: dict, categories: CatSpec | None = None
) -> ColSpec:
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
