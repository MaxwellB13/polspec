"""Spec-file format versions, and how an older file becomes the current one.

A file records the version that wrote it under `version:`. A file with no
such key predates the key, and is version 1. Each migration turns one
version's raw data into the next version's, before any of it is decoded, so
readers only ever see the current shape.

Version history
---------------
1. The original format: a column's tags could also be spelt `category` or
   `categories`; a rule's `when` was a one-column dict; distribution
   parameters used whichever alias the author typed; only self-referencing
   foreign keys were written; checks and validators were not written.
2. `version:` recorded; tags spelt `tags`; rule conditions and checks in the
   predicate data form; distribution parameters canonical; every foreign key
   written with its target's name; `checks:` and `validators:` present when
   written with `col()`; unknown keys are an error. Registry files
   (`specs:` keyed by name, plus `categories:`) appear.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from polspec.distributions import canonicalize_params, normalize_distribution
from polspec.errors import SerializationError
from polspec.expr import Pred, col

FORMAT_VERSION = 2


# ---------------------------------------------------------------------------
# Version 1 vocabulary
#
# A v1 file wrote a rule's `when` as a one-column dict. `ColRule` used to
# accept the same shape directly; it no longer does, so this is the only
# place that understands it -- reading a file old enough to contain one.
# ---------------------------------------------------------------------------

_CONDITION_OPS = (
    "equals",
    "not_equals",
    "in",
    "not_in",
    "lt",
    "lte",
    "gt",
    "gte",
    "between",
    "is_null",
    "is_not_null",
)


def _validate_condition(condition: dict) -> None:
    if not isinstance(condition, dict) or "column" not in condition:
        raise SerializationError(
            "A version 1 rule condition must be a dict like "
            "{'column': 'enum_1', 'in': ['A', 'B']} "
            f"(supported keys: {', '.join(_CONDITION_OPS)})"
        )
    ops_present = [op for op in _CONDITION_OPS if op in condition]
    if len(ops_present) != 1:
        raise SerializationError(
            f"A version 1 rule condition on column {condition['column']!r} must "
            f"have exactly one of {_CONDITION_OPS}, got {ops_present}"
        )
    if "between" in condition:
        b = condition["between"]
        if not (isinstance(b, (list, tuple)) and len(b) == 2 and b[0] <= b[1]):
            raise SerializationError(
                "A version 1 'between' condition requires a 2-element sequence "
                f"[min, max] where min <= max, got {b!r}"
            )
    for op in ("in", "not_in"):
        if op in condition and not isinstance(condition[op], (list, tuple, set)):
            raise SerializationError(
                f"A version 1 {op!r} condition requires a collection, got "
                f"{type(condition[op]).__name__}"
            )


def _condition_to_pred(condition: dict) -> Pred:
    """The predicate a version 1 `{"column": ..., <op>: ...}` condition means."""
    _validate_condition(condition)
    column = col(condition["column"])
    if "equals" in condition:
        return column == condition["equals"]
    if "not_equals" in condition:
        return column != condition["not_equals"]
    if "in" in condition:
        return column.is_in(list(condition["in"]))
    if "not_in" in condition:
        return ~column.is_in(list(condition["not_in"]))
    if "lt" in condition:
        return column < condition["lt"]
    if "lte" in condition:
        return column <= condition["lte"]
    if "gt" in condition:
        return column > condition["gt"]
    if "gte" in condition:
        return column >= condition["gte"]
    if "between" in condition:
        lo, hi = condition["between"]
        return column.is_between(lo, hi)
    if "is_null" in condition:
        return column.is_null() if condition["is_null"] else column.is_not_null()
    if "is_not_null" in condition:
        return column.is_not_null() if condition["is_not_null"] else column.is_null()
    raise SerializationError(f"Unrecognized version 1 condition: {condition}")


def _spec_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    columns = out.get("columns")
    if isinstance(columns, Mapping):
        new_columns: dict[str, Any] = {}
        for name, col in columns.items():
            if not isinstance(col, Mapping):
                new_columns[name] = col
                continue
            col = dict(col)
            for legacy in ("category", "categories"):
                if legacy in col and "tags" not in col:
                    col["tags"] = col.pop(legacy)
                else:
                    col.pop(legacy, None)
            if "distribution" in col and isinstance(col["distribution"], str):
                try:
                    canonical = normalize_distribution(col["distribution"])
                except Exception:  # noqa: BLE001 - leave an unknown name for the decoder to report
                    canonical = None
                if canonical is not None:
                    col["distribution"] = canonical
                    params = col.get("distribution_params")
                    if isinstance(params, Mapping):
                        col["distribution_params"] = canonicalize_params(
                            canonical, {str(k): v for k, v in params.items()}
                        )
            rules = col.get("rules")
            if isinstance(rules, list):
                new_rules = []
                for rule in rules:
                    if isinstance(rule, Mapping) and isinstance(
                        rule.get("when"), Mapping
                    ):
                        when = rule["when"]
                        if "column" in when:
                            rule = {
                                **rule,
                                "when": _condition_to_pred(dict(when)).to_data(),
                            }
                    new_rules.append(rule)
                col["rules"] = new_rules
            new_columns[name] = col
        out["columns"] = new_columns
    return out


def _catspec_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Version 1 also allowed a flat `{NAME: [...] | {...}}` mapping."""
    if "enums" in data or "categoricals" in data or "choices" in data or not data:
        return dict(data)
    enums: dict[str, Any] = {}
    categoricals: dict[str, Any] = {}
    for key, value in data.items():
        if key == "version":
            continue
        if isinstance(value, (list, tuple)):
            enums[key] = list(value)
        elif isinstance(value, (dict, str)):
            categoricals[key] = value
        else:
            raise SerializationError(
                f"Cannot read registry entry {key!r}: expected a list of variants or "
                f"a categorical mapping, got {value!r}"
            )
    out: dict[str, Any] = {}
    if enums:
        out["enums"] = enums
    if categoricals:
        out["categoricals"] = categoricals
    return out


def _registry_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    # Registry files did not exist before version 2; one without a version
    # key is a hand-written current-format file.
    return dict(data)


MIGRATIONS: dict[str, dict[int, Callable[[dict[str, Any]], dict[str, Any]]]] = {
    "spec": {1: _spec_v1_to_v2},
    "catspec": {1: _catspec_v1_to_v2},
    "registry": {1: _registry_v1_to_v2},
}


def migrate(data: Any, kind: str, source: str) -> dict[str, Any]:
    """Brings a file's raw data up to `FORMAT_VERSION`, or refuses it."""
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise SerializationError(
            f"{source}: expected a mapping at the top level, got {type(data).__name__}"
        )
    version = data.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise SerializationError(
            f"{source}: 'version' must be a positive integer, got {version!r}"
        )
    if version > FORMAT_VERSION:
        raise SerializationError(
            f"{source} was written by a newer polspec (format version {version}); "
            f"this version reads up to {FORMAT_VERSION}. Upgrade polspec to read it."
        )
    current = dict(data)
    for step in range(version, FORMAT_VERSION):
        current = MIGRATIONS[kind][step](current)
    current["version"] = FORMAT_VERSION
    return current
