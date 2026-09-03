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
   written with `col()`; unknown keys are an error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from polspec.distributions import canonicalize_params, normalize_distribution
from polspec.errors import SerializationError

FORMAT_VERSION = 2


def _spec_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    from polspec.rules import _condition_to_pred

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


MIGRATIONS: dict[str, dict[int, Callable[[dict[str, Any]], dict[str, Any]]]] = {
    "spec": {1: _spec_v1_to_v2},
    "catspec": {1: _catspec_v1_to_v2},
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
