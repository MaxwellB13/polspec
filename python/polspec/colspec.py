"""Backward compatibility module for polspec.colspec.

All core components have been modularized into:
- polspec.bound
- polspec.constants
- polspec.engine
- polspec.framespec
- polspec.profiler
- polspec.rules
- polspec.serialization
- polspec.spec
"""

from __future__ import annotations

from polspec.bound import Bound
from polspec.catspec import CatSpec
from polspec.check import Check
from polspec.constants import (
    _DEFAULT_FLOAT_BOUND,
    _DEFAULT_NULL_PROBABILITY,
    _DEFAULT_STRING_LEN,
    _DEFAULT_WIDE_INT_BOUND,
    _INT_DTYPE_BOUNDS,
    _MAX_CARTESIAN_ROWS,
)
from polspec.engine import (
    _cast_expr,
    _coverage_values,
    _generate_cartesian,
    _generate_random,
    _resolve_numeric_bounds,
    _to_rust_spec,
)
from polspec.framespec import FrameSchema, FrameSpec
from polspec.profiler import profile_dataframe
from polspec.rules import (
    _CONDITION_OPS,
    ColRule,
    _apply_rules,
    _condition_to_expr,
    _sample_choices,
    _validate_condition,
)
from polspec.serialization import (
    _YAML_DTYPES,
    _YAML_NAME_TO_DTYPE,
    _colspec_from_yaml,
    _colspec_to_yaml,
    _dtype_from_yaml,
    _dtype_to_yaml,
)
from polspec.spec import ColSpec, _column_kind
from polspec.validation import ValidationError, _validate_dataframe

__all__ = [
    "_CONDITION_OPS",
    "_DEFAULT_FLOAT_BOUND",
    "_DEFAULT_NULL_PROBABILITY",
    "_DEFAULT_STRING_LEN",
    "_DEFAULT_WIDE_INT_BOUND",
    "_INT_DTYPE_BOUNDS",
    "_MAX_CARTESIAN_ROWS",
    "_YAML_DTYPES",
    "_YAML_NAME_TO_DTYPE",
    "Bound",
    "CatSpec",
    "Check",
    "ColRule",
    "ColSpec",
    "FrameSchema",
    "FrameSpec",
    "ValidationError",
    "_apply_rules",
    "_cast_expr",
    "_colspec_from_yaml",
    "_colspec_to_yaml",
    "_column_kind",
    "_condition_to_expr",
    "_coverage_values",
    "_dtype_from_yaml",
    "_dtype_to_yaml",
    "_generate_cartesian",
    "_generate_random",
    "_resolve_numeric_bounds",
    "_sample_choices",
    "_to_rust_spec",
    "_validate_condition",
    "_validate_dataframe",
    "profile_dataframe",
]
