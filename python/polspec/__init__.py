from __future__ import annotations

from importlib.metadata import version as _version

from polspec.bound import Bound
from polspec.catspec import CatSpec
from polspec.check import Check
from polspec.drift import DriftFinding, DriftOptions, DriftReport
from polspec.errors import (
    CliError,
    GenerationError,
    MultiValidationError,
    PolspecError,
    RegistryError,
    SerializationError,
    SpecError,
    ValidationError,
)
from polspec.expr import Pred, col
from polspec.foreign_key import ForeignKey
from polspec.framespec import FrameSpec
from polspec.generation import (
    generate,
    generate_batches,
    scan,
    sink_csv,
    sink_ipc,
    sink_ndjson,
    sink_parquet,
)
from polspec.hierarchy import Hierarchy
from polspec.profiler import profile_dataframe
from polspec.registry import Registry
from polspec.rules import ColRule
from polspec.spec import ColSpec
from polspec.tablespec import TableSpec
from polspec.validation import (
    Finding,
    ValidationOptions,
    ValidationReport,
    inspect,
    validate,
)

__version__ = _version("polspec")

__all__ = [
    "Bound",
    "CatSpec",
    "Check",
    "CliError",
    "ColRule",
    "ColSpec",
    "DriftFinding",
    "DriftOptions",
    "DriftReport",
    "Finding",
    "ForeignKey",
    "FrameSpec",
    "GenerationError",
    "Hierarchy",
    "MultiValidationError",
    "PolspecError",
    "Pred",
    "Registry",
    "RegistryError",
    "SerializationError",
    "SpecError",
    "TableSpec",
    "ValidationError",
    "ValidationOptions",
    "ValidationReport",
    "col",
    "generate",
    "generate_batches",
    "inspect",
    "profile_dataframe",
    "scan",
    "sink_csv",
    "sink_ipc",
    "sink_ndjson",
    "sink_parquet",
    "validate",
]
