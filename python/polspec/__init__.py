from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
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
from polspec.expr import Pred, col, lit
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
from polspec.reading import read
from polspec.registry import Registry
from polspec.render import to_markdown, to_mermaid
from polspec.rules import ColRule
from polspec.serialization import from_yaml, to_python, to_yaml
from polspec.sizing import estimated_size
from polspec.spec import ColSpec
from polspec.synthesis import profile, synthesize
from polspec.tablespec import TableSpec
from polspec.validation import (
    Finding,
    ValidationOptions,
    ValidationReport,
    inspect,
    validate,
)

try:
    __version__ = _version("polspec")
except _PackageNotFoundError:
    # Run from a source checkout that was never installed -- the docs build
    # imports it that way, to read the docstrings -- there is no version to
    # report.
    __version__ = "0+unknown"

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
    "estimated_size",
    "from_yaml",
    "generate",
    "generate_batches",
    "inspect",
    "lit",
    "profile",
    "profile_dataframe",
    "read",
    "scan",
    "sink_csv",
    "sink_ipc",
    "sink_ndjson",
    "sink_parquet",
    "synthesize",
    "to_markdown",
    "to_mermaid",
    "to_python",
    "to_yaml",
    "validate",
]
