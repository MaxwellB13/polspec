from __future__ import annotations

from typing import TYPE_CHECKING, Any

# The declarations load with the package: they are what every use of
# polspec starts from, and cost little. Each verb -- generating,
# validating, drift, profiling, spec files, rendering -- loads on first use,
# through `__getattr__` below, so `import polspec` does not pay for what a
# caller never asks for. A type checker reads the imports as written.
from polspec.bound import Bound
from polspec.catspec import CatSpec
from polspec.check import Check
from polspec.errors import (
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
from polspec.hierarchy import Hierarchy
from polspec.rules import ColRule
from polspec.spec import ColSpec
from polspec.tablespec import TableSpec

if TYPE_CHECKING:
    from polspec.asserting import assert_valid, check_pipeline
    from polspec.compare import (
        DriftCode,
        DriftFinding,
        DriftOptions,
        DriftReport,
        Severity,
        diff,
        drift,
    )
    from polspec.generation import (
        generate,
        generate_batches,
        scan,
        sink_csv,
        sink_ipc,
        sink_ndjson,
        sink_parquet,
    )
    from polspec.profiler import ProfileOptions
    from polspec.reading import read
    from polspec.registry import Registry
    from polspec.related import profile_all, synthesize_all
    from polspec.render import to_markdown, to_mermaid
    from polspec.serialization import (
        FORMAT_VERSION,
        from_dict,
        from_yaml,
        to_dict,
        to_python,
        to_yaml,
    )
    from polspec.sizing import estimated_size
    from polspec.strategies import strategy
    from polspec.synthesis import profile, synthesize
    from polspec.validation import (
        FINDING_COLUMN,
        Finding,
        FindingCode,
        ValidationOptions,
        ValidationReport,
        inspect,
        validate,
    )

    __version__: str

# Where each name that loads on first use lives.
_LAZY: dict[str, str] = {
    "assert_valid": "polspec.asserting",
    "check_pipeline": "polspec.asserting",
    **dict.fromkeys(
        ("DriftCode", "DriftFinding", "DriftOptions", "DriftReport", "Severity"),
        "polspec.compare",
    ),
    "diff": "polspec.compare",
    "drift": "polspec.compare",
    **dict.fromkeys(
        (
            "generate",
            "generate_batches",
            "scan",
            "sink_csv",
            "sink_ipc",
            "sink_ndjson",
            "sink_parquet",
        ),
        "polspec.generation",
    ),
    "ProfileOptions": "polspec.profiler",
    "read": "polspec.reading",
    "profile_all": "polspec.related",
    "synthesize_all": "polspec.related",
    "Registry": "polspec.registry",
    "to_markdown": "polspec.render",
    "to_mermaid": "polspec.render",
    **dict.fromkeys(
        ("FORMAT_VERSION", "from_dict", "from_yaml", "to_dict", "to_python", "to_yaml"),
        "polspec.serialization",
    ),
    "estimated_size": "polspec.sizing",
    "strategy": "polspec.strategies",
    "profile": "polspec.synthesis",
    "synthesize": "polspec.synthesis",
    **dict.fromkeys(
        (
            "FINDING_COLUMN",
            "Finding",
            "FindingCode",
            "ValidationOptions",
            "ValidationReport",
            "inspect",
            "validate",
        ),
        "polspec.validation",
    ),
}


def __getattr__(name: str) -> Any:
    """A public name that loads on first use, imported once and kept."""
    if name == "__version__":
        value: Any = _version()
    elif name in _LAZY:
        from importlib import import_module

        value = getattr(import_module(_LAZY[name]), name)
    else:
        raise AttributeError(f"module 'polspec' has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY, "__version__"})


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("polspec")
    except PackageNotFoundError:
        # Run from a source checkout that was never installed -- the docs
        # build imports it that way, to read the docstrings -- there is no
        # version to report.
        return "0+unknown"


__all__ = [
    "FINDING_COLUMN",
    "FORMAT_VERSION",
    "Bound",
    "CatSpec",
    "Check",
    "ColRule",
    "ColSpec",
    "DriftCode",
    "DriftFinding",
    "DriftOptions",
    "DriftReport",
    "Finding",
    "FindingCode",
    "ForeignKey",
    "FrameSpec",
    "GenerationError",
    "Hierarchy",
    "MultiValidationError",
    "PolspecError",
    "Pred",
    "ProfileOptions",
    "Registry",
    "RegistryError",
    "SerializationError",
    "Severity",
    "SpecError",
    "TableSpec",
    "ValidationError",
    "ValidationOptions",
    "ValidationReport",
    "assert_valid",
    "check_pipeline",
    "col",
    "diff",
    "drift",
    "estimated_size",
    "from_dict",
    "from_yaml",
    "generate",
    "generate_batches",
    "inspect",
    "lit",
    "profile",
    "profile_all",
    "read",
    "scan",
    "sink_csv",
    "sink_ipc",
    "sink_ndjson",
    "sink_parquet",
    "strategy",
    "synthesize",
    "synthesize_all",
    "to_dict",
    "to_markdown",
    "to_mermaid",
    "to_python",
    "to_yaml",
    "validate",
]
