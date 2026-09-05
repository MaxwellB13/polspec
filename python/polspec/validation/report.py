"""What validation found, as data.

A `Finding` is one violation of one claim: which check, which columns, how
many rows, a few sample values, and, for row-level findings, a way to get
the offending rows back as a frame. A `ValidationReport` is every finding
for one frame against one spec, with the frame kept lazily so nothing is
materialised until asked for.
"""

from __future__ import annotations

import base64
import datetime
import decimal
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from polspec.errors import ValidationError

if TYPE_CHECKING:
    from polspec.validation import ValidationOptions

FindingCode = Literal[
    "extra_columns",
    "missing_columns",
    "dtype",
    "nullability",
    "choices",
    "bounds",
    "string_length",
    "rule",
    "validator",
    "unique",
    "unique_together",
    "check",
    "foreign_key",
    "foreign_key_unresolved",
]

FINDING_COLUMN = "__polspec_finding"


def json_value(value: Any) -> Any:
    """A JSON-serialisable stand-in for any value a finding may carry."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return str(value)
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_value(v) for v in value]
    if isinstance(value, pl.DataType):
        return str(value)
    return repr(value)


@dataclass(frozen=True, slots=True)
class Finding:
    """One violation of one claim the spec makes.

    Attributes
    ----------
    code : FindingCode
        Which kind of claim was violated.
    key : str
        A stable identifier for the claim within its spec, such as
        `"total__bounds"` or `"check:total_covers_subtotal"`.
    message : str
        The human-readable description of what was violated.
    columns : tuple[str, ...]
        The columns involved; empty for structural findings.
    count : int | None
        How many rows violate the claim; `None` for structural findings.
    samples : tuple
        Up to five offending values (or structs of values, for multi-column
        claims).
    details : Mapping
        Code-specific facts: the expected and actual dtype, the observed
        extremes, the foreign key's target.
    """

    code: FindingCode
    key: str
    message: str
    columns: tuple[str, ...] = ()
    count: int | None = None
    samples: tuple[Any, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    _locate: Callable[[pl.LazyFrame], pl.LazyFrame] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @property
    def row_level(self) -> bool:
        """Whether this finding can name the rows that violate it."""
        return self._locate is not None

    def rows(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        """The rows of `frame` that violate this claim, lazily."""
        if self._locate is None:
            raise ValueError(
                f"Finding {self.key!r} ({self.code}) is structural and has no rows"
            )
        return self._locate(frame)

    def to_dict(self) -> dict[str, Any]:
        """This finding as JSON-ready data."""
        return {
            "code": self.code,
            "key": self.key,
            "message": self.message,
            "columns": list(self.columns),
            "count": self.count,
            "samples": json_value(list(self.samples)),
            "details": json_value(dict(self.details)),
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Every finding for one frame against one spec."""

    spec_name: str
    findings: tuple[Finding, ...]
    frame: pl.LazyFrame = field(repr=False, compare=False)
    options: ValidationOptions = field(repr=False, compare=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))

    @property
    def passed(self) -> bool:
        return not self.findings

    def __bool__(self) -> bool:
        return self.passed

    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self):
        return iter(self.findings)

    def raise_if_failed(self) -> None:
        """Raises `ValidationError` carrying this report, if anything was found."""
        if self.findings:
            raise ValidationError(self)

    def by_column(self) -> dict[str, tuple[Finding, ...]]:
        """Findings grouped by column; structural findings under `""`."""
        grouped: dict[str, list[Finding]] = {}
        for finding in self.findings:
            for column in finding.columns or ("",):
                grouped.setdefault(column, []).append(finding)
        return {k: tuple(v) for k, v in grouped.items()}

    def by_code(self, code: FindingCode) -> tuple[Finding, ...]:
        """Every finding of one kind, such as `"bounds"` or `"foreign_key"`."""
        return tuple(f for f in self.findings if f.code == code)

    def rows(self, finding: Finding) -> pl.LazyFrame:
        """The rows violating one finding, lazily."""
        return finding.rows(self.frame)

    def failing_rows(self) -> pl.LazyFrame:
        """Every row that violates a row-level finding, lazily.

        Adds a `__polspec_finding` column naming the finding's key, so a row
        violating several claims appears once per claim.
        """
        parts = [
            f.rows(self.frame).with_columns(pl.lit(f.key).alias(FINDING_COLUMN))
            for f in self.findings
            if f.row_level
        ]
        if not parts:
            return self.frame.clear().with_columns(
                pl.lit(None, dtype=pl.String).alias(FINDING_COLUMN)
            )
        return pl.concat(parts, how="vertical_relaxed")

    def to_dict(self) -> dict[str, Any]:
        """This report as JSON-ready data: the spec, the verdict, the findings."""
        return {
            "spec": self.spec_name,
            "passed": self.passed,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """This report as a JSON string. `indent=None` for one line."""
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:
        if self.passed:
            return f"Validation passed for DataFrame against '{self.spec_name}'"
        return "\n".join(
            [
                f"Validation failed for DataFrame against '{self.spec_name}' "
                f"({len(self.findings)} error(s) found):",
                *(f"  - {f.message}" for f in self.findings),
            ]
        )
