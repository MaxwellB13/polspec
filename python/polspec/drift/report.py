"""What a comparison found, as data.

A `DriftFinding` is one difference: which kind, which column, whether it
is breaking, and the facts that make it actionable. A `DriftReport` is every
difference between two specs, or between a spec and a frame. The shape
mirrors `polspec.validation.report` on purpose -- same field names, same
`to_dict` -- so the two kinds of report can be read, rendered and gated
the same way. What it leaves out is the frame: a drift report is a
statement about declarations and summary statistics, not about rows.

Severity is one rule, stated once: a finding is *breaking* when a frame
that satisfied the old declaration could fail the new one -- or, against
data, when this frame fails this spec. Everything else is *compatible*.
Direction lives in the code (`domain_widened`, `domain_narrowed`), so a
second axis is not needed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from polspec.validation.report import json_value

if TYPE_CHECKING:
    from pathlib import Path

    from polspec.drift import DriftOptions

DriftCode = Literal[
    "column_added",
    "column_removed",
    "column_renamed",
    "dtype_changed",
    "nullability_changed",
    "domain_widened",
    "domain_narrowed",
    "domain_changed",
    "bounds_exceeded",
    "new_values",
    "format_violated",
    "null_rate_moved",
    "cardinality_moved",
    "constraint_added",
    "constraint_removed",
    "field_changed",
]

Severity = Literal["breaking", "compatible"]
Kind = Literal["diff", "drift"]


@dataclass(frozen=True, slots=True)
class DriftFinding:
    """One difference between two declarations, or a declaration and data.

    Attributes
    ----------
    code : DriftCode
        Which kind of difference.
    severity : "breaking" | "compatible"
        Breaking when data that satisfied the old side could fail the new
        one; compatible otherwise.
    key : str
        A stable identifier within the report, such as `"total__bounds"` or
        `"check:total_covers_subtotal"`.
    message : str
        What changed, naming the column, and what to do about it.
    columns : tuple[str, ...]
        The columns involved; empty for table-level differences.
    details : Mapping
        Code-specific facts: the old and new value, how far a bound was
        exceeded, which values were new.
    """

    code: DriftCode
    severity: Severity
    key: str
    message: str
    columns: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @property
    def breaking(self) -> bool:
        return self.severity == "breaking"

    def to_dict(self) -> dict[str, Any]:
        """This finding as JSON-ready data."""
        return {
            "code": self.code,
            "severity": self.severity,
            "key": self.key,
            "message": self.message,
            "columns": list(self.columns),
            "details": json_value(dict(self.details)),
        }


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Every difference between two specs, or between a spec and a frame.

    `kind` says which: `"diff"` compares `old` to `new`, both declarations;
    `"drift"` compares the declaration `old` to data, and `new` is what the
    data was called. `bool(report)` is `report.unchanged`, the way
    `bool(ValidationReport)` is `passed`.
    """

    kind: Kind
    old: str
    new: str
    findings: tuple[DriftFinding, ...]
    options: DriftOptions | None = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", tuple(self.findings))

    @property
    def unchanged(self) -> bool:
        """True when nothing differs."""
        return not self.findings

    @property
    def breaking(self) -> tuple[DriftFinding, ...]:
        """The findings a CI gate should fail on."""
        return tuple(f for f in self.findings if f.breaking)

    @property
    def compatible(self) -> tuple[DriftFinding, ...]:
        return tuple(f for f in self.findings if not f.breaking)

    def __bool__(self) -> bool:
        return self.unchanged

    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self):
        return iter(self.findings)

    def by_column(self) -> dict[str, tuple[DriftFinding, ...]]:
        """Findings grouped by column; table-level findings under `""`."""
        grouped: dict[str, list[DriftFinding]] = {}
        for finding in self.findings:
            for column in finding.columns or ("",):
                grouped.setdefault(column, []).append(finding)
        return {k: tuple(v) for k, v in grouped.items()}

    def by_code(self, code: DriftCode) -> tuple[DriftFinding, ...]:
        """Every finding of one kind, such as `"domain_narrowed"`."""
        return tuple(f for f in self.findings if f.code == code)

    def to_dict(self) -> dict[str, Any]:
        """This report as JSON-ready data."""
        return {
            "kind": self.kind,
            "old": self.old,
            "new": self.new,
            "unchanged": self.unchanged,
            "breaking": len(self.breaking),
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """This report as a JSON string. `indent=None` for one line."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self, path: str | Path | None = None) -> str:
        """This report as Markdown, written to `path` if given."""
        from polspec.report import drift_to_markdown  # rendering lives in one module

        return drift_to_markdown(self, path)

    def _headline(self) -> str:
        against = f"'{self.old}' and '{self.new}'"
        if self.kind == "drift":
            against = f"{self.new} against '{self.old}'"
        if self.unchanged:
            return f"No drift: {against}"
        return (
            f"Drift: {len(self.breaking)} breaking, {len(self.compatible)} "
            f"compatible, {against}"
        )

    def __str__(self) -> str:
        if self.unchanged:
            return self._headline()
        ordered = (*self.breaking, *self.compatible)
        return "\n".join(
            [self._headline(), *(f"  - [{f.severity}] {f.message}" for f in ordered)]
        )
