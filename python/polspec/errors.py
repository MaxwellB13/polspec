"""The exception hierarchy.

Every error polspec raises about a *spec*, the *data* it checks, a *file* it
reads, or the *generation* it runs derives from `PolspecError`, so a caller can
catch the library's own complaints in one clause and let genuine bugs through:

    try:
        Orders.validate(df)
    except PolspecError as exc:
        ...

Each subclass also inherits the built-in its failure is a kind of
(`ValueError`, `TypeError`, `LookupError`), so an ordinary
`except ValueError` still catches a bad declaration without knowing polspec's
hierarchy. Plain argument misuse -- a negative row count, an unknown
`method=` -- stays a bare `ValueError`, as it would in any Python API.
"""

from __future__ import annotations

from typing import Any


class PolspecError(Exception):
    """Base class for every error polspec raises on its own behalf."""


class SpecError(PolspecError, ValueError, TypeError):
    """A declaration that cannot mean anything.

    Raised while a `ColSpec`, `ColRule`, `Check`, `ForeignKey`, `FrameSpec`
    or `CatSpec` is being built: bounds a dtype cannot hold, a rule pointing
    at a column that does not exist, two columns resolving to one name.
    Inherits both `ValueError` and `TypeError` because it replaces both.
    """


class ValidationError(PolspecError, ValueError):
    """Data does not meet its spec.

    Carries the `ValidationReport` of every violation found as `report`.
    `errors` is the same findings as a plain list of messages, for the common
    case of printing them.
    """

    def __init__(self, report: Any, errors: list[str] | None = None) -> None:
        if isinstance(report, str):
            # A plain message, as raised before reports existed.
            super().__init__(report)
            self.report = None
            self._errors = list(errors or [])
        else:
            super().__init__(str(report))
            self.report = report
            self._errors = [f.message for f in report.findings]

    @property
    def errors(self) -> list[str]:
        return list(self._errors)


class MultiValidationError(ValidationError):
    """Several frames failed validation together, as one registry call.

    `reports` holds the `ValidationReport` of every spec that failed, keyed by
    spec name, so `failing_rows()`, `by_code()` and the rest are reachable for
    each of them. `report` is None: there is no single report here, and
    picking one of several arbitrarily would be worse than saying so. `str()`
    and `errors` read as they always have -- every failing spec's findings,
    one after another.
    """

    def __init__(self, reports: Any) -> None:
        failed = {name: report for name, report in reports.items() if not report.passed}
        super().__init__(
            "\n\n".join(str(report) for report in failed.values()),
            errors=[f.message for report in failed.values() for f in report.findings],
        )
        self.reports: dict[str, Any] = failed


class GenerationError(PolspecError, ValueError):
    """A spec that declares fine cannot be turned into data as asked.

    A dtype the engine cannot fill, a cartesian coverage set past the size
    cap, a foreign key with an empty parent, a `unique` domain smaller than
    the row count. Errors raised inside the Rust extension surface as this.
    """


class SerializationError(PolspecError, ValueError):
    """A spec file cannot be written or read.

    A dtype with no file representation, a key the reader does not know, a
    file written by a newer format version.
    """


class RegistryError(PolspecError, LookupError):
    """A collection of specs is inconsistent.

    An unknown or duplicated spec name, a cycle in the foreign-key graph, two
    specs disagreeing about a shared category.
    """


class CliError(PolspecError):
    """An expected failure on the command line, reported without a traceback."""
