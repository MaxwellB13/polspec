"""The exception hierarchy.

Every error polspec raises about a *spec*, the *data* it checks, a *file* it
reads, or the *generation* it runs derives from `PolspecError`, so a caller can
catch the library's own complaints in one clause and let genuine bugs through:

    try:
        Orders.validate(df)
    except PolspecError as exc:
        ...

Each subclass also inherits the built-in type it replaced (`ValueError`,
`TypeError`, `LookupError`), so code written against earlier versions keeps
working. Plain argument misuse -- a negative row count, an unknown `method=`
-- stays a bare `ValueError`, as it would in any Python API.
"""

from __future__ import annotations


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

    Carries every violation found, as `errors`, rather than only the first.
    """

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


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
