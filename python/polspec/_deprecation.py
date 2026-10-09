"""Deprecation, said one way: a `DeprecationWarning` naming what to use
instead and the release the old name goes in.

A deprecated name keeps working, unchanged, until its removal release. In
0.x that is 1.0 for everything; see CONTRIBUTING, "Deprecations".
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

#: The release every name deprecated in 0.x is removed in.
REMOVAL = "1.0"


def warn_deprecated(
    old: str, *, use: str, removal: str = REMOVAL, stacklevel: int = 3
) -> None:
    """Warns that `old` is deprecated, naming `use` in its place.

    `stacklevel` points the warning at the caller's own code: 3 from a
    function this module wraps, 2 from the deprecated function itself.
    """
    warnings.warn(
        f"{old} is deprecated, and is removed in polspec {removal}: use {use} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def deprecated(
    old: str, *, use: str, removal: str = REMOVAL
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Marks a function deprecated: each call warns, then does what it did.

    The function's docstring gains a first line saying so, which the API
    reference shows.
    """

    def mark(function: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(function)
        def warning(*args: P.args, **kwargs: P.kwargs) -> R:
            warn_deprecated(old, use=use, removal=removal)
            return function(*args, **kwargs)

        note = f"Deprecated: removed in polspec {removal}. Use `{use}` instead."
        warning.__doc__ = f"{note}\n\n{function.__doc__ or ''}"
        setattr(warning, "__deprecated__", note)  # noqa: B010 - a marker, not an API
        return warning

    return mark


def is_deprecated(obj: Any) -> bool:
    """Whether `obj` was marked with `deprecated`."""
    return hasattr(obj, "__deprecated__")
