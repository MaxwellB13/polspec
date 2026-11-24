"""Names 2.0 renamed or removed, each raising an error that names what to
write instead.

2.0 changes these without a 1.x deprecation (PLAN-2.0.0.md, Decide 12), so
the old spelling fails at once, and says what replaced it, rather than
with Python's bare "unexpected keyword" or "has no attribute".

Internal: not part of the public API.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, NoReturn, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def renamed_keywords(**renames: str) -> Callable[[F], F]:
    """Refuses each old keyword, naming the new: `renamed_keywords(categories=
    "catspec")` makes `f(categories=...)` a `TypeError` that says to pass
    `catspec=` instead."""

    def wrap(function: F) -> F:
        name = getattr(function, "__qualname__", repr(function))

        @functools.wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            for old, new in renames.items():
                if old in kwargs:
                    raise TypeError(f"{name}() takes {new}= since 2.0, not {old}=.")
            return function(*args, **kwargs)

        return cast("F", wrapped)

    return wrap


def refuse_keyword(cls: type, old: str, instead: str) -> None:
    """Makes `cls(old=...)` a `TypeError` saying to write `instead`: for a
    keyword with no one-for-one replacement, as `weights=` is folded into
    `choices={value: weight}`. Applied to the built class, so the
    dataclass's own `__init__` -- which `dataclasses.replace` calls too --
    is kept."""
    init = cls.__init__
    message = f"{cls.__name__}() takes {instead} since 2.0, not {old}=."

    @functools.wraps(init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        if old in kwargs:
            raise TypeError(message)
        init(self, *args, **kwargs)

    cls.__init__ = __init__  # ty: ignore[invalid-assignment]


def removed(owner: str, old: str, new: str) -> Callable[..., NoReturn]:
    """A stand-in for a renamed method: calling it raises `AttributeError`
    naming the new one."""

    def stand_in(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise AttributeError(f"{owner}.{old} is {owner}.{new} since 2.0.")

    stand_in.__name__ = old
    stand_in.__doc__ = f"Removed in 2.0: use `{owner}.{new}`."
    return stand_in
