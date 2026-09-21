"""One way to take options: an options object, keyword options, or neither.

`validate()`, `inspect()`, `diff()` and `drift()` each accept an `options=`
object *or* the individual keywords -- alternatives, not base and override,
because a call that passes both has two answers for the same switch. This is
the one place that rule, and its error messages, live.
"""

from __future__ import annotations

import dataclasses
import difflib
from collections.abc import Mapping
from typing import Any, cast


def accepted_options(
    cls: type, *, renames: Mapping[str, str] | None = None
) -> list[str]:
    """The keyword names a caller may pass for `cls`: its fields, with any
    renamed ones spelled the public way instead of the field's."""
    renamed = dict(renames or {})
    hidden = set(renamed.values())
    return sorted(
        {f.name for f in dataclasses.fields(cast("Any", cls)) if f.name not in hidden}
        | set(renamed)
    )


def options_from[T](
    cls: type[T],
    options_obj: T | None,
    options: Mapping[str, Any],
    *,
    what: str,
    renames: Mapping[str, str] | None = None,
) -> T:
    """The options for one call, from an object, keywords, or neither.

    `what` names the kind in messages ("validation", "drift"); `renames`
    maps a public keyword to the field it sets (`validate_unique` ->
    `unique`).
    """
    if options_obj is not None:
        if options:
            raise TypeError(
                "Pass options= or the individual keyword options, not both. "
                f"Given options= alongside {', '.join(sorted(options))}."
            )
        if not isinstance(options_obj, cls):
            raise TypeError(
                f"options= must be a {cls.__name__}, got {type(options_obj).__name__}"
            )
        return options_obj
    accepted = accepted_options(cls, renames=renames)
    unknown = [k for k in options if k not in accepted]
    if unknown:
        # Naming the option the caller meant, rather than letting the
        # dataclass raise about a private class they cannot look up.
        hints = []
        for name in unknown:
            close = difflib.get_close_matches(name, accepted, n=1)
            hints.append(f"{name!r}{f' (did you mean {close[0]!r}?)' if close else ''}")
        raise TypeError(
            f"Unknown {what} option(s): {', '.join(hints)}. "
            f"Accepted: {', '.join(accepted)}."
        )
    renamed = dict(renames or {})
    return cls(**{renamed.get(k, k): v for k, v in options.items()})
