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


def accepted_options(cls: type) -> list[str]:
    """The keyword names a caller may pass for `cls`: its fields."""
    return sorted(f.name for f in dataclasses.fields(cast("Any", cls)))


def options_from[T](
    cls: type[T],
    options_obj: T | None,
    options: Mapping[str, Any],
    *,
    what: str,
) -> T:
    """The options for one call, from an object, keywords, or neither.

    `what` names the kind in messages ("validation", "drift").
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
    accepted = accepted_options(cls)
    unknown = [k for k in options if k not in accepted]
    if unknown:
        # Naming the option the caller meant, rather than letting the
        # dataclass raise about a private class they cannot look up.
        hints = []
        for name in unknown:
            # A prefixed spelling of a real option -- 0.22's `validate_order`
            # for `order` -- is the one a caller upgrading writes, so it comes
            # first: difflib would offer `validators` for it.
            close = [a for a in accepted if name.endswith(f"_{a}")]
            close = close or difflib.get_close_matches(name, accepted, n=1)
            hints.append(f"{name!r}{f' (did you mean {close[0]!r}?)' if close else ''}")
        raise TypeError(
            f"Unknown {what} option(s): {', '.join(hints)}. "
            f"Accepted: {', '.join(accepted)}."
        )
    return cls(**options)
