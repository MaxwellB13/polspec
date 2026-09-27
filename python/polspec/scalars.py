"""Plain forms for the values YAML cannot hold.

A spec file holds values -- bounds, choices, a predicate's literals -- and
YAML has native forms for most of them: numbers, text, dates, datetimes,
bytes. It has none for a time of day or a duration, so those two are
written as a one-key mapping naming what they are:

    {time: "12:30:00.000001"}                              # dt.time
    {duration: {days: 1, seconds: 0, microseconds: 0}}     # dt.timedelta

A time is its ISO form, as `isoformat()` writes it and `fromisoformat()`
reads it. A duration is `timedelta`'s own three fields: lossless, and read
without an ISO 8601 duration parser. Everything else passes through
unchanged, in both directions.

One codec for every place a value is written, so a bound, a choice and a
literal cannot disagree about how a time is spelt. It lives outside
`polspec.serialization` because a predicate (`polspec.expr`) is a
declaration, and writes its own data form.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import Any

TAGS: tuple[str, ...] = ("time", "duration")

_DURATION_FIELDS = ("days", "seconds", "microseconds")


def to_plain(value: Any) -> Any:
    """`value` as YAML can hold it: tagged if it is a time or a duration."""
    if isinstance(value, dt.time):
        return {"time": value.isoformat()}
    if isinstance(value, dt.timedelta):
        return {
            "duration": {
                "days": value.days,
                "seconds": value.seconds,
                "microseconds": value.microseconds,
            }
        }
    return value


def is_tagged(value: Any) -> bool:
    """Whether `value` is one of the tagged forms `to_plain` writes."""
    if not isinstance(value, Mapping) or len(value) != 1:
        return False
    ((tag, payload),) = value.items()
    if tag == "time":
        return isinstance(payload, str)
    if tag == "duration":
        return isinstance(payload, Mapping) and set(payload) <= set(_DURATION_FIELDS)
    return False


def from_plain(value: Any) -> Any:
    """The value a plain form stands for; anything untagged, as it is.

    Raises `ValueError` for a tagged form that does not parse -- a time that
    is not ISO, a duration field that is not an integer -- so a reader can
    say which key held it.
    """
    if not is_tagged(value):
        return value
    ((tag, payload),) = value.items()
    if tag == "time":
        return dt.time.fromisoformat(payload)
    parts = {name: payload.get(name, 0) for name in _DURATION_FIELDS}
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in parts.values()):
        raise ValueError(f"a duration's fields must be integers, got {dict(payload)!r}")
    return dt.timedelta(**parts)
