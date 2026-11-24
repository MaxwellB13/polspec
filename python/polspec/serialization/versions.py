"""Spec-file format versions: the one this polspec reads and writes, and
the refusal of every other.

A file records the version that wrote it under `version:`. polspec 2.0
reads version 4 alone, as every release since 0.14.0 writes it. An older
file -- version 1, 2 or 3, from 0.13 and before -- is refused, naming its
version: polspec 1.x reads one and writes it back as version 4. A newer
file is refused, naming the version it needs. A mapping with no `version:`
key, as a hand-written dict or a registry's inline `categories:` is, is
read as the current format. *Since 2.0; 1.x migrated versions 1 to 3 as
it read them.*

The version history is in *Spec files*. What bumps the version is a key
that changes meaning or shape -- or, as with version 4's
`nan_probability:`, one whose *absence* makes a claim: a float that does
not say otherwise holds no NaN, which a reader too old to know the key
would not check. An *added optional key* is not a new version: a file
without it reads as before, and one with it reads on any polspec that
knows the key.

Internal: not part of the public API.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from polspec.errors import SerializationError

FORMAT_VERSION = 4
"""The spec-file format version this polspec writes, and the only one it reads."""


def versionless(data: Any) -> bool:
    """Whether `data` is a mapping that leaves out `version:`."""
    return isinstance(data, Mapping) and "version" not in data


def current_format(data: Any, where: str) -> dict[str, Any]:
    """A file's raw data, checked to be of `FORMAT_VERSION`, or refused."""
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise SerializationError(
            f"{where}: expected a mapping at the top level, got {type(data).__name__}"
        )
    version = data.get("version", FORMAT_VERSION)
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise SerializationError(
            f"{where}: 'version' must be a positive integer, got {version!r}"
        )
    if version > FORMAT_VERSION:
        raise SerializationError(
            f"{where} was written by a newer polspec (format version {version}); "
            f"this version reads up to {FORMAT_VERSION}. Upgrade polspec to read it."
        )
    if version < FORMAT_VERSION:
        raise SerializationError(
            f"{where} is format version {version}, written by polspec 0.13 or "
            f"earlier; polspec 2.0 reads only version {FORMAT_VERSION}. polspec "
            "1.x reads it and writes it back as version 4: with polspec<2, "
            "load it and save it again (`to_yaml(from_yaml(path), path)`)."
        )
    return {**data, "version": FORMAT_VERSION}
