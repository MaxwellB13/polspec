"""`FrozenDict`: a read-only mapping that pickles.

Every mapping a spec or a report holds -- a table's columns, a struct's
fields, a format's extra values, a finding's details -- is read-only, so a
value that compares equal stays equal. `types.MappingProxyType` gives that,
but cannot be pickled or deep-copied, and so neither could anything holding
one: a spec could not be handed to another process. This is the same
read-only view over its own dict, rebuilt from that dict when unpickled.

Internal: not part of the public API.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any


class FrozenDict[K, V](Mapping[K, V]):
    """A read-only mapping over a dict of its own, in the order given."""

    __slots__ = ("_data",)

    _data: dict[K, V]

    def __init__(self, data: Mapping[K, V] | Iterable[tuple[K, V]] = ()) -> None:
        object.__setattr__(self, "_data", dict(data))

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is read-only")

    def __repr__(self) -> str:
        return repr(self._data)

    def __reduce__(self) -> tuple[type[FrozenDict[K, V]], tuple[dict[K, V]]]:
        return type(self), (dict(self._data),)
