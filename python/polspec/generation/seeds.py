"""One seed per pass, keyed by what the pass is for rather than by when it
was asked for.

The engine seeds each column from the frame seed and the column's seed name
(`sample.rs::seed_for_column`), so inserting or reordering columns never
changes the values of the others. The passes that run afterwards -- rules,
the hierarchy, foreign keys, composite uniqueness, a bounded categorical's
pool -- used to draw their seeds from one `random.Random` in declaration
order, so inserting a rules column shifted every later pass. They now mix
the frame seed with a stable key by the same construction the engine uses,
ported here so the two never disagree about what "keyed by name" means.

Internal: not part of the public API.
"""

from __future__ import annotations

import hashlib

_MASK = (1 << 64) - 1
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def pass_seed(frame_seed: int, key: str) -> int:
    """The seed one pass draws from: `frame_seed` mixed with `key`.

    FNV-1a over the key's UTF-8 bytes, xor'd into the frame seed, then a
    splitmix64 finaliser -- bit for bit what `seed_for_column` does in
    `src/sample.rs`, and pinned against its golden value in the tests.
    """
    digest = _FNV_OFFSET
    for byte in key.encode("utf-8"):
        digest = ((digest ^ byte) * _FNV_PRIME) & _MASK
    z = (frame_seed & _MASK) ^ digest
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def stable_seed(*parts: str) -> int:
    """A seed derived only from `parts`, stable across processes and runs
    (unlike `hash()`, which is salted per-process for strings).

    A second construction beside `pass_seed`, and not interchangeable with
    it: this one seeds what has no frame seed to mix into -- a named
    `Categories` pool, shared across specs whatever seed each is generated
    with, and a registry spec's own seed from the registry's -- and changing
    either would change seeded output. SHA-256 rather than FNV, because it
    is Python's alone; nothing in the engine has to agree with it.
    """
    digest = hashlib.sha256("\0".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")
