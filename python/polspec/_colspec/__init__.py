"""What a `ColSpec` checks about itself where it is declared, by concern.

`ColSpec.__post_init__` calls these in a fixed order: normalising first,
so every check sees the canonical form; then the checks of one field;
then the ones that compare fields. Each takes the column as it stands
and sets a normalised field or raises `SpecError`.

Internal: not part of the public API.
"""

from __future__ import annotations
