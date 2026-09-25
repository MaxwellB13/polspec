"""What several test files need, said once.

On `pythonpath` (see `[tool.pytest.ini_options]`), so a test imports it as
`from helpers import spec_for`.
"""

from __future__ import annotations

from polspec import ColSpec, FrameSpec


def spec_for(column: ColSpec, name: str = "Single") -> type[FrameSpec]:
    """A FrameSpec of one column, `c`, named `name`."""
    return type(name, (FrameSpec,), {"__columns__": {"c": column}})
