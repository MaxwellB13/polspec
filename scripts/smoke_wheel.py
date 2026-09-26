"""Checks an installed polspec, as a user would get it from a wheel.

The test suite runs against `maturin develop` -- an editable build over the
checkout -- so it cannot see what a wheel leaves out. This runs against
whatever `import polspec` finds, in a fresh environment with the wheel
installed, and fails naming the first thing a user would miss:

    python scripts/smoke_wheel.py

It must not import from the checkout: run it from outside the repository,
or with nothing but the installed wheel on `sys.path`.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import polars as pl
import polspec
from polspec import ColSpec, TableSpec, generate, validate


def _require(holds: bool, problem: str) -> None:
    """Fails the check naming the problem. Not `assert`, which `python -O`
    strips: a smoke test that can pass by being optimised away is none."""
    if not holds:
        raise SystemExit(f"smoke test failed: {problem}")


def main() -> None:
    package = Path(polspec.__file__).parent
    for shipped in ("py.typed", "_polspec.pyi"):
        _require((package / shipped).is_file(), f"the wheel does not ship {shipped}")

    installed = importlib.metadata.version("polspec")
    _require(
        polspec.__version__ == installed,
        f"__version__ is {polspec.__version__}, the metadata says {installed}",
    )

    from polspec import _polspec  # the compiled extension loads on this platform

    _require("int64" in _polspec.kinds(), "the extension lists no int64 kind")

    spec = TableSpec(
        "Smoke",
        {
            "id": ColSpec(pl.Int64, unique=True, bounds=(1, 1_000_000)),
            "status": ColSpec(pl.Enum(["NEW", "PAID"])),
            "total": ColSpec(pl.Decimal(10, 2), bounds=(0, 100), nullable=True),
            "tags": ColSpec(pl.List(pl.String), list_length=(0, 3)),
        },
    )
    df = generate(spec, 10_000, seed=1)
    validate(spec, df)
    _require(df.equals(generate(spec, 10_000, seed=1)), "a seed does not reproduce")
    print(f"polspec {installed} from {package}: ok")


if __name__ == "__main__":
    main()
