"""The derive pass: a derived column computed from the frame as it stands.

It runs after every pass writing a column it reads -- rules, foreign keys,
`fixed=` -- as `polspec.pass_order` orders it, so it reads the values
validation will hold it to. Nothing is drawn, so it takes no seed.

Internal: not part of the public API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polspec.errors import GenerationError

if TYPE_CHECKING:
    from polspec.spec import ColSpec


def apply_derived(frame: pl.DataFrame, name: str, spec: ColSpec) -> pl.DataFrame:
    assert spec.derive is not None  # noqa: S101 - only a derived column asks
    try:
        values = frame.select(
            spec.derive.to_expr().cast(spec.dtype, strict=True).alias(name)
        ).to_series()
    except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError) as exc:
        raise GenerationError(
            f"Column {name!r} cannot be derived as {spec.dtype!r} from "
            f"{spec.derive!r}: {exc}"
        ) from exc
    return frame.with_columns(values)
