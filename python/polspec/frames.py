"""The frame plumbing every verb shares.

`generate`, `validate`, the sinks and the `FrameSpec` facade all take the same
two things -- a `references=` mapping of parent frames, and a `method=` -- and
all move frames between the eager and lazy forms. Each module used to declare
its own copy of both aliases and its own coercer, which is four places to
agree with each other the day `references=` learns to take a path, or
`method=` gains a third value.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import polars as pl

__all__ = ["Frame", "Method", "References", "to_eager", "to_lazy"]

Frame = pl.DataFrame | pl.LazyFrame

#: Parent frames for foreign keys, keyed by the spec, its `FrameSpec` class,
#: or its name. `polspec.tablespec.resolve_references` turns any of the three
#: into a name.
References = Mapping[Any, Frame] | None

#: How `generate` draws its rows. See `polspec.generation.generate`.
Method = Literal["random", "cartesian"]


def to_lazy(frame: Frame) -> pl.LazyFrame:
    """`frame` as a LazyFrame, leaving one that already is alone."""
    return frame.lazy() if isinstance(frame, pl.DataFrame) else frame


def to_eager(frame: Frame) -> pl.DataFrame:
    """`frame` as a DataFrame, collecting one that is lazy."""
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame
