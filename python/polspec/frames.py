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


def sample_rows(frame: pl.DataFrame, n: int | None, seed: int = 0) -> pl.DataFrame:
    """`n` of `frame`'s rows, drawn at random from a fixed `seed` -- or the
    whole frame, when `n` is None or not fewer rows. The draw comes back in
    no particular order; nothing that profiles or measures a frame reads one.

    Never the first `n`: a file is often sorted, by date or by key, and its
    head is not a sample of it.
    """
    if n is None:
        return frame
    if n <= 0:
        raise ValueError(f"sample must be a positive row count, got {n}")
    if n >= frame.height:
        return frame
    return frame.sample(n, seed=seed)


def json_casts(schema: Mapping[str, Any], *, error: type[Exception]) -> list[pl.Expr]:
    """The casts that make a frame of `schema` one Polars' JSON writers can
    write -- for `with_columns` -- or `error` naming the column they cannot.

    JSON holds a category as its text, so an `Enum` or `Categorical` -- at
    any depth -- is cast to `String` first: the file is the same, and a list
    of them no longer panics Polars' writer (it does past a few hundred
    values, in 1.x and 2.0 alike). JSON has no bytes, and a `Binary` column
    panics the writer outright, so it is refused before anything is written.
    """
    casts = []
    for name, dtype in schema.items():
        if _holds(dtype, pl.Binary):
            raise error(
                f"Column {name!r} is {dtype}, and JSON has no bytes: write it "
                "to Parquet or Arrow IPC instead"
            )
        text = _categories_as_text(dtype)
        if text != dtype:
            casts.append(pl.col(name).cast(text))
    return casts


def _holds(dtype: Any, kind: Any) -> bool:
    """Whether `dtype` is `kind` or holds one, at any depth."""
    if dtype == kind:
        return True
    if isinstance(dtype, (pl.List, pl.Array)):
        return _holds(dtype.inner, kind)
    if isinstance(dtype, pl.Struct):
        return any(_holds(f.dtype, kind) for f in dtype.fields)
    return False


def _categories_as_text(dtype: Any) -> Any:
    """`dtype` with every `Enum` and `Categorical` in it as `String`."""
    if isinstance(dtype, (pl.Enum, pl.Categorical)) or dtype in (
        pl.Enum,
        pl.Categorical,
    ):
        return pl.String
    if isinstance(dtype, pl.List):
        return pl.List(_categories_as_text(dtype.inner))
    if isinstance(dtype, pl.Array):
        return pl.Array(_categories_as_text(dtype.inner), dtype.size)
    if isinstance(dtype, pl.Struct):
        return pl.Struct(
            [pl.Field(f.name, _categories_as_text(f.dtype)) for f in dtype.fields]
        )
    return dtype
