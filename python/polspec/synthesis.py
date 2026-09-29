"""A fake dataset from a real one.

`profile()` describes a source -- a `DataFrame`, a `LazyFrame`, or a data
file -- as a `TableSpec`: each column's dtype, null rate, extremes, category
frequencies, the distribution its numbers follow, and whether it is a key.
`synthesize()` generates from that spec: a frame of the same schema whose
columns look like the source's, and whose rows are none of its rows.

What carries over from the source is what a spec can say about a column:
category values and their frequencies, numeric and temporal extremes and
fitted shapes, string lengths, null rates, a float's NaN share. Nothing else does -- no row, and
no relationship between columns, which are profiled one at a time. A column
named in `replace=` carries over its shape and not its values: text in it
is never an `Enum` of the values the source holds.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence

import polars as pl

from polspec.dtypes import dtype_value_limits
from polspec.errors import GenerationError
from polspec.generation import generate
from polspec.profiler import profile_dataframe
from polspec.reading import read_file
from polspec.tablespec import TableSpec

__all__ = ["profile", "synthesize"]

Source = pl.DataFrame | pl.LazyFrame | str | os.PathLike[str]


def profile(
    source: Source,
    *,
    name: str = "Profiled",
    replace: Sequence[str] = (),
    sample: int | None = None,
    seed: int = 0,
    max_unique_enum: int = 50,
) -> TableSpec:
    """A spec describing `source` well enough to generate a stand-in for it.

    Parameters
    ----------
    source : pl.DataFrame | pl.LazyFrame | str | os.PathLike
        The data: a frame, a lazy frame (any `scan_*` -- a file, a glob,
        cloud storage), or the path of a CSV, TSV, Parquet, NDJSON, JSON or
        Arrow IPC file.
    name : str, default "Profiled"
        The spec's name.
    replace : sequence of str, default ()
        Columns whose values must not be carried into the spec. A text
        column named here is described by its lengths, never as an `Enum`
        of the values it holds, and carries no weights.
    sample : int, optional
        Profile a random sample of this many rows rather than every row.
    seed : int, default 0
        Seeds the sample and the shape fitting, so a profile is the same
        every run.
    max_unique_enum : int, default 50
        A text column with at most this many distinct values, repeating --
        and not in `replace` -- becomes an `Enum` of them.

    Returns
    -------
    TableSpec
        Everything `FrameSpec.from_dataframe` records, plus each column's
        category frequencies, fitted distribution and uniqueness.
    """
    frame, _ = _load(source, sample, seed)
    columns = profile_dataframe(
        frame,
        weights=True,
        max_unique_enum=max_unique_enum,
        shape=True,
        detect_unique=True,
        replace=replace,
        seed=seed,
    )
    return TableSpec(name, columns)


def synthesize(
    source: Source,
    n: int | None = None,
    *,
    seed: int | None = None,
    replace: Sequence[str] = (),
    sample: int | None = None,
    max_unique_enum: int = 50,
) -> pl.DataFrame:
    """A fake frame that looks like `source`: `generate()` over `profile()`.

    The result has `source`'s schema, and `n` rows -- as many as `source`
    has, by default. Each column follows the profiled column's null rate,
    extremes, category frequencies and fitted distribution; a key column
    stays distinct. Columns are generated independently, so a relationship
    between two columns in `source` is not kept. See `profile` for `replace`,
    `sample` and `max_unique_enum`.

    `seed` seeds the generation, and the sample and fitting behind it, so the
    same source and seed give the same frame. To look at, edit or keep the
    spec the frame came from, call `profile` and `generate` separately.
    """
    fake, _ = synthesized(
        source,
        n,
        seed=seed,
        replace=replace,
        sample=sample,
        max_unique_enum=max_unique_enum,
    )
    return fake


def synthesized(
    source: Source,
    n: int | None = None,
    *,
    seed: int | None = None,
    replace: Sequence[str] = (),
    sample: int | None = None,
    max_unique_enum: int = 50,
    name: str = "Synthesized",
) -> tuple[pl.DataFrame, TableSpec]:
    """`synthesize`, with the spec the frame was generated from -- one
    profile for both, where the command line writes the spec beside the
    data."""
    frame, height = _load(source, sample, seed if seed is not None else 0)
    spec = profile(
        frame,
        name=name,
        replace=replace,
        seed=seed if seed is not None else 0,
        max_unique_enum=max_unique_enum,
    )
    rows = height if n is None else n
    if rows < 0:
        raise ValueError(f"n must be >= 0, got {rows}")
    spec = _room_for_keys(spec, rows)
    fake = generate(spec, rows, seed=seed)
    # The profile narrows a low-cardinality text column to an Enum of its
    # values; a stand-in for the source has the source's own dtypes.
    # Column by column: `DataFrame.cast` skips a mapping's "" key.
    fake = fake.with_columns(
        pl.col(c).cast(dtype)
        for c, dtype in frame.schema.items()
        if fake.schema[c] != dtype
    )
    return fake, spec


def _load(source: Source, sample: int | None, seed: int) -> tuple[pl.DataFrame, int]:
    """`source` as a frame to profile, and how many rows it has in all."""
    if isinstance(source, pl.LazyFrame):
        frame = source.collect()
    elif isinstance(source, pl.DataFrame):
        frame = source
    elif isinstance(source, (str, os.PathLike)):
        frame = read_file(source, infer_dates=True)
    else:
        raise TypeError(
            "source must be a DataFrame, a LazyFrame or the path of a data file, "
            f"got {type(source).__name__}"
        )
    if frame.height == 0:
        raise ValueError("source has no rows to profile")
    height = frame.height
    if sample is not None:
        if sample <= 0:
            raise ValueError(f"sample must be a positive row count, got {sample}")
        if sample < height:
            frame = frame.sample(sample, seed=seed)
    return frame, height


def _room_for_keys(spec: TableSpec, rows: int) -> TableSpec:
    """`spec` with each unique integer key's range wide enough for `rows`.

    A key profiled from a thousand rows spans a thousand values; a stand-in
    ten times the size needs ten times as many. The upper end is raised to
    make room -- the key stays within the dtype, and starts where it did.
    """
    widened = {}
    for name, column in spec.columns.items():
        bounds = column.bounds
        if (
            not column.unique
            or not column.dtype.is_integer()
            or bounds is None
            or bounds.min is None
            or bounds.max is None
            or bounds.max - bounds.min + 1 >= rows
        ):
            continue
        top = bounds.min + rows - 1
        limits = dtype_value_limits(column.dtype)
        if limits is not None and top > limits[1]:
            raise GenerationError(
                f"Column {name!r} is a key of {column.dtype!r} from {bounds.min}, "
                f"which cannot hold {rows:,} distinct values. Generate fewer rows."
            )
        widened[name] = dataclasses.replace(column, bounds=(bounds.min, top))
    return spec.with_columns(widened) if widened else spec
