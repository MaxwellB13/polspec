"""Inferring a spec from data that already exists.

The inverse of generation: given a DataFrame, describe the columns well enough
that `FrameSpec.generate` could produce something like it again.

By default a column is described by its dtype, its null rate, its extremes
and -- for a low-cardinality text column whose values repeat -- its
categories. Three opt-in steps describe more, for when the spec is going to
generate a stand-in for the data (`polspec.synthesize`):

- `shape=True` fits the distribution a numeric or temporal column's values
  follow (`polspec.shape`), so the stand-in is not uniform where the data
  is skewed;
- `detect_unique=True` declares an all-distinct integer or text column
  unique, so a key stays a key;
- `replace=` names columns whose *values* must not be carried into the
  spec: a text column among them is never narrowed to an `Enum` of the
  values it holds, and carries no weights.

The package, by concern: `options` and `rounding` say what is
recorded and how finely; `columns` describes each column, with
`formats`, `shapes` and `text`; `order` reads the rows' order; this
module profiles a frame from them.

Internal: not part of the public API.
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import polars as pl

# Each `X as X` is what other modules take from `polspec.profiler`.
from polspec.profiler.columns import profile_column
from polspec.profiler.options import DESCRIBE_DEFAULTS as DESCRIBE_DEFAULTS
from polspec.profiler.options import ProfileKeywords as ProfileKeywords
from polspec.profiler.options import ProfileOptions
from polspec.profiler.options import SynthesizeKeywords as SynthesizeKeywords
from polspec.profiler.options import profile_options as profile_options
from polspec.profiler.order import detect_order
from polspec.profiler.rounding import rounded_weights as rounded_weights
from polspec.profiler.rounding import step_of as step_of
from polspec.profiler.text import STAYS_TEXT_NOTE, text_note, typed_text
from polspec.profiler.text import TEXT_NOTE as TEXT_NOTE
from polspec.spec import ColSpec
from polspec.tablespec import TableSpec


def profile_columns(df: pl.DataFrame, opts: ProfileOptions) -> dict[str, ColSpec]:
    """Infers ColSpec column definitions by profiling an existing DataFrame.

    Internal: `polspec.profile` is the public way in, and returns these
    columns as a `TableSpec`.

    Warns
    -----
    UserWarning
        Once per float column holding an infinity. The bounds are its finite
        values', so the spec reports the infinities when the data is
        validated against it. A NaN is recorded as `nan_probability`.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame, got {type(df).__name__}")
    replace = opts.replace
    replaced = set(replace)
    unknown = sorted(replaced - set(df.columns))
    if unknown:
        raise ValueError(
            f"replace= names {unknown}, which the data does not have. Its "
            f"columns are {df.columns}."
        )
    for name in sorted(replaced):
        if isinstance(df.schema[name], pl.Enum):
            raise ValueError(
                f"replace= names {name!r}, an Enum: its categories are its "
                "dtype, so they are in the spec however it is profiled. Cast "
                "it to String first to have its values replaced."
            )

    # A shape is fitted between bounds, so without bounds there is none.
    options = dataclasses.replace(opts, shape=opts.shape and opts.calculate_bounds)

    def profiled(name: str) -> tuple[ColSpec, list[tuple[str, int]]]:
        # Each column notes its infinities in a list of its own, so columns
        # profiled at once share nothing, and the notes come out in column
        # order whichever finished first.
        infinite: list[tuple[str, int]] = []
        column = profile_column(
            df[name],
            name,
            total_rows=df.height,
            options=options,
            infinite=infinite,
            imitate=name not in replaced,
            top=True,
        )
        return column, infinite

    results = _for_each_column(profiled, df)
    columns = {
        name: column for name, (column, _) in zip(df.columns, results, strict=True)
    }
    infinite = [note for _, notes in results for note in notes]
    for where, infinities in infinite:
        warnings.warn(
            f"Column '{where}' holds {infinities:,} infinite value(s), which "
            "its bounds -- its finite values' -- leave out, so validating this "
            "data against the spec reports them. Open the bound on that side "
            "(None) if infinities belong there.",
            UserWarning,
            stacklevel=2,
        )
    return columns


# Rows a frame needs before its columns are profiled at once: below it, a
# thread costs more than the column it would profile.
PARALLEL_MIN_ROWS = 50_000


def _for_each_column(profile_one: Callable[[str], Any], df: pl.DataFrame) -> list[Any]:
    """`profile_one` over every column of `df`, in column order: on a pool
    of threads for a frame large enough to gain, since Polars releases the
    GIL while it works, and one after another otherwise."""
    names = df.columns
    if len(names) < 2 or df.height < PARALLEL_MIN_ROWS:
        return [profile_one(name) for name in names]
    with ThreadPoolExecutor(max_workers=min(len(names), os.cpu_count() or 1)) as pool:
        return list(pool.map(profile_one, names))


def profiled_table(frame: pl.DataFrame, opts: ProfileOptions, name: str) -> TableSpec:
    """`frame` described as a `TableSpec`: its text read as `opts` says,
    each column profiled, and the order its rows come in."""
    read = typed_text(frame, opts)
    frame = read.frame
    columns = profile_columns(frame, opts)
    for column, fmt in read.formats.items():
        columns[column] = dataclasses.replace(columns[column], text_format=fmt)
    for column, (dtype, fmt) in read.would.items():
        columns[column] = dataclasses.replace(
            columns[column], description=text_note(dtype, fmt)
        )
    for column, reason in read.stays.items():
        columns[column] = dataclasses.replace(
            columns[column], description=f"{STAYS_TEXT_NOTE}{reason}."
        )
    order = detect_order(frame) if opts.detect_order else None
    order_by, descending = order if order is not None else ((), False)
    return TableSpec(name, columns, order_by=order_by, descending=descending)
