"""Text that holds numbers, flags or dates: read as them with
`parse_text=True`, or noted on the column when not.

Internal: not part of the public API.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from polspec._text import infer as infer_text
from polspec._text import stays_text
from polspec.profiler.options import ProfileOptions

# The start of the description a text column is given when it would parse:
# what `polspec schema infer` looks for to say so on the terminal too.
TEXT_NOTE = "Every value is text that parses as "


# The start of the description of a text column that nearly parses as a
# type, and stays text: what stopped it follows.
STAYS_TEXT_NOTE = "Stays text: "


def text_note(dtype: pl.DataType, text_format: str | None = None) -> str:
    """The description of a text column that would parse as `dtype`,
    written as `text_format` says when it is not the form Polars writes."""
    written = f" ('{text_format}')" if text_format is not None else ""
    return (
        f"{TEXT_NOTE}{type(dtype).__name__}{written}; profile with "
        "parse_text=True (--parse-text) to declare it as one."
    )


@dataclass(frozen=True, slots=True)
class _ReadText:
    """What `typed_text` made of a frame's text: the frame, each column it
    read and the format that read it, each it could have read, and why each
    that nearly parsed stays text."""

    frame: pl.DataFrame
    formats: dict[str, str]
    would: dict[str, tuple[pl.DataType, str | None]]
    stays: dict[str, str]


def typed_text(
    frame: pl.DataFrame, opts: ProfileOptions, *, explain: bool = True
) -> _ReadText:
    """`frame` with each text column read as the type its values hold,
    under `parse_text`, and the format each was read with; without it, the
    type each would be read as. Under `parse_text`, why each column that
    nearly parsed stays text, unless not `explain`.

    A column in `replace` is left as text either way: its values are not
    carried over, and a number's extremes would be.
    """
    parsed: list[pl.Series] = []
    formats: dict[str, str] = {}
    would: dict[str, tuple[pl.DataType, str | None]] = {}
    stays: dict[str, str] = {}
    for name, dtype in frame.schema.items():
        if dtype != pl.String or name in opts.replace:
            continue
        read = infer_text(frame[name])
        if read is None:
            reason = stays_text(frame[name]) if opts.parse_text and explain else None
            if reason is not None:
                stays[name] = reason
            continue
        if opts.parse_text:
            parsed.append(read.values)
            if read.text_format is not None:
                formats[name] = read.text_format
        else:
            would[name] = (read.values.dtype, read.text_format)
    typed = frame.with_columns(parsed) if parsed else frame
    return _ReadText(typed, formats, would, stays)
