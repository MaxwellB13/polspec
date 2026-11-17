"""A text column's format, the extra values beside it, and its pattern.

Internal: not part of the public API.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from polspec import spec as colspec_module
from polspec._frozen import FrozenDict
from polspec.constants import (
    DEFAULT_EXTRA_SHARE,
)
from polspec.errors import SpecError
from polspec.string_formats import lookup as _lookup_format


def validate_format(spec: colspec_module.ColSpec) -> None:
    """Rejects a format on a column it cannot describe, or beside a
    second definition of the same domain.

    A format is the whole story of what a `String` value looks like, so
    anything else that says what the values are -- `choices`, or the
    length a `uuid4` already fixes at 36 -- is a contradiction of the
    kind `_validate_choices_against_domain` already refuses, and is
    refused here in the same voice: drop one or the other.
    """
    if spec.format is None:
        return
    fmt = _lookup_format(spec.format)  # raises, naming the nearest format
    object.__setattr__(spec, "format", fmt.name)
    if spec.value_dtype not in (pl.String, pl.Utf8):
        raise SpecError(
            f"ColSpec.format is only supported for pl.String, got "
            f"{spec.value_dtype!r}. A format describes the text a value is "
            "written as, which no other dtype holds."
        )
    if spec.choices is not None:
        raise SpecError(
            f"ColSpec cannot carry both format={fmt.name!r} and choices: "
            "each is a complete description of the column's domain, and "
            "they cannot both hold. Drop the choices, or the format -- or, "
            "for values the column holds beside the format's own, declare "
            "them as extra_values."
        )
    if spec.string_length is not None:
        raise SpecError(
            f"ColSpec cannot carry both format={fmt.name!r} and "
            "string_length: the format already fixes how long a value is. "
            "Drop the string_length, or the format."
        )


def validate_extra_values(spec: colspec_module.ColSpec) -> None:
    """Normalises `extra_values` to a read-only mapping of value to
    share, and refuses what cannot mean anything.

    The extras widen a format's domain, so they need a format to widen,
    must each add something the format does not already accept -- else
    its share would be counted twice -- and cannot claim more than every
    row. A unique column cannot hold an extra twice, and a share of a
    large frame is many rows of it.
    """
    given: Any = spec.extra_values
    if given is None:
        return
    if spec.format is None:
        raise SpecError(
            "ColSpec.extra_values needs a format to extend. A String column "
            "without one lists its values as choices."
        )
    if isinstance(given, str) or not isinstance(given, (Sequence, Mapping)):
        raise SpecError(
            "ColSpec.extra_values must be a list of values, or a mapping of "
            f"value to share, got {given!r}"
        )
    if isinstance(given, Mapping):
        pairs = list(given.items())
    else:
        pairs = [(value, DEFAULT_EXTRA_SHARE) for value in given]
        repeated = sorted({v for v in given if list(given).count(v) > 1})
        if repeated:
            raise SpecError(f"ColSpec.extra_values repeats {repeated}")
    if not pairs:
        raise SpecError("ColSpec.extra_values must not be empty")
    shares: dict[str, float] = {}
    for value, share in pairs:
        if not isinstance(value, str) or not value:
            raise SpecError(
                f"ColSpec.extra_values holds {value!r}: an extra is a "
                "non-empty string, as a format's own values are"
            )
        share = float(share)
        if not 0.0 < share <= 1.0:
            raise SpecError(
                f"ColSpec.extra_values gives {value!r} a share of {share}; "
                "a share is a fraction of the present rows, above 0 and at "
                "most 1"
            )
        shares[value] = share
    total = sum(shares.values())
    if total > 1.0 + 1e-9:
        raise SpecError(
            f"ColSpec.extra_values shares sum to {total:g}, past every row. "
            "Lower them so they sum to 1 at most."
        )
    fmt = _lookup_format(spec.format)
    accepted = (
        pl.Series(list(shares), dtype=pl.String)
        .to_frame("v")
        .select(fmt.check(pl.col("v")))
        .to_series()
        .to_list()
    )
    already = [v for v, ok in zip(shares, accepted, strict=True) if ok]
    if already:
        raise SpecError(
            f"ColSpec.extra_values {already} already have format "
            f"{fmt.name!r}, so each would be counted twice. Drop them: "
            "the format's own values are generated and accepted already."
        )
    if spec.unique:
        raise SpecError(
            "ColSpec cannot be unique=True and carry extra_values: an "
            "extra drawn on a share of the rows repeats. Drop the extras, "
            "or the uniqueness."
        )
    object.__setattr__(spec, "extra_values", FrozenDict(shares))


def validate_pattern(spec: colspec_module.ColSpec) -> None:
    """Refuses a pattern on a column it cannot describe, beside a format,
    or one Polars cannot compile.

    Compiled by Polars itself rather than Python's `re`: the two dialects
    differ (look-around, for one), and the engine that will run the
    check is the one whose opinion counts.
    """
    if spec.pattern is None:
        return
    if not isinstance(spec.pattern, str) or not spec.pattern:
        raise SpecError(
            f"ColSpec.pattern must be a non-empty string, got {spec.pattern!r}"
        )
    if spec.value_dtype not in (pl.String, pl.Utf8):
        raise SpecError(
            f"ColSpec.pattern is only supported for pl.String, got "
            f"{spec.value_dtype!r}. A pattern describes the text a value is "
            "written as, which no other dtype holds."
        )
    if spec.format is not None:
        raise SpecError(
            f"ColSpec cannot carry both format={spec.format!r} and pattern: "
            "a format is a pattern polspec can also generate. Drop the "
            "pattern, or the format."
        )
    try:
        pl.select(pl.lit("").str.contains(spec.pattern))
    except Exception as exc:
        raise SpecError(
            f"ColSpec.pattern {spec.pattern!r} is not a valid regular "
            f"expression: {' '.join(str(exc).split())}"
        ) from exc


# The strftime directives a date's text may use, and a time's; a datetime's
# may use both. Checked here because Polars panics, rather than raising, on
# a directive its dtype cannot write -- `%d` for a time.
_DATE_DIRECTIVES = frozenset(
    ["Y", "y", "C", "m", "-m", "d", "-d", "e", "b", "B", "h", "a", "A", "j", "-j", "u", "w", "U", "W", "G", "g", "V", "D", "F", "x"]
)  # fmt: skip
_TIME_DIRECTIVES = frozenset(
    ["H", "-H", "k", "I", "-I", "l", "M", "-M", "S", "-S", "p", "P", "f", ".f", "3f", "6f", "9f", ".3f", ".6f", ".9f", "T", "R", "r", "X"]
)  # fmt: skip
_DIRECTIVE = r"%(%|-?[A-Za-z]|\.?[369]?f)"


def validate_text_format(spec: colspec_module.ColSpec) -> None:
    """Refuses a `text_format` on a column that is not a date, a datetime or
    a time, or one with a directive the column's dtype cannot write."""
    fmt = spec.text_format
    if fmt is None:
        return
    if not isinstance(fmt, str) or not fmt:
        raise SpecError(f"ColSpec.text_format must be a non-empty string, got {fmt!r}")
    dtype = spec.dtype
    if dtype == pl.Date:
        allowed, what = _DATE_DIRECTIVES, "a date"
    elif dtype == pl.Time:
        allowed, what = _TIME_DIRECTIVES, "a time"
    elif isinstance(dtype, pl.Datetime):
        allowed, what = _DATE_DIRECTIVES | _TIME_DIRECTIVES, "a datetime"
    else:
        raise SpecError(
            f"ColSpec.text_format is the way a date, a datetime or a time is "
            f"written as text, so it needs a Date, Datetime or Time column, "
            f"not {dtype!r}. A list's values are never text in a file."
        )
    used = [m.group(1) for m in re.finditer(_DIRECTIVE, fmt)]
    unknown = sorted({d for d in used if d != "%" and d not in allowed})
    if unknown:
        raise SpecError(
            f"ColSpec.text_format {fmt!r} uses "
            + ", ".join(f"%{d}" for d in unknown)
            + f", which {what} cannot be written with"
        )
    if not used or used == ["%"] * len(used):
        raise SpecError(
            f"ColSpec.text_format {fmt!r} has no directive, such as %Y or %H, "
            "so it writes every value the same"
        )
