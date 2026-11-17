"""The profiling flags: each `ProfileOptions` field the command line sets,
declared once for every command that profiles.

`add_profile_flags` gives a parser the flags, and `profile_keywords` reads
them back as `ProfileOptions`' keywords, so a new option is one entry in
`_FLAGS`.

Internal: not part of the public API.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from polspec.profiler import step_of


def _rounding(text: str) -> object:
    """`--round-bounds` and `--round-rates`: a step (`1k`, `0.01`), `auto`
    or `exact`, read as `profile()` reads one."""
    try:
        return step_of(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _same(value: Any) -> Any:
    return value


def _negated(value: Any) -> bool:
    return not value


@dataclass(frozen=True, slots=True)
class _Flag:
    """One flag: the `ProfileOptions` field it sets, how its value reads as
    that field's, and what argparse is given."""

    option: str
    flag: str
    help: str
    to_option: Callable[[Any], Any] = _same
    argparse: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dest(self) -> str:
        return self.flag.removeprefix("--").replace("-", "_")


# In the order `--help` lists them.
_FLAGS = (
    _Flag(
        "weights",
        "--weights",
        "Record each category's observed frequency",
        argparse={"action": "store_true"},
    ),
    _Flag(
        "max_unique_enum",
        "--max-unique-enum",
        "Max distinct values for a text column to keep its values, as an "
        "Enum, when they repeat (default: 50)",
        argparse={"type": int, "default": 50, "metavar": "N"},
    ),
    _Flag(
        "formats",
        "--no-formats",
        "Do not name the format a text column's values have (email, uuid4, ...)",
        _negated,
        {"action": "store_true"},
    ),
    _Flag(
        "patterns",
        "--no-patterns",
        "Do not write the pattern a code-like text column's values share "
        "(ORD-[0-9]{6}, ...)",
        _negated,
        {"action": "store_true"},
    ),
    _Flag(
        "parse_text",
        "--parse-text",
        "Read a text column as the number, flag or date every value holds "
        "(Boolean, Int64, Float64, Date, Datetime)",
        argparse={"action": "store_true"},
    ),
    _Flag(
        "calculate_bounds",
        "--no-bounds",
        "Skip computing numeric/temporal bounds and string lengths",
        _negated,
        {"action": "store_true"},
    ),
    _Flag(
        "sample",
        "--sample",
        "Profile a random sample of N rows (seeded, so the same every run)",
        argparse={"type": int, "metavar": "N"},
    ),
    _Flag(
        "shape",
        "--shape",
        "Fit the distribution each numeric or temporal column follows",
        argparse={"action": "store_true"},
    ),
    _Flag(
        "detect_unique",
        "--keys",
        "Declare an all-distinct integer or text column unique",
        argparse={"action": "store_true"},
    ),
    _Flag(
        "detect_order",
        "--order",
        "Declare order_by when the rows come sorted by a column",
        argparse={"action": "store_true"},
    ),
    _Flag(
        "round_bounds",
        "--round-bounds",
        "Round each numeric column's bounds outward to STEP (1000, 1k, 10k, "
        "1M, 0.01), within its dtype, so the data stays inside; 'auto' keeps "
        "two significant figures of the span and whole seconds; 'exact' "
        "writes them as observed (default: auto)",
        argparse={"default": "auto", "type": _rounding, "metavar": "STEP"},
    ),
    _Flag(
        "round_rates",
        "--round-rates",
        "Round each null and NaN rate up to STEP (0.01), and each --weights "
        "frequency to its nearest STEP; 'auto' keeps two significant "
        "figures; 'exact' writes them as observed. A rate of 0 stays 0 "
        "(default: auto)",
        argparse={"default": "auto", "type": _rounding, "metavar": "STEP"},
    ),
    _Flag(
        "replace",
        "--replace",
        "Columns whose values must not be carried over: text in them is "
        "described by its lengths, never by the values the source holds",
        lambda columns: tuple(columns or ()),
        {"nargs": "+", "action": "extend", "metavar": "COL"},
    ),
)

# What `synthesize` profiles with: the source imitated as it stands, so
# no switch that would describe it less than whole.
SYNTHESIZE_OPTIONS = (
    "max_unique_enum",
    "formats",
    "patterns",
    "parse_text",
    "sample",
    "replace",
)


def add_profile_flags(
    parser: argparse.ArgumentParser, options: Sequence[str] | None = None
) -> None:
    """`parser` given the flag of each of `options` -- by default, every
    one -- in `_FLAGS`' order."""
    wanted = set(options) if options is not None else None
    for flag in _FLAGS:
        if wanted is None or flag.option in wanted:
            parser.add_argument(flag.flag, help=flag.help, **flag.argparse)


def profile_keywords(args: argparse.Namespace) -> dict[str, Any]:
    """`ProfileOptions`' keywords for each profiling flag `args` has."""
    return {
        flag.option: flag.to_option(getattr(args, flag.dest))
        for flag in _FLAGS
        if hasattr(args, flag.dest)
    }
