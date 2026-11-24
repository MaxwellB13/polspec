"""`ProfileOptions`, the keywords that stand for it, and `describe=`'s
defaults.

Internal: not part of the public API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypedDict

from polspec._options import options_from
from polspec.profiler.rounding import Rounding, step_of


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileOptions:
    """What `profile()` infers from data, and how finely it writes it.

    `profile`, `FrameSpec.from_dataframe` and `synthesize` take these as an
    `options=` object or as the keywords themselves -- one or the other, as
    `validate` takes `ValidationOptions`. Every one of them, and
    `polspec schema infer`, has these defaults; `describe=True` (`--describe`)
    records less unless asked: `weights`, `shape`, `detect_unique` and
    `detect_order` off. *Since 2.0: 1.x's `FrameSpec.from_dataframe` and
    `schema infer` recorded less by default, and `schema infer` rounded
    bounds and rates.*

    Parameters
    ----------
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
    formats : bool, default True
        Name the `format` a text column's values have -- an email, a UUID,
        an IP, a country code -- so its fake values have it too: when at
        least 20 distinct values have one of `email`, `uuid4`, `ipv4`,
        `ipv6`, `mac`, `hostname`, `iso_country` or `iso_currency`, and
        every value does -- or 99% do, and the rest are at most five
        repeated stand-ins, which become its `extra_values`.
    patterns : bool, default True
        Write the `pattern` a code-like text column's values share -- an
        order reference, a SKU, a postcode -- so its fake values have that
        shape: each run of digits, capitals or small letters a class of the
        lengths seen (`ORD-004211`, `ORD-918200` give `^ORD-[0-9]{6}$`), and
        a part every value shares kept as written. Only for a column of at
        least 20 distinct values that no format fits and that would not be
        an `Enum`, when every value has one of at most three shapes; for one
        of distinct values, which may be a key, only when the pattern holds
        ten times as many. Not for a column in `replace`, whose shared parts
        are real text, nor for numbers, dates or booleans written as text.
    weights : bool, default True
        Record each category's observed frequency, for categorical, enum and
        boolean columns.
    shape : bool, default True
        Fit the distribution each numeric or temporal column follows, so it
        generates that shape rather than uniformly between its extremes.
        Needs `calculate_bounds`.
    detect_unique : bool, default True
        Declare an integer or text column `unique=True` when every one of at
        least a hundred values is distinct.
    calculate_bounds : bool, default True
        Record the observed extremes of numeric and temporal columns, and the
        length range of text and binary ones.
    detect_order : bool, default True
        Declare `order_by` when the rows come sorted by a column of single
        values -- either way, nulls last, with at least 20 distinct values,
        so not by chance -- so a stand-in comes sorted too. The first such
        column, in column order. A sample keeps the source's row order, so
        a sorted file's sample shows it.
    round_bounds : number, "auto" or None, default None
        Write each numeric column's bounds rounded outward to this step --
        `1000`, `0.01` -- the minimum down and the maximum up, within what
        its dtype holds, so the data stays inside them. "auto" keeps two
        significant figures of the column's span (`[3.17, 9871.23]` becomes
        `[0, 9900]`) and rounds a datetime, time or duration outward to the
        second. None writes them exactly as observed.
    round_rates : number, "auto" or None, default None
        Write each null, NaN and list-element null rate rounded up to this
        step, never past 1; "auto" keeps two significant figures. A rate of
        0 stays 0. Category frequencies (`weights`) are written at the same
        precision, each to its nearest step and never to 0. None writes them
        exactly as observed.
    parse_text : bool, default False
        Read a text column as the type its values hold, for a frame whose
        types were switched off (`pl.read_csv(..., infer_schema=False)`):
        `Boolean`, `Int64`, `Float64`, `Date` or `Datetime`, the first that
        every present value parses as exactly. Text is stripped and empty
        text is a null. A column with a value written with a leading zero
        (`007`) stays text, being a code, as does one in `replace`. Left
        False, such a column stays text and its description says what it
        would parse as. Validate the same frame with `parse_text=True` too.


    Examples
    --------
    >>> house = ProfileOptions(round_bounds="auto", replace=["name"])
    >>> frame = pl.DataFrame({"name": ["Ada", "Grace"] * 50, "total": [3.17, 9871.23] * 50})
    >>> profile(frame, options=house)["total"].bounds
    Bound(min=0.0, max=9900.0)
    """

    replace: Sequence[str] = ()
    sample: int | None = None
    seed: int = 0
    max_unique_enum: int = 50
    formats: bool = True
    patterns: bool = True
    weights: bool = True
    shape: bool = True
    detect_unique: bool = True
    calculate_bounds: bool = True
    detect_order: bool = True
    round_bounds: Rounding = None
    round_rates: Rounding = None
    parse_text: bool = False

    def __post_init__(self) -> None:
        replace: object = self.replace
        if isinstance(replace, str):
            raise TypeError(
                f"replace takes a sequence of column names, got the string {replace!r}"
            )
        object.__setattr__(self, "replace", tuple(self.replace))
        object.__setattr__(self, "round_bounds", step_of(self.round_bounds))
        object.__setattr__(self, "round_rates", step_of(self.round_rates))


class SynthesizeKeywords(TypedDict, total=False):
    """`ProfileOptions`' fields as `synthesize()` takes them: all but
    `seed`, which there seeds the stand-in as well as its profile."""

    replace: Sequence[str]
    sample: int | None
    max_unique_enum: int
    formats: bool
    patterns: bool
    weights: bool
    shape: bool
    detect_unique: bool
    calculate_bounds: bool
    detect_order: bool
    round_bounds: Rounding
    round_rates: Rounding
    parse_text: bool


class ProfileKeywords(SynthesizeKeywords, total=False):
    """`ProfileOptions`' fields, as the keywords `profile()` takes them."""

    seed: int


def profile_options(
    options: ProfileOptions | None,
    keywords: Mapping[str, Any],
    *,
    describe: bool = False,
) -> ProfileOptions:
    """The options for one call: the object, or the keywords -- over
    `DESCRIBE_DEFAULTS` when `describe`, the lean form."""
    if describe:
        if options is not None:
            raise TypeError(
                "describe=True sets what the keywords default to; with "
                "options=, set its fields instead: ProfileOptions(weights=False, "
                "shape=False, detect_unique=False, detect_order=False)."
            )
        keywords = {**DESCRIBE_DEFAULTS, **keywords}
    return options_from(ProfileOptions, options, keywords, what="profiling")


# What `describe=True` leaves out unless asked: the shape of the data, not
# a stand-in for it. *1.x's defaults for `FrameSpec.from_dataframe`.*
DESCRIBE_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {"weights": False, "shape": False, "detect_unique": False, "detect_order": False}
)
