"""Which named `format` -- an email, a UUID, a country code -- a text
column's values have.

Internal: not part of the public API.
"""

from __future__ import annotations

import polars as pl

from polspec import frames
from polspec.string_formats import lookup as _lookup_format

# A text column is named a `format` only when at least this many distinct
# values have it: three rows of "a@b.co" are a short Enum, not evidence.
FORMAT_MIN_DISTINCT = 20


# The formats a column is tried against, most specific first; the first
# that fits wins. A finite format (a country, a currency) is tried only where
# the column would not narrow to an Enum of the codes it holds, which keeps
# their frequencies -- see `_profile_textual`.
FORMAT_ORDER = (
    "uuid4",
    "uuid7",
    "ulid",
    "ipv6",
    "ipv4",
    "mac",
    "url",
    "iso_country",
    "iso_country3",
    "iso_currency",
    "email",
    "hostname",
)


# A near miss: a column where nearly every value has the format and the rest
# are a few repeated stand-ins -- `N/A`, `UK (ISO)` -- is that format with
# those values as `extra_values`, at the share each is observed at.
FORMAT_MIN_SHARE = 0.99


FORMAT_MAX_EXTRAS = 5


# Rows the format probe reads its distinct values from.
FORMAT_PROBE_ROWS = 20_000


def _mostly_dotted(non_null: pl.Series, total: int) -> bool:
    """Whether enough rows hold a dot for a hostname to be named at all: it
    is named only when its dotted values cover `FORMAT_MIN_SHARE` of the
    rows. Read without hashing a value, since a column of single words
    passes the hostname check and would otherwise be counted whole."""
    dotted = int(non_null.str.contains(".", literal=True).sum())
    return dotted >= FORMAT_MIN_SHARE * total


def named_format(
    non_null: pl.Series, *, keep_values: bool, finite: bool = True, distinct: int
) -> tuple[str, dict[str, float]] | None:
    """The format `non_null`'s values have, and the stand-ins beside it with
    their shares; None when no format fits.

    Asked of the distinct values, weighted by how often each occurs, with
    the check validation uses -- so a spec that names a format accepts the
    data it was named from. A near miss is kept only with `keep_values`:
    its stand-ins are values of the source. A finite format is tried only
    with `finite`. `distinct` is how many distinct values `non_null` holds.
    """
    value, count = frames.PLAIN, frames.COUNT
    total = len(non_null)
    # A format tolerates at most FORMAT_MAX_EXTRAS distinct misses -- none,
    # without `keep_values` -- so a probe of any thousand distinct values
    # holding more rules it out exactly, before free text is counted whole.
    # The probe is the distinct values the column starts with, so free text
    # is ruled out without counting a single value of it.
    allowed_misses = FORMAT_MAX_EXTRAS if keep_values else 0
    probe = (
        non_null.head(FORMAT_PROBE_ROWS)
        .unique(maintain_order=True)
        .head(1_000)
        .alias(value)
        .to_frame()
    )
    counts: pl.DataFrame | None = None
    for name in FORMAT_ORDER:
        fmt = _lookup_format(name)
        if fmt.is_finite and not finite:
            continue
        # What the format accepts, and what counts as evidence of it. They
        # differ for a hostname: a single word passes the check -- so a
        # column of first names would be hostnames -- and so does a decimal,
        # `0.45` being host `0` in domain `45`. Evidence is a dotted value
        # whose last label is not all digits, as no real hostname's is (RFC
        # 1123). A value that passes is no stand-in, though: the format
        # accepts it, and `extra_values` may hold only what it does not.
        fits = fmt.check(frames.column(value)).fill_null(False)
        evidence = fits
        if name == "hostname":
            text = frames.column(value)
            evidence = (
                fits
                & text.str.contains(".", literal=True)
                & ~text.str.contains(r"\.[0-9]+$")
            )
        misses = probe.select((~fits).sum()).item()
        if misses > allowed_misses:
            continue
        if (
            misses == 0
            and name != "hostname"
            and distinct >= FORMAT_MIN_DISTINCT
            and non_null.alias(value).to_frame().select(fits.all()).item()
        ):
            # Every value fits: the column is the format, with no stand-in,
            # which is what counting every distinct value would conclude.
            # One pass over the column, not a count of it. A hostname needs
            # evidence beyond fitting, so it is counted.
            return name, {}
        if name == "hostname" and not _mostly_dotted(non_null, total):
            continue  # single words: evidence of no format, however many
        if counts is None:
            counts = frames.value_counts(non_null)
        judged = counts.with_columns(
            fits.alias("__polspec_fits"), evidence.alias("__polspec_evidence")
        )
        passing = judged.filter(pl.col("__polspec_evidence"))
        if passing.height < FORMAT_MIN_DISTINCT:
            continue
        passing_rows = int(passing[count].sum())
        # Most frequent first, then by value: `value_counts` keeps no order,
        # and a spec file should not change between runs on the same data.
        failing = judged.filter(~pl.col("__polspec_fits")).sort(
            [count, value], descending=[True, False]
        )
        if failing.height == 0 and passing_rows >= FORMAT_MIN_SHARE * total:
            return name, {}
        if (
            keep_values
            and passing_rows >= FORMAT_MIN_SHARE * total
            and failing.height <= FORMAT_MAX_EXTRAS
            and bool((failing[count] >= 2).all())
        ):
            return name, {
                str(extra): int(seen) / total
                for extra, seen in zip(
                    failing[value].to_list(), failing[count].to_list(), strict=True
                )
            }
    return None
