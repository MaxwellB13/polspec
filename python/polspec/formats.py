"""Named string formats: what each one looks like, said once for both sides.

A `String` column can say how long its values are; `format=` lets it say what
they look like. Each format below is one `Format`: the *template* the engine
fills a value from, and the *check* validation runs over a column. They sit
in the same declaration so that a reviewer sees them agree, and so that the
round-trip test -- generate, then validate -- is what keeps them agreeing.

The engine knows nothing about emails or UUIDs. `src/format.rs` fills a
template, which is a run of parts: a literal, a stretch of characters drawn
from an alphabet, or one value from a list. That is enough to spell every
format here, and it keeps the knowledge of what a format *is* in this file.

The two ISO formats are finite domains rather than templates. They take the
same path `choices` does -- the engine samples indices, Python gathers the
codes -- so a `unique=True` column over one of them is refused, by name, when
the frame outgrows the list.

The set is closed on purpose. A named format is twenty lines with an
unambiguous validator; a `pattern=<regex>` would need a regex-to-sampler
compiler, and has no answer for `.*`. What a format promises is syntax:
`format="email"` generates a well-formed address, not a deliverable one, and
validates the shape, not the existence.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from polspec.errors import SpecError

# One tuple shape for every part, matching `RawPart` in `src/format.rs`:
# `(kind, strings, lo, hi)`. `strings` is the literal text, the alphabet, or
# the values; `lo`/`hi` are the length range and are read only by `chars`.
RawPart = tuple[str, list[str], int, int]


def lit(text: str) -> RawPart:
    """A part emitted as written."""
    return ("lit", [text], 0, 0)


def chars(alphabet: str, min_len: int, max_len: int | None = None) -> RawPart:
    """`min_len..=max_len` characters, each drawn uniformly from `alphabet`."""
    return ("chars", [alphabet], min_len, max_len if max_len is not None else min_len)


def one_of(values: Sequence[str]) -> RawPart:
    """One of `values`, uniformly."""
    return ("one_of", list(values), 0, 0)


@dataclass(frozen=True, slots=True)
class Format:
    """One named format: how to fill a value, and how to check one.

    Exactly one of `template` and `values` is set. A template format is
    sampled by the engine and checked against `pattern` (and `max_length`,
    where the pattern alone cannot say); a finite format is sampled as an
    index into `values` and checked by membership.
    """

    name: str
    summary: str
    template: tuple[RawPart, ...] | None = None
    pattern: str | None = None
    max_length: int | None = None
    values: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if (self.template is None) == (self.values is None):
            raise ValueError(f"Format {self.name!r} needs a template or values")
        if self.template is not None and self.pattern is None:
            raise ValueError(f"Format {self.name!r} has a template but no pattern")

    @property
    def is_finite(self) -> bool:
        """Whether this format is a list of values rather than a shape."""
        return self.values is not None

    def check(self, column: pl.Expr) -> pl.Expr:
        """True where a value of `column` has this format; null stays null."""
        if self.values is not None:
            return column.is_in(list(self.values))
        if self.pattern is None:  # pragma: no cover - a format has values or a pattern
            raise ValueError(f"format {self.name!r} has neither values nor a pattern")
        matches = column.str.contains(self.pattern)
        if self.max_length is not None:
            matches = matches & (column.str.len_chars() <= self.max_length)
        return matches

    def __str__(self) -> str:
        return f"{self.name} ({self.summary})"


# ---------------------------------------------------------------------------
# The formats
# ---------------------------------------------------------------------------

_LOWER = "abcdefghijklmnopqrstuvwxyz"
_DIGITS = "0123456789"
_HEX = _DIGITS + "abcdef"
_OCTETS = [str(i) for i in range(256)]
_TLDS = (".com", ".org", ".net", ".io", ".dev", ".co", ".edu", ".gov")

# RFC 1123: a label is 1-63 alphanumerics, hyphens allowed inside.
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"


def _hex_groups(width: int, count: int, sep: str) -> tuple[RawPart, ...]:
    """`count` groups of `width` hex characters, joined by `sep`."""
    parts: list[RawPart] = []
    for i in range(count):
        if i:
            parts.append(lit(sep))
        parts.append(chars(_HEX, width))
    return tuple(parts)


# ISO 3166-1 alpha-2, the 249 officially assigned codes.
_ISO_COUNTRIES = [
    "AD",
    "AE",
    "AF",
    "AG",
    "AI",
    "AL",
    "AM",
    "AO",
    "AQ",
    "AR",
    "AS",
    "AT",
    "AU",
    "AW",
    "AX",
    "AZ",
    "BA",
    "BB",
    "BD",
    "BE",
    "BF",
    "BG",
    "BH",
    "BI",
    "BJ",
    "BL",
    "BM",
    "BN",
    "BO",
    "BQ",
    "BR",
    "BS",
    "BT",
    "BV",
    "BW",
    "BY",
    "BZ",
    "CA",
    "CC",
    "CD",
    "CF",
    "CG",
    "CH",
    "CI",
    "CK",
    "CL",
    "CM",
    "CN",
    "CO",
    "CR",
    "CU",
    "CV",
    "CW",
    "CX",
    "CY",
    "CZ",
    "DE",
    "DJ",
    "DK",
    "DM",
    "DO",
    "DZ",
    "EC",
    "EE",
    "EG",
    "EH",
    "ER",
    "ES",
    "ET",
    "FI",
    "FJ",
    "FK",
    "FM",
    "FO",
    "FR",
    "GA",
    "GB",
    "GD",
    "GE",
    "GF",
    "GG",
    "GH",
    "GI",
    "GL",
    "GM",
    "GN",
    "GP",
    "GQ",
    "GR",
    "GS",
    "GT",
    "GU",
    "GW",
    "GY",
    "HK",
    "HM",
    "HN",
    "HR",
    "HT",
    "HU",
    "ID",
    "IE",
    "IL",
    "IM",
    "IN",
    "IO",
    "IQ",
    "IR",
    "IS",
    "IT",
    "JE",
    "JM",
    "JO",
    "JP",
    "KE",
    "KG",
    "KH",
    "KI",
    "KM",
    "KN",
    "KP",
    "KR",
    "KW",
    "KY",
    "KZ",
    "LA",
    "LB",
    "LC",
    "LI",
    "LK",
    "LR",
    "LS",
    "LT",
    "LU",
    "LV",
    "LY",
    "MA",
    "MC",
    "MD",
    "ME",
    "MF",
    "MG",
    "MH",
    "MK",
    "ML",
    "MM",
    "MN",
    "MO",
    "MP",
    "MQ",
    "MR",
    "MS",
    "MT",
    "MU",
    "MV",
    "MW",
    "MX",
    "MY",
    "MZ",
    "NA",
    "NC",
    "NE",
    "NF",
    "NG",
    "NI",
    "NL",
    "NO",
    "NP",
    "NR",
    "NU",
    "NZ",
    "OM",
    "PA",
    "PE",
    "PF",
    "PG",
    "PH",
    "PK",
    "PL",
    "PM",
    "PN",
    "PR",
    "PS",
    "PT",
    "PW",
    "PY",
    "QA",
    "RE",
    "RO",
    "RS",
    "RU",
    "RW",
    "SA",
    "SB",
    "SC",
    "SD",
    "SE",
    "SG",
    "SH",
    "SI",
    "SJ",
    "SK",
    "SL",
    "SM",
    "SN",
    "SO",
    "SR",
    "SS",
    "ST",
    "SV",
    "SX",
    "SY",
    "SZ",
    "TC",
    "TD",
    "TF",
    "TG",
    "TH",
    "TJ",
    "TK",
    "TL",
    "TM",
    "TN",
    "TO",
    "TR",
    "TT",
    "TV",
    "TW",
    "TZ",
    "UA",
    "UG",
    "UM",
    "US",
    "UY",
    "UZ",
    "VA",
    "VC",
    "VE",
    "VG",
    "VI",
    "VN",
    "VU",
    "WF",
    "WS",
    "YE",
    "YT",
    "ZA",
    "ZM",
    "ZW",
]

# ISO 4217 alpha-3, the codes of currencies in circulation. Fund codes
# (`BOV`, `CLF`, `USN`, ...), precious metals (`XAU`, ...), the SDR and the
# `XTS`/`XXX` placeholders are left out: none is a currency a price is in.
_ISO_CURRENCIES = [
    "AED",
    "AFN",
    "ALL",
    "AMD",
    "ANG",
    "AOA",
    "ARS",
    "AUD",
    "AWG",
    "AZN",
    "BAM",
    "BBD",
    "BDT",
    "BGN",
    "BHD",
    "BIF",
    "BMD",
    "BND",
    "BOB",
    "BRL",
    "BSD",
    "BTN",
    "BWP",
    "BYN",
    "BZD",
    "CAD",
    "CDF",
    "CHF",
    "CLP",
    "CNY",
    "COP",
    "CRC",
    "CUP",
    "CVE",
    "CZK",
    "DJF",
    "DKK",
    "DOP",
    "DZD",
    "EGP",
    "ERN",
    "ETB",
    "EUR",
    "FJD",
    "FKP",
    "GBP",
    "GEL",
    "GHS",
    "GIP",
    "GMD",
    "GNF",
    "GTQ",
    "GYD",
    "HKD",
    "HNL",
    "HTG",
    "HUF",
    "IDR",
    "ILS",
    "INR",
    "IQD",
    "IRR",
    "ISK",
    "JMD",
    "JOD",
    "JPY",
    "KES",
    "KGS",
    "KHR",
    "KMF",
    "KPW",
    "KRW",
    "KWD",
    "KYD",
    "KZT",
    "LAK",
    "LBP",
    "LKR",
    "LRD",
    "LSL",
    "LYD",
    "MAD",
    "MDL",
    "MGA",
    "MKD",
    "MMK",
    "MNT",
    "MOP",
    "MRU",
    "MUR",
    "MVR",
    "MWK",
    "MXN",
    "MYR",
    "MZN",
    "NAD",
    "NGN",
    "NIO",
    "NOK",
    "NPR",
    "NZD",
    "OMR",
    "PAB",
    "PEN",
    "PGK",
    "PHP",
    "PKR",
    "PLN",
    "PYG",
    "QAR",
    "RON",
    "RSD",
    "RUB",
    "RWF",
    "SAR",
    "SBD",
    "SCR",
    "SDG",
    "SEK",
    "SGD",
    "SHP",
    "SLE",
    "SOS",
    "SRD",
    "SSP",
    "STN",
    "SVC",
    "SYP",
    "SZL",
    "THB",
    "TJS",
    "TMT",
    "TND",
    "TOP",
    "TRY",
    "TTD",
    "TWD",
    "TZS",
    "UAH",
    "UGX",
    "USD",
    "UYU",
    "UZS",
    "VED",
    "VES",
    "VND",
    "VUV",
    "WST",
    "XAF",
    "XCD",
    "XCG",
    "XOF",
    "XPF",
    "YER",
    "ZAR",
    "ZMW",
    "ZWG",
]


FORMATS: dict[str, Format] = {
    f.name: f
    for f in (
        Format(
            "uuid4",
            "a version-4 UUID in canonical 8-4-4-4-12 form",
            template=(
                chars(_HEX, 8),
                lit("-"),
                chars(_HEX, 4),
                lit("-4"),
                chars(_HEX, 3),
                lit("-"),
                one_of(("8", "9", "a", "b")),
                chars(_HEX, 3),
                lit("-"),
                chars(_HEX, 12),
            ),
            pattern=(
                r"(?i)^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-"
                r"[0-9a-f]{12}$"
            ),
        ),
        Format(
            "email",
            "an address with one '@' and a dotted domain",
            template=(
                chars(_LOWER + _DIGITS, 3, 12),
                lit("@"),
                chars(_LOWER, 3, 10),
                one_of(_TLDS),
            ),
            pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        ),
        Format(
            "ipv4",
            "four dotted decimal octets, each 0-255",
            template=(
                one_of(_OCTETS),
                lit("."),
                one_of(_OCTETS),
                lit("."),
                one_of(_OCTETS),
                lit("."),
                one_of(_OCTETS),
            ),
            pattern=(
                r"^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}"
                r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])$"
            ),
        ),
        Format(
            "ipv6",
            "eight colon-separated hextets, uncompressed",
            template=_hex_groups(4, 8, ":"),
            pattern=r"(?i)^(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}$",
        ),
        Format(
            "mac",
            "six colon-separated hex pairs",
            template=_hex_groups(2, 6, ":"),
            pattern=r"(?i)^(?:[0-9a-f]{2}:){5}[0-9a-f]{2}$",
        ),
        Format(
            "hostname",
            "dot-separated RFC 1123 labels, at most 253 characters",
            template=(
                chars(_LOWER + _DIGITS, 2, 12),
                one_of(("", ".internal", ".corp", ".svc", ".eu", ".us", ".staging")),
                one_of(_TLDS),
            ),
            pattern=rf"(?i)^{_LABEL}(?:\.{_LABEL})*$",
            max_length=253,
        ),
        Format(
            "iso_country",
            "an ISO 3166-1 alpha-2 country code",
            values=tuple(_ISO_COUNTRIES),
        ),
        Format(
            "iso_currency",
            "an ISO 4217 alpha-3 currency code",
            values=tuple(_ISO_CURRENCIES),
        ),
    )
}


def names() -> list[str]:
    """Every format `ColSpec(format=...)` accepts, in the order they are documented."""
    return list(FORMATS)


def lookup(name: str) -> Format:
    """The `Format` called `name`, or a `SpecError` naming the nearest one."""
    fmt = FORMATS.get(name)
    if fmt is not None:
        return fmt
    close = difflib.get_close_matches(str(name), list(FORMATS), n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    raise SpecError(
        f"ColSpec.format {name!r} is not a known format.{hint} "
        f"Known formats: {', '.join(FORMATS)}"
    )
