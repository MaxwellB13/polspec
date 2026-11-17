"""`tolerate=`: findings a pipeline accepts up to a share of the rows.

A finding that counts rows carries `share`, the fraction of the frame's rows
it covers. A tolerance says how large a share is acceptable: for every
finding, or for one kind of finding, one column, or one kind of finding in
one column. A finding at or under its tolerance stays in the report,
marked `tolerated`, and does not fail it. One that judges the whole frame
-- a missing column, a wrong dtype -- has no share, and is never tolerated.

Internal: not part of the public API.
"""

from __future__ import annotations

import dataclasses
import difflib
import math
from collections.abc import Mapping, Sequence
from typing import Any, get_args

from polspec._frozen import FrozenDict
from polspec.validation.report import Finding, FindingCode

__all__ = ["CODES", "Tolerance", "checked_tolerance", "tolerated"]

CODES: frozenset[str] = frozenset(get_args(FindingCode))


class _Tolerances(FrozenDict):
    """`tolerate=` as a mapping, hashable, so `ValidationOptions` holding one
    hashes as it always has."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())

    def __hash__(self) -> int:
        return hash(frozenset(self.items()))


type TolerateKey = str | tuple[str, str]
type Tolerance = float | Mapping[TolerateKey, float] | None


def _share(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"tolerate= {where} must be a share of the rows, a number from 0 "
            f"to 1, got {type(value).__name__}"
        )
    if not (math.isfinite(value) and 0 <= value <= 1):
        raise ValueError(
            f"tolerate= {where} must be a share of the rows from 0 to 1, "
            f"got {value!r}; 0.001 tolerates up to 0.1% of rows"
        )
    return float(value)


def checked_tolerance(value: object) -> Tolerance:
    """`tolerate=` as `ValidationOptions` holds it: a share, a frozen
    mapping of keys to shares, or None. The keys' names are checked against
    a spec when one is validated (`tolerated`)."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return _share(value, "as one number")
    checked: dict[TolerateKey, float] = {}
    for key, share in value.items():
        if isinstance(key, tuple):
            if len(key) != 2 or not all(isinstance(part, str) for part in key):
                raise TypeError(
                    f"tolerate= takes a (column, code) pair as a key, got {key!r}"
                )
            if key[1] not in CODES:
                raise ValueError(_unknown_code(key[1], f"in the key {key!r}"))
        elif not isinstance(key, str):
            raise TypeError(
                "tolerate= keys are finding codes, column names or (column, "
                f"code) pairs, got {type(key).__name__}"
            )
        checked[key] = _share(share, f"for {key!r}")
    return _Tolerances(checked)


def _unknown_code(code: str, where: str) -> str:
    close = difflib.get_close_matches(code, sorted(CODES), n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    return f"tolerate= names the finding code {code!r} {where}, which is none.{hint}"


def _check_keys(tolerance: Mapping[TolerateKey, float], columns: Sequence[str]) -> None:
    """Each key names a code or a column of the spec, and not both."""
    known = set(columns)
    for key in tolerance:
        if isinstance(key, tuple):
            if key[0] not in known:
                raise ValueError(_unknown_column(key[0], columns))
            continue
        if key in CODES and key in known:
            raise ValueError(
                f"tolerate= key {key!r} is both a finding code and a column; "
                f"say which with ({key!r}, code) for the column, or name the "
                "code's column too"
            )
        if key not in CODES and key not in known:
            close = difflib.get_close_matches(key, sorted(CODES | known), n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise ValueError(
                f"tolerate= key {key!r} is neither a finding code nor a column "
                f"of the spec.{hint}"
            )


def _unknown_column(column: str, columns: Sequence[str]) -> str:
    close = difflib.get_close_matches(column, list(columns), n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    return f"tolerate= names the column {column!r}, which the spec has none of.{hint}"


def _limit(finding: Finding, tolerance: Mapping[TolerateKey, float]) -> float | None:
    """The most specific tolerance for `finding`: its code in one of its
    columns, then one of its columns, then its code. The largest, where
    several of its columns have one."""
    levels = (
        [tolerance.get((column, finding.code)) for column in finding.columns],
        [tolerance.get(column) for column in finding.columns],
        [tolerance.get(finding.code)],
    )
    for level in levels:
        found = [share for share in level if share is not None]
        if found:
            return max(found)
    return None


def tolerated(
    findings: Sequence[Finding], tolerance: Tolerance, columns: Sequence[str]
) -> list[Finding]:
    """`findings`, each one a tolerance covers marked `tolerated`."""
    if tolerance is None:
        return list(findings)
    if isinstance(tolerance, Mapping):
        _check_keys(tolerance, columns)
    marked: list[Finding] = []
    for finding in findings:
        limit: Any = (
            tolerance
            if not isinstance(tolerance, Mapping)
            else _limit(finding, tolerance)
        )
        if finding.share is not None and limit is not None and finding.share <= limit:
            finding = dataclasses.replace(finding, tolerated=True)
        marked.append(finding)
    return marked
