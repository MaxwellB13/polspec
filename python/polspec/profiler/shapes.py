"""The pattern a code-like text column's values share: each run of
digits, capitals or small letters a class of the lengths seen.

Internal: not part of the public API.
"""

from __future__ import annotations

import polars as pl

from polspec.profiler.formats import FORMAT_PROBE_ROWS

# The shapes a code-like column's values may have between them, each an
# alternative of its pattern. More is free text, or a column of mixed codes
# a pattern would only list.
SHAPE_MAX = 3


# The longest run of other characters -- a separator, a prefix -- a shape
# keeps. A longer one is prose.
SHAPE_MAX_LITERAL = 8


# A part of a shape every value of it shares is written as it is when at
# least this many values share it, or when one value has the shape alone (a
# stand-in, `N/A`). A part fewer share is chance, and written as its class.
SHAPE_MIN_SHARED = 10


# How many times its own count of values a key's shape must hold: room for a
# stand-in up to ten times the size.
SHAPE_KEY_ROOM = 10


# Values read for their shapes before any more are.
SHAPE_PROBE_ROWS = 1_000


# How many characters each class holds.
_CLASS_SIZES = {"[0-9]": 10, "[A-Z]": 26, "[a-z]": 26}


# Each character class, and the mark a run of it leaves in a value's shape:
# a control character, so no value's own text is read as one.
_CLASSES = (("[0-9]", "\x01"), ("[A-Z]", "\x02"), ("[a-z]", "\x03"))


_MARKS = r"[\x00-\x03]"


# Characters a regular expression reads as syntax outside a class.
_REGEX_SYNTAX = frozenset("\\.+*?()|[]{}^$")


def shape_pattern(non_null: pl.Series) -> tuple[str, int] | None:
    """The pattern `non_null`'s values share, when every one has one of at
    most `SHAPE_MAX` shapes. A shape is a value's runs of digits, capitals,
    small letters and other characters. Each run of a class is written as
    the class with the lengths seen, or as it is when enough values share it
    (`SHAPE_MIN_SHARED`); each run of other characters as it is. None when
    the values have more shapes than that, or a run of other characters is
    longer than `SHAPE_MAX_LITERAL`. Beside the pattern, how many values it
    holds. Classes, literals and one alternation are always within the
    generator's reach.

    The shapes are read from the first values, and the pattern drawn from
    them; only when a later value fails it are the lengths read from every
    value. A value of no shape the first showed means more shapes.
    """
    # Most columns that are not codes show more shapes in their first values.
    if _shapes(non_null.head(SHAPE_PROBE_ROWS)).n_unique() > SHAPE_MAX:
        return None
    head = non_null.head(FORMAT_PROBE_ROWS)
    if head.str.contains(_MARKS).any():
        return None  # a value holding a mark would be misread
    shapes = _shapes(head).unique().sort()
    if len(shapes) > SHAPE_MAX:
        return None  # most columns that are not codes, at once
    runs = [_shape_runs(shape) for shape in shapes]
    if any(r is None for r in runs):
        return None  # prose, not a code
    shaped = [r for r in runs if r is not None]
    drawn = _shape_drawn(shaped, head.unique())
    if drawn is not None and (
        len(head) == len(non_null) or non_null.str.contains(drawn[0]).all()
    ):
        return drawn
    if non_null.str.contains(_MARKS).any():
        return None
    return _shape_drawn(shaped, non_null.unique(), every=True)


def _shape_drawn(
    shapes: list[list[str]], values: pl.Series, *, every: bool = False
) -> tuple[str, int] | None:
    """The pattern of `values`, each of one of `shapes` (its runs), and how
    many values it holds. With `every`, None when a value has none of them."""
    alternatives = []
    for runs in shapes:
        matcher = (
            "^" + "".join(f"({r}+)" if r in _CLASS_SIZES else r for r in runs) + "$"
        )
        groups = values.str.extract_groups(matcher).struct.unnest().drop_nulls()
        alternatives.append((runs, groups))
    if every and sum(groups.height for _, groups in alternatives) != len(values):
        return None  # a shape the first values did not show
    # The commonest shape first; the shapes come sorted, which a tie keeps.
    alternatives.sort(key=lambda a: -a[1].height)
    written, holds = [], 0
    for runs, groups in alternatives:
        part, held = _shape_alternative(runs, groups)
        written.append(part)
        holds += held
    body = written[0] if len(written) == 1 else f"(?:{'|'.join(written)})"
    return f"^{body}$", holds


def _shapes(values: pl.Series) -> pl.Series:
    """Each value with every run of a class replaced by that class's mark."""
    for cls, mark in _CLASSES:
        values = values.str.replace_all(f"{cls}+", mark)
    return values


def _shape_runs(shape: str) -> list[str] | None:
    """`shape`'s runs: each class, and each run of other characters escaped
    as a regular expression; None when one of those is prose."""
    classes = {mark: cls for cls, mark in _CLASSES}
    runs: list[str] = []
    literal = ""
    for c in [*shape, None]:
        if c is not None and c not in classes:
            literal += c
            continue
        if len(literal) > SHAPE_MAX_LITERAL:
            return None
        if literal:
            runs.append("".join(f"\\{x}" if x in _REGEX_SYNTAX else x for x in literal))
            literal = ""
        if c is not None:
            runs.append(classes[c])
    return runs


def _shape_alternative(runs: list[str], groups: pl.DataFrame) -> tuple[str, int]:
    """The pattern of the values of one shape, from its `runs` and each
    class run's values (`groups`, a column per run in order), and how many
    values it holds."""
    shared_by_enough = groups.height == 1 or groups.height >= SHAPE_MIN_SHARED
    captured = iter(groups.columns)
    parts, holds = [], 1
    for cls in runs:
        if cls not in _CLASS_SIZES:
            parts.append(cls)  # other characters, as they are
            continue
        run = groups[next(captured)]
        lengths = run.str.len_chars()
        low, high = int(lengths.min()), int(lengths.max())  # ty: ignore[invalid-argument-type]
        first = run[0]
        if shared_by_enough and high <= SHAPE_MAX_LITERAL and (run == first).all():
            parts.append(first)  # letters or digits: nothing to escape
            continue
        holds *= sum(_CLASS_SIZES[cls] ** n for n in range(low, high + 1))
        if low == high == 1:
            parts.append(cls)
        else:
            parts.append(f"{cls}{{{low}}}" if low == high else f"{cls}{{{low},{high}}}")
    return "".join(parts), holds
