"""Related tables profiled together.

`profile_all()` profiles a set of frames, each as `profile()` would, and
finds the foreign keys between them from their values: a column whose
every value is a value of a unique column elsewhere, with a name that says
so, or text with too many values to be contained by chance. The result is
a `Registry`, so `generate_all` and `validate_all` take it as it is.

Internal: not part of the public API.
"""

from __future__ import annotations

import dataclasses
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Unpack

import polars as pl

from polspec.dtypes import dtype_value_limits
from polspec.errors import SpecError
from polspec.foreign_key import ForeignKey
from polspec.profiler import (
    ProfileKeywords,
    ProfileOptions,
    profile_options,
    typed_text,
)
from polspec.registry import Registry
from polspec.spec import ColSpec
from polspec.synthesis import Source, load_source, profile
from polspec.tablespec import TableSpec

__all__ = ["profile_all"]

# Distinct values a text column needs before containment alone makes it a
# key: the bar a format clears. Fewer, and a column of a handful of codes
# is contained in a parent's by chance as often as by design.
TEXT_KEY_MIN_DISTINCT = 20

# Values of a declared key's orphans an error shows.
ORPHANS_SHOWN = 5

Family = Literal["integer", "text", "date"]

Keys = Mapping[str, Sequence[ForeignKey]]


def profile_all(
    frames: Mapping[str, Source],
    *,
    keys: Keys | Literal[False] | None = None,
    options: ProfileOptions | None = None,
    **keywords: Unpack[ProfileKeywords],
) -> Registry:
    """A `Registry` describing a set of related tables, with the foreign
    keys between them found from their values.

    Each source is profiled as `profile` profiles it, with the same
    options. Then a column becomes a foreign key into a column of another
    table, or of its own, when:

    - the parent column is an integer, text or date column with no nulls
      and no value twice, in all of its rows;
    - every non-null value of the child is one of the parent's, again in
      all of its rows, whatever `sample` profiled;
    - and either the names link them -- `customer_id` or `customer` into
      `Customers.id`, `customer_id` into `Customers.customer_id`,
      `parent_id` into its own table's `id` -- or the child is text with
      at least 20 distinct values. An integer column with no name link is
      never a key: a quantity of 1 to 20 is contained in every `id`.

    A child that could point at more than one parent, with nothing to
    choose between them, gets no key, and a warning names the candidates;
    so does a key that would close a cycle between tables. The child column
    of a key takes its parent's dtype and value claims, which are what
    generation fills it with, and keeps its own null rate and uniqueness;
    its description says the key was inferred.

    Parameters
    ----------
    frames : Mapping[str, DataFrame | LazyFrame | str | PathLike]
        Each table, by the name its spec takes.
    keys : Mapping[str, Sequence[ForeignKey]] | False, optional
        Keys to declare rather than infer, by the name of the table that
        holds them: `{"Orders": [ForeignKey("buyer", references="Customers",
        ref_columns="id")]}`. A declared key replaces any key inferred on
        its columns, and one the data does not satisfy raises `SpecError`.
        `False` infers none.
    options : ProfileOptions, optional
        How every table is profiled -- or pass its fields as keywords. A
        column in `replace` need only be in one of the tables.

    Returns
    -------
    Registry
        A spec per table, in the order of `frames`, resolved.

    Examples
    --------
    >>> customers = pl.DataFrame({"id": range(1, 101)})
    >>> orders = pl.DataFrame({"id": range(1, 301), "customer_id": [1, 2, 3] * 100})
    >>> registry = profile_all({"Customers": customers, "Orders": orders})
    >>> registry["Orders"].foreign_keys[0].references
    'Customers'
    >>> registry["Orders"]["customer_id"].bounds
    Bound(min=1, max=100)

    *Added in 1.5.*
    """
    if not isinstance(frames, Mapping) or not frames:
        raise TypeError("frames must be a non-empty mapping of table name to data")
    if keys is not None and keys is not False and not isinstance(keys, Mapping):
        raise TypeError(
            "keys takes a mapping of table name to the ForeignKeys it holds, "
            f"or False, got {type(keys).__name__}"
        )
    opts = profile_options(options, keywords)
    loaded = {}
    for name, source in frames.items():
        frame = load_source(source)
        if frame.height == 0:
            raise ValueError(f"Table {name!r} has no rows to profile")
        loaded[name] = frame
    every_column = {c for frame in loaded.values() for c in frame.columns}
    unknown = sorted(set(opts.replace) - every_column)
    if unknown:
        raise ValueError(
            f"replace= names {unknown}, which none of the tables has. Its "
            "names are the columns to profile without their values."
        )
    specs, typed = {}, {}
    for name, frame in loaded.items():
        own = dataclasses.replace(
            opts, replace=[c for c in opts.replace if c in frame.columns]
        )
        specs[name] = profile(frame, name=name, options=own)
        # Keys compare values as the profile read them: with `parse_text`,
        # a number as a number, and empty text as no value.
        typed[name] = typed_text(frame, own)[0]
    declared = _declared(keys or {}, typed)
    found = {} if keys is False else _inferred(typed, declared)
    chosen = _acyclic(typed, declared, found)
    return Registry(
        *(_keyed(specs, name, chosen[name], typed[name].schema) for name in typed)
    ).resolve()


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Key:
    """One foreign key: the child columns, and the parent columns they point
    at. `declared` is the key as `keys=` gave it; an inferred key has none."""

    table: str
    columns: tuple[str, ...]
    parent: str
    parent_columns: tuple[str, ...]
    declared: ForeignKey | None = None

    @property
    def inferred(self) -> bool:
        return self.declared is None

    def foreign_key(self) -> ForeignKey:
        if self.declared is not None:
            return self.declared
        references = "self" if self.parent == self.table else self.parent
        return ForeignKey(
            self.columns, references=references, ref_columns=self.parent_columns
        )

    def label(self) -> str:
        return f"{self.parent}.{'/'.join(self.parent_columns)}"


@dataclass(frozen=True, slots=True)
class _Parent:
    """A column every key into it can trust: no nulls, no value twice."""

    table: str
    column: str
    family: Family
    values: pl.Series
    low: Any
    high: Any


def _family(dtype: pl.DataType) -> Family | None:
    """What a key's values are compared as, or None for a column that is
    never a key: a float, a flag, a timestamp, anything nested."""
    if dtype.is_integer():
        return "integer"
    if dtype == pl.String or isinstance(dtype, (pl.Enum, pl.Categorical)):
        return "text"
    if dtype == pl.Date:
        return "date"
    return None


def _comparable(series: pl.Series) -> pl.Series:
    """`series` as its values compare: an Enum's or a Categorical's as text."""
    return series.cast(pl.String) if _family(series.dtype) == "text" else series


def _parents(loaded: Mapping[str, pl.DataFrame]) -> list[_Parent]:
    out = []
    for table, frame in loaded.items():
        for column, dtype in frame.schema.items():
            family = _family(dtype)
            series = frame[column]
            if (
                family is None
                or series.null_count()
                or series.n_unique() != series.len()
            ):
                continue
            values = _comparable(series)
            out.append(
                _Parent(table, column, family, values, values.min(), values.max())
            )
    return out


def _contained(child: pl.Series, parent: _Parent) -> bool:
    """Whether every value of `child` -- distinct, comparable and with no
    nulls -- is one of `parent`'s. The extremes rule most pairs out first."""
    low, high = child.min(), child.max()
    if low < parent.low or high > parent.high:
        return False
    # Within the parent's extremes, so its dtype holds every value.
    held = parent.values.to_frame("v")
    within = child.cast(parent.values.dtype).to_frame("v")
    return within.join(held, on="v", how="anti").height == 0


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def _snake(name: str) -> str:
    """`name` as lower-case words joined by `_`: `OrderLines` -> `order_lines`."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"[^a-z0-9]+", "_", spaced.lower()).strip("_")


def _singulars(table: str) -> set[str]:
    """What one row of `table` is called, as far as an English plural can
    be undone without a dictionary: every reading is kept, and a name only
    has to match one."""
    word = _snake(table)
    out = {word}
    if word.endswith("ies"):
        out.add(word[:-3] + "y")
    if word.endswith("es"):
        out.add(word[:-2])
    if word.endswith("s") and not word.endswith("ss"):
        out.add(word[:-1])
    return {w for w in out if w}


def _linked(table: str, column: str, parent: str, parent_column: str) -> bool:
    """Whether the names say `table.column` points at `parent.parent_column`."""
    child, ref = _snake(column), _snake(parent_column)
    if table == parent:
        return child in {f"parent_{ref}", "parent"}
    for row in _singulars(parent):
        if child in {f"{row}_{ref}", row}:
            return True
        if child == ref and f"_{row}_" in f"_{ref}_":
            return True
    return False


# ---------------------------------------------------------------------------
# Choosing the keys
# ---------------------------------------------------------------------------


def _inferred(
    loaded: Mapping[str, pl.DataFrame], declared: Mapping[str, list[_Key]]
) -> dict[str, list[_Key]]:
    """The keys the data and the names support, for every column a declared
    key leaves free."""
    parents = _parents(loaded)
    taken = {
        (k.table, c) for keys in declared.values() for k in keys for c in k.columns
    }
    unique_columns = {(p.table, p.column) for p in parents}
    out: dict[str, list[_Key]] = {}
    for table, frame in loaded.items():
        for column, dtype in frame.schema.items():
            family = _family(dtype)
            if family is None or (table, column) in taken:
                continue
            values = _comparable(frame[column]).drop_nulls().unique()
            if values.len() < 2:
                continue
            linked, contained = [], []
            for parent in parents:
                if parent.family != family or (parent.table, parent.column) == (
                    table,
                    column,
                ):
                    continue
                link = _linked(table, column, parent.table, parent.column)
                by_text = family == "text" and values.len() >= TEXT_KEY_MIN_DISTINCT
                if not (link or by_text) or not _contained(values, parent):
                    continue
                if (
                    not link
                    and (table, column) in unique_columns
                    and values.len() == parent.values.len()
                ):
                    # Two unique columns holding the same values: nothing
                    # says which of the two points at the other.
                    continue
                (linked if link else contained).append(parent)
            candidates = linked or contained
            if len(candidates) == 1:
                parent = candidates[0]
                key = _Key(table, (column,), parent.table, (parent.column,))
                out.setdefault(table, []).append(key)
            elif candidates:
                named = ", ".join(f"{p.table}.{p.column}" for p in candidates)
                warnings.warn(
                    f"Column '{table}.{column}' could be a key into any of "
                    f"{named}, so it is left without one. Declare the one it "
                    "is with keys=.",
                    UserWarning,
                    stacklevel=3,
                )
    return out


def _declared(keys: Keys, loaded: Mapping[str, pl.DataFrame]) -> dict[str, list[_Key]]:
    """`keys`, each checked against the data it describes."""
    out: dict[str, list[_Key]] = {}
    for table, foreign_keys in keys.items():
        if table not in loaded:
            raise SpecError(
                f"keys= declares keys on {table!r}, which is not one of the "
                f"tables: {sorted(loaded)}"
            )
        if isinstance(foreign_keys, ForeignKey):
            foreign_keys = [foreign_keys]
        for fk in foreign_keys:
            if not isinstance(fk, ForeignKey):
                raise TypeError(
                    f"keys= takes ForeignKeys for each table, got "
                    f"{type(fk).__name__} for {table!r}"
                )
            parent = table if fk.references == "self" else fk.references
            if parent not in loaded:
                raise SpecError(
                    f"A key on {table!r} references {parent!r}, which is not one "
                    f"of the tables: {sorted(loaded)}"
                )
            frame, parent_frame = loaded[table], loaded[parent]
            for column, ref in zip(fk.columns, fk.ref_columns, strict=True):
                for where, name, holder in (
                    (table, column, frame),
                    (parent, ref, parent_frame),
                ):
                    if name not in holder.columns:
                        raise SpecError(
                            f"A key on {table!r} names {name!r}, which "
                            f"{where!r} does not have"
                        )
            _check_contained(fk, table, parent, frame, parent_frame)
            out.setdefault(table, []).append(
                _Key(table, fk.columns, parent, fk.ref_columns, declared=fk)
            )
    return out


def _check_contained(
    fk: ForeignKey,
    table: str,
    parent: str,
    frame: pl.DataFrame,
    parent_frame: pl.DataFrame,
) -> None:
    """Refuses a declared key some of whose values the parent lacks."""
    columns, refs = list(fk.columns), list(fk.ref_columns)
    child = frame.select(_comparable(frame[c]) for c in columns).drop_nulls().unique()
    held = parent_frame.select(
        _comparable(parent_frame[r]).alias(c)
        for c, r in zip(columns, refs, strict=True)
    )
    try:
        held = held.cast(child.schema)
        orphans = child.join(held, on=columns, how="anti", nulls_equal=False)
    except (pl.exceptions.PolarsError, TypeError) as exc:
        raise SpecError(
            f"The key {'/'.join(columns)} on {table!r} cannot be compared with "
            f"{'/'.join(refs)} on {parent!r}: {exc}"
        ) from exc
    if orphans.height:
        shown = orphans.head(ORPHANS_SHOWN).rows()
        values = ", ".join(repr(r[0] if len(r) == 1 else r) for r in shown)
        raise SpecError(
            f"The key {'/'.join(columns)} on {table!r} holds {orphans.height:,} "
            f"value(s) {parent!r} does not have in {'/'.join(refs)}, such as "
            f"{values}. A key's every value must be its parent's."
        )


def _acyclic(
    loaded: Mapping[str, pl.DataFrame],
    declared: Mapping[str, list[_Key]],
    found: Mapping[str, list[_Key]],
) -> dict[str, list[_Key]]:
    """The declared keys, then each inferred one that closes no cycle
    between tables: parents are generated before their children, so a
    cycle has no order to generate in."""
    edges: dict[str, set[str]] = {name: set() for name in loaded}
    chosen: dict[str, list[_Key]] = {name: [] for name in loaded}
    for keys in declared.values():
        for key in keys:
            chosen[key.table].append(key)
            if key.parent != key.table:
                edges[key.table].add(key.parent)
    for keys in found.values():
        for key in keys:
            if key.parent != key.table and _reaches(edges, key.parent, key.table):
                warnings.warn(
                    f"Column '{key.table}.{key.columns[0]}' looks like a key into "
                    f"{key.label()}, but {key.parent!r} already depends on "
                    f"{key.table!r}, so it is left without one. Declare the "
                    "keys with keys=.",
                    UserWarning,
                    stacklevel=3,
                )
                continue
            chosen[key.table].append(key)
            if key.parent != key.table:
                edges[key.table].add(key.parent)
    return chosen


def _reaches(edges: Mapping[str, set[str]], start: str, goal: str) -> bool:
    seen, todo = set(), [start]
    while todo:
        name = todo.pop()
        if name == goal:
            return True
        if name not in seen:
            seen.add(name)
            todo.extend(edges[name])
    return False


# ---------------------------------------------------------------------------
# The specs
# ---------------------------------------------------------------------------


def _keyed(
    specs: Mapping[str, TableSpec],
    table: str,
    keys: list[_Key],
    schema: pl.Schema,
) -> TableSpec:
    """`table`'s spec with `keys` declared, each child column taking its
    parent's claims: a key fills its column with the parent's values, so
    the child's own extremes or lengths would refuse some of them."""
    spec = specs[table]
    if not keys:
        return spec
    columns = {}
    for key in keys:
        for name, ref in zip(key.columns, key.parent_columns, strict=True):
            child = spec.columns[name]
            parent = specs[key.parent].columns[ref]
            description = child.description
            if key.inferred:
                note = f"A key into {key.label()}, inferred from its values."
                description = f"{description} {note}" if description else note
            columns[name] = dataclasses.replace(
                parent,
                dtype=_child_dtype(child.dtype, schema[name], parent),
                col_name=child.col_name,
                seed_name=child.seed_name,
                nullable=child.nullable,
                null_probability=child.null_probability,
                unique=child.unique,
                optional=child.optional,
                default=child.default,
                tags=child.tags,
                description=description,
                weights=None,
                distribution=None,
                distribution_params=None,
            )
    return spec.with_columns(columns).with_foreign_keys(
        *(key.foreign_key() for key in keys)
    )


def _child_dtype(
    profiled: pl.DataType, source: pl.DataType, parent: ColSpec
) -> pl.DataType:
    """The dtype a key's child column is declared with: its own where every
    value of the parent fits, and the parent's otherwise.

    Its own is the child's profiled dtype for a number or a date -- which
    `parse_text` may have read from text, as it read the parent's -- and
    the source's for text, which a profile narrows to an `Enum` of the
    child's own values. An `Enum` in the source would refuse the parent's
    other values too.
    """
    if _family(parent.dtype) == "text":
        text = _family(source) == "text" and not isinstance(source, pl.Enum)
        return source if text else parent.dtype
    if not (profiled.is_integer() and parent.dtype.is_integer()):
        return parent.dtype
    limits, bounds = dtype_value_limits(profiled), parent.bounds
    fits = (
        limits is not None
        and bounds is not None
        and bounds.min is not None
        and bounds.max is not None
        and limits[0] <= bounds.min
        and bounds.max <= limits[1]
    )
    return profiled if fits else parent.dtype
