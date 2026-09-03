"""`TableSpec` -- a spec as a value.

A `FrameSpec` class body is the convenient way to *write* a spec. This is the
thing it builds: an immutable record of the columns, checks, composite keys
and foreign keys, with the declaration-time validation that keeps them
consistent, and pure operations that derive one spec from another.

Every verb the library offers -- generate, validate, serialize, render --
takes a `TableSpec`. `FrameSpec` classmethods forward to them with `cls.spec`.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from polspec.check import Check
from polspec.errors import SpecError
from polspec.foreign_key import ForeignKey, _default_fk_name
from polspec.spec import ColSpec, _column_kind

if TYPE_CHECKING:
    from polspec.catspec import CatSpec


def _parse_unique_together(val: Any) -> tuple[tuple[str, ...], ...]:
    """Normalises the loose forms `__unique_together__` accepts.

    A single group may be written as a list of names; several groups as a list
    of lists; a lone string means a one-column group.
    """
    out: list[tuple[str, ...]] = []

    def add(group: tuple[str, ...]) -> None:
        if group and group not in out:
            out.append(group)

    if val is None:
        return ()
    if isinstance(val, str):
        add((val,))
    elif isinstance(val, Sequence):
        if not val:
            return ()
        if all(isinstance(x, str) for x in val):
            add(tuple(val))
        else:
            for item in val:
                if isinstance(item, str):
                    add((item,))
                elif isinstance(item, Sequence):
                    add(tuple(str(x) for x in item))
                else:
                    raise SpecError(
                        "Elements of unique_together must be strings or sequences "
                        f"of strings, got {type(item).__name__}"
                    )
    else:
        raise SpecError(
            "unique_together must be a sequence of column names or sequence of "
            f"sequences, got {type(val).__name__}"
        )
    return tuple(out)


def _dedupe[T](items: Sequence[T], kind: type[T], label: str) -> tuple[T, ...]:
    out: list[T] = []
    for item in items:
        if not isinstance(item, kind):
            raise SpecError(
                f"Items in {label} must be {kind.__name__} instances, got "
                f"{type(item).__name__}"
            )
        if item not in out:
            out.append(item)
    return tuple(out)


def _fk_kind_bucket(kind: str) -> str:
    # String/Enum/Categorical columns are all textual domains that can
    # reasonably reference one another (a plain String key pointing at an
    # Enum primary key), so they share one bucket.
    return "string" if kind in ("string", "enum", "categorical") else kind


@dataclass(frozen=True, slots=True)
class TableSpec:
    """The columns and constraints of one table, as an immutable value.

    Parameters
    ----------
    name : str
        What the table is called: the class name for a `FrameSpec`, the
        `name:` key for a file. Foreign keys refer to a spec by this name.
    columns : Mapping[str, ColSpec]
        Column name to declaration, in the order columns should appear.
    checks : Sequence[Check]
        Multi-column invariants; see `FrameSpec.__checks__`.
    unique_together : Sequence[Sequence[str]]
        Composite unique keys. A single group may be given as a flat list.
    foreign_keys : Sequence[ForeignKey]
        Referential-integrity constraints.

    Everything a `FrameSpec` class body validates at declaration is validated
    here, so a `TableSpec` that constructs is one that can be used.
    """

    name: str
    columns: Mapping[str, ColSpec] = field(default_factory=dict)
    checks: Sequence[Check] = ()
    unique_together: Sequence[Sequence[str]] = ()
    foreign_keys: Sequence[ForeignKey] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise SpecError(
                f"TableSpec.name must be a non-empty string, got {self.name!r}"
            )
        if not isinstance(self.columns, Mapping):
            raise SpecError(
                "TableSpec.columns must be a mapping of column name to ColSpec, "
                f"got {type(self.columns).__name__}"
            )
        columns: dict[str, ColSpec] = {}
        for raw_key, value in self.columns.items():
            key = str(raw_key)
            if not isinstance(value, ColSpec):
                raise SpecError(
                    f"columns[{key!r}] must be a ColSpec, got {type(value).__name__}"
                )
            if value.col_name is not None and value.col_name != key:
                raise SpecError(
                    f"columns[{key!r}] declares col_name={value.col_name!r}, which "
                    "conflicts with its key. The key already is the column name."
                )
            columns[key] = value
        object.__setattr__(self, "columns", MappingProxyType(columns))
        object.__setattr__(self, "checks", _dedupe(tuple(self.checks), Check, "checks"))
        object.__setattr__(
            self, "unique_together", _parse_unique_together(self.unique_together)
        )
        object.__setattr__(
            self,
            "foreign_keys",
            _dedupe(tuple(self.foreign_keys), ForeignKey, "foreign_keys"),
        )

        self._validate_rules()
        self._validate_validators()
        self._validate_unique_together()
        self._validate_checks()
        self._validate_foreign_keys()

    # ------------------------------------------------------------------
    # Declaration-time validation
    # ------------------------------------------------------------------

    def _validate_rules(self) -> None:
        for col_name, spec in self.columns.items():
            for rule in spec.rules:
                for ref_col in sorted(rule.when.root_names()):
                    if ref_col not in self.columns:
                        raise SpecError(
                            f"ColRule on column {col_name!r} references unknown "
                            f"column {ref_col!r}"
                        )

    def _validate_validators(self) -> None:
        for col_name, spec in self.columns.items():
            for validator in spec.validators:
                other_cols = set(validator.expr.meta.root_names()) - {col_name}
                if other_cols:
                    raise SpecError(
                        f"ColSpec.validators on column {col_name!r} references "
                        f"other column(s) {sorted(other_cols)}: {validator.expr!r}. "
                        "A ColSpec validator may only reference its own column -- "
                        "use FrameSpec.__checks__ for cross-column invariants."
                    )

    def _validate_unique_together(self) -> None:
        for group in self.unique_together:
            for col_name in group:
                if col_name not in self.columns:
                    raise SpecError(
                        f"Composite unique key {group} references unknown column "
                        f"{col_name!r}"
                    )

    def _validate_checks(self) -> None:
        seen: dict[str, Check] = {}
        for check in self.checks:
            prior = seen.get(check.name)
            if prior is not None:
                raise SpecError(
                    f"Duplicate Check name {check.name!r} on {self.name}: "
                    f"{prior.expr!r} vs {check.expr!r}. Give each Check a "
                    "distinct name, or reuse the exact same Check instance/"
                    "definition to inherit it unchanged."
                )
            seen[check.name] = check

    def _validate_foreign_keys(self) -> None:
        seen: dict[str, ForeignKey] = {}
        for fk in self.foreign_keys:
            prior = seen.get(fk.name)
            if prior is not None and prior != fk:
                raise SpecError(
                    f"Duplicate ForeignKey name {fk.name!r} on {self.name} "
                    "points at two different targets/columns. Give each "
                    "ForeignKey a distinct name."
                )
            seen[fk.name] = fk

            for col in fk.columns:
                if col not in self.columns:
                    raise SpecError(
                        f"ForeignKey {fk.name!r} on {self.name!r} references "
                        f"unknown local column {col!r}"
                    )

            target = self.resolve_target(fk)
            if target is None:
                # A name with no spec behind it yet (a key read from a file).
                # The registry resolves and checks it later.
                continue
            target_name = "self" if fk.references == "self" else target.name
            for ref_col in fk.ref_columns:
                if ref_col not in target.columns:
                    raise SpecError(
                        f"ForeignKey {fk.name!r} on {self.name!r} references "
                        f"unknown column {ref_col!r} on {target_name!r}"
                    )
            for col, ref_col in zip(fk.columns, fk.ref_columns, strict=True):
                local_kind = _fk_kind_bucket(_column_kind(self.columns[col].dtype))
                ref_kind = _fk_kind_bucket(_column_kind(target.columns[ref_col].dtype))
                if local_kind != ref_kind:
                    raise SpecError(
                        f"ForeignKey {fk.name!r} on {self.name!r}: column "
                        f"{col!r} ({self.columns[col].dtype}) is not "
                        f"dtype-compatible with referenced column {ref_col!r} "
                        f"({target.columns[ref_col].dtype}) on {target_name!r}"
                    )

    def resolve_target(self, fk: ForeignKey) -> TableSpec | None:
        """The spec a foreign key points at.

        This spec for `"self"`, the bound target when the key was declared
        against a spec object, `None` for a bare name nothing has resolved.
        """
        if fk.references == "self":
            return self
        return fk.target

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> ColSpec:
        try:
            return self.columns[name]
        except KeyError:
            raise KeyError(f"{self.name} has no column {name!r}") from None

    def __contains__(self, name: object) -> bool:
        return name in self.columns

    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def schema(self) -> pl.Schema:
        return pl.Schema({name: spec.dtype for name, spec in self.columns.items()})

    def tag(
        self,
        *tags: str | Sequence[str],
        match: Literal["any", "all"] = "any",
    ) -> list[str]:
        """Column names carrying any (or all) of the tags, in declaration order."""
        flattened: list[str] = []
        for t in tags:
            if isinstance(t, str):
                flattened.append(t)
            elif isinstance(t, Sequence):
                flattened.extend(str(item) for item in t)
            else:
                raise TypeError(
                    f"Tag must be a string or sequence of strings, got {type(t).__name__}"
                )
        if not flattened:
            return []
        target_tags = set(flattened)
        if match == "any":
            return [
                name
                for name, spec in self.columns.items()
                if any(t in spec.tags for t in target_tags)
            ]
        if match == "all":
            return [
                name
                for name, spec in self.columns.items()
                if target_tags.issubset(spec.tags)
            ]
        raise ValueError(f"Invalid match mode: {match!r}. Expected 'any' or 'all'.")

    # ------------------------------------------------------------------
    # Structural operations -- each returns a new TableSpec
    # ------------------------------------------------------------------

    def with_name(self, name: str) -> TableSpec:
        return dataclasses.replace(self, name=name)

    def with_columns(
        self,
        mapping: Mapping[str, ColSpec] | None = None,
        /,
        **columns: ColSpec,
    ) -> TableSpec:
        """Adds columns, or replaces existing ones of the same name in place."""
        merged = dict(self.columns)
        merged.update(mapping or {})
        merged.update(columns)
        return dataclasses.replace(self, columns=merged)

    def with_checks(self, *checks: Check) -> TableSpec:
        return dataclasses.replace(self, checks=(*self.checks, *checks))

    def with_foreign_keys(self, *foreign_keys: ForeignKey) -> TableSpec:
        return dataclasses.replace(
            self, foreign_keys=(*self.foreign_keys, *foreign_keys)
        )

    def with_unique_together(self, *groups: Sequence[str]) -> TableSpec:
        return dataclasses.replace(
            self,
            unique_together=(*self.unique_together, *(tuple(g) for g in groups)),
        )

    def _require_columns(self, names: Sequence[str], verb: str) -> None:
        unknown = [n for n in names if n not in self.columns]
        if unknown:
            raise SpecError(
                f"Cannot {verb} {unknown} from {self.name}: no such column(s)"
            )

    def _without(self, dropped: set[str]) -> dict[str, Any]:
        """Constraint fields with every mention of `dropped` columns removed.

        A composite key or foreign key that loses a member is removed whole.
        A rule on a surviving column that points at a dropped one is left for
        validation to reject: silently dropping a rule would change what the
        surviving column generates.
        """
        return {
            "unique_together": tuple(
                g for g in self.unique_together if not dropped & set(g)
            ),
            "foreign_keys": tuple(
                fk for fk in self.foreign_keys if not dropped & set(fk.columns)
            ),
        }

    def drop(self, *names: str) -> TableSpec:
        """Removes columns, and any composite or foreign key that used them."""
        self._require_columns(names, "drop")
        dropped = set(names)
        columns = {k: v for k, v in self.columns.items() if k not in dropped}
        return dataclasses.replace(self, columns=columns, **self._without(dropped))

    def select(self, *names: str) -> TableSpec:
        """Keeps only the named columns, in the order given."""
        self._require_columns(names, "select")
        dropped = set(self.columns) - set(names)
        columns = {n: self.columns[n] for n in names}
        return dataclasses.replace(self, columns=columns, **self._without(dropped))

    def rename(self, mapping: Mapping[str, str]) -> TableSpec:
        """Renames columns, rewriting every constraint that names them.

        Rules, composite keys and foreign keys are rewritten. A column
        carrying `validators` cannot be renamed: a validator is a Polars
        expression that names the column, and rewriting expressions is not
        something this library does.
        """
        self._require_columns(list(mapping), "rename")
        for old in mapping:
            if self.columns[old].validators:
                raise SpecError(
                    f"Cannot rename {old!r}: it carries validators, and a validator "
                    "is an expression naming the column. Drop the validators, "
                    "rename, then re-declare them against the new name."
                )
        taken = set(self.columns) - set(mapping)
        targets = list(mapping.values())
        for old, new in mapping.items():
            if new in taken or targets.count(new) > 1:
                raise SpecError(f"Cannot rename {old!r} to {new!r}: that name is taken")

        def new_name(n: str) -> str:
            return mapping.get(n, n)

        columns: dict[str, ColSpec] = {}
        for key, spec in self.columns.items():
            updates: dict[str, Any] = {}
            if spec.col_name is not None:
                updates["col_name"] = None
            if any(r.when.root_names() & set(mapping) for r in spec.rules):
                updates["rules"] = tuple(
                    dataclasses.replace(r, when=r.when.rename(mapping))
                    for r in spec.rules
                )
            columns[new_name(key)] = (
                dataclasses.replace(spec, **updates) if updates else spec
            )

        foreign_keys: list[ForeignKey] = []
        for fk in self.foreign_keys:
            cols = tuple(new_name(c) for c in fk.columns)
            ref_cols = (
                tuple(new_name(c) for c in fk.ref_columns)
                if fk.references == "self"
                else fk.ref_columns
            )
            keep_name = fk.name != _default_fk_name(fk.columns, fk.references)
            foreign_keys.append(
                ForeignKey(
                    cols,
                    references=fk.target if fk.target is not None else fk.references,
                    ref_columns=ref_cols,
                    name=fk.name if keep_name else None,
                )
            )
        return dataclasses.replace(
            self,
            columns=columns,
            unique_together=tuple(
                tuple(new_name(c) for c in g) for g in self.unique_together
            ),
            foreign_keys=tuple(foreign_keys),
        )

    def with_catspec(self, catspec: CatSpec) -> TableSpec:
        """Re-points columns at the registry's Enum and Categorical types.

        A column whose name resolves in the registry (exactly, or by case)
        takes the registry's dtype; everything else it declared carries over.
        `choices` a new Enum cannot hold, and `weights` over a domain that
        changed size, are dropped with a warning.
        """
        new_columns: dict[str, ColSpec] = {}
        for col_name, spec in self.columns.items():
            resolved = catspec.resolve_key(col_name)
            if resolved is None:
                new_columns[col_name] = spec
                continue
            kind, key = resolved
            if kind == "enum":
                new_columns[col_name] = _retype_column(
                    col_name, spec, pl.Enum(catspec.get_enum(key))
                )
            else:
                new_columns[col_name] = _retype_column(
                    col_name,
                    spec,
                    pl.Categorical(catspec.get_categorical(key)),
                    choices=catspec.get_choices(key) or spec.choices,
                )
        return dataclasses.replace(self, columns=new_columns)


def _retype_column(
    col_name: str,
    spec: ColSpec,
    dtype: pl.DataType,
    *,
    choices: Sequence[Any] | None = None,
) -> ColSpec:
    """Re-points a ColSpec at a new dtype, keeping everything else it declared.

    `dataclasses.replace` rather than a field-by-field rebuild: a rebuild
    silently drops whatever field it forgets to list.

    Two fields a dtype change can genuinely invalidate are handled explicitly:
    `choices`, which may name values outside the new dtype's domain, and
    `weights`, which is positional over a domain that just changed size. Each
    is dropped with a warning rather than carried into a confusing ColSpec
    error further down.
    """
    updates: dict[str, Any] = {"dtype": dtype}

    new_choices = tuple(choices) if choices is not None else spec.choices
    if choices is None and new_choices is not None and isinstance(dtype, pl.Enum):
        valid = set(dtype.categories.to_list())
        stale = [c for c in new_choices if c not in valid]
        if stale:
            warnings.warn(
                f"Column {col_name!r}: dropping choices {stale} while re-typing "
                f"to {dtype!r}, whose categories do not include them. The new "
                "dtype's own categories become the column's domain.",
                stacklevel=4,
            )
            new_choices = None
    updates["choices"] = new_choices

    if new_choices is not None:
        new_domain = len(new_choices)
    elif isinstance(dtype, pl.Enum):
        new_domain = len(dtype.categories)
    else:
        new_domain = None

    if spec.weights is not None and (
        new_domain is None or new_domain != len(spec.weights)
    ):
        warnings.warn(
            f"Column {col_name!r}: dropping {len(spec.weights)} weight(s) while "
            f"re-typing to {dtype!r}, which defines "
            f"{new_domain if new_domain is not None else 'no'} value(s) for them "
            "to apply to. Re-declare weights against the new domain if the "
            "distribution mattered.",
            stacklevel=4,
        )
        updates["weights"] = None

    return dataclasses.replace(spec, **updates)


def as_table_spec(obj: Any) -> TableSpec:
    """The `TableSpec` behind `obj`: itself, or a `FrameSpec` subclass's `.spec`."""
    if isinstance(obj, TableSpec):
        return obj
    spec = getattr(obj, "spec", None)
    if isinstance(obj, type) and isinstance(spec, TableSpec):
        return spec
    raise TypeError(
        f"Expected a TableSpec or a FrameSpec subclass, got {type(obj).__name__}"
    )


def as_spec_name(obj: Any) -> str:
    """The name a `references=` key stands for: a spec, a class, or the name itself."""
    if isinstance(obj, str):
        return obj
    return as_table_spec(obj).name


def resolve_references(
    references: Mapping[Any, pl.DataFrame | pl.LazyFrame] | None,
    coerce: Callable[[pl.DataFrame | pl.LazyFrame], Any],
) -> dict[str, Any]:
    """Keys a `references=` mapping by spec name, coercing each frame."""
    if not references:
        return {}
    return {as_spec_name(key): coerce(frame) for key, frame in references.items()}
