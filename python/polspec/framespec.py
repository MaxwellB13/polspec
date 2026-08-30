from __future__ import annotations

import dataclasses
import random
import warnings
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, overload

import polars as pl
import yaml

from polspec.catspec import CatSpec
from polspec.check import Check
from polspec.engine import _generate_cartesian, _generate_random
from polspec.foreign_key import ForeignKey, _apply_foreign_keys
from polspec.profiler import profile_dataframe
from polspec.rules import _apply_rules
from polspec.serialization import (
    _colspec_from_yaml,
    _colspec_to_yaml,
    _foreignkey_from_yaml,
    _foreignkey_to_yaml,
)
from polspec.spec import ColSpec, _column_kind
from polspec.validation import _validate_dataframe


def _retype_column(
    col_name: str,
    spec: ColSpec,
    dtype: pl.DataType,
    *,
    choices: Sequence[Any] | None = None,
) -> ColSpec:
    """Re-points a ColSpec at a new dtype, keeping everything else it declared.

    `dataclasses.replace` rather than a field-by-field rebuild: a rebuild
    silently drops whatever field it forgets to list, and quietly stopped
    honouring `unique` and `string_length` for every re-typed column.

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
                stacklevel=3,
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
            stacklevel=3,
        )
        updates["weights"] = None

    return dataclasses.replace(spec, **updates)


class FrameSpec:
    """Base class for declaring a DataFrame/LazyFrame specification.

    Subclass it and assign `ColSpec` instances as class attributes, in the
    order columns should appear:

        class DataSource(FrameSpec):
            string_1 = ColSpec(dtype=pl.String, nullable=False)
            enum_1 = ColSpec(dtype=pl.Enum(["mammal", "reptile", "insect"]), nullable=True)
            int_1 = ColSpec(dtype=pl.Int64, bounds=Bound(-100, 100), nullable=True)

        df = DataSource.generate(1_000_000, seed=42)
        df = DataSource.generate(n=1_000_000, method="cartesian", seed=42)

    Columns whose names cannot be class attributes -- a leading underscore, or
    a collision with one of this class's own methods such as `schema` -- are
    declared through `__columns__` instead, which is not looked up as an
    attribute:

        class DataSource(FrameSpec):
            __columns__ = {"_id": ColSpec(pl.Int64), "schema": ColSpec(pl.String)}

    `from_dataframe` and `from_yaml` build specs this way, since their column
    names come from data rather than from a class body.
    """

    _columns: ClassVar[dict[str, ColSpec]] = {}
    _checks: ClassVar[tuple[Check, ...]] = ()
    _unique_together: ClassVar[tuple[tuple[str, ...], ...]] = ()
    _foreign_keys: ClassVar[tuple[ForeignKey, ...]] = ()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        columns: dict[str, ColSpec] = {}
        checks_list: list[Check] = []
        unique_together_list: list[tuple[str, ...]] = []
        foreign_keys_list: list[ForeignKey] = []

        for base in reversed(cls.__mro__):
            # Collect checks from base classes
            for attr_name in ("__checks__", "checks"):
                if attr_name in vars(base):
                    val = vars(base)[attr_name]
                    if isinstance(val, (classmethod, staticmethod)) or callable(val):
                        continue
                    if isinstance(val, Check):
                        if val not in checks_list:
                            checks_list.append(val)
                    elif isinstance(val, (list, tuple, Sequence)):
                        for item in val:
                            if isinstance(item, Check):
                                if item not in checks_list:
                                    checks_list.append(item)
                            else:
                                raise TypeError(
                                    f"Items in __checks__ must be Check instances, got {type(item).__name__}"
                                )
                    elif val is not None:
                        raise TypeError(
                            f"__checks__ must be a Check or sequence of Check instances, got {type(val).__name__}"
                        )

            # Collect unique_together from base classes
            for attr_name in ("__unique_together__", "unique_together"):
                if attr_name in vars(base):
                    val = vars(base)[attr_name]
                    if isinstance(val, (classmethod, staticmethod)) or callable(val):
                        continue
                    cls._parse_unique_together(val, unique_together_list)

            # Collect foreign keys from base classes
            for attr_name in ("__foreign_keys__", "foreign_keys"):
                if attr_name in vars(base):
                    val = vars(base)[attr_name]
                    if isinstance(val, (classmethod, staticmethod)) or callable(val):
                        continue
                    if isinstance(val, ForeignKey):
                        if val not in foreign_keys_list:
                            foreign_keys_list.append(val)
                    elif isinstance(val, (list, tuple, Sequence)):
                        for item in val:
                            if isinstance(item, ForeignKey):
                                if item not in foreign_keys_list:
                                    foreign_keys_list.append(item)
                            else:
                                raise TypeError(
                                    f"Items in __foreign_keys__ must be ForeignKey instances, got {type(item).__name__}"
                                )
                    elif val is not None:
                        raise TypeError(
                            f"__foreign_keys__ must be a ForeignKey or sequence of ForeignKey instances, got {type(val).__name__}"
                        )

            # Collect columns declared as ordinary class attributes
            for name, value in vars(base).items():
                if name.startswith("_"):
                    continue
                if isinstance(value, ColSpec):
                    if name in _RESERVED_ATTRS:
                        # A warning, not an error: `tag`, `schema` and friends
                        # are ordinary column names in real data, and polspec's
                        # choice of method names is no reason to refuse someone
                        # else's schema. The column works either way -- what
                        # breaks is only the shadowed accessor, and declaring
                        # the column through __columns__ keeps both.
                        warnings.warn(
                            f"Column {name!r} on {cls.__name__} shadows "
                            f"FrameSpec.{name}, so {cls.__name__}.{name}() is "
                            f"no longer callable on this spec. The {name!r} "
                            "column itself is unaffected. Declare it through "
                            "__columns__ = {...} to keep the method as well.",
                            stacklevel=3,
                        )
                    columns[name] = value
                elif name in columns:
                    del columns[name]

            # Collect columns declared explicitly. Names here never pass
            # through attribute lookup, so they may start with an underscore
            # or match one of this class's own methods -- which is what makes
            # this the right channel for names that come from data (a profiled
            # DataFrame, a YAML file) rather than from someone's class body.
            declared = vars(base).get("__columns__")
            if declared is not None:
                if not isinstance(declared, Mapping):
                    raise TypeError(
                        "__columns__ must be a mapping of column name to "
                        f"ColSpec, got {type(declared).__name__}"
                    )
                for name, value in declared.items():
                    if not isinstance(value, ColSpec):
                        raise TypeError(
                            f"__columns__[{name!r}] must be a ColSpec, got "
                            f"{type(value).__name__}"
                        )
                    columns[str(name)] = value

        cls._columns = columns
        cls._checks = tuple(checks_list)
        cls._unique_together = tuple(unique_together_list)
        cls._foreign_keys = tuple(foreign_keys_list)
        cls._validate_rules()
        cls._validate_validators()
        cls._validate_unique_together()
        cls._validate_checks()
        cls._validate_foreign_keys()

    @classmethod
    def _parse_unique_together(
        cls,
        val: Any,
        out: list[tuple[str, ...]],
    ) -> None:
        if val is None:
            return
        if isinstance(val, str):
            t = (val,)
            if t not in out:
                out.append(t)
        elif isinstance(val, (list, tuple, Sequence)):
            if not val:
                return
            if all(isinstance(x, str) for x in val):
                t = tuple(val)
                if t not in out:
                    out.append(t)
            else:
                for item in val:
                    if isinstance(item, str):
                        t = (item,)
                        if t not in out:
                            out.append(t)
                    elif isinstance(item, (list, tuple, Sequence)):
                        t = tuple(str(x) for x in item)
                        if t not in out:
                            out.append(t)
                    else:
                        raise TypeError(
                            f"Elements of unique_together must be strings or sequences of strings, got {type(item).__name__}"
                        )
        else:
            raise TypeError(
                f"unique_together must be a sequence of column names or sequence of sequences, got {type(val).__name__}"
            )

    @classmethod
    def _validate_unique_together(cls) -> None:
        for group in cls._unique_together:
            for col_name in group:
                if col_name not in cls._columns:
                    raise ValueError(
                        f"Composite unique key {group} references unknown column {col_name!r}"
                    )

    @classmethod
    def _validate_checks(cls) -> None:
        seen: dict[str, Check] = {}
        for check in cls._checks:
            prior = seen.get(check.name)
            if prior is not None:
                raise ValueError(
                    f"Duplicate Check name {check.name!r} on {cls.__name__}: "
                    f"{prior.expr!r} vs {check.expr!r}. Give each Check a "
                    "distinct name, or reuse the exact same Check instance/"
                    "definition to inherit it unchanged."
                )
            seen[check.name] = check

    @staticmethod
    def _fk_kind_bucket(kind: str) -> str:
        # String/Enum/Categorical columns are all textual domains that can
        # reasonably reference one another (e.g. a plain String FK column
        # pointing at an Enum primary key), so they share one bucket here.
        return "string" if kind in ("string", "enum", "categorical") else kind

    @classmethod
    def _validate_foreign_keys(cls) -> None:
        seen: dict[str, ForeignKey] = {}
        for fk in cls._foreign_keys:
            prior = seen.get(fk.name)
            if prior is not None and prior != fk:
                raise ValueError(
                    f"Duplicate ForeignKey name {fk.name!r} on {cls.__name__} "
                    f"points at two different targets/columns. Give each "
                    "ForeignKey a distinct name."
                )
            seen[fk.name] = fk

            for col in fk.columns:
                if col not in cls._columns:
                    raise ValueError(
                        f"ForeignKey {fk.name!r} on {cls.__name__!r} references "
                        f"unknown local column {col!r}"
                    )

            target = cls if fk.references == "self" else fk.references
            if not (isinstance(target, type) and hasattr(target, "_columns")):
                raise TypeError(
                    f"ForeignKey {fk.name!r} on {cls.__name__!r}: references must "
                    f"be a FrameSpec subclass or 'self', got {fk.references!r}"
                )
            target_name = "self" if fk.references == "self" else target.__name__
            target_columns: dict[str, ColSpec] = target._columns

            for ref_col in fk.ref_columns:
                if ref_col not in target_columns:
                    raise ValueError(
                        f"ForeignKey {fk.name!r} on {cls.__name__!r} references "
                        f"unknown column {ref_col!r} on {target_name!r}"
                    )

            for col, ref_col in zip(fk.columns, fk.ref_columns):
                local_kind = cls._fk_kind_bucket(_column_kind(cls._columns[col].dtype))
                ref_kind = cls._fk_kind_bucket(
                    _column_kind(target_columns[ref_col].dtype)
                )
                if local_kind != ref_kind:
                    raise ValueError(
                        f"ForeignKey {fk.name!r} on {cls.__name__!r}: column "
                        f"{col!r} ({cls._columns[col].dtype}) is not "
                        f"dtype-compatible with referenced column {ref_col!r} "
                        f"({target_columns[ref_col].dtype}) on {target_name!r}"
                    )

    @classmethod
    def checks(cls) -> tuple[Check, ...]:
        """Returns the tuple of Check constraints defined on this FrameSpec."""
        return cls._checks

    @classmethod
    def foreign_keys(cls) -> tuple[ForeignKey, ...]:
        """Returns the tuple of ForeignKey constraints defined on this FrameSpec."""
        return cls._foreign_keys

    @classmethod
    def unique_together(cls) -> tuple[tuple[str, ...], ...]:
        """Returns the tuple of composite unique column groups defined on this FrameSpec."""
        return cls._unique_together

    @classmethod
    def _validate_rules(cls) -> None:
        for col_name, spec in cls._columns.items():
            for rule in spec.rules:
                ref_col = rule.when.get("column")
                if ref_col not in cls._columns:
                    raise ValueError(
                        f"ColRule on column {col_name!r} references unknown column {ref_col!r}"
                    )

    @classmethod
    def _validate_validators(cls) -> None:
        for col_name, spec in cls._columns.items():
            for validator in spec.validators:
                other_cols = set(validator.expr.meta.root_names()) - {col_name}
                if other_cols:
                    raise ValueError(
                        f"ColSpec.validators on column {col_name!r} references "
                        f"other column(s) {sorted(other_cols)}: {validator.expr!r}. "
                        "A ColSpec validator may only reference its own column -- "
                        "use FrameSpec.__checks__ for cross-column invariants."
                    )

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema({name: spec.dtype for name, spec in cls._columns.items()})

    @classmethod
    def tag(
        cls,
        *tags: str | Sequence[str],
        match: Literal["any", "all"] = "any",
    ) -> list[str]:
        """Returns the list of column names matching the specified tag or tags.

        Parameters
        ----------
        *tags : str | Sequence[str]
            One or more tag names or sequences of tag names to match.
        match : {"any", "all"}, default "any"
            Whether to match columns having any of the given tags ("any")
            or all of the given tags ("all").

        Returns
        -------
        list[str]
            List of column names with matching tags, in declaration order.
        """
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
                for name, spec in cls._columns.items()
                if any(t in spec.tags for t in target_tags)
            ]
        elif match == "all":
            return [
                name
                for name, spec in cls._columns.items()
                if target_tags.issubset(spec.tags)
            ]
        else:
            raise ValueError(f"Invalid match mode: {match!r}. Expected 'any' or 'all'.")

    @classmethod
    def catspec(cls) -> CatSpec:
        """Extracts a CatSpec registry from this FrameSpec's column definitions."""
        return CatSpec.from_framespec(cls)

    @classmethod
    def generate_catspec(cls) -> CatSpec:
        """Extracts a CatSpec registry from this FrameSpec's column definitions (alias for `catspec`)."""
        return cls.catspec()

    @classmethod
    def write_catspec(cls, source: str | Path) -> None:
        """Extracts and writes a CatSpec YAML registry for this FrameSpec to `source`."""
        cls.catspec().to_yaml(source)

    @classmethod
    def infer_catspec(
        cls,
        data: pl.DataFrame | pl.LazyFrame | None = None,
        *,
        max_enum_cardinality: int = 30,
        max_categorical_cardinality: int = 10_000,
        max_categorical_ratio: float = 0.20,
        include_columns: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] | None = (
            r"(?:^|.*_)id$",
            r"(?:^|.*_)uuid$",
            r"(?:^|.*_)hash$",
            r"(?:^|.*_)url$",
            r"(?:^|.*_)key$",
        ),
        default_physical: pl.DataType | None = None,
    ) -> CatSpec:
        """Infers an optimal CatSpec registry from string columns using heuristic rules."""
        if data is not None:
            return CatSpec.infer_from_dataframe(
                data,
                max_enum_cardinality=max_enum_cardinality,
                max_categorical_cardinality=max_categorical_cardinality,
                max_categorical_ratio=max_categorical_ratio,
                include_columns=include_columns,
                exclude_patterns=exclude_patterns,
                default_physical=default_physical,
            )
        return CatSpec.infer_from_framespec(
            cls,
            max_enum_cardinality=max_enum_cardinality,
            max_categorical_cardinality=max_categorical_cardinality,
            include_columns=include_columns,
            exclude_patterns=exclude_patterns,
            default_physical=default_physical,
        )

    @classmethod
    def with_catspec(
        cls,
        catspec: CatSpec,
        *,
        name: str | None = None,
    ) -> type[FrameSpec]:
        """Creates a new FrameSpec subclass with columns re-typed using the provided CatSpec."""
        new_columns: dict[str, ColSpec] = {}
        for col_name, spec in cls._columns.items():
            enum_key = (
                catspec._resolve_enum_key(col_name)
                if hasattr(catspec, "_resolve_enum_key")
                else (col_name if col_name in catspec.enums else None)
            )
            cat_key = (
                catspec._resolve_cat_key(col_name)
                if hasattr(catspec, "_resolve_cat_key")
                else (col_name if col_name in catspec.categoricals else None)
            )

            if enum_key is not None:
                new_columns[col_name] = _retype_column(
                    col_name, spec, pl.Enum(catspec.get_enum(enum_key))
                )
            elif cat_key is not None:
                new_columns[col_name] = _retype_column(
                    col_name,
                    spec,
                    pl.Categorical(catspec.get_categorical(cat_key)),
                    choices=catspec.get_choices(cat_key) or spec.choices,
                )
            else:
                new_columns[col_name] = spec

        subclass_name = name or f"{cls.__name__}WithCatSpec"
        class_attrs: dict[str, Any] = {"__columns__": new_columns}
        if cls._checks:
            class_attrs["__checks__"] = cls._checks
        if cls._unique_together:
            class_attrs["__unique_together__"] = cls._unique_together
        if cls._foreign_keys:
            class_attrs["__foreign_keys__"] = cls._foreign_keys
        return type(subclass_name, (FrameSpec,), class_attrs)

    @classmethod
    def with_inferred_catspec(
        cls,
        catspec: CatSpec | None = None,
        *,
        data: pl.DataFrame | pl.LazyFrame | None = None,
        name: str | None = None,
        max_enum_cardinality: int = 30,
        max_categorical_cardinality: int = 10_000,
        max_categorical_ratio: float = 0.20,
        include_columns: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] | None = (
            r"(?:^|.*_)id$",
            r"(?:^|.*_)uuid$",
            r"(?:^|.*_)hash$",
            r"(?:^|.*_)url$",
            r"(?:^|.*_)key$",
        ),
        default_physical: pl.DataType | None = None,
    ) -> type[FrameSpec]:
        """Infers a CatSpec and returns a new FrameSpec subclass with optimized Enum and Categorical columns."""
        if catspec is None:
            catspec = cls.infer_catspec(
                data=data,
                max_enum_cardinality=max_enum_cardinality,
                max_categorical_cardinality=max_categorical_cardinality,
                max_categorical_ratio=max_categorical_ratio,
                include_columns=include_columns,
                exclude_patterns=exclude_patterns,
                default_physical=default_physical,
            )
        subclass_name = name or f"{cls.__name__}Optimized"
        return cls.with_catspec(catspec, name=subclass_name)

    @classmethod
    def to_yaml(cls, source: str | Path) -> None:
        """Writes this spec's columns to a human-readable YAML file at `source`."""
        if not cls._columns:
            raise ValueError(f"{cls.__name__} declares no ColSpec columns")
        if cls._checks:
            check_names = ", ".join(repr(c.name) for c in cls._checks)
            warnings.warn(
                f"{cls.__name__} declares {len(cls._checks)} __checks__ "
                f"({check_names}) that cannot be represented in YAML (a Check "
                "wraps an arbitrary polars.Expr) and will NOT be written to "
                f"{source!s}. They will be lost on FrameSpec.from_yaml() unless "
                "re-declared on a subclass of the loaded spec.",
                stacklevel=2,
            )
        external_fks = [fk for fk in cls._foreign_keys if fk.references != "self"]
        if external_fks:
            fk_names = ", ".join(repr(fk.name) for fk in external_fks)
            warnings.warn(
                f"{cls.__name__} declares {len(external_fks)} ForeignKey(s) "
                f"({fk_names}) referencing another FrameSpec class, which has no "
                f"stable name to persist and will NOT be written to {source!s}. "
                "Only self-referencing ForeignKeys (references='self') survive a "
                "YAML round-trip; re-declare the others on a subclass of the "
                "loaded spec.",
                stacklevel=2,
            )
        validator_names = [
            f"{col_name}.{v.name}"
            for col_name, spec in cls._columns.items()
            for v in spec.validators
        ]
        if validator_names:
            warnings.warn(
                f"{cls.__name__} declares {len(validator_names)} column-level "
                f"validator(s) ({', '.join(repr(n) for n in validator_names)}) "
                "that cannot be represented in YAML (a validator wraps an "
                f"arbitrary polars.Expr) and will NOT be written to {source!s}. "
                "They will be lost on FrameSpec.from_yaml() unless re-declared "
                "on a subclass of the loaded spec.",
                stacklevel=2,
            )
        data: dict[str, Any] = {
            "name": cls.__name__,
            "columns": {
                name: _colspec_to_yaml(spec) for name, spec in cls._columns.items()
            },
        }
        if cls._unique_together:
            data["unique_together"] = [list(group) for group in cls._unique_together]
        self_fks = [fk for fk in cls._foreign_keys if fk.references == "self"]
        if self_fks:
            data["foreign_keys"] = [_foreignkey_to_yaml(fk) for fk in self_fks]
        p = Path(source)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    @classmethod
    def from_yaml(
        cls,
        source: str | Path,
        *,
        categories: CatSpec | str | Path | None = None,
    ) -> type[FrameSpec]:
        """Builds a new FrameSpec subclass from a YAML file written by `to_yaml`.

        Parameters
        ----------
        source : str | Path
            Path to the YAML specification file.
        categories : CatSpec | str | Path | None, optional
            A CatSpec registry or file path to resolve shared Enums and Categoricals.
            If omitted and the YAML specifies a `categories` property, it is loaded automatically.

        Examples
        --------
        >>> DataSource = FrameSpec.from_yaml(source="spec.yaml")
        >>> df = DataSource.generate(1_000, seed=42)
        """
        source_path = Path(source)
        data = yaml.safe_load(source_path.read_text(encoding="utf-8"))

        catspec: CatSpec | None = None
        if categories is not None:
            if isinstance(categories, (str, Path)):
                catspec = CatSpec.from_yaml(categories)
            else:
                catspec = categories
        elif "categories" in data:
            cat_source = data["categories"]
            if isinstance(cat_source, (str, Path)):
                cat_path = Path(cat_source)
                if not cat_path.is_absolute():
                    cat_path = source_path.parent / cat_path
                catspec = CatSpec.from_yaml(cat_path)
            elif isinstance(cat_source, dict):
                catspec = CatSpec.from_dict(cat_source)

        columns_data = data.get("columns") or {}
        if not columns_data:
            raise ValueError(f"{source} declares no columns")
        columns = {
            name: _colspec_from_yaml(col_data, categories=catspec)
            for name, col_data in columns_data.items()
        }
        class_attrs: dict[str, Any] = {"__columns__": columns}
        if "unique_together" in data:
            class_attrs["__unique_together__"] = data["unique_together"]
        if "foreign_keys" in data:
            class_attrs["__foreign_keys__"] = [
                _foreignkey_from_yaml(fk_data) for fk_data in data["foreign_keys"]
            ]
        return type(data.get("name", "LoadedFrameSpec"), (FrameSpec,), class_attrs)

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        *,
        name: str = "ProfiledFrameSpec",
        weights: bool = False,
        max_unique_enum: int = 50,
        max_unique: int | None = None,
        calculate_bounds: bool = True,
        bounds: bool | None = None,
    ) -> type[FrameSpec]:
        """Infers and builds a new FrameSpec subclass by profiling an existing DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            The DataFrame to profile.
        name : str, default "ProfiledFrameSpec"
            The name of the generated FrameSpec subclass.
        weights : bool, default False
            If True, calculates empirical frequency weights for categorical, enum,
            and boolean columns.
        max_unique_enum : int, default 50
            Maximum number of unique non-null values for a string or categorical
            column to be converted into an Enum. Can also be set via `max_unique`.
        max_unique : int | None, optional
            Alias for `max_unique_enum`.
        calculate_bounds : bool, default True
            If True, computes (min, max) bounds for numeric and temporal columns,
            and (min_len, max_len) for string and binary columns. Can also be set via `bounds`.
        bounds : bool | None, optional
            Alias for `calculate_bounds`.
        """
        columns = profile_dataframe(
            df,
            weights=weights,
            max_unique_enum=max_unique_enum,
            max_unique=max_unique,
            calculate_bounds=calculate_bounds,
            bounds=bounds,
        )
        return type(name, (FrameSpec,), {"__columns__": columns})

    @overload
    @classmethod
    def generate(
        cls,
        n: int,
        *,
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
        references: Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None = None,
        lazy: Literal[False] = False,
    ) -> pl.DataFrame: ...

    @overload
    @classmethod
    def generate(
        cls,
        n: int,
        *,
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
        references: Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None = None,
        lazy: Literal[True],
    ) -> pl.LazyFrame: ...

    @classmethod
    def generate(
        cls,
        n: int,
        *,
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
        references: Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None = None,
        lazy: bool = False,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Generates a DataFrame (or LazyFrame) matching this spec.

        method="random" (default): `n` rows, each column drawn independently.

        method="cartesian": guarantees a minimum level of coverage. Builds
        the cartesian product of every Enum/Boolean column's full set of
        values, crossed with the negative/zero/positive/null partitions of
        every bounded numeric column -- so every enum combination appears
        alongside every numeric sign/null case. `n` is then a *minimum*: if
        that coverage set has fewer than `n` rows, it's padded with ordinary
        random rows up to `n`; if it already has more, all of it is kept.

        Any ColSpec.rules are applied next, as a vectorized overwrite pass
        over the fully-generated DataFrame (see ColRule), regardless of
        method.

        Any `ForeignKey` this spec declares is then made referentially
        consistent, but only where data for its target is actually
        available: self-referencing keys (`references="self"`) always are,
        sampled from this same generated DataFrame; a key referencing
        another FrameSpec only is if a matching entry is supplied via
        `references={OtherSpec: other_df}` -- otherwise that column is left
        exactly as freely generated, same as before this existed. Composite
        keys are sampled as one joint pick per row so multi-column keys stay
        internally consistent; a single-column key whose ColSpec is
        `unique=True` samples without replacement when the parent has enough
        distinct rows to cover `n` (a one-to-one relationship), otherwise
        (or for composite keys) sampling is with replacement.

        lazy=True returns a `pl.LazyFrame` around the generated DataFrame.
        """
        if not cls._columns:
            raise ValueError(f"{cls.__name__} declares no ColSpec columns")
        if n < 0:
            raise ValueError("n must be >= 0")

        rng = random.Random(seed)
        gen_seed = rng.randrange(2**63)
        if method == "random":
            df = _generate_random(cls._columns, n, gen_seed)
        elif method == "cartesian":
            df = _generate_cartesian(cls._columns, n, gen_seed)
        else:
            raise ValueError(
                f"Unknown method {method!r}; expected 'random' or 'cartesian'"
            )

        res = _apply_rules(df, cls._columns, rng.randrange(2**63))

        if cls._foreign_keys:
            resolved_foreign_keys: list[tuple[ForeignKey, pl.DataFrame | None]] = []
            for fk in cls._foreign_keys:
                if fk.references == "self":
                    resolved_foreign_keys.append((fk, res))
                    continue
                parent = references.get(fk.references) if references else None
                if parent is None:
                    resolved_foreign_keys.append((fk, None))
                    continue
                parent_df = (
                    parent.collect() if isinstance(parent, pl.LazyFrame) else parent
                )
                resolved_foreign_keys.append((fk, parent_df))
            res = _apply_foreign_keys(
                res, cls._columns, resolved_foreign_keys, rng.randrange(2**63)
            )

        return res.lazy() if lazy else res

    @classmethod
    def generate_batches(
        cls,
        n: int,
        *,
        batch_size: int = 100_000,
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
        references: Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None = None,
    ) -> Iterator[pl.DataFrame]:
        """Yields chunks of generated DataFrames without holding all rows in memory.

        Parameters
        ----------
        n : int
            Total number of rows to generate across all batches.
        batch_size : int, default 100_000
            Maximum number of rows per batch. Must be > 0.
        method : Literal["random", "cartesian"], default "random"
            Generation strategy ("random" or "cartesian").
        seed : int | None, optional
            Random seed for reproducible batch generation.
        references : Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None, optional
            Parent DataFrames/LazyFrames for any `ForeignKey` referencing
            another FrameSpec, as in `generate()`. Note each batch samples
            independently, so a `unique=True` FK column is only sampled
            without replacement *within* a batch, not across the whole `n`.

        Yields
        ------
        pl.DataFrame
            Batches of generated DataFrames matching the spec schema.
        """
        if not cls._columns:
            raise ValueError(f"{cls.__name__} declares no ColSpec columns")
        if n < 0:
            raise ValueError("n must be >= 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if n == 0:
            return

        rng = random.Random(seed)

        if method == "cartesian":
            first_batch = cls.generate(
                min(n, batch_size),
                method="cartesian",
                seed=rng.randrange(2**63),
                references=references,
            )
            # The coverage set built for the first batch can be far larger
            # than batch_size (it's a cross-product, not a row count cap), so
            # slice it before yielding to honor the per-batch memory bound.
            for offset in range(0, first_batch.height, batch_size):
                yield first_batch.slice(offset, batch_size)
            rows_remaining = max(0, n - first_batch.height)
            while rows_remaining > 0:
                current_batch_size = min(rows_remaining, batch_size)
                yield cls.generate(
                    current_batch_size,
                    method="random",
                    seed=rng.randrange(2**63),
                    references=references,
                )
                rows_remaining -= current_batch_size
        elif method == "random":
            rows_remaining = n
            while rows_remaining > 0:
                current_batch_size = min(rows_remaining, batch_size)
                yield cls.generate(
                    current_batch_size,
                    method="random",
                    seed=rng.randrange(2**63),
                    references=references,
                )
                rows_remaining -= current_batch_size
        else:
            raise ValueError(
                f"Unknown method {method!r}; expected 'random' or 'cartesian'"
            )

    @classmethod
    def sink_parquet(
        cls,
        path: str | Path,
        n: int,
        *,
        batch_size: int = 100_000,
        compression: str = "zstd",
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
        references: Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None = None,
        **kwargs,
    ) -> None:
        """Generates `n` rows and streams them directly to a Parquet file in batches.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        n : int
            Total number of rows to generate and sink.
        batch_size : int, default 100_000
            Number of rows per generated batch.
        compression : str, default "zstd"
            Parquet compression codec (e.g. "zstd", "snappy", "gzip", "none").
        method : Literal["random", "cartesian"], default "random"
            Generation strategy ("random" or "cartesian").
        seed : int | None, optional
            Random seed for reproducibility.
        references : Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None, optional
            Parent DataFrames/LazyFrames for any `ForeignKey` referencing
            another FrameSpec, as in `generate()`.
        **kwargs
            Additional arguments passed to `pyarrow.parquet.ParquetWriter`.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for sink_parquet(). Please install pyarrow."
            ) from exc

        if not cls._columns:
            raise ValueError(f"{cls.__name__} declares no ColSpec columns")
        if n < 0:
            raise ValueError("n must be >= 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        writer = None
        try:
            for batch_df in cls.generate_batches(
                n,
                batch_size=batch_size,
                method=method,
                seed=seed,
                references=references,
            ):
                arrow_table = batch_df.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(
                        str(path),
                        arrow_table.schema,
                        compression=compression,
                        **kwargs,
                    )
                writer.write_table(arrow_table)
        finally:
            if writer is not None:
                writer.close()
            elif n == 0:
                empty_df = cls.generate(0, references=references)
                empty_table = empty_df.to_arrow()
                writer = pq.ParquetWriter(
                    str(path),
                    empty_table.schema,
                    compression=compression,
                    **kwargs,
                )
                writer.close()

    @classmethod
    def sink_csv(
        cls,
        path: str | Path,
        n: int,
        *,
        batch_size: int = 100_000,
        include_header: bool = True,
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
        references: Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None = None,
        **kwargs,
    ) -> None:
        """Generates `n` rows and streams them directly to a CSV file in batches.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        n : int
            Total number of rows to generate and sink.
        batch_size : int, default 100_000
            Number of rows per generated batch.
        include_header : bool, default True
            Whether to include the CSV header row.
        method : Literal["random", "cartesian"], default "random"
            Generation strategy ("random" or "cartesian").
        seed : int | None, optional
            Random seed for reproducibility.
        references : Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None, optional
            Parent DataFrames/LazyFrames for any `ForeignKey` referencing
            another FrameSpec, as in `generate()`.
        **kwargs
            Additional arguments passed to `pl.DataFrame.write_csv`.
        """
        if not cls._columns:
            raise ValueError(f"{cls.__name__} declares no ColSpec columns")
        if n < 0:
            raise ValueError("n must be >= 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        header_needed = include_header
        with open(path, "wb") as f:
            if n == 0:
                empty_df = cls.generate(0, references=references)
                if include_header:
                    empty_df.write_csv(f, include_header=True, **kwargs)
                return

            for batch_df in cls.generate_batches(
                n,
                batch_size=batch_size,
                method=method,
                seed=seed,
                references=references,
            ):
                batch_df.write_csv(f, include_header=header_needed, **kwargs)
                header_needed = False

    @classmethod
    def sink_ipc(
        cls,
        path: str | Path,
        n: int,
        *,
        batch_size: int = 100_000,
        compression: str | None = "zstd",
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
        references: Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None = None,
        **kwargs,
    ) -> None:
        """Generates `n` rows and streams them directly to an Arrow IPC / Feather file in batches.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        n : int
            Total number of rows to generate and sink.
        batch_size : int, default 100_000
            Number of rows per generated batch.
        compression : str | None, default "zstd"
            Compression codec (e.g. "zstd", "lz4", "uncompressed", None).
        method : Literal["random", "cartesian"], default "random"
            Generation strategy ("random" or "cartesian").
        seed : int | None, optional
            Random seed for reproducibility.
        references : Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None, optional
            Parent DataFrames/LazyFrames for any `ForeignKey` referencing
            another FrameSpec, as in `generate()`.
        **kwargs
            Additional arguments passed to `pyarrow.ipc.new_file`.
        """
        try:
            from pyarrow import ipc
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required for sink_ipc(). Please install pyarrow."
            ) from exc

        if not cls._columns:
            raise ValueError(f"{cls.__name__} declares no ColSpec columns")
        if n < 0:
            raise ValueError("n must be >= 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        writer = None
        with open(path, "wb") as f:
            try:
                for batch_df in cls.generate_batches(
                    n,
                    batch_size=batch_size,
                    method=method,
                    seed=seed,
                    references=references,
                ):
                    arrow_table = batch_df.to_arrow()
                    if writer is None:
                        writer = ipc.new_file(
                            f,
                            arrow_table.schema,
                            options=ipc.IpcWriteOptions(compression=compression),
                            **kwargs,
                        )
                    writer.write_table(arrow_table)
            finally:
                if writer is not None:
                    writer.close()
                elif n == 0:
                    empty_df = cls.generate(0, references=references)
                    empty_table = empty_df.to_arrow()
                    writer = ipc.new_file(
                        f,
                        empty_table.schema,
                        options=ipc.IpcWriteOptions(compression=compression),
                        **kwargs,
                    )
                    writer.close()

    @classmethod
    def sink_ndjson(
        cls,
        path: str | Path,
        n: int,
        *,
        batch_size: int = 100_000,
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
        references: Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None = None,
        **kwargs,
    ) -> None:
        """Generates `n` rows and streams them directly to a newline-delimited JSON (NDJSON) file in batches.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        n : int
            Total number of rows to generate and sink.
        batch_size : int, default 100_000
            Number of rows per generated batch.
        method : Literal["random", "cartesian"], default "random"
            Generation strategy ("random" or "cartesian").
        seed : int | None, optional
            Random seed for reproducibility.
        references : Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None, optional
            Parent DataFrames/LazyFrames for any `ForeignKey` referencing
            another FrameSpec, as in `generate()`.
        **kwargs
            Additional arguments passed to `pl.DataFrame.write_ndjson`.
        """
        if not cls._columns:
            raise ValueError(f"{cls.__name__} declares no ColSpec columns")
        if n < 0:
            raise ValueError("n must be >= 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            if n == 0:
                return
            for batch_df in cls.generate_batches(
                n,
                batch_size=batch_size,
                method=method,
                seed=seed,
                references=references,
            ):
                batch_df.write_ndjson(f, **kwargs)

    @classmethod
    def to_markdown(
        cls,
        path: str | Path | None = None,
        *,
        title: str | None = None,
    ) -> str:
        """Generates a Markdown data dictionary document for this FrameSpec.

        Parameters
        ----------
        path : str | Path | None, optional
            If specified, writes the generated Markdown to this file path.
        title : str | None, optional
            Custom title for the data dictionary. Defaults to the FrameSpec class name.

        Returns
        -------
        str
            The formatted Markdown string.
        """
        doc_title = title or cls.__name__
        lines: list[str] = [
            f"# {doc_title}",
            "",
            "## Overview",
            f"- **Schema:** `{cls.__name__}`",
            f"- **Total Columns:** {len(cls._columns)}",
        ]

        if cls._unique_together:
            ut_str = ", ".join(f"`{list(g)}`" for g in cls._unique_together)
            lines.append(f"- **Composite Unique Keys:** {ut_str}")

        if cls._checks:
            lines.append(
                f"- **Custom Invariants / Checks:** {len(cls._checks)} check(s)"
            )

        if cls._foreign_keys:
            lines.append(f"- **Foreign Keys:** {len(cls._foreign_keys)} key(s)")

        lines.extend(
            [
                "",
                "## Columns",
                "",
                "| Column | Type | Nullable | Bounds | Domain / Choices | String Length | Tags | Rules | Unique |",
                "|:---|:---|:---|:---|:---|:---|:---|:---|:---|",
            ]
        )

        for col_name, spec in cls._columns.items():
            dtype_str = str(spec.dtype)
            if isinstance(spec.dtype, pl.Enum):
                cats_preview = list(spec.dtype.categories)
                if len(cats_preview) <= 4:
                    dtype_str = f"Enum({cats_preview})"
                else:
                    dtype_str = (
                        f"Enum({cats_preview[:3]} + {len(cats_preview) - 3} more)"
                    )
            elif isinstance(spec.dtype, pl.Categorical):
                cat_obj = getattr(spec.dtype, "categories", None)
                if cat_obj and hasattr(cat_obj, "name") and cat_obj.name():
                    dtype_str = f"Categorical({cat_obj.name()})"
                else:
                    dtype_str = "Categorical"

            nullable_str = "Yes" if spec.nullable else "No"
            bounds_str = str(spec.bounds) if spec.bounds else "-"

            if spec.choices is not None:
                ch_list = list(spec.choices)
                if len(ch_list) <= 4:
                    choices_str = str(ch_list)
                else:
                    choices_str = f"[{', '.join(str(c) for c in ch_list[:3])}, ... ({len(ch_list)} total)]"
            else:
                choices_str = "-"

            str_len_str = (
                f"[{spec.string_length.min}, {spec.string_length.max}]"
                if spec.string_length
                else "-"
            )
            tags_str = ", ".join(f"`{t}`" for t in spec.tags) if spec.tags else "-"
            rules_str = f"{len(spec.rules)} rule(s)" if spec.rules else "-"
            unique_str = "Yes" if spec.unique else "No"

            lines.append(
                f"| `{col_name}` | `{dtype_str}` | {nullable_str} | {bounds_str} | {choices_str} | {str_len_str} | {tags_str} | {rules_str} | {unique_str} |"
            )

        has_constraints = bool(
            cls._checks
            or cls._unique_together
            or cls._foreign_keys
            or any(s.rules for s in cls._columns.values())
            or any(s.validators for s in cls._columns.values())
        )
        if has_constraints:
            lines.extend(
                [
                    "",
                    "## Constraints & Invariants",
                ]
            )

            if cls._unique_together:
                lines.extend(
                    [
                        "",
                        "### Composite Uniqueness",
                    ]
                )
                for group in cls._unique_together:
                    lines.append(f"- Key: `{list(group)}`")

            if cls._checks:
                lines.extend(
                    [
                        "",
                        "### Multi-Column Checks",
                    ]
                )
                for chk in cls._checks:
                    desc_line = (
                        f"\n  - *Description:* {chk.description}"
                        if chk.description
                        else ""
                    )
                    lines.append(f"- **`{chk.name}`**: `{chk.expr}`{desc_line}")

            if cls._foreign_keys:
                lines.extend(
                    [
                        "",
                        "### Foreign Keys",
                    ]
                )
                for fk in cls._foreign_keys:
                    target_label = (
                        cls.__name__
                        if fk.references == "self"
                        else fk.references.__name__
                    )
                    lines.append(
                        f"- **`{fk.name}`**: `{list(fk.columns)}` -> "
                        f"`{target_label}.{list(fk.ref_columns)}`"
                    )

            cols_with_rules = [
                (name, spec) for name, spec in cls._columns.items() if spec.rules
            ]
            if cols_with_rules:
                lines.extend(
                    [
                        "",
                        "### Conditional Rules (`ColRule`)",
                    ]
                )
                for name, spec in cols_with_rules:
                    lines.append(f"- **Column `{name}`**:")
                    for r_idx, rule in enumerate(spec.rules, 1):
                        lines.append(
                            f"  {r_idx}. When `{rule.when}` -> Choices: `{list(rule.choices)}`"
                        )

            cols_with_validators = [
                (name, spec) for name, spec in cls._columns.items() if spec.validators
            ]
            if cols_with_validators:
                lines.extend(
                    [
                        "",
                        "### Column Validators",
                    ]
                )
                for name, spec in cols_with_validators:
                    lines.append(f"- **Column `{name}`**:")
                    for validator in spec.validators:
                        desc_line = (
                            f" -- {validator.description}"
                            if validator.description
                            else ""
                        )
                        lines.append(
                            f"  - **`{validator.name}`**: `{validator.expr}`{desc_line}"
                        )

        content = "\n".join(lines) + "\n"
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        return content

    @classmethod
    def to_mermaid(
        cls,
        path: str | Path | None = None,
        *,
        title: str | None = None,
    ) -> str:
        """Generates a Mermaid Entity-Relationship (ER) diagram for this FrameSpec.

        Parameters
        ----------
        path : str | Path | None, optional
            If specified, writes the generated Mermaid diagram to this file path.
        title : str | None, optional
            Entity name in the diagram. Defaults to the FrameSpec class name.

        Returns
        -------
        str
            The formatted Mermaid diagram definition.
        """
        entity_name = title or cls.__name__
        entity_name = "".join(
            c if c.isalnum() or c == "_" else "_" for c in entity_name
        )

        fk_columns: dict[str, ForeignKey] = {}
        for fk in cls._foreign_keys:
            for col in fk.columns:
                fk_columns.setdefault(col, fk)

        lines: list[str] = [
            "erDiagram",
            f"    {entity_name} {{",
        ]

        for col_name, spec in cls._columns.items():
            dtype = spec.dtype
            if isinstance(dtype, pl.Enum):
                type_name = "Enum"
            elif isinstance(dtype, pl.Categorical):
                type_name = "Categorical"
            elif isinstance(dtype, pl.Datetime):
                type_name = "Datetime"
            elif isinstance(dtype, pl.Duration):
                type_name = "Duration"
            elif isinstance(dtype, pl.List):
                type_name = "List"
            elif isinstance(dtype, pl.Struct):
                type_name = "Struct"
            elif isinstance(dtype, pl.Array):
                type_name = "Array"
            else:
                type_name = type(dtype).__name__

            key_token = ""
            if spec.unique:
                key_token = "PK"
            elif any(col_name in group for group in cls._unique_together):
                key_token = "UK"
            elif col_name in fk_columns:
                key_token = "FK"

            comments: list[str] = []
            if spec.nullable:
                comments.append("nullable")
            if spec.bounds is not None:
                comments.append(f"bounds: {spec.bounds}")
            elif spec.choices is not None:
                ch = list(spec.choices)
                if len(ch) <= 3:
                    comments.append(f"choices: [{', '.join(str(c) for c in ch)}]")
                else:
                    comments.append(f"choices: [{len(ch)} items]")
            if spec.tags:
                comments.append(f"tags: [{', '.join(spec.tags)}]")
            if spec.string_length is not None:
                comments.append(
                    f"len: [{spec.string_length.min}, {spec.string_length.max}]"
                )

            comment_body = ", ".join(comments).replace('"', "'")
            comment_str = f' "{comment_body}"' if comments else ""
            key_str = f" {key_token}" if key_token else ""
            lines.append(f"        {type_name} {col_name}{key_str}{comment_str}")

        lines.append("    }")

        if cls._foreign_keys:
            for fk in cls._foreign_keys:
                if fk.references == "self":
                    target_name = entity_name
                else:
                    target_name = "".join(
                        c if c.isalnum() or c == "_" else "_"
                        for c in fk.references.__name__
                    )
                fk_label = fk.name.replace('"', "'") if fk.name else "references"
                lines.append(f'    {target_name} ||--o{{ {entity_name} : "{fk_label}"')

        content = "\n".join(lines) + "\n"

        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        return content

    @overload
    @classmethod
    def validate(
        cls,
        df: pl.DataFrame,
        *,
        extra_cols: Literal["drop", "allow", "raise"] = "raise",
        missing_cols: Literal["add", "allow", "raise"] = "raise",
        strict_dtypes: bool = False,
        validate_rules: bool = True,
        validate_validators: bool = True,
        validate_unique: bool = True,
        validate_checks: bool = True,
        validate_foreign_keys: bool = True,
        references: Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None = None,
        cast: bool = False,
        streaming: bool = False,
    ) -> pl.DataFrame: ...

    @overload
    @classmethod
    def validate(
        cls,
        df: pl.LazyFrame,
        *,
        extra_cols: Literal["drop", "allow", "raise"] = "raise",
        missing_cols: Literal["add", "allow", "raise"] = "raise",
        strict_dtypes: bool = False,
        validate_rules: bool = True,
        validate_validators: bool = True,
        validate_unique: bool = True,
        validate_checks: bool = True,
        validate_foreign_keys: bool = True,
        references: Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None = None,
        cast: bool = False,
        streaming: bool = False,
    ) -> pl.LazyFrame: ...

    @classmethod
    def validate(
        cls,
        df: pl.DataFrame | pl.LazyFrame,
        *,
        extra_cols: Literal["drop", "allow", "raise"] = "raise",
        missing_cols: Literal["add", "allow", "raise"] = "raise",
        strict_dtypes: bool = False,
        validate_rules: bool = True,
        validate_validators: bool = True,
        validate_unique: bool = True,
        validate_checks: bool = True,
        validate_foreign_keys: bool = True,
        references: Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None = None,
        cast: bool = False,
        streaming: bool = False,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Validates a DataFrame or LazyFrame against this spec's schema and constraints.

        Parameters
        ----------
        df : pl.DataFrame | pl.LazyFrame
            The DataFrame or LazyFrame to validate.
        extra_cols : Literal["drop", "allow", "raise"], default "raise"
            How to handle columns present in `df` but not declared in this spec:
            - "raise": raise a ValidationError containing all extra columns.
            - "drop": drop extra columns from the returned DataFrame/LazyFrame.
            - "allow": retain extra columns in the returned DataFrame/LazyFrame.
        missing_cols : Literal["add", "allow", "raise"], default "raise"
            How to handle columns declared in this spec but missing from `df`:
            - "raise": raise a ValidationError containing all missing columns.
            - "add": add missing columns populated with nulls of the declared dtype.
            - "allow": skip missing columns without raising an error.
        strict_dtypes : bool, default False
            Whether to strictly enforce identical data types (True) or allow compatible
            types like widened integers, floats, or string representations (False).
        validate_rules : bool, default True
            Whether to validate conditional `ColRule` expressions defined on columns.
        validate_validators : bool, default True
            Whether to validate single-column `ColSpec.validators` predicates.
        validate_unique : bool, default True
            Whether to validate single-column (`unique=True`) and composite unique constraints (`__unique_together__`).
        validate_checks : bool, default True
            Whether to validate multi-column `Check` constraints (`__checks__`).
        validate_foreign_keys : bool, default True
            Whether to validate `ForeignKey` referential-integrity constraints
            (`__foreign_keys__`).
        references : Mapping[type[FrameSpec], pl.DataFrame | pl.LazyFrame] | None, optional
            Parent DataFrames/LazyFrames for any `ForeignKey` that references
            another FrameSpec, keyed by that FrameSpec class. Not needed for
            self-referencing ForeignKeys (`references="self"`), which are
            checked against `df` itself.
        cast : bool, default False
            If True, casts validated columns to the declared `ColSpec.dtype`.
        streaming : bool, default False
            If True, uses Polars' streaming execution engine for evaluating LazyFrames.

        Returns
        -------
        pl.DataFrame | pl.LazyFrame
            The validated (and optionally transformed) DataFrame or LazyFrame.

        Raises
        ------
        ValidationError
            If any structural or column-level constraints are violated, collecting
            all violations across all columns before raising.
        ValueError
            If invalid options are supplied for `extra_cols` or `missing_cols`, or if
            a declared ForeignKey references another FrameSpec but no matching
            DataFrame was supplied via `references`.
        """
        if not cls._columns:
            raise ValueError(f"{cls.__name__} declares no ColSpec columns")

        resolved_foreign_keys: list[tuple[ForeignKey, pl.LazyFrame | None]] = []
        if validate_foreign_keys:
            for fk in cls._foreign_keys:
                if fk.references == "self":
                    resolved_foreign_keys.append((fk, None))
                    continue
                target = fk.references
                parent = references.get(target) if references else None
                if parent is None:
                    raise ValueError(
                        f"{cls.__name__}.validate(): ForeignKey {fk.name!r} "
                        f"references {target.__name__!r}, but no DataFrame for it "
                        "was supplied via validate(references={...})"
                    )
                parent_lf = (
                    parent.lazy() if isinstance(parent, pl.DataFrame) else parent
                )
                resolved_foreign_keys.append((fk, parent_lf))

        return _validate_dataframe(
            cls._columns,
            cls.__name__,
            df,
            extra_cols=extra_cols,
            missing_cols=missing_cols,
            strict_dtypes=strict_dtypes,
            validate_rules=validate_rules,
            validate_validators=validate_validators,
            validate_unique=validate_unique,
            validate_checks=validate_checks,
            validate_foreign_keys=validate_foreign_keys,
            checks=cls._checks if validate_checks else None,
            unique_together=cls._unique_together if validate_unique else None,
            foreign_keys=resolved_foreign_keys if validate_foreign_keys else None,
            cast=cast,
            streaming=streaming,
        )


# Names a ColSpec attribute may not take: assigning one shadows the classmethod
# it collides with, so MySpec.schema() would resolve to a ColSpec instance.
_RESERVED_ATTRS = frozenset(
    name for name in vars(FrameSpec) if not name.startswith("_")
)

FrameSchema = FrameSpec
