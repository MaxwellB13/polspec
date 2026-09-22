"""`CatSpec` -- the `Enum` and `Categorical` domains several specs share.

One registry, written once, so two tables that both hold a status agree on
what a status is -- and, for a `Categorical`, on the physical codes that let
them be joined.

There are two ways to write one and they build the same value:

    class Categories(CatSpec):              # a class body, one entry per line
        STATUS   = pl.Enum(["NEW", "PAID"])
        CURRENCY = pl.Categorical(pl.Categories("CURRENCY", physical=pl.UInt8))

    categories = CatSpec(                   # a constructor, for names from data
        enums={"STATUS": ["NEW", "PAID"]},
        categoricals={"CURRENCY": pl.Categories("CURRENCY", physical=pl.UInt8)},
    )

A class body is read by `_CatSpecMeta`, which lifts the entries out of the
namespace before the class exists -- the same trick `_FrameSpecMeta` plays with
`ColSpec` columns. Two things follow. An entry may be named anything, `get`
included, because a method is never shadowed by one. And an entry is reached
the same way whichever form declared it:

    Categories.STATUS      # pl.Enum(["NEW", "PAID"])
    Categories.spec        # the CatSpec value behind the class
    categories.STATUS      # pl.Enum(["NEW", "PAID"])

Naming an entry always gives back the *dtype*, ready for `ColSpec(...)`. The
pieces underneath are reached by asking for them: `get_enum` for a category
list, `get_categorical` for a `pl.Categories`, `get_choices` for a domain pool.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import polars as pl

from polspec.dtypes import DtypeLike
from polspec.errors import SpecError
from polspec.serialization.dtypes import categories_from_data, physical_from_name
from polspec.spec import _is_categorical_dtype
from polspec.tablespec import TableSpec, as_table_spec

if TYPE_CHECKING:
    from polspec.framespec import FrameSpec

Kind = Literal["enum", "categorical"]

DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    r"(?:^|.*_)id$",
    r"(?:^|.*_)uuid$",
    r"(?:^|.*_)hash$",
    r"(?:^|.*_)url$",
    r"(?:^|.*_)key$",
)


def _refuse_case_clash(names: Sequence[str]) -> None:
    """Lookup is case-insensitive -- a column `status` finds an entry
    `STATUS` -- so two entries differing only in case would be one name with
    two answers. Refused where they are declared, naming both."""
    by_fold: dict[str, str] = {}
    for name in names:
        other = by_fold.setdefault(name.casefold(), name)
        if other != name:
            raise SpecError(
                f"CatSpec entries {other!r} and {name!r} differ only in case, and "
                "lookup is case-insensitive: a column would bind to either. "
                "Rename one."
            )


def _matches_patterns(name: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pat, name, re.IGNORECASE) for pat in patterns)


def _auto_physical(n_unique: int) -> DtypeLike:
    """The narrowest physical dtype that holds `n_unique` distinct codes."""
    if n_unique < 256:
        return pl.UInt8
    if n_unique < 65536:
        return pl.UInt16
    return pl.UInt32


def _declared_categories_from(value: object) -> pl.Categories | None:
    """The Categories a value declares, if it declares one at all.

    Accepts either a bare `pl.Categories(...)` or a `pl.Categorical(...)`
    wrapping one -- the latter is what `ColSpec.dtype` already expects, so a
    value can be copied straight from a CatSpec declaration into a ColSpec
    without unwrapping it first.
    """
    if isinstance(value, pl.Categories):
        return value
    if isinstance(value, pl.Categorical):
        cats = value.categories
        if isinstance(cats, pl.Categories):
            return cats
    return None


def _categories_identity(cats: pl.Categories) -> tuple[str, str, str]:
    """What makes two `pl.Categories` the same shared registry."""
    return (cats.name(), cats.namespace(), str(cats.physical()))


# ---------------------------------------------------------------------------
# Typed accessors
# ---------------------------------------------------------------------------


class _Accessor:
    """Attribute, item and call access to one kind of entry.

    `cats.enum.STATUS`, `cats.enum["STATUS"]` and `cats.enum("STATUS")` are the
    same lookup. Naming the kind is what distinguishes these from `cats.STATUS`:
    they refuse an entry of the other kind rather than quietly returning it.
    """

    __slots__ = ("_kind", "_spec")

    def __init__(self, spec: CatSpec, kind: Kind) -> None:
        self._spec = spec
        self._kind = kind

    def __call__(self, name: str) -> pl.DataType:
        return self[name]

    def __getitem__(self, name: str) -> pl.DataType:
        if self._kind == "enum":
            return pl.Enum(self._spec.get_enum(name))
        return pl.Categorical(self._spec.get_categorical(name))

    def __getattr__(self, name: str) -> pl.DataType:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(str(exc)) from None

    def __repr__(self) -> str:
        entries = self._spec.enums if self._kind == "enum" else self._spec.categoricals
        return f"<{self._kind} accessor: {list(entries)}>"


# ---------------------------------------------------------------------------
# The class body
# ---------------------------------------------------------------------------


def _entry_of(key: str, value: Any, owner: str) -> tuple[Kind, Any] | None:
    """What a class-body assignment declares, or None if it declares nothing."""
    if isinstance(value, pl.Enum):
        return ("enum", value.categories.to_list())
    categories = _declared_categories_from(value)
    if categories is None:
        return None
    if not categories.name():
        raise SpecError(
            f"{owner}.{key} has no name, but a CatSpec entry needs one to act "
            f"as a shared registry key -- give it one: pl.Categories({key!r}, ...)"
        )
    return ("categorical", categories)


class _CatSpecMeta(type):
    """Reads the entries out of a `CatSpec` subclass's body.

    They are removed from the namespace, so the class keeps every method it
    inherits and an entry named `get` is an entry rather than a collision.
    What is left reaches them through `cls.spec`.
    """

    # What `__new__` sets on every class it builds.
    _declared_enums: dict[str, list[str]]
    _declared_categoricals: dict[str, pl.Categories]
    _spec_cache: CatSpec | None

    def __new__(
        mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs
    ) -> _CatSpecMeta:
        enums: dict[str, list[str]] = {}
        categoricals: dict[str, pl.Categories] = {}
        for base in bases:
            enums.update(getattr(base, "_declared_enums", {}))
            categoricals.update(getattr(base, "_declared_categoricals", {}))

        declared: dict[str, tuple[Kind, Any]] = {}
        for key, value in list(namespace.items()):
            if key.startswith("_"):
                continue
            entry = _entry_of(key, value, name)
            if entry is not None:
                declared[key] = entry
        for key, (kind, payload) in declared.items():
            del namespace[key]
            # A redeclaration replaces the entry whatever kind it had before,
            # so a subclass can turn a base's Enum into a Categorical.
            enums.pop(key, None)
            categoricals.pop(key, None)
            if kind == "enum":
                enums[key] = payload
            else:
                categoricals[key] = payload

        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        cls._declared_enums = enums
        cls._declared_categoricals = categoricals
        cls._spec_cache = None
        return cls

    @property
    def spec(cls) -> CatSpec:
        """The `CatSpec` value this class body declares."""
        if cls._spec_cache is None:
            cls._spec_cache = CatSpec(
                enums=cls._declared_enums, categoricals=cls._declared_categoricals
            )
        return cls._spec_cache

    def __getattr__(cls, name: str) -> Any:
        # Only reached when ordinary lookup fails, so a method always wins.
        if name.startswith("_"):
            raise AttributeError(name)
        dtype = cls.spec.get(name)
        if dtype is None:
            raise AttributeError(f"{cls.__name__} has no entry named {name!r}")
        return dtype


# ---------------------------------------------------------------------------
# The value
# ---------------------------------------------------------------------------


class CatSpec(metaclass=_CatSpecMeta):
    """A set of shared `Enum` and `Categorical` domains, as a value.

    Parameters
    ----------
    enums : Mapping[str, Sequence[str]], optional
        Entry name to its ordered category list.
    categoricals : Mapping[str, pl.Categories | dict | str | pl.DataType], optional
        Entry name to its shared `pl.Categories`. A physical dtype (`pl.UInt8`)
        or its name (`"UInt8"`) is shorthand for a registry of that name; a
        mapping is the file form, and its `categories` key becomes `choices`.
    choices : Mapping[str, Sequence[Any]], optional
        The pool of values an entry draws from when generating. An `Enum`'s
        categories already are its pool; this is for `Categorical` entries,
        whose registry names the domain without listing it.

    Notes
    -----
    Naming an entry -- `cats.STATUS`, `cats["STATUS"]`, `cats.get("STATUS")` --
    gives back the dtype, ready to hand to `ColSpec`. Lookup is
    case-insensitive, so a column named `status` finds `STATUS`, and an `Enum`
    wins over a `Categorical` of the same name.

    A subclass of `CatSpec` declares its entries in the class body; see the
    module docstring. Instantiating one takes those entries as defaults, so
    `Categories(enums={"REASON": [...]})` extends rather than replaces.

    Examples
    --------
    >>> cats = CatSpec(enums={"STATUS": ["PENDING", "COMPLETED"]})
    >>> cats.STATUS
    Enum(categories=['PENDING', 'COMPLETED'])
    >>> cats.get_enum("status")
    ['PENDING', 'COMPLETED']
    """

    __slots__ = (
        "_cat_accessor",
        "_categoricals",
        "_choices",
        "_enum_accessor",
        "_enums",
    )

    # Filled in by the metaclass from a subclass's class body; empty on CatSpec
    # itself and on any subclass that declares nothing.
    _declared_enums: ClassVar[dict[str, list[str]]] = {}
    _declared_categoricals: ClassVar[dict[str, pl.Categories]] = {}
    _spec_cache: ClassVar[CatSpec | None] = None

    def __init__(
        self,
        *,
        enums: Mapping[str, Sequence[str]] | None = None,
        categoricals: Mapping[str, pl.Categories | dict[str, Any] | str | pl.DataType]
        | None = None,
        choices: Mapping[str, Sequence[Any]] | None = None,
    ) -> None:
        merged_enums = {**type(self)._declared_enums, **(enums or {})}
        merged_cats = {**type(self)._declared_categoricals, **(categoricals or {})}

        self._enums = {str(k): [str(x) for x in v] for k, v in merged_enums.items()}
        self._categoricals: dict[str, pl.Categories] = {}
        self._choices: dict[str, list[Any]] = {}
        _refuse_case_clash([*self._enums, *(str(k) for k in merged_cats)])

        for key, value in merged_cats.items():
            name = str(key)
            self._categoricals[name] = self._as_categories(name, value)
            if isinstance(value, dict) and "categories" in value:
                self._choices[name] = list(value["categories"])

        for key, value in (choices or {}).items():
            self._choices[str(key)] = list(value)

        self._enum_accessor = _Accessor(self, "enum")
        self._cat_accessor = _Accessor(self, "categorical")

    @staticmethod
    def _as_categories(name: str, value: Any) -> pl.Categories:
        """One `categoricals=` value in every form the constructor accepts."""
        if isinstance(value, pl.Categories):
            return value
        if isinstance(value, str):
            return pl.Categories(name, physical=physical_from_name(value))
        if isinstance(value, pl.DataType):
            declared = _declared_categories_from(value)
            return (
                declared
                if declared is not None
                else pl.Categories(name, physical=value)
            )
        if isinstance(value, dict):
            return categories_from_data(value, None, name)
        raise SpecError(
            f"Unsupported categorical specification for {name!r}: {value!r}"
        )

    # ------------------------------------------------------------------
    # What is in the registry
    # ------------------------------------------------------------------

    @property
    def enums(self) -> Mapping[str, list[str]]:
        """Every `Enum` entry: name to its ordered category list."""
        return MappingProxyType(self._enums)

    @property
    def categoricals(self) -> Mapping[str, pl.Categories]:
        """Every `Categorical` entry: name to its shared `pl.Categories`."""
        return MappingProxyType(self._categoricals)

    @property
    def choices(self) -> Mapping[str, list[Any]]:
        """Every recorded domain pool, keyed by entry name."""
        return MappingProxyType(self._choices)

    @property
    def enum(self) -> _Accessor:
        """Entries as `pl.Enum` dtypes: `cats.enum.STATUS`."""
        return self._enum_accessor

    @property
    def categorical(self) -> _Accessor:
        """Entries as `pl.Categorical` dtypes: `cats.categorical.CURRENCY`."""
        return self._cat_accessor

    # ------------------------------------------------------------------
    # Lookup -- one resolution rule, and everything else built on it
    # ------------------------------------------------------------------

    def resolve_key(self, name: str) -> tuple[Kind, str] | None:
        """Which entry a name binds to, if any.

        Exact match first, then the upper- and lower-case forms, so a column
        named `status` finds `STATUS`. An `Enum` wins over a `Categorical` of
        the same name.
        """
        for kind, entries in (
            ("enum", self._enums),
            ("categorical", self._categoricals),
        ):
            for candidate in (name, name.upper(), name.lower()):
                if candidate in entries:
                    return (kind, candidate)  # type: ignore[return-value]
        return None

    def dtype_of(self, name: str) -> pl.DataType | None:
        """The dtype registered under `name`, or None if nothing is."""
        resolved = self.resolve_key(name)
        if resolved is None:
            return None
        kind, key = resolved
        if kind == "enum":
            return pl.Enum(self._enums[key])
        return pl.Categorical(self._categoricals[key])

    def get_enum(self, name: str) -> list[str]:
        """The category list of an `Enum` entry."""
        resolved = self.resolve_key(name)
        if resolved is None or resolved[0] != "enum":
            raise KeyError(f"CatSpec has no Enum named {name!r}")
        return list(self._enums[resolved[1]])

    def get_categorical(self, name: str) -> pl.Categories:
        """The shared `pl.Categories` of a `Categorical` entry."""
        for candidate in (name, name.upper(), name.lower()):
            if candidate in self._categoricals:
                return self._categoricals[candidate]
        raise KeyError(f"CatSpec has no Categorical named {name!r}")

    def get_choices(self, name: str) -> list[Any] | None:
        """The pool of values an entry draws from, if it has one.

        An `Enum`'s categories are its pool. A `Categorical`'s comes from the
        `choices` it was registered with, since a `pl.Categories` names a shared
        domain without listing what is in it.
        """
        if name in self._choices:
            return list(self._choices[name])
        resolved = self.resolve_key(name)
        if resolved is None:
            return None
        kind, key = resolved
        if kind == "enum":
            return list(self._enums[key])
        return list(self._choices[key]) if key in self._choices else None

    def get(self, name: str, default: Any = None) -> Any:
        """The dtype registered under `name`, or `default` if nothing is."""
        dtype = self.dtype_of(name)
        return default if dtype is None else dtype

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        dtype = self.dtype_of(name)
        if dtype is None:
            raise AttributeError(f"CatSpec has no Enum or Categorical named {name!r}")
        return dtype

    def __getitem__(self, name: str) -> pl.DataType:
        dtype = self.dtype_of(name)
        if dtype is None:
            raise KeyError(f"CatSpec has no Enum or Categorical named {name!r}")
        return dtype

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.resolve_key(name) is not None

    def __iter__(self) -> Iterator[str]:
        """Entry names, `Enum`s first, in declaration order within each kind."""
        seen: set[str] = set()
        for key in (*self._enums, *self._categoricals):
            if key not in seen:
                seen.add(key)
                yield key

    def __len__(self) -> int:
        return len(set(self._enums) | set(self._categoricals))

    # ------------------------------------------------------------------
    # Value semantics
    # ------------------------------------------------------------------

    def _identity(self) -> tuple:
        return (
            tuple((k, tuple(v)) for k, v in self._enums.items()),
            tuple((k, _categories_identity(v)) for k, v in self._categoricals.items()),
            tuple((k, tuple(map(repr, v))) for k, v in self._choices.items()),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CatSpec):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        return hash(self._identity())

    def __repr__(self) -> str:
        return (
            f"CatSpec(enums={list(self._enums)}, "
            f"categoricals={list(self._categoricals)})"
        )

    # ------------------------------------------------------------------
    # Reading a registry out of what already exists
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame | pl.LazyFrame) -> CatSpec:
        """The `Enum` and `Categorical` columns a frame already declares.

        Reads what is there; `infer` is what looks at String columns and
        decides what *could* be one.
        """
        frame = _as_frame(df)
        enums: dict[str, list[str]] = {}
        categoricals: dict[str, Any] = {}
        choices: dict[str, list[Any]] = {}

        for col_name, dtype in frame.schema.items():
            if isinstance(dtype, pl.Enum):
                enums[col_name] = dtype.categories.to_list()
            elif _is_categorical_dtype(dtype):
                categoricals[col_name] = _categories_of(dtype, col_name)
                non_null = frame[col_name].drop_nulls()
                if len(non_null) > 0:
                    choices[col_name] = non_null.unique().sort().to_list()

        return CatSpec(enums=enums, categoricals=categoricals, choices=choices)

    @classmethod
    def from_framespec(cls, spec: TableSpec | type[FrameSpec]) -> CatSpec:
        """The `Enum` and `Categorical` columns a spec already declares."""
        table = as_table_spec(spec)
        enums: dict[str, list[str]] = {}
        categoricals: dict[str, Any] = {}
        choices: dict[str, list[Any]] = {}

        for col_name, col_spec in table.columns.items():
            dtype = col_spec.dtype
            if isinstance(dtype, pl.Enum):
                enums[col_name] = dtype.categories.to_list()
            elif _is_categorical_dtype(dtype):
                categoricals[col_name] = _categories_of(dtype, col_name)
                if col_spec.choices:
                    choices[col_name] = list(col_spec.choices)
            elif col_spec.choices:
                choices[col_name] = list(col_spec.choices)

        return CatSpec(enums=enums, categoricals=categoricals, choices=choices)

    @classmethod
    def infer(
        cls,
        target: pl.DataFrame | pl.LazyFrame | TableSpec | type[FrameSpec],
        *,
        max_enum_cardinality: int = 30,
        max_categorical_cardinality: int = 10_000,
        max_categorical_ratio: float = 0.20,
        include_columns: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] | None = DEFAULT_EXCLUDE_PATTERNS,
        default_physical: pl.DataType | None = None,
    ) -> CatSpec:
        """A registry of the domains `target` looks like it has.

        Existing `Enum` and `Categorical` columns are kept as declared. A
        String column becomes an `Enum` when it holds few enough distinct
        values, or a `Categorical` when it holds many but repeats them; one
        that looks like an identifier is skipped.

        Parameters
        ----------
        target : pl.DataFrame | pl.LazyFrame | TableSpec | type[FrameSpec]
            Data to measure, or a spec to read declarations from.
        max_enum_cardinality : int
            At most this many distinct values makes a column an `Enum`.
        max_categorical_cardinality : int
            Beyond this many, a column is left as String.
        max_categorical_ratio : float
            Distinct values as a fraction of rows, above which a column is too
            close to unique to be worth a category registry. Frames only -- a
            spec has no row count to measure against.
        include_columns : Sequence[str], optional
            Consider only these columns, exempting them from `exclude_patterns`.
        exclude_patterns : Sequence[str], optional
            Regexes for names to skip; defaults to identifier-shaped names
            (`*_id`, `*_uuid`, `*_hash`, `*_url`, `*_key`).
        default_physical : pl.DataType, optional
            Physical dtype for new `Categorical` registries, instead of the
            narrowest one that fits.
        """
        if isinstance(target, (pl.DataFrame, pl.LazyFrame)):
            return _infer_from_frame(
                _as_frame(target),
                max_enum_cardinality=max_enum_cardinality,
                max_categorical_cardinality=max_categorical_cardinality,
                max_categorical_ratio=max_categorical_ratio,
                include_columns=include_columns,
                exclude_patterns=exclude_patterns,
                default_physical=default_physical,
            )
        return _infer_from_spec(
            as_table_spec(target),
            max_enum_cardinality=max_enum_cardinality,
            max_categorical_cardinality=max_categorical_cardinality,
            include_columns=include_columns,
            exclude_patterns=exclude_patterns,
            default_physical=default_physical,
        )

    # ------------------------------------------------------------------
    # Files and diagrams
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, strict: bool = True) -> CatSpec:
        """A registry read from the data form `to_dict` writes."""
        from polspec.serialization import catspec_from_dict

        if not isinstance(data, dict):
            raise TypeError(
                f"Expected dict for CatSpec data, got {type(data).__name__}"
            )
        return catspec_from_dict(data, strict=strict)

    @classmethod
    def from_yaml(cls, source: str | Path, *, strict: bool = True) -> CatSpec:
        """A registry read from a YAML file written by `to_yaml`."""
        from polspec.serialization import catspec_from_yaml

        return catspec_from_yaml(source, strict=strict)

    def to_dict(self) -> dict[str, Any]:
        """This registry as plain data, without the file's `version` key."""
        from polspec.serialization import catspec_to_dict

        return catspec_to_dict(self)

    def to_yaml(self, source: str | Path | None = None) -> str | None:
        """Writes this registry as YAML to `source`, or returns the text."""
        from polspec.serialization import catspec_to_yaml

        return catspec_to_yaml(self, source)

    def to_markdown(
        self, path: str | Path | None = None, *, title: str | None = None
    ) -> str:
        """A Markdown table of every entry; written to `path` when given."""
        from polspec.report import catspec_to_markdown

        return catspec_to_markdown(self, path, title=title)

    def to_mermaid(
        self, path: str | Path | None = None, *, title: str | None = None
    ) -> str:
        """A Mermaid class diagram of every entry; written to `path` when given."""
        from polspec.report import catspec_to_mermaid

        return catspec_to_mermaid(self, path, title=title)


def as_catspec(obj: CatSpec | type[CatSpec]) -> CatSpec:
    """The `CatSpec` behind `obj`: itself, or a subclass's declared `spec`."""
    if isinstance(obj, CatSpec):
        return obj
    if isinstance(obj, type) and issubclass(obj, CatSpec):
        return obj.spec
    raise TypeError(
        f"Expected a CatSpec or a CatSpec subclass, got {type(obj).__name__}"
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _as_frame(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    if not isinstance(df, pl.DataFrame):
        raise TypeError(
            f"Expected pl.DataFrame or pl.LazyFrame, got {type(df).__name__}"
        )
    return df


def _categories_of(
    dtype: pl.DataType, name: str, physical: DtypeLike | None = None
) -> pl.Categories:
    """The shared registry a Categorical column already names, or a new one."""
    declared = _declared_categories_from(dtype)
    if declared is not None:
        return declared
    return pl.Categories(name, physical=physical or pl.UInt32)


def _skipped(
    name: str,
    include: set[str] | None,
    exclude_patterns: Sequence[str] | None,
) -> bool:
    """Whether a column is out of scope for inference.

    An explicit `include_columns` entry is never skipped by a pattern: naming a
    column is a stronger statement than the default exclusions.
    """
    if include is not None:
        return name not in include
    return bool(exclude_patterns) and _matches_patterns(name, exclude_patterns)


def _infer_from_frame(
    df: pl.DataFrame,
    *,
    max_enum_cardinality: int,
    max_categorical_cardinality: int,
    max_categorical_ratio: float,
    include_columns: Sequence[str] | None,
    exclude_patterns: Sequence[str] | None,
    default_physical: pl.DataType | None,
) -> CatSpec:
    """Inference from data: cardinality and repetition decide."""
    enums: dict[str, list[str]] = {}
    categoricals: dict[str, Any] = {}
    choices: dict[str, list[Any]] = {}

    total_rows = df.height
    include = set(include_columns) if include_columns is not None else None

    for col_name, dtype in df.schema.items():
        if isinstance(dtype, pl.Enum):
            enums[col_name] = dtype.categories.to_list()
            continue

        if _is_categorical_dtype(dtype):
            non_null = df[col_name].drop_nulls()
            physical = default_physical or _auto_physical(non_null.n_unique())
            categoricals[col_name] = _categories_of(dtype, col_name, physical)
            if len(non_null) > 0:
                choices[col_name] = non_null.unique().sort().to_list()
            continue

        if dtype not in (pl.String, pl.Utf8):
            continue
        if _skipped(col_name, include, exclude_patterns):
            continue

        non_null = df[col_name].drop_nulls()
        if len(non_null) == 0:
            continue

        n_unique = non_null.n_unique()
        ratio = n_unique / total_rows if total_rows > 0 else 1.0

        if 0 < n_unique <= max_enum_cardinality:
            enums[col_name] = [str(x) for x in non_null.unique().sort().to_list()]
        elif n_unique <= max_categorical_cardinality and (
            include is not None or ratio <= max_categorical_ratio or n_unique <= 256
        ):
            categoricals[col_name] = pl.Categories(
                col_name, physical=default_physical or _auto_physical(n_unique)
            )
            choices[col_name] = non_null.unique().sort().to_list()

    return CatSpec(enums=enums, categoricals=categoricals, choices=choices)


def _infer_from_spec(
    table: TableSpec,
    *,
    max_enum_cardinality: int,
    max_categorical_cardinality: int,
    include_columns: Sequence[str] | None,
    exclude_patterns: Sequence[str] | None,
    default_physical: pl.DataType | None,
) -> CatSpec:
    """Inference from declarations: a column's own `choices` decide.

    Two columns are left alone that data-driven inference cannot see: a
    `unique` column, whose values are distinct by declaration, and one whose
    declared strings are too long to be worth a category registry.
    """
    enums: dict[str, list[str]] = {}
    categoricals: dict[str, Any] = {}
    choices: dict[str, list[Any]] = {}
    include = set(include_columns) if include_columns is not None else None

    for col_name, col_spec in table.columns.items():
        dtype = col_spec.dtype
        if isinstance(dtype, pl.Enum):
            enums[col_name] = dtype.categories.to_list()
            continue

        if _is_categorical_dtype(dtype):
            categoricals[col_name] = _categories_of(
                dtype, col_name, default_physical or pl.UInt8
            )
            if col_spec.choices:
                choices[col_name] = list(col_spec.choices)
            continue

        if dtype not in (pl.String, pl.Utf8) or col_spec.unique:
            continue
        length = col_spec.string_length
        if length is not None and max(length.min or 0, length.max or 0) > 255:
            continue
        if _skipped(col_name, include, exclude_patterns):
            continue
        if not col_spec.choices:
            continue

        declared = list(col_spec.choices)
        n_unique = len(declared)
        if 0 < n_unique <= max_enum_cardinality:
            enums[col_name] = [str(x) for x in declared]
        elif n_unique <= max_categorical_cardinality:
            categoricals[col_name] = pl.Categories(
                col_name, physical=default_physical or _auto_physical(n_unique)
            )
            choices[col_name] = declared

    return CatSpec(enums=enums, categoricals=categoricals, choices=choices)
