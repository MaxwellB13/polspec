from __future__ import annotations

import re
import warnings
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import polars as pl

from polspec.errors import SpecError
from polspec.serialization.dtypes import categories_from_data, physical_from_name
from polspec.spec import _is_categorical_dtype
from polspec.tablespec import TableSpec, as_table_spec

if TYPE_CHECKING:
    from polspec.framespec import FrameSpec

DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    r"(?:^|.*_)id$",
    r"(?:^|.*_)uuid$",
    r"(?:^|.*_)hash$",
    r"(?:^|.*_)url$",
    r"(?:^|.*_)key$",
)


def _matches_patterns(name: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pat, name, re.IGNORECASE) for pat in patterns)


def _auto_physical(n_unique: int) -> pl.DataType:
    if n_unique < 256:
        return pl.UInt8
    elif n_unique < 65536:
        return pl.UInt16
    return pl.UInt32


def _declared_categories_from(value: object) -> pl.Categories | None:
    """The Categories a class-body value declares, if it declares one at all.

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


class _EnumAccessor:
    """Helper accessor supporting both callable and attribute/item lookup for pl.Enum data types.

    Example:
        categories.enum("ORDER_STATUS")
        categories.enum.ORDER_STATUS
        categories.enum["ORDER_STATUS"]
    """

    def __init__(self, spec: CatSpec) -> None:
        self._spec = spec

    def __call__(self, name: str) -> pl.Enum:
        return pl.Enum(self._spec.get_enum(name))

    def __getattr__(self, name: str) -> pl.Enum:
        if name.startswith("_"):
            raise AttributeError(name)
        return pl.Enum(self._spec.get_enum(name))

    def __getitem__(self, name: str) -> pl.Enum:
        return pl.Enum(self._spec.get_enum(name))

    def __repr__(self) -> str:
        return f"_EnumAccessor({list(self._spec.enums.keys())})"


class _CategoricalAccessor:
    """Helper accessor supporting both callable and attribute/item lookup for pl.Categorical data types.

    Example:
        categories.categorical("CURRENCY")
        categories.categorical.CURRENCY
        categories.categorical["CURRENCY"]
    """

    def __init__(self, spec: CatSpec) -> None:
        self._spec = spec

    def __call__(self, name: str) -> pl.Categorical:
        return pl.Categorical(self._spec.get_categorical(name))

    def __getattr__(self, name: str) -> pl.Categorical:
        if name.startswith("_"):
            raise AttributeError(name)
        return pl.Categorical(self._spec.get_categorical(name))

    def __getitem__(self, name: str) -> pl.Categorical:
        return pl.Categorical(self._spec.get_categorical(name))

    def __repr__(self) -> str:
        return f"_CategoricalAccessor({list(self._spec.categoricals.keys())})"


class CatSpec:
    """A centralized registry for Enums and Categoricals across pipelines and FrameSpecs.

    Stores Enum variant sequences (for `pl.Enum`) and named Categorical definitions
    (for `pl.Categorical` / `pl.Categories`), enabling seamless schema reuse, join
    compatibility, and clean YAML serialization.

    The declarative way to write one by hand is to subclass it, one line per
    entry, in the same vocabulary `ColSpec.dtype` already accepts:

        class Categories(CatSpec):
            STATUS   = pl.Enum(["NEW", "PAID", "SHIPPED"])
            CURRENCY = pl.Categorical(pl.Categories("CURRENCY", physical=pl.UInt8))

        ColSpec(Categories.STATUS)
        ColSpec(Categories.CURRENCY)

    `Categories.STATUS` is a plain class attribute -- ordinary Python
    attribute lookup, nothing polspec-specific -- so it can be handed
    straight to `ColSpec()`. Instantiating the class (`Categories()`) also
    still works, and behaves exactly like the keyword-argument form below,
    for the registry-level operations (`to_yaml`, `get_enum`,
    case-insensitive lookup, ...) that need an instance.

    This makes `.STATUS` mean two different things depending on how the
    registry was built, which is worth being deliberate about rather than
    surprised by: on a class-body subclass, plain attribute access returns
    the dtype exactly as written (a `pl.Enum`/`pl.Categorical`), because that
    is a real class attribute and nothing intercepts it. On a registry built
    from `enums=`/`categoricals=` dicts, `.STATUS` goes through `__getattr__`
    instead and returns the raw category list -- `pl.Enum(cats.STATUS)` is
    how that form turns it into a dtype. `get_enum()`, `get_categorical()`,
    `[...]` and the `.enum`/`.categorical` accessors behave identically
    either way, since those always go through the registry, never through
    plain attribute lookup.

    The `enums=`/`categoricals=`/`choices=` constructor below remains the
    form `CatSpec.infer()`, `from_dataframe()` and `from_yaml()` build
    programmatically, since their names come from data rather than from a
    class body someone writes by hand -- the two are not in competition.

    Examples
    --------
    >>> cats = CatSpec(
    ...     enums={"STATUS": ["PENDING", "COMPLETED"]},
    ...     categoricals={"CURRENCY": pl.Categories("CURRENCY", physical=pl.UInt8)},
    ... )
    >>> pl.Enum(cats.STATUS)
    Enum(categories=['PENDING', 'COMPLETED'])
    >>> pl.Categorical(cats.CURRENCY)
    Categorical(Categories(name="CURRENCY", namespace="", physical=pl.UInt8))
    >>> cats.enum.STATUS
    Enum(categories=['PENDING', 'COMPLETED'])
    >>> cats.categorical.CURRENCY
    Categorical(Categories(name="CURRENCY", namespace="", physical=pl.UInt8))
    """

    __slots__ = (
        "_cat_accessor",
        "_categoricals",
        "_choices",
        "_enum_accessor",
        "_enums",
    )

    # Populated by __init_subclass__ from bare pl.Enum / pl.Categorical class
    # attributes; empty for CatSpec itself and for any subclass declaring none.
    _declared_enums: ClassVar[dict[str, list[str]]] = {}
    _declared_categoricals: ClassVar[dict[str, pl.Categories]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Collects this subclass's pl.Enum / pl.Categorical class attributes.

        Walks the MRO like `FrameSpec.__init_subclass__` collects `ColSpec`
        columns, so a subclass inherits and can override entries the same
        way. Anything that isn't a `pl.Enum`, a `pl.Categorical` wrapping a
        named `pl.Categories`, or a bare `pl.Categories` is left alone --
        including CatSpec's own methods, which are never instances of either.
        """
        super().__init_subclass__(**kwargs)
        enums: dict[str, list[str]] = {}
        categoricals: dict[str, pl.Categories] = {}

        for base in reversed(cls.__mro__):
            for name, value in vars(base).items():
                if name.startswith("_"):
                    continue

                if isinstance(value, pl.Enum):
                    target, entry = enums, value.categories.to_list()
                else:
                    categories_obj = _declared_categories_from(value)
                    if categories_obj is None:
                        continue
                    if not categories_obj.name():
                        raise SpecError(
                            f"{cls.__name__}.{name} has no name, but a CatSpec "
                            "entry needs one to act as a shared registry key -- "
                            f"give it one: pl.Categories({name!r}, ...)"
                        )
                    target, entry = categoricals, categories_obj

                if name in _RESERVED_ATTRS:
                    # A warning, not an error, for the same reason FrameSpec's
                    # column/method collision is a warning: the entry itself
                    # still works (Categories.get, say, is still a valid
                    # attribute), only the shadowed method is lost.
                    warnings.warn(
                        f"{cls.__name__}.{name} shadows CatSpec.{name}, so "
                        f"{cls.__name__}().{name} no longer resolves to the "
                        "method. Access it via .get(...), the .enum/"
                        ".categorical accessors, or rename the entry.",
                        stacklevel=2,
                    )

                target[name] = entry

        cls._declared_enums = enums
        cls._declared_categoricals = categoricals

    def __init__(
        self,
        *,
        enums: Mapping[str, Sequence[str]] | None = None,
        categoricals: Mapping[str, pl.Categories | dict[str, Any] | str | pl.DataType]
        | None = None,
        choices: Mapping[str, Sequence[Any]] | None = None,
    ) -> None:
        # A subclass declaring pl.Enum/pl.Categorical class attributes (see
        # __init_subclass__) seeds these as defaults; explicit keyword
        # arguments here still add to or override them, per key.
        enums = {**type(self)._declared_enums, **(enums or {})}
        categoricals = {**type(self)._declared_categoricals, **(categoricals or {})}

        self._enums: dict[str, list[str]] = {}
        if enums:
            for k, v in enums.items():
                self._enums[str(k)] = [str(x) for x in v]

        self._categoricals: dict[str, pl.Categories] = {}
        self._choices: dict[str, list[Any]] = {}

        if categoricals:
            for k, v in categoricals.items():
                k_str = str(k)
                if isinstance(v, pl.Categories):
                    self._categoricals[k_str] = v
                elif isinstance(v, str):
                    self._categoricals[k_str] = pl.Categories(
                        k_str, physical=physical_from_name(v)
                    )
                elif isinstance(v, pl.DataType):
                    self._categoricals[k_str] = pl.Categories(k_str, physical=v)
                elif isinstance(v, dict):
                    self._categoricals[k_str] = categories_from_data(v, None, k_str)
                    if "categories" in v:
                        self._choices[k_str] = list(v["categories"])
                else:
                    raise SpecError(
                        f"Unsupported categorical specification for {k!r}: {v!r}"
                    )

        if choices:
            for k, v in choices.items():
                self._choices[str(k)] = list(v)

        self._enum_accessor = _EnumAccessor(self)
        self._cat_accessor = _CategoricalAccessor(self)

    @property
    def enums(self) -> dict[str, list[str]]:
        """Dictionary of registered Enum variant lists."""
        return dict(self._enums)

    @property
    def categoricals(self) -> dict[str, pl.Categories]:
        """Dictionary of registered Categorical definitions."""
        return dict(self._categoricals)

    @property
    def enum(self) -> _EnumAccessor:
        """Accessor for pl.Enum dtypes."""
        return self._enum_accessor

    @property
    def categorical(self) -> _CategoricalAccessor:
        """Accessor for pl.Categorical dtypes."""
        return self._cat_accessor

    def resolve_key(
        self, name: str
    ) -> tuple[Literal["enum", "categorical"], str] | None:
        """Which registry entry a column name binds to, if any.

        Exact match first, then upper- and lower-case forms. Enums win over
        categoricals of the same name.
        """
        key = self._resolve_enum_key(name)
        if key is not None:
            return ("enum", key)
        key = self._resolve_cat_key(name)
        if key is not None:
            return ("categorical", key)
        return None

    def _resolve_enum_key(self, name: str) -> str | None:
        if name in self._enums:
            return name
        if name.upper() in self._enums:
            return name.upper()
        if name.lower() in self._enums:
            return name.lower()
        return None

    def _resolve_cat_key(self, name: str) -> str | None:
        if name in self._categoricals:
            return name
        if name.upper() in self._categoricals:
            return name.upper()
        if name.lower() in self._categoricals:
            return name.lower()
        return None

    def get_enum(self, name: str) -> list[str]:
        """Returns the list of categories/variants for an enum."""
        key = self._resolve_enum_key(name)
        if key is not None:
            return list(self._enums[key])
        raise KeyError(f"No enum named {name!r} in CatSpec")

    def get_categorical(self, name: str) -> pl.Categories:
        """Returns the pl.Categories definition for a categorical."""
        key = self._resolve_cat_key(name)
        if key is not None:
            return self._categoricals[key]
        raise KeyError(f"No categorical named {name!r} in CatSpec")

    def get_choices(self, name: str) -> list[Any] | None:
        """Returns domain choices if defined for this enum or categorical."""
        if name in self._choices:
            return list(self._choices[name])
        cat_key = self._resolve_cat_key(name)
        if cat_key is not None and cat_key in self._choices:
            return list(self._choices[cat_key])
        enum_key = self._resolve_enum_key(name)
        if enum_key is not None:
            return list(self._enums[enum_key])
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._categoricals:
            return self._categoricals[name]
        if name in self._enums:
            return list(self._enums[name])
        cat_key = self._resolve_cat_key(name)
        if cat_key is not None:
            return self._categoricals[cat_key]
        enum_key = self._resolve_enum_key(name)
        if enum_key is not None:
            return list(self._enums[enum_key])
        raise AttributeError(f"CatSpec has no Enum or Categorical named {name!r}")

    def __getitem__(self, name: str) -> Any:
        if name in self._categoricals:
            return self._categoricals[name]
        if name in self._enums:
            return list(self._enums[name])
        cat_key = self._resolve_cat_key(name)
        if cat_key is not None:
            return self._categoricals[cat_key]
        enum_key = self._resolve_enum_key(name)
        if enum_key is not None:
            return list(self._enums[enum_key])
        raise KeyError(f"CatSpec has no Enum or Categorical named {name!r}")

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return (
            name in self._categoricals
            or name in self._enums
            or self._resolve_cat_key(name) is not None
            or self._resolve_enum_key(name) is not None
        )

    def __len__(self) -> int:
        return len(self._categoricals) + len(self._enums)

    def __iter__(self) -> Iterator[str]:
        seen = set()
        for k in self._categoricals:
            seen.add(k)
            yield k
        for k in self._enums:
            if k not in seen:
                yield k

    def get(self, name: str, default: Any = None) -> Any:
        if name in self._categoricals:
            return self._categoricals[name]
        if name in self._enums:
            return list(self._enums[name])
        cat_key = self._resolve_cat_key(name)
        if cat_key is not None:
            return self._categoricals[cat_key]
        enum_key = self._resolve_enum_key(name)
        if enum_key is not None:
            return list(self._enums[enum_key])
        return default

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame | pl.LazyFrame) -> CatSpec:
        """Constructs a CatSpec from existing Enum and Categorical columns in a DataFrame or LazyFrame."""
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        if not isinstance(df, pl.DataFrame):
            raise TypeError(
                f"Expected pl.DataFrame or pl.LazyFrame, got {type(df).__name__}"
            )

        enums: dict[str, list[str]] = {}
        categoricals: dict[str, Any] = {}
        choices: dict[str, list[Any]] = {}

        for col_name, dtype in df.schema.items():
            if isinstance(dtype, pl.Enum):
                enums[col_name] = dtype.categories.to_list()
            elif _is_categorical_dtype(dtype):
                if hasattr(dtype, "categories") and isinstance(
                    dtype.categories, pl.Categories
                ):
                    categoricals[col_name] = dtype.categories
                else:
                    categoricals[col_name] = pl.Categories(col_name, physical=pl.UInt32)
                non_null = df[col_name].drop_nulls()
                if len(non_null) > 0:
                    choices[col_name] = non_null.unique().sort().to_list()

        return cls(enums=enums, categoricals=categoricals, choices=choices)

    @classmethod
    def from_framespec(cls, spec: TableSpec | type[FrameSpec]) -> CatSpec:
        """Constructs a CatSpec from a TableSpec or a FrameSpec subclass."""
        table = as_table_spec(spec)

        enums: dict[str, list[str]] = {}
        categoricals: dict[str, Any] = {}
        choices: dict[str, list[Any]] = {}

        for col_name, col_spec in table.columns.items():
            dtype = col_spec.dtype
            if isinstance(dtype, pl.Enum):
                enums[col_name] = dtype.categories.to_list()
            elif _is_categorical_dtype(dtype):
                if hasattr(dtype, "categories") and isinstance(
                    dtype.categories, pl.Categories
                ):
                    categoricals[col_name] = dtype.categories
                else:
                    categoricals[col_name] = pl.Categories(col_name, physical=pl.UInt32)
                if col_spec.choices:
                    choices[col_name] = list(col_spec.choices)
            elif col_spec.choices:
                choices[col_name] = list(col_spec.choices)

        return cls(enums=enums, categoricals=categoricals, choices=choices)

    @classmethod
    def infer_from_dataframe(
        cls,
        df: pl.DataFrame | pl.LazyFrame,
        *,
        max_enum_cardinality: int = 30,
        max_categorical_cardinality: int = 10_000,
        max_categorical_ratio: float = 0.20,
        include_columns: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] | None = DEFAULT_EXCLUDE_PATTERNS,
        default_physical: pl.DataType | None = None,
    ) -> CatSpec:
        """Infers an optimal CatSpec from String and Categorical columns in a DataFrame or LazyFrame."""
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        if not isinstance(df, pl.DataFrame):
            raise TypeError(
                f"Expected pl.DataFrame or pl.LazyFrame, got {type(df).__name__}"
            )

        enums: dict[str, list[str]] = {}
        categoricals: dict[str, Any] = {}
        choices: dict[str, list[Any]] = {}

        total_rows = df.height
        include_set = set(include_columns) if include_columns is not None else None

        for col_name, dtype in df.schema.items():
            if isinstance(dtype, pl.Enum):
                enums[col_name] = dtype.categories.to_list()
                continue
            elif _is_categorical_dtype(dtype):
                non_null = df[col_name].drop_nulls()
                if hasattr(dtype, "categories") and isinstance(
                    dtype.categories, pl.Categories
                ):
                    categoricals[col_name] = dtype.categories
                else:
                    n_u = non_null.n_unique()
                    phys = default_physical or _auto_physical(n_u)
                    categoricals[col_name] = pl.Categories(col_name, physical=phys)
                if len(non_null) > 0:
                    choices[col_name] = non_null.unique().sort().to_list()
                continue

            if dtype not in (pl.String, pl.Utf8):
                continue

            is_explicit_include = include_set is not None and col_name in include_set
            if include_set is not None and not is_explicit_include:
                continue
            if (
                not is_explicit_include
                and exclude_patterns
                and _matches_patterns(col_name, exclude_patterns)
            ):
                continue

            non_null = df[col_name].drop_nulls()
            if len(non_null) == 0:
                continue

            n_unique = non_null.n_unique()
            ratio = n_unique / total_rows if total_rows > 0 else 1.0

            if 0 < n_unique <= max_enum_cardinality:
                unique_vals = non_null.unique().sort().to_list()
                enums[col_name] = [str(x) for x in unique_vals]
            elif (is_explicit_include and n_unique <= max_categorical_cardinality) or (
                n_unique <= max_categorical_cardinality
                and (ratio <= max_categorical_ratio or n_unique <= 256)
            ):
                phys = default_physical or _auto_physical(n_unique)
                categoricals[col_name] = pl.Categories(col_name, physical=phys)
                choices[col_name] = non_null.unique().sort().to_list()

        return cls(enums=enums, categoricals=categoricals, choices=choices)

    @classmethod
    def infer_from_framespec(
        cls,
        spec: TableSpec | type[FrameSpec],
        *,
        max_enum_cardinality: int = 30,
        max_categorical_cardinality: int = 10_000,
        include_columns: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] | None = DEFAULT_EXCLUDE_PATTERNS,
        default_physical: pl.DataType | None = None,
    ) -> CatSpec:
        """Infers an optimal CatSpec from a TableSpec or FrameSpec subclass's declarations."""
        table = as_table_spec(spec)

        enums: dict[str, list[str]] = {}
        categoricals: dict[str, Any] = {}
        choices: dict[str, list[Any]] = {}
        include_set = set(include_columns) if include_columns is not None else None

        for col_name, col_spec in table.columns.items():
            dtype = col_spec.dtype
            if isinstance(dtype, pl.Enum):
                enums[col_name] = dtype.categories.to_list()
                continue
            elif _is_categorical_dtype(dtype):
                if hasattr(dtype, "categories") and isinstance(
                    dtype.categories, pl.Categories
                ):
                    categoricals[col_name] = dtype.categories
                else:
                    categoricals[col_name] = pl.Categories(
                        col_name, physical=default_physical or pl.UInt8
                    )
                if col_spec.choices:
                    choices[col_name] = list(col_spec.choices)
                continue

            if dtype not in (pl.String, pl.Utf8):
                continue

            if getattr(col_spec, "unique", False):
                continue

            if col_spec.string_length and (
                (
                    col_spec.string_length.min is not None
                    and col_spec.string_length.min > 255
                )
                or (
                    col_spec.string_length.max is not None
                    and col_spec.string_length.max > 255
                )
            ):
                continue

            if include_set is not None:
                if col_name not in include_set:
                    continue
            elif exclude_patterns and _matches_patterns(col_name, exclude_patterns):
                continue

            if col_spec.choices:
                c_list = list(col_spec.choices)
                n_unique = len(c_list)
                if 0 < n_unique <= max_enum_cardinality:
                    enums[col_name] = [str(x) for x in c_list]
                elif n_unique <= max_categorical_cardinality:
                    phys = default_physical or _auto_physical(n_unique)
                    categoricals[col_name] = pl.Categories(col_name, physical=phys)
                    choices[col_name] = c_list

        return cls(enums=enums, categoricals=categoricals, choices=choices)

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
        """Infers an optimal CatSpec from a DataFrame, LazyFrame, or FrameSpec."""
        if isinstance(target, (pl.DataFrame, pl.LazyFrame)):
            return cls.infer_from_dataframe(
                target,
                max_enum_cardinality=max_enum_cardinality,
                max_categorical_cardinality=max_categorical_cardinality,
                max_categorical_ratio=max_categorical_ratio,
                include_columns=include_columns,
                exclude_patterns=exclude_patterns,
                default_physical=default_physical,
            )
        return cls.infer_from_framespec(
            target,
            max_enum_cardinality=max_enum_cardinality,
            max_categorical_cardinality=max_categorical_cardinality,
            include_columns=include_columns,
            exclude_patterns=exclude_patterns,
            default_physical=default_physical,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, strict: bool = True) -> CatSpec:
        """Constructs a CatSpec from its data form (any format version)."""
        from polspec.serialization import catspec_from_dict

        if not isinstance(data, dict):
            raise TypeError(
                f"Expected dict for CatSpec data, got {type(data).__name__}"
            )
        return catspec_from_dict(data, strict=strict)

    @classmethod
    def from_yaml(cls, source: str | Path, *, strict: bool = True) -> CatSpec:
        """Loads a CatSpec from a YAML file."""
        from polspec.serialization import catspec_from_yaml

        return catspec_from_yaml(source, strict=strict)

    @property
    def choices(self) -> dict[str, list[Any]]:
        """Every recorded choices list, keyed by column or registry name."""
        return {k: list(v) for k, v in self._choices.items()}

    def to_dict(self) -> dict[str, Any]:
        """The data form of this registry, without the file's `version` key."""
        from polspec.serialization import catspec_to_dict

        return catspec_to_dict(self)

    def to_yaml(self, source: str | Path | None = None) -> str | None:
        """Writes this registry as YAML to `source`, or returns the text when no path is given."""
        from polspec.serialization import catspec_to_yaml

        return catspec_to_yaml(self, source)

    def to_markdown(
        self,
        path: str | Path | None = None,
        *,
        title: str | None = None,
    ) -> str:
        """Generates a Markdown documentation table of this CatSpec registry.

        Parameters
        ----------
        path : str | Path | None, optional
            File destination to write. If None, returns the Markdown string.
        title : str | None, optional
            Custom title for the document. Defaults to 'Categorical & Enum Registry'.

        Returns
        -------
        str
            The formatted Markdown string.
        """
        from polspec.report import catspec_to_markdown

        return catspec_to_markdown(self, path, title=title)

    def to_mermaid(
        self,
        path: str | Path | None = None,
        *,
        title: str | None = None,
    ) -> str:
        """Generates a Mermaid class diagram definition for this CatSpec registry."""
        from polspec.report import catspec_to_mermaid

        return catspec_to_mermaid(self, path, title=title)

    def __repr__(self) -> str:
        enum_keys = list(self._enums.keys())
        cat_keys = list(self._categoricals.keys())
        return f"CatSpec(enums={enum_keys}, categoricals={cat_keys})"


# Names a class-body entry may not take: assigning one shadows the CatSpec
# method it collides with, e.g. an entry named "get" would shadow
# CatSpec.get(). Computed here, after CatSpec's own body is complete, and
# referenced (as a module global, resolved lazily) from __init_subclass__
# above -- which only ever runs for a subclass defined later, by which point
# this already exists.
_RESERVED_ATTRS = frozenset(name for name in vars(CatSpec) if not name.startswith("_"))
