from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import yaml

from polspec.serialization import _YAML_DTYPES, _YAML_NAME_TO_DTYPE
from polspec.spec import _is_categorical_dtype

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

    def __init__(
        self,
        *,
        enums: Mapping[str, Sequence[str]] | None = None,
        categoricals: Mapping[str, pl.Categories | dict[str, Any] | str | pl.DataType]
        | None = None,
        choices: Mapping[str, Sequence[Any]] | None = None,
    ) -> None:
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
                    phys = _YAML_NAME_TO_DTYPE.get(v, pl.UInt32)
                    self._categoricals[k_str] = pl.Categories(k_str, physical=phys)
                elif isinstance(v, pl.DataType):
                    self._categoricals[k_str] = pl.Categories(k_str, physical=v)
                elif isinstance(v, dict):
                    cat_name = str(v.get("name", k_str))
                    phys_raw = v.get("physical", "UInt32")
                    if isinstance(phys_raw, str):
                        phys = _YAML_NAME_TO_DTYPE.get(phys_raw, pl.UInt32)
                    elif isinstance(phys_raw, pl.DataType):
                        phys = phys_raw
                    else:
                        phys = pl.UInt32
                    namespace = str(v.get("namespace", ""))
                    self._categoricals[k_str] = pl.Categories(
                        cat_name,
                        namespace=namespace,
                        physical=phys,
                    )
                    if "categories" in v:
                        self._choices[k_str] = list(v["categories"])
                else:
                    raise TypeError(
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
    def from_framespec(cls, spec: type[FrameSpec] | FrameSpec) -> CatSpec:
        """Constructs a CatSpec from a declared FrameSpec class or instance."""
        spec_cls = spec if isinstance(spec, type) else type(spec)
        if not hasattr(spec_cls, "_columns"):
            raise TypeError(f"Expected FrameSpec subclass, got {spec_cls.__name__}")

        enums: dict[str, list[str]] = {}
        categoricals: dict[str, Any] = {}
        choices: dict[str, list[Any]] = {}

        for col_name, col_spec in spec_cls._columns.items():
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
        spec: type[FrameSpec] | FrameSpec,
        *,
        max_enum_cardinality: int = 30,
        max_categorical_cardinality: int = 10_000,
        include_columns: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] | None = DEFAULT_EXCLUDE_PATTERNS,
        default_physical: pl.DataType | None = None,
    ) -> CatSpec:
        """Infers an optimal CatSpec from a declared FrameSpec class using schema definitions."""
        spec_cls = spec if isinstance(spec, type) else type(spec)
        if not hasattr(spec_cls, "_columns"):
            raise TypeError(f"Expected FrameSpec subclass, got {spec_cls.__name__}")

        enums: dict[str, list[str]] = {}
        categoricals: dict[str, Any] = {}
        choices: dict[str, list[Any]] = {}
        include_set = set(include_columns) if include_columns is not None else None

        for col_name, col_spec in spec_cls._columns.items():
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
        target: pl.DataFrame | pl.LazyFrame | type[FrameSpec] | FrameSpec,
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
    def from_dict(cls, data: dict[str, Any]) -> CatSpec:
        """Constructs a CatSpec from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected dict for CatSpec data, got {type(data).__name__}"
            )

        if "enums" in data or "categoricals" in data:
            return cls(
                enums=data.get("enums"),
                categoricals=data.get("categoricals"),
                choices=data.get("choices"),
            )

        # Smart inference for flat format
        enums: dict[str, list[str]] = {}
        categoricals: dict[str, Any] = {}
        choices: dict[str, list[Any]] = {}

        for k, v in data.items():
            if isinstance(v, (list, tuple)):
                enums[k] = list(v)
            elif isinstance(v, (dict, str, pl.Categories, pl.DataType)):
                categoricals[k] = v
            else:
                enums[k] = [str(v)]

        return cls(enums=enums, categoricals=categoricals, choices=choices)

    @classmethod
    def from_yaml(cls, source: str | Path) -> CatSpec:
        """Loads a CatSpec from a YAML file."""
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"CatSpec file not found: {source}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            return cls()
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the CatSpec to a dictionary structure."""
        out: dict[str, Any] = {}
        if self._enums:
            out["enums"] = {k: list(v) for k, v in self._enums.items()}
        if self._categoricals:
            cats_dict = {}
            for k, cat in self._categoricals.items():
                cat_info: dict[str, Any] = {"name": cat.name()}
                ns = cat.namespace()
                if ns:
                    cat_info["namespace"] = ns
                phys = cat.physical()
                if phys != pl.UInt32:
                    cat_info["physical"] = _YAML_DTYPES.get(phys, str(phys))
                if k in self._choices:
                    cat_info["categories"] = list(self._choices[k])
                cats_dict[k] = cat_info
            out["categoricals"] = cats_dict
        return out

    def to_yaml(self, source: str | Path | None = None) -> str | None:
        """Serializes the CatSpec to YAML format.

        Parameters
        ----------
        source : str | Path | None, optional
            File destination. If None, returns the YAML text string.
        """
        data = self.to_dict()
        dumped = yaml.safe_dump(data, sort_keys=False)
        if source is not None:
            p = Path(source)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(dumped, encoding="utf-8")
            return None
        return dumped

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
        doc_title = title or "Categorical & Enum Registry"
        lines: list[str] = [
            f"# {doc_title}",
            "",
            "## Summary",
            f"- **Enums:** {len(self._enums)}",
            f"- **Categoricals:** {len(self._categoricals)}",
            "",
        ]

        if self._enums:
            lines.extend(
                [
                    "## Enums (`pl.Enum`)",
                    "",
                    "| Name | Variants Count | Allowed Variants |",
                    "|:---|:---|:---|",
                ]
            )
            for k, variants in self._enums.items():
                var_str = f"[{', '.join(repr(v) for v in variants[:6])}{', ...' if len(variants) > 6 else ''}]"
                lines.append(f"| `{k}` | {len(variants)} | `{var_str}` |")
            lines.append("")

        if self._categoricals:
            lines.extend(
                [
                    "## Categoricals (`pl.Categorical`)",
                    "",
                    "| Key | Registry Name | Physical Dtype | Namespace | Domain Choices Pool |",
                    "|:---|:---|:---|:---|:---|",
                ]
            )
            for k, cat in self._categoricals.items():
                phys = _YAML_DTYPES.get(cat.physical(), str(cat.physical()))
                ns = cat.namespace() or "-"
                choices = self._choices.get(k)
                if choices:
                    ch_str = f"[{', '.join(repr(c) for c in choices[:6])}{', ...' if len(choices) > 6 else ''}] ({len(choices)} total)"
                else:
                    ch_str = "-"
                lines.append(
                    f"| `{k}` | `{cat.name()}` | `{phys}` | `{ns}` | `{ch_str}` |"
                )
            lines.append("")

        content = "\n".join(lines).rstrip() + "\n"
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        return content

    def to_mermaid(
        self,
        path: str | Path | None = None,
        *,
        title: str | None = None,
    ) -> str:
        """Generates a Mermaid class diagram definition for this CatSpec registry."""
        lines: list[str] = [
            "classDiagram",
        ]
        if title:
            lines.insert(0, f"%% {title}")

        for k, variants in self._enums.items():
            clean_k = "".join(c if c.isalnum() or c == "_" else "_" for c in k)
            lines.append(f"    class {clean_k} {{")
            lines.append("        <<enumeration>>")
            for v in variants[:10]:
                clean_v = "".join(c if c.isalnum() or c == "_" else "_" for c in str(v))
                lines.append(f"        +{clean_v}")
            if len(variants) > 10:
                lines.append(f"        +... ({len(variants) - 10} more)")
            lines.append("    }")

        for k, cat in self._categoricals.items():
            clean_k = "".join(c if c.isalnum() or c == "_" else "_" for c in k)
            phys = _YAML_DTYPES.get(cat.physical(), str(cat.physical()))
            lines.append(f"    class {clean_k} {{")
            lines.append(f"        <<categorical: {phys}>>")
            ns = cat.namespace()
            if ns:
                lines.append(f"        +namespace: {ns}")
            choices = self._choices.get(k)
            if choices:
                for c in choices[:5]:
                    clean_c = "".join(
                        c if c.isalnum() or c == "_" else "_" for c in str(c)
                    )
                    lines.append(f"        +{clean_c}")
                if len(choices) > 5:
                    lines.append(f"        +... ({len(choices) - 5} more)")
            lines.append("    }")

        content = "\n".join(lines) + "\n"
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        return content

    def __repr__(self) -> str:
        enum_keys = list(self._enums.keys())
        cat_keys = list(self._categoricals.keys())
        return f"CatSpec(enums={enum_keys}, categoricals={cat_keys})"
