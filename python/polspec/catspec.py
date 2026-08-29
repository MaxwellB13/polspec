from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence
import yaml
import polars as pl

from polspec.serialization import _YAML_DTYPES, _YAML_NAME_TO_DTYPE

if TYPE_CHECKING:
    pass


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

    __slots__ = ("_enums", "_categoricals", "_choices", "_enum_accessor", "_cat_accessor")

    def __init__(
        self,
        *,
        enums: Mapping[str, Sequence[str]] | None = None,
        categoricals: Mapping[str, pl.Categories | dict[str, Any] | str | pl.DataType] | None = None,
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

    def get_enum(self, name: str) -> list[str]:
        """Returns the list of categories/variants for an enum."""
        if name in self._enums:
            return list(self._enums[name])
        raise KeyError(f"No enum named {name!r} in CatSpec")

    def get_categorical(self, name: str) -> pl.Categories:
        """Returns the pl.Categories definition for a categorical."""
        if name in self._categoricals:
            return self._categoricals[name]
        raise KeyError(f"No categorical named {name!r} in CatSpec")

    def get_choices(self, name: str) -> list[Any] | None:
        """Returns domain choices if defined for this enum or categorical."""
        if name in self._choices:
            return list(self._choices[name])
        if name in self._enums:
            return list(self._enums[name])
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._categoricals:
            return self._categoricals[name]
        if name in self._enums:
            return list(self._enums[name])
        raise AttributeError(f"CatSpec has no Enum or Categorical named {name!r}")

    def __getitem__(self, name: str) -> Any:
        if name in self._categoricals:
            return self._categoricals[name]
        if name in self._enums:
            return list(self._enums[name])
        raise KeyError(f"CatSpec has no Enum or Categorical named {name!r}")

    def __contains__(self, name: object) -> bool:
        return name in self._categoricals or name in self._enums

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
        return default

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CatSpec:
        """Constructs a CatSpec from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for CatSpec data, got {type(data).__name__}")

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
        raw = yaml.safe_load(path.read_text())
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
            Path(source).write_text(dumped)
            return None
        return dumped

    def __repr__(self) -> str:
        enum_keys = list(self._enums.keys())
        cat_keys = list(self._categoricals.keys())
        return f"CatSpec(enums={enum_keys}, categoricals={cat_keys})"
