from __future__ import annotations

import random
from pathlib import Path
from typing import ClassVar, Literal

import polars as pl
import yaml

from polspec.engine import _generate_cartesian, _generate_random
from polspec.profiler import profile_dataframe
from polspec.rules import _apply_rules
from polspec.serialization import _colspec_from_yaml, _colspec_to_yaml
from polspec.spec import ColSpec


class DfSpec:
    """Base class for declaring a DataFrame specification.

    Subclass it and assign `ColSpec` instances as class attributes, in the
    order columns should appear:

        class DataSource(DfSpec):
            string_1 = ColSpec(dtype=pl.String, nullable=False)
            enum_1 = ColSpec(dtype=pl.Enum(["mammal", "reptile", "insect"]), nullable=True)
            int_1 = ColSpec(dtype=pl.Int64, bounds=Bound(-100, 100), nullable=True)

        df = DataSource.generate(1_000_000, seed=42)
        df = DataSource.generate(n=1_000_000, method="cartesian", seed=42)
    """

    _columns: ClassVar[dict[str, ColSpec]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        columns: dict[str, ColSpec] = {}
        for base in reversed(cls.__mro__):
            for name, value in vars(base).items():
                if name.startswith("_"):
                    continue
                if isinstance(value, ColSpec):
                    columns[name] = value
                elif name in columns:
                    del columns[name]
        cls._columns = columns
        cls._validate_rules()

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
    def schema(cls) -> pl.Schema:
        return pl.Schema({name: spec.dtype for name, spec in cls._columns.items()})

    @classmethod
    def to_yaml(cls, source: str | Path) -> None:
        """Writes this spec's columns to a human-readable YAML file at `source`."""
        if not cls._columns:
            raise ValueError(f"{cls.__name__} declares no ColSpec columns")
        data = {
            "name": cls.__name__,
            "columns": {
                name: _colspec_to_yaml(spec) for name, spec in cls._columns.items()
            },
        }
        Path(source).write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def from_yaml(cls, source: str | Path) -> type[DfSpec]:
        """Builds a new DfSpec subclass from a YAML file written by `to_yaml`.

        DataSource = DfSpec.from_yaml(source="spec.yaml")
        df = DataSource.generate(1_000, seed=42)
        """
        data = yaml.safe_load(Path(source).read_text())
        columns_data = data.get("columns") or {}
        if not columns_data:
            raise ValueError(f"{source} declares no columns")
        columns = {
            name: _colspec_from_yaml(col_data)
            for name, col_data in columns_data.items()
        }
        return type(data.get("name", "LoadedDfSpec"), (DfSpec,), columns)

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        *,
        name: str = "ProfiledDfSpec",
        weights: bool = False,
        max_unique_enum: int = 50,
        max_unique: int | None = None,
        calculate_bounds: bool = True,
        bounds: bool | None = None,
    ) -> type[DfSpec]:
        """Infers and builds a new DfSpec subclass by profiling an existing DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            The DataFrame to profile.
        name : str, default "ProfiledDfSpec"
            The name of the generated DfSpec subclass.
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
        return type(name, (DfSpec,), columns)

    @classmethod
    def generate(
        cls,
        n: int,
        *,
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
    ) -> pl.DataFrame:
        """Generates a DataFrame matching this spec.

        method="random" (default): `n` rows, each column drawn independently.

        method="cartesian": guarantees a minimum level of coverage. Builds
        the cartesian product of every Enum/Boolean column's full set of
        values, crossed with the negative/zero/positive/null partitions of
        every bounded numeric column -- so every enum combination appears
        alongside every numeric sign/null case. `n` is then a *minimum*: if
        that coverage set has fewer than `n` rows, it's padded with ordinary
        random rows up to `n`; if it already has more, all of it is kept.

        Any ColSpec.rules are applied last, as a vectorized overwrite pass
        over the fully-generated DataFrame (see ColRule), regardless of
        method.
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

        return _apply_rules(df, cls._columns, rng.randrange(2**63))
