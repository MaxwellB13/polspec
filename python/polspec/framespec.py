from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar, Literal, Sequence, overload

import polars as pl
import yaml

from polspec.catspec import CatSpec
from polspec.engine import _generate_cartesian, _generate_random
from polspec.profiler import profile_dataframe
from polspec.rules import _apply_rules
from polspec.serialization import _colspec_from_yaml, _colspec_to_yaml
from polspec.spec import ColSpec
from polspec.validation import _validate_dataframe


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
                enum_dt = pl.Enum(catspec.get_enum(enum_key))
                new_columns[col_name] = ColSpec(
                    dtype=enum_dt,
                    nullable=spec.nullable,
                    tags=spec.tags,
                    null_probability=spec.null_probability,
                    weights=spec.weights,
                    rules=spec.rules,
                )
            elif cat_key is not None:
                cat_dt = pl.Categorical(catspec.get_categorical(cat_key))
                choices = catspec.get_choices(cat_key) or spec.choices
                new_columns[col_name] = ColSpec(
                    dtype=cat_dt,
                    nullable=spec.nullable,
                    tags=spec.tags,
                    null_probability=spec.null_probability,
                    choices=choices,
                    weights=spec.weights,
                    rules=spec.rules,
                )
            else:
                new_columns[col_name] = spec

        subclass_name = name or f"{cls.__name__}WithCatSpec"
        return type(subclass_name, (FrameSpec,), new_columns)

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
        data = {
            "name": cls.__name__,
            "columns": {
                name: _colspec_to_yaml(spec) for name, spec in cls._columns.items()
            },
        }
        Path(source).write_text(yaml.safe_dump(data, sort_keys=False))

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
        data = yaml.safe_load(source_path.read_text())

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
        return type(data.get("name", "LoadedFrameSpec"), (FrameSpec,), columns)

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
        return type(name, (FrameSpec,), columns)

    @overload
    @classmethod
    def generate(
        cls,
        n: int,
        *,
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
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
        lazy: Literal[True],
    ) -> pl.LazyFrame: ...

    @classmethod
    def generate(
        cls,
        n: int,
        *,
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
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

        Any ColSpec.rules are applied last, as a vectorized overwrite pass
        over the fully-generated DataFrame (see ColRule), regardless of
        method.

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
        return res.lazy() if lazy else res

    @classmethod
    def generate_batches(
        cls,
        n: int,
        *,
        batch_size: int = 100_000,
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
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
                min(n, batch_size), method="cartesian", seed=rng.randrange(2**63)
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
                    current_batch_size, method="random", seed=rng.randrange(2**63)
                )
                rows_remaining -= current_batch_size
        elif method == "random":
            rows_remaining = n
            while rows_remaining > 0:
                current_batch_size = min(rows_remaining, batch_size)
                yield cls.generate(
                    current_batch_size, method="random", seed=rng.randrange(2**63)
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
                n, batch_size=batch_size, method=method, seed=seed
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
                empty_df = cls.generate(0)
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
                empty_df = cls.generate(0)
                if include_header:
                    empty_df.write_csv(f, include_header=True, **kwargs)
                return

            for batch_df in cls.generate_batches(
                n, batch_size=batch_size, method=method, seed=seed
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
                    n, batch_size=batch_size, method=method, seed=seed
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
                    empty_df = cls.generate(0)
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
                n, batch_size=batch_size, method=method, seed=seed
            ):
                batch_df.write_ndjson(f, **kwargs)

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
            If invalid options are supplied for `extra_cols` or `missing_cols`.
        """
        if not cls._columns:
            raise ValueError(f"{cls.__name__} declares no ColSpec columns")
        return _validate_dataframe(
            cls._columns,
            cls.__name__,
            df,
            extra_cols=extra_cols,
            missing_cols=missing_cols,
            strict_dtypes=strict_dtypes,
            validate_rules=validate_rules,
            cast=cast,
            streaming=streaming,
        )


FrameSchema = FrameSpec
