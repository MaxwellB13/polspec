"""`FrameSpec` -- declaring a `TableSpec` as a class body, and the facade over it.

    class Orders(FrameSpec):
        order_id = ColSpec(pl.Int64, unique=True)
        status = ColSpec(pl.Enum(["NEW", "PAID"]))
        __checks__ = [Check(...)]

    Orders.spec            # the TableSpec the class body built
    Orders.generate(1_000) # every verb forwards to a function over `Orders.spec`

The metaclass takes the `ColSpec` attributes and the dunder declarations out
of the class namespace *before* the class exists, so a column can be named
`schema` or `tag` without shadowing the method of the same name. Column
attributes are still reachable as `Orders.order_id`, through a fallback that
only runs when ordinary attribute lookup fails; `Orders.col("schema")` and
`Orders.spec["schema"]` reach a column whatever it is called.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, overload

import polars as pl

from polspec import generation, serialization, validation
from polspec.catspec import CatSpec
from polspec.check import Check
from polspec.errors import SpecError
from polspec.foreign_key import ForeignKey
from polspec.profiler import profile_dataframe
from polspec.report import framespec_to_markdown, framespec_to_mermaid
from polspec.spec import ColSpec
from polspec.tablespec import TableSpec, _parse_unique_together

References = Mapping[Any, pl.DataFrame | pl.LazyFrame] | None


def _is_declaration_value(value: Any) -> bool:
    """Whether a class-body value under a declaration name is data, not code.

    `checks`, `foreign_keys` and `unique_together` double as accessor names,
    so a method or descriptor under one of them is left alone.
    """
    return not (
        value is None
        or callable(value)
        or isinstance(value, (classmethod, staticmethod, property))
    )


def _pop_declaration[T](
    ns: dict[str, Any], names: tuple[str, ...], kind: type[T]
) -> list[T]:
    """Removes `__checks__`/`checks`-style declarations from a namespace.

    Accepts a single instance or a sequence of them, under either spelling.
    """
    out: list[T] = []
    for attr in names:
        if attr not in ns or not _is_declaration_value(ns[attr]):
            continue
        value = ns.pop(attr)
        if isinstance(value, kind):
            items: Sequence[T] = [value]
        elif isinstance(value, Sequence) and not isinstance(value, str):
            items = value
        else:
            raise SpecError(
                f"{names[0]} must be a {kind.__name__} or sequence of "
                f"{kind.__name__} instances, got {type(value).__name__}"
            )
        for item in items:
            if not isinstance(item, kind):
                raise SpecError(
                    f"Items in {names[0]} must be {kind.__name__} instances, got "
                    f"{type(item).__name__}"
                )
            if item not in out:
                out.append(item)
    return out


def _pop_unique_together(ns: dict[str, Any]) -> Any:
    for attr in ("__unique_together__", "unique_together"):
        if attr in ns and _is_declaration_value(ns[attr]):
            return ns.pop(attr)
    return None


def _build_table_spec(
    name: str,
    parents: Sequence[_FrameSpecMeta],
    own_columns: dict[str, ColSpec],
    declared: Mapping[Any, Any] | None,
    removed: set[str],
    checks: Sequence[Check],
    foreign_keys: Sequence[ForeignKey],
    unique_together: Any,
) -> tuple[TableSpec, dict[str, str]]:
    """Merges inherited specs with a class body's own declarations.

    Returns the spec and the attribute-name to column-name mapping that a
    subclass needs in order to override or remove a column by attribute.
    """
    columns: dict[str, ColSpec] = {}
    attr_to_col: dict[str, str] = {}
    all_checks: list[Check] = []
    all_fks: list[ForeignKey] = []
    all_unique: list[tuple[str, ...]] = []

    for parent in reversed(parents):
        columns.update(parent.spec.columns)
        attr_to_col.update(parent._attr_to_col)
        for check in parent.spec.checks:
            if check not in all_checks:
                all_checks.append(check)
        for fk in parent.spec.foreign_keys:
            if fk not in all_fks:
                all_fks.append(fk)
        for group in parent.spec.unique_together:
            if tuple(group) not in all_unique:
                all_unique.append(tuple(group))

    for attr in removed:
        columns.pop(attr_to_col.pop(attr), None)

    for attr, colspec in own_columns.items():
        final = colspec.col_name if colspec.col_name is not None else attr
        colliding = next(
            (a for a, c in attr_to_col.items() if c == final and a != attr), None
        )
        if colliding is not None:
            raise SpecError(
                f"Columns {colliding!r} and {attr!r} on {name} both resolve to "
                f"the column name {final!r} (via col_name). Give each a distinct "
                "col_name, or rename one of the attributes."
            )
        prior = attr_to_col.get(attr)
        if prior is not None and prior != final:
            del columns[prior]
        columns[final] = colspec
        attr_to_col[attr] = final

    if declared is not None:
        if not isinstance(declared, Mapping):
            raise SpecError(
                "__columns__ must be a mapping of column name to ColSpec, got "
                f"{type(declared).__name__}"
            )
        for raw_key, value in declared.items():
            key = str(raw_key)
            if not isinstance(value, ColSpec):
                raise SpecError(
                    f"__columns__[{key!r}] must be a ColSpec, got {type(value).__name__}"
                )
            if value.col_name is not None and value.col_name != key:
                raise SpecError(
                    f"__columns__[{key!r}] declares col_name={value.col_name!r}, "
                    "which conflicts with its __columns__ key. col_name only "
                    "applies to columns declared as class attributes -- for "
                    "__columns__, the dict key already is the column name."
                )
            columns[key] = value

    for check in checks:
        if check not in all_checks:
            all_checks.append(check)
    for fk in foreign_keys:
        if fk not in all_fks:
            all_fks.append(fk)
    for group in _parse_unique_together(unique_together):
        if group not in all_unique:
            all_unique.append(group)

    spec = TableSpec(
        name,
        columns,
        checks=all_checks,
        unique_together=all_unique,
        foreign_keys=all_fks,
    )
    return spec, attr_to_col


class _FrameSpecMeta(type):
    """Builds `cls.spec` from a class body, keeping columns off the class itself."""

    spec: TableSpec
    _attr_to_col: dict[str, str]

    def __new__(
        mcls, name: str, bases: tuple[type, ...], ns: dict[str, Any], **kwargs: Any
    ) -> _FrameSpecMeta:
        prebuilt: TableSpec | None = ns.pop("__tablespec__", None)
        parents = [b for b in bases if isinstance(b, _FrameSpecMeta)]

        own_columns: dict[str, ColSpec] = {}
        for attr in list(ns):
            if not attr.startswith("_") and isinstance(ns[attr], ColSpec):
                own_columns[attr] = ns.pop(attr)
        declared = ns.pop("__columns__", None)
        checks = _pop_declaration(ns, ("__checks__", "checks"), Check)
        foreign_keys = _pop_declaration(
            ns, ("__foreign_keys__", "foreign_keys"), ForeignKey
        )
        unique_together = _pop_unique_together(ns)
        # A non-ColSpec value assigned over an inherited column attribute
        # removes that column; the value itself stays an ordinary attribute.
        removed = {
            attr
            for attr in ns
            if not attr.startswith("_") and any(attr in p._attr_to_col for p in parents)
        }

        cls = super().__new__(mcls, name, bases, ns, **kwargs)
        if prebuilt is not None:
            cls.spec = prebuilt if prebuilt.name == name else prebuilt.with_name(name)
            cls._attr_to_col = {}
        else:
            cls.spec, cls._attr_to_col = _build_table_spec(
                name,
                parents,
                own_columns,
                declared,
                removed,
                checks,
                foreign_keys,
                unique_together,
            )
        return cls

    def __getattr__(cls, item: str) -> Any:
        # Reached only when ordinary lookup fails, so a method always wins
        # over a column of the same name.
        if not item.startswith("__"):
            spec = cls.__dict__.get("spec")
            if spec is not None:
                col = spec.columns.get(item)
                if col is not None:
                    return col
        raise AttributeError(f"type object {cls.__name__!r} has no attribute {item!r}")


class FrameSpec(metaclass=_FrameSpecMeta):
    """Base class for declaring a DataFrame/LazyFrame specification.

    Subclass it and assign a `ColSpec` per column, in the order columns should
    appear:

        class DataSource(FrameSpec):
            string_1 = ColSpec(pl.String)
            enum_1 = ColSpec(pl.Enum(["mammal", "reptile"]), nullable=True)
            int_1 = ColSpec(pl.Int64, bounds=(-100, 100), nullable=True)

        df = DataSource.generate(1_000_000, seed=42)

    The class body builds `DataSource.spec`, a `TableSpec`; every classmethod
    here forwards to a function over it. A column may take any name: one that
    collides with a method (`schema`, `tag`, ...) is reachable as
    `DataSource.col("schema")` while the method keeps working. Names that
    cannot be attributes at all -- a leading underscore, or one straight from
    data -- go through `__columns__`:

        class Raw(FrameSpec):
            __columns__ = {"_id": ColSpec(pl.Int64), "Unit Price": ColSpec(pl.Float64)}
    """

    spec: ClassVar[TableSpec]
    _attr_to_col: ClassVar[dict[str, str]]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_spec(cls, spec: TableSpec, *, name: str | None = None) -> type[FrameSpec]:
        """A `FrameSpec` subclass wrapping an existing `TableSpec`."""
        return _FrameSpecMeta(name or spec.name, (FrameSpec,), {"__tablespec__": spec})

    @classmethod
    def from_yaml(
        cls,
        source: str | Path,
        *,
        categories: CatSpec | str | Path | None = None,
        strict: bool = True,
    ) -> type[FrameSpec]:
        """Builds a new FrameSpec subclass from a YAML file written by `to_yaml`.

        `categories` is a CatSpec registry, or a path to one, used to resolve
        shared Enums and Categoricals; when omitted, a `categories:` key in
        the file is loaded automatically. An unknown key in the file is an
        error unless `strict=False`, which downgrades it to a warning.
        """
        return cls.from_spec(
            serialization.from_yaml(source, categories=categories, strict=strict)
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        *,
        name: str = "ProfiledFrameSpec",
        weights: bool = False,
        max_unique_enum: int = 50,
        calculate_bounds: bool = True,
    ) -> type[FrameSpec]:
        """Infers a spec by profiling an existing DataFrame.

        `weights=True` records empirical frequencies for categorical, enum and
        boolean columns. A string or categorical column with at most
        `max_unique_enum` distinct values becomes an `Enum`. `calculate_bounds`
        records observed `(min, max)` for numeric and temporal columns and
        `(min_len, max_len)` for strings and binary.
        """
        columns = profile_dataframe(
            df,
            weights=weights,
            max_unique_enum=max_unique_enum,
            calculate_bounds=calculate_bounds,
        )
        return cls.from_spec(TableSpec(name, columns))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @classmethod
    def col(cls, name: str) -> ColSpec:
        """The declaration of one column, whatever it is called."""
        return cls.spec[name]

    @classmethod
    def schema(cls) -> pl.Schema:
        return cls.spec.schema()

    @classmethod
    def tag(
        cls,
        *tags: str | Sequence[str],
        match: Literal["any", "all"] = "any",
    ) -> list[str]:
        """Column names carrying any (or all) of the tags, in declaration order."""
        return cls.spec.tag(*tags, match=match)

    @classmethod
    def checks(cls) -> tuple[Check, ...]:
        """The Check constraints defined on this FrameSpec."""
        return tuple(cls.spec.checks)

    @classmethod
    def foreign_keys(cls) -> tuple[ForeignKey, ...]:
        """The ForeignKey constraints defined on this FrameSpec."""
        return tuple(cls.spec.foreign_keys)

    @classmethod
    def unique_together(cls) -> tuple[tuple[str, ...], ...]:
        """The composite unique column groups defined on this FrameSpec."""
        return tuple(tuple(g) for g in cls.spec.unique_together)

    # ------------------------------------------------------------------
    # Shared categories
    # ------------------------------------------------------------------

    @classmethod
    def catspec(cls) -> CatSpec:
        """The CatSpec registry this spec's Enum and Categorical columns imply."""
        return CatSpec.from_framespec(cls.spec)

    @classmethod
    def with_catspec(
        cls,
        catspec: CatSpec,
        *,
        name: str | None = None,
    ) -> type[FrameSpec]:
        """A new FrameSpec subclass with columns re-typed against `catspec`."""
        return cls.from_spec(
            cls.spec.with_catspec(catspec), name=name or f"{cls.__name__}WithCatSpec"
        )

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    @classmethod
    def to_yaml(cls, source: str | Path) -> None:
        """Writes this spec to a human-readable YAML file at `source`."""
        serialization.to_yaml(cls.spec, source)

    @classmethod
    def to_python(cls, source: str | Path) -> None:
        """Writes this spec as an importable Python module defining a subclass."""
        serialization.to_python(cls.spec, source)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @overload
    @classmethod
    def generate(
        cls,
        n: int,
        *,
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
        references: References = None,
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
        references: References = None,
        lazy: Literal[True],
    ) -> pl.LazyFrame: ...

    @classmethod
    def generate(
        cls,
        n: int,
        *,
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
        references: References = None,
        lazy: bool = False,
    ) -> pl.DataFrame | pl.LazyFrame:
        """Generates a DataFrame (or LazyFrame) matching this spec.

        See `polspec.generation.generate` for the full contract.
        """
        return generation.generate(
            cls.spec, n, method=method, seed=seed, references=references, lazy=lazy
        )

    @classmethod
    def generate_batches(
        cls,
        n: int,
        *,
        batch_size: int = 100_000,
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
        references: References = None,
    ) -> Iterator[pl.DataFrame]:
        """Yields chunks of generated rows without holding all `n` in memory."""
        return generation.generate_batches(
            cls.spec,
            n,
            batch_size=batch_size,
            method=method,
            seed=seed,
            references=references,
        )

    @classmethod
    def sink_parquet(cls, path: str | Path, n: int, **kwargs: Any) -> None:
        """Generates `n` rows and streams them to a Parquet file in batches."""
        generation.sink_parquet(cls.spec, path, n, **kwargs)

    @classmethod
    def sink_csv(cls, path: str | Path, n: int, **kwargs: Any) -> None:
        """Generates `n` rows and streams them to a CSV file in batches."""
        generation.sink_csv(cls.spec, path, n, **kwargs)

    @classmethod
    def sink_ipc(cls, path: str | Path, n: int, **kwargs: Any) -> None:
        """Generates `n` rows and streams them to an Arrow IPC file in batches."""
        generation.sink_ipc(cls.spec, path, n, **kwargs)

    @classmethod
    def sink_ndjson(cls, path: str | Path, n: int, **kwargs: Any) -> None:
        """Generates `n` rows and streams them to an NDJSON file in batches."""
        generation.sink_ndjson(cls.spec, path, n, **kwargs)

    # ------------------------------------------------------------------
    # Documentation
    # ------------------------------------------------------------------

    @classmethod
    def to_markdown(
        cls, path: str | Path | None = None, *, title: str | None = None
    ) -> str:
        """A Markdown data dictionary for this spec, written to `path` if given."""
        return framespec_to_markdown(cls.spec, path, title=title)

    @classmethod
    def to_mermaid(
        cls, path: str | Path | None = None, *, title: str | None = None
    ) -> str:
        """A Mermaid entity-relationship diagram for this spec."""
        return framespec_to_mermaid(cls.spec, path, title=title)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @classmethod
    def inspect(
        cls, df: pl.DataFrame | pl.LazyFrame, **options: Any
    ) -> validation.ValidationReport:
        """Everything this spec has to say about `df`, as a `ValidationReport`.

        Never raises for a frame that fails: each violation is a `Finding`
        with a code, a count, samples and, for row-level findings, the
        offending rows reachable lazily through `report.rows(finding)` or
        `report.failing_rows()`. Takes the same options as `validate`.
        """
        return validation.inspect(cls.spec, df, **options)

    @overload
    @classmethod
    def validate(cls, df: pl.DataFrame, **options: Any) -> pl.DataFrame: ...

    @overload
    @classmethod
    def validate(cls, df: pl.LazyFrame, **options: Any) -> pl.LazyFrame: ...

    @classmethod
    def validate(
        cls, df: pl.DataFrame | pl.LazyFrame, **options: Any
    ) -> pl.DataFrame | pl.LazyFrame:
        """Validates a DataFrame or LazyFrame against this spec.

        Raises `ValidationError` carrying a `ValidationReport` of every
        violation, or returns the (optionally transformed) frame. Use
        `inspect` for the report without the exception. See
        `polspec.validation.validate` for
        every option: `extra_cols`, `missing_cols`, `strict_dtypes`,
        `validate_rules`, `validate_validators`, `validate_unique`,
        `validate_checks`, `validate_foreign_keys`, `references`, `cast`,
        `streaming`.
        """
        return validation.validate(cls.spec, df, **options)
