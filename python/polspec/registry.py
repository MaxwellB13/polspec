"""Several specs that belong together.

A `ForeignKey` names the spec it points at, and nothing about a single spec
knows which other specs exist. A `Registry` is that knowledge: a declared set
of `TableSpec`s that resolves every cross-spec key, orders the set so parents
come before children, generates or validates the whole set in one call, and
draws the relationships between them.

    registry = Registry(Customers, Orders, OrderLines)
    frames = registry.generate_all(1_000, seed=1)
    registry.validate_all(frames)

A registry is declared, not global: two notebooks or test modules may each
define an `Orders`, and neither should silently see the other's.
"""

from __future__ import annotations

import dataclasses
import difflib
import importlib.util
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from polspec import generation, validation
from polspec.catspec import CatSpec, _declared_categories_from
from polspec.engine import _stable_seed
from polspec.errors import RegistryError, SpecError, ValidationError
from polspec.tablespec import TableSpec, as_spec_name, as_table_spec, resolve_references

if TYPE_CHECKING:
    from polspec.validation import ValidationReport

__all__ = ["Registry"]

Frames = Mapping[Any, pl.DataFrame | pl.LazyFrame]


def _collect(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def _spec_of(value: Any) -> TableSpec | None:
    """The `TableSpec` a module-level value stands for, if it stands for one."""
    from polspec.framespec import FrameSpec

    if isinstance(value, TableSpec):
        return value
    if (
        isinstance(value, type)
        and issubclass(value, FrameSpec)
        and value is not FrameSpec
    ):
        return value.spec
    return None


def _load_module(path: Path) -> ModuleType:
    """Imports a Python file as a throwaway module."""
    module_name = f"_polspec_registry_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RegistryError(f"could not load {path} as a Python module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RegistryError(f"error importing {path}: {exc}") from exc
    return module


def _is_registry_file(path: Path) -> bool:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return isinstance(raw, Mapping) and "specs" in raw


class Registry:
    """A declared set of specs, with everything that needs more than one.

    Parameters
    ----------
    *specs : TableSpec | type[FrameSpec]
        The specs, in any order. Each is stored under its name; two different
        specs with one name are an error.
    categories : CatSpec | None
        A shared category registry the specs are expected to agree with.
        When given, `resolve()` checks every `Enum`/`Categorical` column that
        binds to one of its entries against it, and the registry file carries
        it. When omitted, `catspec()` derives one from the specs themselves.
    """

    def __init__(
        self, *specs: TableSpec | type, categories: CatSpec | None = None
    ) -> None:
        self._specs: dict[str, TableSpec] = {}
        self._categories = categories
        for spec in specs:
            self.add(spec)

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------

    def add(self, spec: TableSpec | type) -> Registry:
        """Adds a spec, returning the registry so calls chain."""
        table = as_table_spec(spec)
        prior = self._specs.get(table.name)
        if prior is not None and prior != table:
            raise RegistryError(
                f"Two different specs are both named {table.name!r}. A registry "
                "holds one spec per name; rename one with TableSpec.with_name()."
            )
        self._specs[table.name] = table
        return self

    @property
    def names(self) -> tuple[str, ...]:
        """Spec names in the order they were added."""
        return tuple(self._specs)

    @property
    def specs(self) -> tuple[TableSpec, ...]:
        return tuple(self._specs.values())

    @property
    def categories(self) -> CatSpec | None:
        """The shared category registry this one was declared with, if any."""
        return self._categories

    def __getitem__(self, key: Any) -> TableSpec:
        name = as_spec_name(key)
        try:
            return self._specs[name]
        except KeyError:
            close = difflib.get_close_matches(name, self._specs, n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            have = ", ".join(self._specs) or "it is empty"
            raise RegistryError(
                f"No spec named {name!r} in the registry{hint}; it holds: {have}"
            ) from None

    def __contains__(self, key: object) -> bool:
        try:
            return as_spec_name(key) in self._specs
        except TypeError:
            return False

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __repr__(self) -> str:
        return f"Registry({', '.join(self._specs)})"

    # ------------------------------------------------------------------
    # Building one from what already exists
    # ------------------------------------------------------------------

    @classmethod
    def from_module(
        cls,
        module: ModuleType,
        *,
        own_only: bool = False,
        categories: CatSpec | None = None,
    ) -> Registry:
        """Every `FrameSpec` subclass and `TableSpec` bound in a module.

        `own_only=True` keeps only classes the module itself defines, leaving
        out ones it imported.
        """
        registry = cls(categories=categories)
        for value in vars(module).values():
            table = _spec_of(value)
            if table is None:
                continue
            if (
                own_only
                and isinstance(value, type)
                and value.__module__ != module.__name__
            ):
                continue
            registry.add(table)
        return registry

    @classmethod
    def discover(
        cls,
        *paths: str | Path,
        categories: CatSpec | None = None,
        strict: bool = True,
    ) -> Registry:
        """Every spec found under the given files and directories.

        A `.py` file is imported and searched like `from_module`; a `.yaml`
        file is read as a spec, or as a registry file when it has a `specs:`
        key; a directory is walked for both. Importing a Python file runs
        it, so point this only at files you would import anyway.
        """
        from polspec import serialization

        registry = cls(categories=categories)
        for path in _spec_files(paths):
            if path.suffix == ".py":
                found = cls.from_module(_load_module(path))
            elif _is_registry_file(path):
                found = serialization.registry_from_yaml(path, strict=strict)
            else:
                found = cls(
                    serialization.from_yaml(path, categories=categories, strict=strict)
                )
            for spec in found.specs:
                registry.add(spec)
            if registry._categories is None and found._categories is not None:
                registry._categories = found._categories
        return registry

    # ------------------------------------------------------------------
    # Consistency
    # ------------------------------------------------------------------

    def resolve(self) -> Registry:
        """A registry whose every cross-spec key is bound to its target.

        Binding runs the checks a key declared against a class gets at
        declaration -- the referenced columns exist and are dtype-compatible
        -- for keys that were declared against a bare name or read from a
        file. Also refuses a key whose target is not in the registry, a cycle
        between specs, and a column disagreeing with `categories`.
        """
        resolved = Registry(categories=self._categories)
        for name, spec in self._specs.items():
            keys = []
            for fk in spec.foreign_keys:
                if fk.references == "self":
                    keys.append(fk)
                    continue
                target = self._specs.get(fk.references)
                if target is None:
                    raise RegistryError(
                        f"ForeignKey {fk.name!r} on {name!r} references "
                        f"{fk.references!r}, which is not in the registry "
                        f"({', '.join(self._specs)})"
                    )
                keys.append(dataclasses.replace(fk, target=target))
            try:
                resolved._specs[name] = dataclasses.replace(
                    spec, foreign_keys=tuple(keys)
                )
            except SpecError as exc:
                raise RegistryError(str(exc)) from exc
        resolved._check_categories()
        resolved.order()
        return resolved

    def _check_categories(self) -> None:
        cats = self._categories
        if cats is None:
            return
        for spec in self._specs.values():
            for col_name, cs in spec.columns.items():
                bound = cats.resolve_key(col_name)
                if bound is None:
                    continue
                kind, key = bound
                dtype = cs.dtype
                if kind == "enum" and isinstance(dtype, pl.Enum):
                    declared = dtype.categories.to_list()
                    expected = cats.get_enum(key)
                    if declared != expected:
                        raise RegistryError(
                            f"{spec.name}.{col_name} declares Enum {declared}, but "
                            f"the shared categories define {key!r} as {expected}"
                        )
                elif kind == "categorical" and isinstance(dtype, pl.Categorical):
                    declared_cats = _declared_categories_from(dtype)
                    expected_cats = cats.get_categorical(key)
                    if declared_cats is not None and (
                        declared_cats.name() != expected_cats.name()
                        or declared_cats.physical() != expected_cats.physical()
                    ):
                        raise RegistryError(
                            f"{spec.name}.{col_name} declares Categories "
                            f"{declared_cats.name()!r} ({declared_cats.physical()}), "
                            f"but the shared categories define {key!r} as "
                            f"{expected_cats.name()!r} ({expected_cats.physical()})"
                        )

    def parents(self, key: Any) -> tuple[str, ...]:
        """Names of the specs one spec's foreign keys point at, self excluded."""
        spec = self[key]
        seen: dict[str, None] = {}
        for fk in spec.foreign_keys:
            if fk.references != "self":
                seen.setdefault(fk.references, None)
        return tuple(seen)

    def order(self) -> tuple[str, ...]:
        """Every spec name, parents before children.

        Specs with no dependency between them keep the order they were
        added in. A target outside the registry imposes no order; a cycle
        is an error.
        """
        pending = list(self._specs)
        done: set[str] = set()
        ordered: list[str] = []
        while pending:
            ready = [
                name
                for name in pending
                if all(p in done or p not in self._specs for p in self.parents(name))
            ]
            if not ready:
                raise RegistryError(
                    "Foreign keys form a cycle between specs: "
                    + ", ".join(pending)
                    + ". A registry can only order specs whose references form "
                    "a tree; make one side of the cycle a self-reference or "
                    "drop it."
                )
            ordered.extend(ready)
            done.update(ready)
            pending = [name for name in pending if name not in done]
        return tuple(ordered)

    def ancestors(self, key: Any) -> tuple[str, ...]:
        """Every spec a spec depends on, directly or through other specs."""
        root = as_spec_name(key)
        seen: dict[str, None] = {}
        stack = list(self.parents(root))
        while stack:
            name = stack.pop()
            if name in seen or name not in self._specs:
                continue
            seen[name] = None
            stack.extend(self.parents(name))
        return tuple(name for name in self.order() if name in seen)

    # ------------------------------------------------------------------
    # Generating
    # ------------------------------------------------------------------

    def _counts(
        self, n: int | Mapping[Any, int], names: tuple[str, ...]
    ) -> dict[str, int]:
        if isinstance(n, int):
            return dict.fromkeys(names, n)
        counts = {as_spec_name(k): v for k, v in n.items()}
        unknown = [k for k in counts if k not in self._specs]
        if unknown:
            raise RegistryError(
                f"Row counts given for specs not in the registry: {unknown}"
            )
        missing = [name for name in names if name not in counts]
        if missing:
            raise RegistryError(
                f"No row count given for {missing}; pass an int for every spec, "
                "or a mapping with an entry for each"
            )
        return {name: counts[name] for name in names}

    def _generate(
        self,
        names: tuple[str, ...],
        n: int | Mapping[Any, int],
        *,
        seed: int | None,
        method: Literal["random", "cartesian"],
        references: Frames | None,
    ) -> dict[str, pl.DataFrame]:
        counts = self._counts(n, names)
        supplied = resolve_references(references, _collect)
        for name in names:
            for parent in self.parents(name):
                if parent not in self._specs and parent not in supplied:
                    raise RegistryError(
                        f"{name!r} references {parent!r}, which is neither in the "
                        "registry nor supplied via references="
                    )
        frames: dict[str, pl.DataFrame] = {}
        for name in self.order():
            if name not in names:
                continue
            if name in supplied:
                frames[name] = supplied[name]
                continue
            spec_seed = None if seed is None else _stable_seed(str(seed), name)
            frames[name] = generation.generate(
                self._specs[name],
                counts[name],
                method=method,
                seed=spec_seed,
                references={**supplied, **frames},
            )
        return frames

    def generate_all(
        self,
        n: int | Mapping[Any, int],
        *,
        seed: int | None = None,
        method: Literal["random", "cartesian"] = "random",
        references: Frames | None = None,
    ) -> dict[str, pl.DataFrame]:
        """One frame per spec, parents generated first and threaded into
        their children, so every foreign key is satisfied by construction.

        `n` is a row count for every spec, or a mapping from spec (or name)
        to its own count. Each spec's seed is derived from `seed` and its
        name, so adding a spec to the registry never changes the rows another
        one produces. A frame in `references` is used as-is in place of
        generating that spec, and also serves parents outside the registry.
        """
        return self._generate(
            self.names, n, seed=seed, method=method, references=references
        )

    def generate_related(
        self,
        key: Any,
        n: int | Mapping[Any, int],
        *,
        seed: int | None = None,
        method: Literal["random", "cartesian"] = "random",
        references: Frames | None = None,
    ) -> dict[str, pl.DataFrame]:
        """`generate_all` restricted to one spec and everything it depends on."""
        root = as_spec_name(key)
        self[root]
        needed = tuple(
            name
            for name in self.order()
            if name == root or name in self.ancestors(root)
        )
        return self._generate(
            needed, n, seed=seed, method=method, references=references
        )

    # ------------------------------------------------------------------
    # Validating
    # ------------------------------------------------------------------

    def _named(self, frames: Frames) -> dict[str, pl.DataFrame | pl.LazyFrame]:
        named = {as_spec_name(k): v for k, v in frames.items()}
        for name in named:
            self[name]
        return named

    def inspect_all(
        self, frames: Frames, *, references: Frames | None = None, **options: Any
    ) -> dict[str, ValidationReport]:
        """A `ValidationReport` per frame, each spec seeing every other frame
        as a possible parent. Takes the options `validate()` does.
        """
        named = self._named(frames)
        parents = {**resolve_references(references, lambda f: f), **named}
        return {
            name: validation.inspect(
                self._specs[name], named[name], references=parents, **options
            )
            for name in self.order()
            if name in named
        }

    def validate_all(
        self, frames: Frames, *, references: Frames | None = None, **options: Any
    ) -> dict[str, pl.DataFrame | pl.LazyFrame]:
        """Validates every frame, raising one `ValidationError` that lists
        every spec's findings, or returning the frames with the structural
        transformations `validate()` applies.
        """
        named = self._named(frames)
        reports = self.inspect_all(named, references=references, **options)
        failed = [report for report in reports.values() if not report.passed]
        if failed:
            raise ValidationError(
                "\n\n".join(str(report) for report in failed),
                errors=[f.message for report in failed for f in report.findings],
            )
        return {
            name: validation._transformed(self._specs[name], named[name], report)
            for name, report in reports.items()
        }

    # ------------------------------------------------------------------
    # Categories, files, diagrams
    # ------------------------------------------------------------------

    def catspec(self) -> CatSpec:
        """The categories these specs share: the one declared, or one merged
        from every spec's Enum and Categorical columns.

        Merging refuses two specs that define the same name differently;
        pass `categories=` to the registry to settle which is right.
        """
        if self._categories is not None:
            return self._categories
        enums: dict[str, list[str]] = {}
        categoricals: dict[str, pl.Categories] = {}
        choices: dict[str, list[Any]] = {}
        origin: dict[str, str] = {}

        def merge(
            store: dict[str, Any], key: str, value: Any, spec: str, same: Any
        ) -> None:
            if key in store and not same(store[key], value):
                raise RegistryError(
                    f"{spec} and {origin[key]} disagree about {key!r} "
                    f"({value!r} vs {store[key]!r}). Pass categories= to the "
                    "Registry to say which is right."
                )
            store.setdefault(key, value)
            origin.setdefault(key, spec)

        def same_categories(a: pl.Categories, b: pl.Categories) -> bool:
            return (a.name(), a.namespace(), a.physical()) == (
                b.name(),
                b.namespace(),
                b.physical(),
            )

        for spec in self._specs.values():
            part = CatSpec.from_framespec(spec)
            for key, value in part.enums.items():
                merge(enums, key, value, spec.name, lambda a, b: a == b)
            for key, cats in part.categoricals.items():
                merge(categoricals, key, cats, spec.name, same_categories)
            for key, value in part.choices.items():
                merge(choices, key, value, spec.name, lambda a, b: a == b)
        return CatSpec(enums=enums, categoricals=categoricals, choices=choices)

    def to_dict(self) -> dict[str, Any]:
        from polspec import serialization

        return serialization.registry_to_dict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, strict: bool = True) -> Registry:
        from polspec import serialization

        return serialization.registry_from_dict(data, strict=strict)

    def to_yaml(self, source: str | Path) -> None:
        """Writes every spec, and the declared categories, to one file."""
        from polspec import serialization

        serialization.registry_to_yaml(self, source)

    @classmethod
    def from_yaml(cls, source: str | Path, *, strict: bool = True) -> Registry:
        from polspec import serialization

        return serialization.registry_from_yaml(source, strict=strict)

    def to_mermaid(self, path: str | Path | None = None) -> str:
        """One entity-relationship diagram with every spec and every key."""
        from polspec.report import registry_to_mermaid

        return registry_to_mermaid(self.specs, path)


def _spec_files(paths: tuple[str | Path, ...]) -> list[Path]:
    files: list[Path] = []
    for given in paths:
        path = Path(given)
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.suffix.lower() not in (".py", ".yaml", ".yml"):
                    continue
                if any(
                    part.startswith((".", "__"))
                    for part in candidate.relative_to(path).parts
                ):
                    continue
                if candidate.name.startswith(("_", "test_")):
                    continue
                files.append(candidate)
        elif path.is_file():
            if path.suffix.lower() not in (".py", ".yaml", ".yml"):
                raise RegistryError(
                    f"don't know how to read specs from {path.suffix!r} files ({path})"
                )
            files.append(path)
        else:
            raise RegistryError(f"no such file or directory: {path}")
    return files
