"""What every verb shares: reading data files, loading spec files, writing
generated Python.

One reader and one writer per suffix, so what `generate` writes, `validate`
and `drift` read back; a suffix in one map and not the other is a red test.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import subprocess
import sys
import types
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl

from polspec import FrameSpec
from polspec.errors import CliError
from polspec.foreign_key import ForeignKey
from polspec.frames import json_casts, sample_rows, text_format_casts
from polspec.profiler import STAYS_TEXT_NOTE, TEXT_NOTE
from polspec.reading import SUFFIXES, parse_declared_text, read_file
from polspec.registry import Registry, load_module
from polspec.serialization import registry_to_python

if TYPE_CHECKING:
    from polspec.tablespec import TableSpec

# Written through `json_casts`: categories as text, no bytes.
_JSON_SUFFIXES = frozenset({".ndjson", ".jsonl", ".json"})

_DATA_WRITERS = {
    ".csv": pl.DataFrame.write_csv,
    ".tsv": lambda df, p: df.write_csv(p, separator="\t"),
    ".parquet": pl.DataFrame.write_parquet,
    ".pq": pl.DataFrame.write_parquet,
    ".ndjson": pl.DataFrame.write_ndjson,
    ".jsonl": pl.DataFrame.write_ndjson,
    ".json": pl.DataFrame.write_json,
    ".arrow": pl.DataFrame.write_ipc,
    ".ipc": pl.DataFrame.write_ipc,
    ".feather": pl.DataFrame.write_ipc,
}


def _existing(path_text: str, *, what: str = "file") -> Path:
    path = Path(path_text)
    if not path.exists():
        raise CliError(f"no such {what}: {path}")
    return path


def _read_data_file(
    path: Path,
    sample: int | None,
    spec: TableSpec | None = None,
    *,
    infer_dates: bool = False,
) -> pl.DataFrame:
    """A data file as a frame, read in `spec`'s terms when there is one.

    With a spec, this is `polspec.read`. Without one, `infer_dates` has
    Polars recognise dates in a CSV itself -- `schema infer`'s case, where
    there is no declaration to go by.
    """
    if path.suffix.lower() not in SUFFIXES:
        raise CliError(
            f"don't know how to read {path.suffix!r} files ({path}). "
            f"Supported: {', '.join(sorted(SUFFIXES))}"
        )
    try:
        df = read_file(path, infer_dates=infer_dates)
    except Exception as exc:
        raise CliError(f"could not read {path}: {exc}") from exc
    try:
        df = sample_rows(df, sample)
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    return parse_declared_text(df, spec) if spec is not None else df


# The suffixes whose files hold every value as text, where a column's
# `text_format` says how a date is written.
_TEXT_SUFFIXES = frozenset({".csv", ".tsv", *_JSON_SUFFIXES})


def _write_data_file(
    df: pl.DataFrame, path: Path, spec: TableSpec | None = None
) -> None:
    """`df` written to `path`, by its suffix; a text file writes `spec`'s
    dates, datetimes and times as their `text_format` says."""
    writer = _DATA_WRITERS.get(path.suffix.lower())
    if writer is None:
        raise CliError(
            f"don't know how to write {path.suffix!r} files ({path}). "
            f"Supported: {', '.join(sorted(_DATA_WRITERS))}"
        )
    if spec is not None and path.suffix.lower() in _TEXT_SUFFIXES:
        df = df.with_columns(text_format_casts(spec.columns))
    if path.suffix.lower() in _JSON_SUFFIXES:
        df = df.with_columns(json_casts(df.schema, error=CliError))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer(df, path)
    except Exception as exc:
        raise CliError(f"could not write {path}: {exc}") from exc


def _registry_from(source: Path) -> Registry:
    """Every spec under `source`, with its foreign keys bound to each other."""
    registry = Registry.discover(source)
    if not registry.names:
        raise CliError(f"no specs found under {source}")
    return registry


def _data_file_for(directory: Path, name: str) -> Path | None:
    """The one data file under `directory` named after the spec `name`, or
    None when it has none there: `<name>.<suffix>` for a suffix the readers
    know, or a file whose stem gives that spec name, as several-source
    `schema infer` names a spec after its file (`order_lines.csv` for
    `OrderLines`). Matched by name, not by the file system, so a directory
    `synthesize` wrote is found on any platform."""
    found = sorted(
        path
        for path in directory.iterdir()
        if path.suffix in SUFFIXES
        and path.is_file()
        and name in (path.stem, _class_name_from(path.stem))
    )
    if len(found) > 1:
        raise CliError(
            f"{name} has several data files under {directory}: "
            f"{', '.join(p.name for p in found)}; keep one"
        )
    return found[0] if found else None


def frames_named_after_specs(
    registry: Registry, data_dir: Path, source: Path
) -> dict[str, pl.DataFrame]:
    """The data file named after each spec, for every spec that has one.

    Shared by the `--all` verbs: a directory of specs is matched to a
    directory of data by name, and a run with nothing to read at all is an
    error rather than a silent pass.
    """
    frames = {
        name: _read_data_file(path, None, registry[name])
        for name in registry.names
        if (path := _data_file_for(data_dir, name)) is not None
    }
    if not frames:
        raise CliError(
            f"no data file under {data_dir} is named after a spec in {source} "
            f"(looked for {', '.join(registry.names)} with a known suffix)"
        )
    return frames


def _single_spec(source: Path, class_name: str | None) -> type[FrameSpec]:
    """The one FrameSpec a file defines, or the one `--class` picks out."""
    specs = _loaded_specs(source, class_name, source)
    if len(specs) != 1:
        names = ", ".join(name for name, _, _ in specs)
        raise CliError(
            f"{source} defines several specs ({names}); pick one with --class"
        )
    return specs[0][1]


def _references_from(items: list[str] | None) -> dict[str, pl.DataFrame] | None:
    """`--references NAME=PATH ...` as the mapping `references=` takes."""
    references: dict[str, pl.DataFrame] = {}
    for item in items or ():
        name, path = _parse_reference(item)
        if not path.exists():
            raise CliError(f"no such file for reference {name!r}: {path}")
        references[name] = _read_data_file(path, None)
    return references or None


def _parse_reference(text: str) -> tuple[str, Path]:
    """`Customers=customers.parquet` -> ("Customers", Path("customers.parquet"))."""
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise CliError(
            f"--references expects NAME=PATH, got {text!r} "
            "(the spec name a foreign key points at, and a data file for it)"
        )
    return name, Path(path)


def _class_name_from(text: str) -> str:
    """A PascalCase identifier from an arbitrary file stem or name."""
    words = re.findall(r"[A-Za-z0-9]+", text) or ["Spec"]
    name = "".join(w[:1].upper() + w[1:] for w in words)
    return name if name[0].isalpha() else f"Spec{name}"


def _tables(paths: list[Path]) -> dict[str, pl.DataFrame]:
    """Each data file, whole, under the spec name its stem gives it -- the
    tables a several-source `schema infer` or `synthesize` profiles."""
    named: dict[str, Path] = {}
    for path in paths:
        name = _class_name_from(path.stem)
        if name in named:
            raise CliError(
                f"{named[name]} and {path} would both be the spec {name!r}; "
                "rename one of the files"
            )
        named[name] = path
    return {
        name: _read_data_file(path, None, infer_dates=True)
        for name, path in named.items()
    }


def _declared_keys(
    args: argparse.Namespace,
) -> dict[str, list[ForeignKey]] | Literal[False] | None:
    """`--foreign-key`s as the `keys=` a profile of related tables takes:
    None for none, False for `--no-foreign-keys`."""
    if args.no_foreign_keys:
        if args.foreign_key:
            raise CliError("--foreign-key and --no-foreign-keys contradict each other")
        return False
    keys: dict[str, list[ForeignKey]] = {}
    for text in args.foreign_key or ():
        left, sep, right = text.partition("=")
        table, _, column = left.partition(".")
        parent, _, ref = right.partition(".")
        if not (sep and table and column and parent and ref):
            raise CliError(
                "--foreign-key expects SPEC.COLUMN=SPEC.COLUMN, the child column "
                f"then the parent column, each under its file's spec name; got {text!r}"
            )
        references = "self" if parent == table else parent
        keys.setdefault(table, []).append(
            ForeignKey(column, references=references, ref_columns=ref)
        )
    return keys or None


def _add_key_flags(parser: argparse.ArgumentParser) -> None:
    """`--foreign-key` and `--no-foreign-keys`, which `schema infer` and
    `synthesize` take for several sources."""
    parser.add_argument(
        "--foreign-key",
        action="append",
        metavar="SPEC.COLUMN=SPEC.COLUMN",
        help=(
            "With several sources, declare a foreign key, child then parent, "
            "each under its file's spec name: Orders.buyer=Customers.id"
        ),
    )
    parser.add_argument(
        "--no-foreign-keys",
        action="store_true",
        help="With several sources, find no foreign keys between them",
    )


def _refuse_related_flags(args: argparse.Namespace) -> None:
    """The flags that only mean something across several sources."""
    for flag, given in (
        ("--foreign-key", args.foreign_key),
        ("--no-foreign-keys", args.no_foreign_keys),
    ):
        if given:
            raise CliError(f"{flag} is for two or more sources, related by keys")


def _write_registry(registry: Registry, path: Path) -> None:
    """`registry` as a spec file: Python for `.py`, YAML otherwise."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _warnings_printed():
        if path.suffix.lower() == ".py":
            registry_to_python(registry, path)
        else:
            registry.to_yaml(path)
    _maybe_format(path)


def _snake_case(name: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return s.lower()


def _require_identifier(name: str, *, what: str) -> None:
    if not name.isidentifier():
        raise CliError(f"{what} {name!r} is not a valid Python identifier")


def _load_module_from_path(path: Path) -> types.ModuleType:
    return load_module(path, error=CliError, prefix="cli")


def _frame_specs_in_module(module: types.ModuleType) -> dict[str, type[FrameSpec]]:
    """FrameSpec subclasses this module itself defines, excluding imports."""
    return {
        name: value
        for name, value in vars(module).items()
        if isinstance(value, type)
        and issubclass(value, FrameSpec)
        and value is not FrameSpec
        and value.__module__ == module.__name__
    }


@contextlib.contextmanager
def _warnings_printed() -> Iterator[None]:
    """Prints each warning raised inside as a `warning:` line on stderr --
    a spec file that cannot hold a value, a column profiled around its NaNs
    -- where a Python warning would be a traceback-looking line, or an
    error under `-W error`."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for warning in caught:
        print(f"warning: {warning.message}", file=sys.stderr)


def _note_text(spec: TableSpec) -> None:
    """Says on stderr which text columns would parse as another type, which
    their descriptions say too, and the flag that reads them so; and, under
    --parse-text, why each column that nearly parsed stays text."""
    noted = [
        name
        for name, column in spec.columns.items()
        if (column.description or "").startswith(TEXT_NOTE)
    ]
    if noted:
        print(
            f"note: {len(noted)} text column(s) hold numbers, flags or dates "
            f"throughout: {', '.join(noted)}. --parse-text reads them as those "
            "types; their descriptions say which.",
            file=sys.stderr,
        )
    for name, column in spec.columns.items():
        description = column.description or ""
        if description.startswith(STAYS_TEXT_NOTE):
            reason = description.removeprefix(STAYS_TEXT_NOTE)
            print(f"note: column {name!r} stays text: {reason}", file=sys.stderr)


def _maybe_format(path: Path) -> None:
    """Runs ruff format on generated Python, best-effort.

    Not fatal if ruff is missing -- the file is already valid Python without
    it, just less consistently spaced.
    """
    if path.suffix != ".py":
        return
    with contextlib.suppress(OSError):
        subprocess.run(  # noqa: S603 - fixed argv, no shell, path came from our own writer
            [sys.executable, "-m", "ruff", "format", str(path)],
            check=False,
            capture_output=True,
        )


def _loaded_specs(
    source: Path, class_name: str | None, output: Path
) -> list[tuple[str, type[FrameSpec], str]]:
    """The FrameSpec classes to generate tests for, and how to load each in
    the generated file.

    Returns (name, class, loader_snippet) triples, where the snippet is
    Python source that binds `name` in the generated test module. Loading
    happens exactly once here -- a .py source is only ever imported a single
    time, so any side effect its import causes only happens once.
    """
    rel = _relative_to_output(source, output)

    if source.suffix.lower() in (".yaml", ".yml"):
        spec_cls = FrameSpec.from_yaml(source)
        name = class_name or spec_cls.__name__
        loader = f"{name} = FrameSpec.from_yaml(Path(__file__).parent / {rel!r})"
        return [(name, spec_cls, loader)]

    if source.suffix.lower() == ".py":
        module = _load_module_from_path(source)
        found = _frame_specs_in_module(module)
        if class_name is not None:
            if class_name not in found:
                raise CliError(
                    f"no FrameSpec class {class_name!r} in {source} "
                    f"(found: {', '.join(sorted(found)) or 'none'})"
                )
            found = {class_name: found[class_name]}
        if not found:
            raise CliError(f"no FrameSpec subclasses defined in {source}")
        return [
            (
                name,
                spec_cls,
                f"{name} = _load_spec_module(Path(__file__).parent / {rel!r}).{name}",
            )
            for name, spec_cls in found.items()
        ]

    raise CliError(
        f"don't know how to load a spec from {source.suffix!r} files "
        f"({source}). Expected .yaml, .yml or .py"
    )


def _relative_to_output(path: Path, output: Path) -> str:
    """`path` relative to where `output` will live, else absolute.

    The generated test resolves this path against `Path(__file__).parent` at
    *its own* run time -- so the reference point has to be the output file's
    directory, not the current working directory the CLI happens to run
    from. Falls back to an absolute path when the two are on different
    drives, where no relative path exists.
    """
    try:
        return os.path.relpath(path.resolve(), start=output.resolve().parent)
    except ValueError:
        return str(path.resolve())


def _display_path(path: str) -> str:
    """Forward slashes, for a path shown inside a plain string body.

    A Windows path embedded raw between quotes turns a run like `\\U` into
    the start of a unicode escape, which fails to parse when Python
    re-reads the file it just wrote. Anywhere a path is quoted through
    `!r` this is not a concern -- `repr()` already escapes it -- but a
    docstring or comment inserts the text directly, so it needs to already
    be escape-free.
    """
    return path.replace("\\", "/")
