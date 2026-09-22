"""What every verb shares: reading data files, loading spec files, writing
generated Python.

One reader and one writer per suffix, so what `generate` writes, `validate`
and `drift` read back; a suffix in one map and not the other is a red test.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import polars as pl

from polspec import FrameSpec
from polspec.errors import CliError
from polspec.registry import Registry

_DATA_READERS = {
    ".csv": pl.read_csv,
    ".tsv": lambda p: pl.read_csv(p, separator="\t"),
    ".parquet": pl.read_parquet,
    ".pq": pl.read_parquet,
    ".ndjson": pl.read_ndjson,
    ".jsonl": pl.read_ndjson,
    ".json": pl.read_json,
    ".arrow": pl.read_ipc,
    ".ipc": pl.read_ipc,
    ".feather": pl.read_ipc,
}


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


def _read_data_file(path: Path, sample: int | None) -> pl.DataFrame:
    reader = _DATA_READERS.get(path.suffix.lower())
    if reader is None:
        raise CliError(
            f"don't know how to read {path.suffix!r} files ({path}). "
            f"Supported: {', '.join(sorted(_DATA_READERS))}"
        )
    try:
        df = reader(path)
    except ImportError as exc:
        hint = ' Try: pip install "polspec[arrow]"' if "pyarrow" in str(exc) else ""
        raise CliError(f"could not read {path}: {exc}.{hint}") from exc
    except Exception as exc:
        raise CliError(f"could not read {path}: {exc}") from exc
    return df.head(sample) if sample is not None else df


def _write_data_file(df: pl.DataFrame, path: Path) -> None:
    writer = _DATA_WRITERS.get(path.suffix.lower())
    if writer is None:
        raise CliError(
            f"don't know how to write {path.suffix!r} files ({path}). "
            f"Supported: {', '.join(sorted(_DATA_WRITERS))}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer(df, path)
    except ImportError as exc:
        hint = ' Try: pip install "polspec[arrow]"' if "pyarrow" in str(exc) else ""
        raise CliError(f"could not write {path}: {exc}.{hint}") from exc


def _registry_from(source: Path) -> Registry:
    """Every spec under `source`, with its foreign keys bound to each other."""
    registry = Registry.discover(source)
    if not registry.names:
        raise CliError(f"no specs found under {source}")
    return registry


def _data_file_for(directory: Path, name: str) -> Path | None:
    """`directory/<name>.<suffix>` for the one suffix the readers know, or
    None when the spec has no file there."""
    found = [
        candidate
        for suffix in _DATA_READERS
        if (candidate := directory / f"{name}{suffix}").exists()
    ]
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
        name: _read_data_file(path, None)
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


def _snake_case(name: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return s.lower()


def _require_identifier(name: str, *, what: str) -> None:
    if not name.isidentifier():
        raise CliError(f"{what} {name!r} is not a valid Python identifier")


def _load_module_from_path(path: Path) -> types.ModuleType:
    module_name = f"_polspec_cli_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise CliError(f"could not load {path} as a Python module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise CliError(f"error importing {path}: {exc}") from exc
    return module


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
