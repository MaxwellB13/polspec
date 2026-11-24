"""Spec files: loading a spec, a registry or a module of specs, and
writing generated Python beside them.
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

from polspec import FrameSpec
from polspec.errors import CliError
from polspec.registry import Registry, load_module
from polspec.serialization import registry_to_python

if TYPE_CHECKING:
    pass
from polspec.cli._files import _warnings_printed


def _registry_from(source: Path) -> Registry:
    """Every spec under `source`, with its foreign keys bound to each other."""
    registry = Registry.discover(source)
    if not registry.names:
        raise CliError(f"no specs found under {source}")
    return registry


def _single_spec(source: Path, class_name: str | None) -> type[FrameSpec]:
    """The one FrameSpec a file defines, or the one `--class` picks out."""
    specs = _loaded_specs(source, class_name, source)
    if len(specs) != 1:
        names = ", ".join(name for name, _, _ in specs)
        raise CliError(
            f"{source} defines several specs ({names}); pick one with --class"
        )
    return specs[0][1]


def _class_name_from(text: str) -> str:
    """A PascalCase identifier from an arbitrary file stem or name."""
    words = re.findall(r"[A-Za-z0-9]+", text) or ["Spec"]
    name = "".join(w[:1].upper() + w[1:] for w in words)
    return name if name[0].isalpha() else f"Spec{name}"


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
