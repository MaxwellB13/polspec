"""What you can import is what polspec promises.

The public API is `polspec`'s `__all__`, plus the `__all__` of three public
submodules: `polspec.drift`, `polspec.validation` and
`polspec.serialization`. Every other module says, in its docstring, that it
is internal. A module or package whose name starts with an underscore is
private by its name already. These tests hold the code and the docs to that
line:
- the docs teach only public names;
- every public name has a docstring;
- every other module says it is internal;
- no module takes another's private names, and the CLI uses polspec the way
  a user would.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import pkgutil
import re
from pathlib import Path

import polspec
import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python" / "polspec"
PUBLIC = ("polspec", "polspec.drift", "polspec.validation", "polspec.serialization")
INTERNAL = "Internal: not part of the public API."

# `from polspec.x import a, b` or `from polspec.x import (\n a,\n b,\n)`,
# and `import polspec.x`.
FROM_IMPORT = re.compile(r"^\s*from (polspec[\w.]*) import (\([^)]*\)|[^\n]+)", re.M)
PLAIN_IMPORT = re.compile(r"^\s*import (polspec[\w.]*)", re.M)


def _private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def _private_module(part: str) -> bool:
    """`_io`, and `__main__`, which `python -m polspec` runs."""
    return _private(part) or part == "__main__"


def _modules() -> list[str]:
    names = ["polspec"]
    names += [m.name for m in pkgutil.walk_packages(polspec.__path__, "polspec.")]
    return names


def _internal_modules() -> list[str]:
    return [
        name
        for name in _modules()
        if name not in PUBLIC and not any(_private_module(p) for p in name.split("."))
    ]


def _imported_names(clause: str) -> list[str]:
    clause = re.sub(r"#[^\n]*", "", clause).strip().strip("()")
    return [part.split(" as ")[0].strip() for part in clause.split(",") if part.strip()]


DOCUMENTS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


@pytest.mark.parametrize(
    "document", DOCUMENTS, ids=lambda p: p.relative_to(ROOT).as_posix()
)
def test_the_docs_import_only_public_names(document: Path):
    text = document.read_text(encoding="utf-8")
    wrong = [module for module in PLAIN_IMPORT.findall(text) if module not in PUBLIC]
    for module, clause in FROM_IMPORT.findall(text):
        if module not in PUBLIC:
            wrong.append(module)
            continue
        public = importlib.import_module(module).__all__
        wrong += [
            f"{module}.{name}"
            for name in _imported_names(clause)
            if name != "*" and name not in public
        ]
    assert not wrong, f"{document.name} imports names outside the public API"


@pytest.mark.parametrize("module", PUBLIC)
def test_every_public_name_is_real_and_has_a_docstring(module: str):
    """Each class and function, that is: a constant or a `Literal` alias
    cannot carry one, and its API page describes it."""
    loaded = importlib.import_module(module)
    assert loaded.__all__ == sorted(loaded.__all__)
    undocumented = []
    for name in loaded.__all__:
        value = getattr(loaded, name)  # an AttributeError is a stale __all__
        if (inspect.isclass(value) or inspect.isroutine(value)) and not value.__doc__:
            undocumented.append(name)
    assert not undocumented


@pytest.mark.parametrize("module", _internal_modules())
def test_every_other_module_says_it_is_internal(module: str):
    assert INTERNAL in (importlib.import_module(module).__doc__ or "")


def _sources() -> list[tuple[str, Path]]:
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        found.append((".".join(parts), path))
    return found


def _allowed(importer: str, target: str, name: str) -> bool:
    """Whether `importer` may take `name`, private, from `target`.

    A private module's names belong to the package it sits in, so its
    siblings may share them: `polspec.cli._io` serves the rest of
    `polspec.cli`. A private name in a module with a public name belongs to
    that module alone. A private submodule, imported as a module, is a
    module, not a name.
    """
    package = importlib.import_module(target)
    if hasattr(package, "__path__") and importlib.util.find_spec(f"{target}.{name}"):
        return True
    parts = target.split(".")
    for depth, part in enumerate(parts):
        if _private(part):
            owner = ".".join(parts[:depth])
            return importer == owner or importer.startswith(f"{owner}.")
    return False


def test_no_module_imports_another_modules_private_names():
    crossings = []
    for importer, path in _sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith("polspec"):
                continue
            crossings += [
                f"{importer}:{node.lineno} takes {node.module}.{alias.name}"
                for alias in node.names
                if _private(alias.name)
                and node.module != importer
                and not _allowed(importer, node.module, alias.name)
            ]
    assert not crossings


# argparse has no public name for the type `add_subparsers()` returns.
_STDLIB_PRIVATE = {"argparse._SubParsersAction"}


def test_the_cli_reaches_into_nothing_private():
    """The command line uses polspec as a user would: no `report._…`.
    (Inside the library, the declarations share private helpers between
    their own modules; that is not held here.)"""
    reaches = []
    for module, path in _sources():
        if not module.startswith("polspec.cli"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Attribute) or not _private(node.attr):
                continue
            if isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}:
                continue
            if ast.unparse(node) not in _STDLIB_PRIVATE:
                reaches.append(f"{module}:{node.lineno} {ast.unparse(node)}")
    assert not reaches
