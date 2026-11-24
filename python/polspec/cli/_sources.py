"""Several sources at once: data files named after the specs they hold,
`--reference` frames, and the foreign-key flags of a related set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl

from polspec.errors import CliError
from polspec.foreign_key import ForeignKey
from polspec.reading import SUFFIXES
from polspec.registry import Registry

if TYPE_CHECKING:
    pass
from polspec.cli._files import _read_data_file
from polspec.cli._specs import _class_name_from


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
