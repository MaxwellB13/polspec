"""`polspec validate` and `polspec generate`: a spec against a data file, and
a data file from a spec -- or, with `--all`, every spec a directory holds
against a directory of data files named after them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from polspec.cli._io import (
    _DATA_READERS,
    _DATA_WRITERS,
    _existing,
    _read_data_file,
    _references_from,
    _single_spec,
    _write_data_file,
)
from polspec.errors import CliError
from polspec.registry import Registry


def _cmd_validate(args: argparse.Namespace) -> int:
    """Checks a data file against a spec, printing findings or JSON.

    Exit status 0 when the data passes, 1 when it does not; a problem with
    the arguments or files is reported like any other CLI error.
    """
    if args.all:
        return _validate_all(args)
    source = _existing(args.spec)
    data_path = _existing(args.data)
    spec_cls = _single_spec(source, args.cls)
    references = _references_from(args.references)

    df = _read_data_file(data_path, None)
    report = spec_cls.inspect(
        df,
        references=references,
        extra_cols="allow" if args.allow_extra else "raise",
        missing_cols="allow" if args.allow_missing else "raise",
        strict_dtypes=args.strict_dtypes,
    )

    if args.json:
        print(report.to_json())
    else:
        print(str(report))
    return 0 if report.passed else 1


def _cmd_generate(args: argparse.Namespace) -> int:
    """Generates rows from a spec and writes them to one data file.

    Eager: the frame is built in memory and written once. The streaming
    sinks (`sink_parquet` and friends) stay a Python surface -- a file too
    large to hold is a file too large to inspect at the shell anyway.
    """
    if args.all:
        return _generate_all(args)
    source = _existing(args.spec)
    if args.rows < 0:
        raise CliError(f"-n/--rows must be non-negative, got {args.rows}")
    spec_cls = _single_spec(source, args.cls)
    references = _references_from(args.references)
    output = Path(args.output)
    if output.suffix.lower() not in _DATA_WRITERS:
        raise CliError(
            f"don't know how to write {output.suffix!r} files ({output}). "
            f"Supported: {', '.join(sorted(_DATA_WRITERS))}"
        )
    df = spec_cls.generate(
        args.rows, method=args.method, seed=args.seed, references=references
    )
    _write_data_file(df, output)
    print(f"Wrote {df.height} row(s) of {spec_cls.__name__} to {output}")
    return 0


# ---------------------------------------------------------------------------
# --all: a registry of specs, a directory of files named after them
# ---------------------------------------------------------------------------


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


def _generate_all(args: argparse.Namespace) -> int:
    """Every spec under a directory, parents first, one file each."""
    source = _existing(args.spec, what="file or directory")
    if args.rows < 0:
        raise CliError(f"-n/--rows must be non-negative, got {args.rows}")
    suffix = f".{args.format.lstrip('.')}"
    if suffix not in _DATA_WRITERS:
        raise CliError(
            f"don't know how to write {suffix!r} files. "
            f"Supported: {', '.join(sorted(_DATA_WRITERS))}"
        )
    output = Path(args.output)
    if output.suffix:
        raise CliError(
            f"with --all, -o/--output is a directory, got a file: {output}. "
            "Pick the format with --format"
        )
    registry = _registry_from(source)
    references = _references_from(args.references)
    frames = registry.generate_all(
        args.rows, seed=args.seed, method=args.method, references=references
    )
    for name in registry.order():
        if name not in frames:
            continue
        path = output / f"{name}{suffix}"
        _write_data_file(frames[name], path)
        print(f"Wrote {frames[name].height} row(s) of {name} to {path}")
    return 0


def _validate_all(args: argparse.Namespace) -> int:
    """Every spec under a directory against the file named after it, each
    seeing the others as parents; exit 1 when any fails."""
    source = _existing(args.spec, what="file or directory")
    data_dir = _existing(args.data, what="directory")
    if not data_dir.is_dir():
        raise CliError(f"with --all, DATA is a directory of data files, got {data_dir}")
    registry = _registry_from(source)
    frames: dict[str, pl.DataFrame] = {}
    for name in registry.names:
        path = _data_file_for(data_dir, name)
        if path is not None:
            frames[name] = _read_data_file(path, None)
    if not frames:
        raise CliError(
            f"no data file under {data_dir} is named after a spec in {source} "
            f"(looked for {', '.join(registry.names)} with a known suffix)"
        )
    references = _references_from(args.references)
    reports = registry.inspect_all(
        frames,
        references=references,
        extra_cols="allow" if args.allow_extra else "raise",
        missing_cols="allow" if args.allow_missing else "raise",
        strict_dtypes=args.strict_dtypes,
    )
    if args.json:
        print(json.dumps({name: r.to_dict() for name, r in reports.items()}, indent=2))
    else:
        for name, report in reports.items():
            print(f"== {name}")
            print(str(report))
        skipped = [n for n in registry.names if n not in frames]
        if skipped:
            print(f"(no data file for: {', '.join(skipped)})")
    return 0 if all(r.passed for r in reports.values()) else 1
