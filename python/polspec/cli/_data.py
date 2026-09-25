"""`polspec validate` and `polspec generate`: a spec against a data file, and
a data file from a spec -- or, with `--all`, every spec a directory holds
against a directory of data files named after them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polspec.cli._io import (
    _DATA_WRITERS,
    _existing,
    _read_data_file,
    _references_from,
    _registry_from,
    _single_spec,
    _write_data_file,
    frames_named_after_specs,
)
from polspec.constants import _LARGE_FRAME_BYTES
from polspec.errors import CliError
from polspec.generation import _describe_bytes
from polspec.tablespec import TableSpec
from polspec.validation import validate

if TYPE_CHECKING:
    import polars as pl

    from polspec.validation import ValidationReport


def _cmd_validate(args: argparse.Namespace) -> int:
    """Checks a data file against a spec, printing findings or JSON.

    Exit status 0 when the data passes, 1 when it does not; a problem with
    the arguments or files is reported like any other CLI error.
    """
    if args.all:
        if args.output or args.failing:
            raise CliError(
                "--output and --failing each take one file, so they do not "
                "combine with --all"
            )
        return _validate_all(args)
    for target in (args.output, args.failing):
        if target is not None:
            _writable(Path(target))
    source = _existing(args.spec)
    data_path = _existing(args.data)
    spec_cls = _single_spec(source, args.cls)
    references = _references_from(args.references)
    options = _validation_options(args)

    df = _read_data_file(data_path, None, spec_cls.spec)
    report = spec_cls.inspect(df, references=references, **options)

    if args.json:
        print(report.to_json())
    else:
        print(str(report))
    if args.output or args.failing:
        _write_split(report, spec_cls.spec, args, references, options)
    return 0 if report.passed else 1


def _validation_options(args: argparse.Namespace) -> dict[str, Any]:
    """The `validate` options the command line sets, as `inspect` takes them."""
    return {
        "extra_cols": "allow" if args.allow_extra else "raise",
        "missing_cols": "allow" if args.allow_missing else "raise",
        "strict_dtypes": args.strict_dtypes,
        **_skipped(args),
    }


def _writable(path: Path) -> None:
    """Refuses an output path whose format polspec cannot write, before any
    reading is done."""
    if path.suffix.lower() not in _DATA_WRITERS:
        raise CliError(
            f"don't know how to write {path.suffix!r} files ({path}). "
            f"Supported: {', '.join(sorted(_DATA_WRITERS))}"
        )


def _write_split(
    report: ValidationReport,
    spec: TableSpec,
    args: argparse.Namespace,
    references: dict[str, pl.DataFrame] | None,
    options: dict[str, Any],
) -> None:
    """`--output` and `--failing`: the file split into what passed and what did
    not. Notes go to stderr, so `--json` output stays one document.

    The passing rows are validated again, cast to the declared dtypes: the
    file `--output` writes is one the spec accepts, typed as it declares. A
    structural finding judges the whole frame, so neither file is written.
    """
    structural = [f.key for f in report if not f.row_level]
    if structural:
        print(
            f"note: {', '.join(structural)} judge the whole frame, so no "
            "--output or --failing file was written",
            file=sys.stderr,
        )
        return
    if args.output:
        passing = report.passing_rows().collect()
        typed = validate(spec, passing, references=references, cast=True, **options)
        _write_data_file(typed, Path(args.output))
        print(f"Wrote {typed.height} passing row(s) to {args.output}", file=sys.stderr)
    if args.failing:
        failing = report.failing_rows().collect()
        _write_data_file(failing, Path(args.failing))
        print(
            f"Wrote {failing.height} failing row(s) to {args.failing}", file=sys.stderr
        )


def _skipped(args: argparse.Namespace) -> dict[str, Any]:
    """`--skip NAME ...` as the `validate_NAME=False` switches `inspect` takes."""
    return {f"validate_{name}": False for name in args.skip or ()}


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
    _say_if_large(spec_cls.spec, args.rows)
    df = spec_cls.generate(
        args.rows,
        method=args.method,
        seed=args.seed,
        references=references,
        max_bytes=0,  # said above already; one warning is enough
    )
    _write_data_file(df, output)
    print(f"Wrote {df.height} row(s) of {spec_cls.__name__} to {output}")
    return 0


def _say_if_large(spec: TableSpec, rows: int) -> None:
    """Says how big the frame will be before it is built, when that is worth
    saying. `generate` builds the whole frame, so a shell caller who did not
    expect gigabytes should hear it before the machine starts swapping."""
    estimate = spec.estimated_size(rows)
    if estimate > _LARGE_FRAME_BYTES:
        print(
            f"note: {spec.name} at {rows:,} rows is an estimated "
            f"{_describe_bytes(estimate)}, held in memory before it is written",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# --all: a registry of specs, a directory of files named after them
# ---------------------------------------------------------------------------


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
    frames = frames_named_after_specs(registry, data_dir, source)
    references = _references_from(args.references)
    reports = registry.inspect_all(
        frames, references=references, **_validation_options(args)
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
