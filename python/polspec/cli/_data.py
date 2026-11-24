"""`polspec validate` and `polspec generate`: a spec against a data file, and
a data file from a spec -- or, with `--all`, every spec a directory holds
against a directory of data files named after them."""

from __future__ import annotations

import argparse
import decimal
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec._text import parse_series, text_parser
from polspec.cli._files import (
    _DATA_WRITERS,
    _existing,
    _read_data_file,
    _write_data_file,
)
from polspec.cli._sources import _references_from, frames_named_after_specs
from polspec.cli._specs import _registry_from, _single_spec
from polspec.constants import LARGE_FRAME_BYTES
from polspec.errors import CliError, PolspecError
from polspec.generation.pipeline import describe_bytes
from polspec.sizing import estimated_size
from polspec.tablespec import TableSpec
from polspec.validation import FINDING_COLUMN, validate
from polspec.validation.report import SCHEMA_VERSION as REPORT_SCHEMA_VERSION
from polspec.validation.tolerance import CODES

if TYPE_CHECKING:
    from polspec.registry import Registry
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
        "fill_defaults": args.fill_defaults,
        # The file is read in the spec's terms already (`read()`); asking
        # again names the values of a column that stayed text.
        "parse_text": True,
        **_skipped(args),
        **_tolerance(args),
    }


def _tolerance(args: argparse.Namespace) -> dict[str, Any]:
    """`--tolerate [KEY=]SHARE ...` as the `tolerate=` `inspect` takes: one
    share for every finding, or a mapping by code, column, or `column:code`.
    A share may be written as a percentage."""
    given = getattr(args, "tolerate", None)
    if not given:
        return {}
    every: float | None = None
    keyed: dict[Any, float] = {}
    for entry in given:
        key, sep, text = entry.rpartition("=")
        share = _share_text(text, entry)
        if not sep:
            every = share
            continue
        column, colon, code = key.rpartition(":")
        keyed[(column, code) if colon and code in CODES else key] = share
    if every is not None and keyed:
        raise CliError(
            "--tolerate takes one share for every finding, or shares by "
            f"KEY=SHARE, not both (got {', '.join(given)})"
        )
    return {"tolerate": every if every is not None else keyed}


def _share_text(text: str, entry: str) -> float:
    """A share as `--tolerate` writes it: `0.001`, or `0.1%`."""
    percent = text.endswith("%")
    try:
        # Through a decimal, so `0.1%` is exactly the share `0.001` is.
        value = decimal.Decimal(text.removesuffix("%"))
        if not value.is_finite():
            raise ValueError(text)
    except (ValueError, decimal.InvalidOperation):
        raise CliError(
            f"--tolerate {entry!r}: {text!r} is not a share of the rows; write "
            "0.001 or 0.1%"
        ) from None
    return float(value / 100 if percent else value)


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
    if args.failing and FINDING_COLUMN in report.frame.collect_schema().names():
        # Before either file: a half-written split is worse than none.
        raise CliError(
            f"the data has a column named {FINDING_COLUMN!r}, the column "
            "--failing names each row's broken claims in; rename it, or "
            "leave out --failing"
        )
    if args.output:
        try:
            passing = report.passing_rows().collect()
        except (
            ValueError,
            PolspecError,
        ) as exc:  # too few rows pass for the spec's `rows`
            raise CliError(str(exc)) from exc
        typed = validate(spec, passing, references=references, cast=True, **options)
        _write_data_file(typed, Path(args.output), spec)
        print(f"Wrote {typed.height} passing row(s) to {args.output}", file=sys.stderr)
    if args.failing:
        # Each row once, naming every claim it breaks: with --output, the
        # file is split exactly, whatever a row breaks.
        failing = report.failing_rows(once=True).collect()
        _write_data_file(failing, Path(args.failing), spec)
        print(
            f"Wrote {failing.height} failing row(s) to {args.failing}", file=sys.stderr
        )


def _skipped(args: argparse.Namespace) -> dict[str, Any]:
    """`--skip NAME ...` as the `NAME=False` switches `inspect` takes."""
    return dict.fromkeys(args.skip or (), False)


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
        fixed=_fixed_from(args.fixed or [], spec_cls.spec),
    )
    _write_data_file(df, output, spec_cls.spec)
    print(f"Wrote {df.height} row(s) of {spec_cls.__name__} to {output}")
    return 0


def _pin_parts(pin: str) -> tuple[str, str]:
    name, sep, text = pin.partition("=")
    if not sep or not name:
        raise CliError(f"--fixed takes COLUMN=VALUE, got {pin!r}")
    return name, text


def _fixed_from(pins: list[str], spec: TableSpec) -> dict[str, Any]:
    """`--fixed COLUMN=VALUE` as `generate(fixed=)` takes it: each value read
    as its column's dtype, as `validate --parse-text` reads text, and empty
    text as a null."""
    fixed: dict[str, Any] = {}
    for pin in pins:
        name, text = _pin_parts(pin)
        if name not in spec.columns:
            raise CliError(
                f"--fixed names {name!r}, which {spec.name} does not declare. "
                f"Its columns: {', '.join(spec.columns)}"
            )
        dtype = spec.columns[name].dtype
        if text_parser(dtype) is None:
            # Text, an Enum, a Categorical: the value is the text itself.
            fixed[name] = text or None
            continue
        # In the column's own text_format, as its file holds it, or in the
        # form Polars writes.
        written = spec.columns[name].text_format
        parsed, _ = parse_series(pl.Series([text]), dtype, written)
        if parsed is None and written is not None:
            parsed, _ = parse_series(pl.Series([text]), dtype)
        if parsed is None:
            raise CliError(f"--fixed {pin!r}: {text!r} does not read as {dtype!r}")
        fixed[name] = parsed[0]
    return fixed


def _fixed_by_spec(pins: list[str], registry: Registry) -> dict[str, dict[str, Any]]:
    """`--all --fixed SPEC.COLUMN=VALUE`, by spec."""
    by_spec: dict[str, list[str]] = {}
    for pin in pins:
        target, _ = _pin_parts(pin)
        spec_name, dot, _ = target.partition(".")
        if not dot or spec_name not in registry.names:
            raise CliError(
                f"with --all, --fixed takes SPEC.COLUMN=VALUE for one of "
                f"{', '.join(registry.names)}; got {pin!r}"
            )
        by_spec.setdefault(spec_name, []).append(pin.partition(".")[2])
    return {
        name: _fixed_from(spec_pins, registry[name])
        for name, spec_pins in by_spec.items()
    }


def _say_if_large(spec: TableSpec, rows: int) -> None:
    """Says how big the frame will be before it is built, when that is worth
    saying. `generate` builds the whole frame, so a shell caller who did not
    expect gigabytes should hear it before the machine starts swapping."""
    estimate = estimated_size(spec, rows)
    if estimate > LARGE_FRAME_BYTES:
        print(
            f"note: {spec.name} at {rows:,} rows is an estimated "
            f"{describe_bytes(estimate)}, held in memory before it is written",
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
        args.rows,
        seed=args.seed,
        method=args.method,
        references=references,
        fixed=_fixed_by_spec(args.fixed or [], registry),
    )
    for name in registry.order():
        if name not in frames:
            continue
        path = output / f"{name}{suffix}"
        _write_data_file(frames[name], path, registry[name])
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
        # Keyed under `reports`: a spec may be called anything, `schema_version`
        # included.
        document = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "reports": {name: r.to_dict() for name, r in reports.items()},
        }
        print(json.dumps(document, indent=2))
    else:
        for name, report in reports.items():
            print(f"== {name}")
            print(str(report))
        skipped = [n for n in registry.names if n not in frames]
        if skipped:
            print(f"(no data file for: {', '.join(skipped)})")
    return 0 if all(r.passed for r in reports.values()) else 1
