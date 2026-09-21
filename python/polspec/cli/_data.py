"""`polspec validate` and `polspec generate`: a spec against a data file, and
a data file from a spec."""

from __future__ import annotations

import argparse
from pathlib import Path

from polspec.cli._io import (
    _DATA_WRITERS,
    _existing,
    _read_data_file,
    _references_from,
    _single_spec,
    _write_data_file,
)
from polspec.errors import CliError


def _cmd_validate(args: argparse.Namespace) -> int:
    """Checks a data file against a spec, printing findings or JSON.

    Exit status 0 when the data passes, 1 when it does not; a problem with
    the arguments or files is reported like any other CLI error.
    """
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
