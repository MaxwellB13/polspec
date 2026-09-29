"""`polspec synthesize`: a fake data file from a real one."""

from __future__ import annotations

import argparse
from pathlib import Path

from polspec.cli._io import (
    _DATA_WRITERS,
    _class_name_from,
    _existing,
    _maybe_format,
    _require_identifier,
    _warnings_printed,
    _write_data_file,
)
from polspec.errors import CliError
from polspec.framespec import FrameSpec
from polspec.synthesis import synthesized


def _cmd_synthesize(args: argparse.Namespace) -> int:
    """Profiles a data file and writes a fake one like it -- and, with
    `--spec`, the spec the fake one was generated from."""
    source = _existing(args.source)
    output = Path(args.output)
    if output.suffix.lower() not in _DATA_WRITERS:
        raise CliError(
            f"don't know how to write {output.suffix!r} files ({output}). "
            f"Supported: {', '.join(sorted(_DATA_WRITERS))}"
        )
    if args.rows is not None and args.rows < 0:
        raise CliError(f"-n/--rows must be non-negative, got {args.rows}")
    name = args.name or _class_name_from(source.stem)
    _require_identifier(name, what="--name")

    with _warnings_printed():
        try:
            fake, spec = synthesized(
                source,
                args.rows,
                seed=args.seed,
                replace=args.replace or (),
                sample=args.sample,
                max_unique_enum=args.max_unique_enum,
                formats=not args.no_formats,
                name=name,
            )
        except ValueError as exc:  # a name in --replace the file lacks, and the like
            raise CliError(str(exc)) from exc
    _write_data_file(fake, output)
    print(f"Wrote {fake.height:,} synthesized row(s) like {source} to {output}")

    if args.spec:
        spec_path = Path(args.spec)
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_cls = FrameSpec.from_spec(spec)
        writer = (
            spec_cls.to_python
            if spec_path.suffix.lower() == ".py"
            else spec_cls.to_yaml
        )
        with _warnings_printed():
            writer(spec_path)
        _maybe_format(spec_path)
        print(f"Wrote the spec it was generated from to {spec_path}")
    return 0


def add_synthesize_parser(subparsers: argparse._SubParsersAction) -> None:
    synthesize = subparsers.add_parser(
        "synthesize", help="Write a fake data file that looks like a real one"
    )
    synthesize.add_argument(
        "source", help="Path to a CSV, TSV, Parquet, NDJSON, JSON or IPC file"
    )
    synthesize.add_argument(
        "-o",
        "--output",
        required=True,
        help="File to write; the extension picks the format (.parquet, .csv, ...)",
    )
    synthesize.add_argument(
        "-n",
        "--rows",
        type=int,
        metavar="N",
        help="Rows to generate (default: as many as the source has)",
    )
    synthesize.add_argument(
        "--seed", type=int, help="Generation seed (default: random)"
    )
    synthesize.add_argument(
        "--replace",
        nargs="+",
        action="extend",
        metavar="COL",
        help=(
            "Columns whose values must not be carried over: text in them is "
            "generated from its lengths, never from the values the source holds"
        ),
    )
    synthesize.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Profile a random sample of N rows rather than every row",
    )
    synthesize.add_argument(
        "--max-unique-enum",
        type=int,
        default=50,
        metavar="N",
        help="Max distinct values for a text column to keep its values, when they repeat (default: 50)",
    )
    synthesize.add_argument(
        "--no-formats",
        action="store_true",
        help="Do not name the format a text column's values have (email, uuid4, ...)",
    )
    synthesize.add_argument(
        "--spec",
        metavar="PATH",
        help="Also write the spec the data was generated from (.yaml, or .py)",
    )
    synthesize.add_argument(
        "--name", help="The spec's class name (default: derived from source)"
    )
    synthesize.set_defaults(func=_cmd_synthesize)
