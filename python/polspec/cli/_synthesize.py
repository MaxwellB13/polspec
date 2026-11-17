"""`polspec synthesize`: a fake data file from a real one."""

from __future__ import annotations

import argparse
from pathlib import Path

from polspec.cli._io import (
    _DATA_WRITERS,
    _add_key_flags,
    _class_name_from,
    _declared_keys,
    _existing,
    _maybe_format,
    _note_text,
    _refuse_related_flags,
    _require_identifier,
    _tables,
    _warnings_printed,
    _write_data_file,
    _write_registry,
)
from polspec.cli._profiling import (
    SYNTHESIZE_OPTIONS,
    add_profile_flags,
    profile_keywords,
)
from polspec.errors import CliError
from polspec.framespec import FrameSpec
from polspec.related import synthesized_all
from polspec.synthesis import synthesized


def _cmd_synthesize(args: argparse.Namespace) -> int:
    """Profiles a data file and writes a fake one like it -- and, with
    `--spec`, the spec the fake one was generated from."""
    sources = [_existing(text) for text in args.source]
    if args.rows is not None and args.rows < 0:
        raise CliError(f"-n/--rows must be non-negative, got {args.rows}")
    if args.rows is not None and args.scale is not None:
        raise CliError("pass -n/--rows or --scale, not both")
    if len(sources) > 1:
        return _synthesize_related(args, sources)
    _refuse_related_flags(args)
    if args.format:
        raise CliError("--format is for two or more sources; -o names the file")
    source = sources[0]
    output = Path(args.output)
    if output.suffix.lower() not in _DATA_WRITERS:
        raise CliError(
            f"don't know how to write {output.suffix!r} files ({output}). "
            f"Supported: {', '.join(sorted(_DATA_WRITERS))}"
        )
    name = args.name or _class_name_from(source.stem)
    _require_identifier(name, what="--name")

    with _warnings_printed():
        try:
            fake, spec = synthesized(
                source,
                args.rows,
                scale=args.scale,
                seed=args.seed,
                **profile_keywords(args),
                name=name,
            )
        except ValueError as exc:  # a name in --replace the file lacks, and the like
            raise CliError(str(exc)) from exc
    _note_text(spec)
    _write_data_file(fake, output, spec)
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


def _synthesize_related(args: argparse.Namespace, sources: list[Path]) -> int:
    """`synthesize` over several files: a fake file for each, in one
    directory, with the keys between them kept."""
    if args.name:
        raise CliError(
            "--name names one spec; with several sources each is named after its file"
        )
    out_dir = Path(args.output)
    if out_dir.suffix.lower() in _DATA_WRITERS:
        raise CliError(
            f"with several sources, -o names a directory to write them in, got {out_dir}"
        )
    suffix = _suffix(args.format) if args.format else None
    for source in sources:
        if (suffix or source.suffix.lower()) not in _DATA_WRITERS:
            raise CliError(
                f"don't know how to write {source.suffix!r} files ({source}); "
                f"pick one with --format: {', '.join(sorted(_DATA_WRITERS))}"
            )
    keys = _declared_keys(args)
    tables = _tables(sources)
    with _warnings_printed():
        try:
            fake, registry = synthesized_all(
                tables,
                args.rows,
                scale=args.scale,
                seed=args.seed,
                keys=keys,
                **profile_keywords(args),
            )
        except ValueError as exc:
            raise CliError(str(exc)) from exc
    for name in registry:
        _note_text(registry[name])
    for source, (name, frame) in zip(sources, fake.items(), strict=True):
        path = out_dir / f"{source.stem}{suffix or source.suffix.lower()}"
        _write_data_file(frame, path, registry[name])
        print(f"Wrote {frame.height:,} synthesized row(s) like {source} to {path}")
    found = sum(len(registry[name].foreign_keys) for name in registry)
    print(f"Kept {found} foreign key(s) between them")
    if args.spec:
        spec_path = Path(args.spec)
        _write_registry(registry, spec_path)
        print(f"Wrote the specs they were generated from to {spec_path}")
    return 0


def _suffix(text: str) -> str:
    suffix = text.lower() if text.startswith(".") else f".{text.lower()}"
    if suffix not in _DATA_WRITERS:
        raise CliError(
            f"--format {text!r} is not a format synthesize writes. "
            f"Supported: {', '.join(sorted(_DATA_WRITERS))}"
        )
    return suffix


def add_synthesize_parser(subparsers: argparse._SubParsersAction) -> None:
    synthesize = subparsers.add_parser(
        "synthesize", help="Write a fake data file that looks like a real one"
    )
    synthesize.add_argument(
        "source",
        nargs="+",
        help=(
            "Path to a CSV, TSV, Parquet, NDJSON, JSON or IPC file; several are "
            "related tables, synthesized with the foreign keys between them"
        ),
    )
    synthesize.add_argument(
        "-o",
        "--output",
        required=True,
        help=(
            "File to write, whose extension picks the format (.parquet, .csv, ...); "
            "with several sources, a directory to write one file each in"
        ),
    )
    synthesize.add_argument(
        "-n",
        "--rows",
        type=int,
        metavar="N",
        help="Rows to generate (default: as many as the source has)",
    )
    synthesize.add_argument(
        "--scale",
        type=float,
        metavar="X",
        help=(
            "Rows as a multiple of each source's: 0.1 for a tenth, keeping the "
            "ratios between related tables"
        ),
    )
    synthesize.add_argument(
        "--seed", type=int, help="Generation seed (default: random)"
    )
    synthesize.add_argument(
        "--format",
        metavar="EXT",
        help="With several sources, write each as EXT (parquet, csv, ...) (default: its own)",
    )
    _add_key_flags(synthesize)
    add_profile_flags(synthesize, SYNTHESIZE_OPTIONS)
    synthesize.add_argument(
        "--spec",
        metavar="PATH",
        help=(
            "Also write the spec the data was generated from (.yaml, or .py); "
            "with several sources, the registry of them"
        ),
    )
    synthesize.add_argument(
        "--name", help="The spec's class name (default: derived from source)"
    )
    synthesize.set_defaults(func=_cmd_synthesize)
