"""`polspec schema infer` and `polspec schema new`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polspec import FrameSpec
from polspec.cli._io import (
    _class_name_from,
    _declared_keys,
    _existing,
    _maybe_format,
    _note_text,
    _read_data_file,
    _refuse_related_flags,
    _require_identifier,
    _tables,
    _warnings_printed,
    _write_registry,
)
from polspec.cli._profiling import profile_keywords
from polspec.errors import CliError, PolspecError
from polspec.profiler import ProfileOptions
from polspec.related import profile_all
from polspec.serialization import registry_to_yaml, to_yaml


def _cmd_schema_infer(args: argparse.Namespace) -> int:
    sources = [_existing(text) for text in args.source]
    if len(sources) > 1:
        return _infer_related(args, sources)
    _refuse_related_flags(args)
    source = sources[0]

    df = _read_data_file(source, args.sample, infer_dates=True)
    if df.height == 0:
        raise CliError(f"{source} has no rows to profile")

    name = args.name or _class_name_from(source.stem)
    _require_identifier(name, what="--name")

    with _warnings_printed():
        try:
            spec_cls = FrameSpec.from_dataframe(
                df,
                name=name,
                **profile_keywords(args),
            )
        except (ValueError, PolspecError) as exc:  # a name in --replace the file lacks
            raise CliError(str(exc)) from exc

    _note_text(spec_cls.spec)
    summary = (
        f"Inferred {len(spec_cls.spec.columns)} column(s) from "
        f"{df.height:,} row(s) of {source}"
    )
    if args.output == "-":
        # The spec is the output: YAML on stdout, everything else on stderr,
        # so `polspec schema infer data.csv -o - > spec.yaml` holds only it.
        with _warnings_printed():
            text = to_yaml(spec_cls.spec)
        sys.stdout.write(text)
        print(summary, file=sys.stderr)
        return 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = spec_cls.to_python if output.suffix.lower() == ".py" else spec_cls.to_yaml
    with _warnings_printed():
        writer(output)
    _maybe_format(output)

    print(f"{summary} -> {output}")
    return 0


def _infer_related(args: argparse.Namespace, sources: list[Path]) -> int:
    """`schema infer` over several files: a registry, keys found between them."""
    if args.name:
        raise CliError(
            "--name names one spec; with several sources each is named after its file"
        )
    keys = _declared_keys(args)
    tables = _tables(sources)
    options = ProfileOptions(**profile_keywords(args))
    with _warnings_printed():
        try:
            registry = profile_all(tables, keys=keys, options=options)
        except (
            ValueError,
            PolspecError,
        ) as exc:  # a name in --replace no file has, an empty file
            raise CliError(str(exc)) from exc
    for name in registry:
        _note_text(registry[name])
    found = sum(len(registry[name].foreign_keys) for name in registry)
    summary = (
        f"Inferred {len(registry)} spec(s), with {found} foreign key(s) "
        f"between them, from {', '.join(str(s) for s in sources)}"
    )
    if args.output == "-":
        with _warnings_printed():
            text = registry_to_yaml(registry)
        sys.stdout.write(text)
        print(summary, file=sys.stderr)
        return 0
    output = Path(args.output)
    _write_registry(registry, output)
    print(f"{summary} -> {output}")
    return 0


_NEW_SPEC_TEMPLATE = '''"""Declares the {name} schema."""

import polars as pl
from polspec import FrameSpec


class {name}(FrameSpec):
    # Declare one ColSpec per column, in the order columns should appear.
    # Examples:
    #     id     = ColSpec(pl.Int64, bounds=(1, None), unique=True)
    #     status = ColSpec(pl.Enum(["NEW", "PAID", "SHIPPED"]))
    #     total  = ColSpec(pl.Float64, bounds=(0.0, None))
    pass
'''


def _cmd_schema_new(args: argparse.Namespace) -> int:
    _require_identifier(args.name, what="NAME")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_NEW_SPEC_TEMPLATE.format(name=args.name), encoding="utf-8")
    _maybe_format(output)
    print(f"Wrote a starter FrameSpec to {output}")
    return 0
