"""`polspec schema infer` and `polspec schema new`."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from polspec import FrameSpec
from polspec.cli._io import (
    _class_name_from,
    _maybe_format,
    _read_data_file,
    _require_identifier,
)
from polspec.errors import CliError


def _cmd_schema_infer(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if not source.exists():
        raise CliError(f"no such file: {source}")

    df = _read_data_file(source, args.sample, infer_dates=True)
    if df.height == 0:
        raise CliError(f"{source} has no rows to profile")

    name = args.name or _class_name_from(source.stem)
    _require_identifier(name, what="--name")

    spec_cls = FrameSpec.from_dataframe(
        df,
        name=name,
        weights=args.weights,
        max_unique_enum=args.max_unique_enum,
        calculate_bounds=not args.no_bounds,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = spec_cls.to_python if output.suffix.lower() == ".py" else spec_cls.to_yaml
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        writer(output)
    for warning in caught:
        print(f"warning: {warning.message}", file=sys.stderr)
    _maybe_format(output)

    print(
        f"Inferred {len(spec_cls.spec.columns)} column(s) from "
        f"{df.height:,} row(s) of {source} -> {output}"
    )
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
