"""`polspec diff` and `polspec drift`: a report, and an exit status decided
by `--fail-on` -- for one spec, or with `--all` for every spec a directory
holds against the data files named after them."""

from __future__ import annotations

import argparse
import json

from polspec.cli._io import (
    _existing,
    _read_data_file,
    _registry_from,
    _single_spec,
    frames_named_after_specs,
)
from polspec.drift import DriftOptions, DriftReport, diff, drift
from polspec.errors import CliError

_FAIL_ON = ("breaking", "any", "none")


def _print_drift(report: DriftReport, args: argparse.Namespace) -> int:
    """Prints a drift report the way `--json`/`--markdown` ask, and decides
    the exit status from `--fail-on`: 1 when a finding at or above that
    level is present, 0 otherwise. `none` never fails, for posting a report
    without gating on it.
    """
    if args.json:
        print(report.to_json())
    elif args.markdown:
        print(report.to_markdown(), end="")
    else:
        print(str(report))
    if args.fail_on == "none":
        return 0
    failing = report.findings if args.fail_on == "any" else report.breaking
    return 1 if failing else 0


def _parse_rename(text: str) -> tuple[str, str]:
    old, sep, new = text.partition("=")
    if not sep or not old or not new:
        raise CliError(
            f"--rename expects OLD=NEW, got {text!r} "
            "(a column's name in the first spec, and its name in the second)"
        )
    return old, new


def _cmd_diff(args: argparse.Namespace) -> int:
    """Compares two spec files; exit 1 when the change is breaking."""
    old_cls = _single_spec(_existing(args.old), args.cls)
    new_cls = _single_spec(_existing(args.new), args.cls)
    renames = dict(_parse_rename(item) for item in args.rename or ())
    report = diff(
        old_cls,
        new_cls,
        renames=renames or None,
        options=DriftOptions(strict_dtypes=args.strict_dtypes),
    )
    return _print_drift(report, args)


def _drift_options(args: argparse.Namespace) -> DriftOptions:
    return DriftOptions(
        null_rate_tolerance=args.null_rate_tolerance,
        unseen_values=not args.no_unseen,
        strict_dtypes=args.strict_dtypes,
        max_samples=args.max_samples,
    )


def _drift_all(args: argparse.Namespace) -> int:
    """Every spec under a directory against the file named after it; exit 1
    when any report gates under `--fail-on`."""
    source = _existing(args.spec, what="file or directory")
    data_dir = _existing(args.data, what="directory")
    if not data_dir.is_dir():
        raise CliError(f"with --all, DATA is a directory of data files, got {data_dir}")
    registry = _registry_from(source)
    frames = frames_named_after_specs(registry, data_dir, source)
    options = _drift_options(args)
    reports = {
        name: drift(registry[name], frame, options=options)
        for name, frame in frames.items()
    }
    if args.json:
        print(json.dumps({n: r.to_dict() for n, r in reports.items()}, indent=2))
        status = 0
    else:
        status = 0
        for name, report in reports.items():
            print(f"== {name}")
            status |= _print_drift(report, args)
        skipped = [n for n in registry.names if n not in frames]
        if skipped:
            print(f"(no data file for: {', '.join(skipped)})")
        return status
    if args.fail_on == "none":
        return 0
    failing = any(
        (report.findings if args.fail_on == "any" else report.breaking)
        for report in reports.values()
    )
    return 1 if failing else 0


def _cmd_drift(args: argparse.Namespace) -> int:
    """Measures a data file against a spec; exit 1 when the data fails it."""
    if args.all:
        return _drift_all(args)
    source = _existing(args.spec)
    data_path = _existing(args.data)
    spec_cls = _single_spec(source, args.cls)
    df = _read_data_file(data_path, args.sample)
    return _print_drift(drift(spec_cls, df, options=_drift_options(args)), args)
