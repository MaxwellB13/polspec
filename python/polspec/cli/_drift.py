"""`polspec diff` and `polspec drift`: a report, and an exit status decided
by `--fail-on`."""

from __future__ import annotations

import argparse

from polspec.cli._io import _existing, _read_data_file, _single_spec
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


def _cmd_drift(args: argparse.Namespace) -> int:
    """Measures a data file against a spec; exit 1 when the data fails it."""
    source = _existing(args.spec)
    data_path = _existing(args.data)
    spec_cls = _single_spec(source, args.cls)
    df = _read_data_file(data_path, args.sample)
    options = DriftOptions(
        null_rate_tolerance=args.null_rate_tolerance,
        unseen_values=not args.no_unseen,
        strict_dtypes=args.strict_dtypes,
        max_samples=args.max_samples,
    )
    return _print_drift(drift(spec_cls, df, options=options), args)
