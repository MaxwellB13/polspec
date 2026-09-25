"""``polspec`` command-line interface.

Two things a spec is useful for beyond Python code: turning existing data into
a starting declaration, and turning a declaration into a test that would have
caught this session's own round-trip bugs. Both are thin wrappers over
`FrameSpec` methods that already exist -- `from_dataframe`, `to_yaml`,
`to_python`, `generate`, `validate` -- so this module's job is argument
parsing and templating, not new behaviour.

    polspec schema infer orders.parquet -o orders.yaml
    polspec schema infer orders.parquet -o orders.py
    polspec schema new Orders -o orders.py
    polspec test orders.yaml -o test_orders.py
    polspec validate orders.yaml orders.parquet --references Customers=customers.parquet
    polspec generate orders.yaml -n 1000 -o orders.parquet --seed 1
    polspec diff orders_v1.yaml orders_v2.yaml --markdown
    polspec drift orders.yaml orders.parquet --fail-on breaking
"""

from __future__ import annotations

import argparse
import sys

from polspec.cli._data import _cmd_generate, _cmd_validate
from polspec.cli._drift import _FAIL_ON, _cmd_diff, _cmd_drift
from polspec.cli._schema import _cmd_schema_infer, _cmd_schema_new
from polspec.cli._test import _cmd_test
from polspec.errors import PolspecError
from polspec.validation import _SWITCHES

try:
    from importlib.metadata import version as _pkg_version

    _VERSION = _pkg_version("polspec")
except Exception:  # noqa: BLE001 - version detection must never break the CLI
    _VERSION = "unknown"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polspec",
        description=(
            "Generate a schema from data, a test from a schema, or check data "
            "against a schema."
        ),
    )
    parser.add_argument("--version", action="version", version=f"polspec {_VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema = subparsers.add_parser("schema", help="Create or infer a FrameSpec")
    schema_sub = schema.add_subparsers(dest="schema_command", required=True)

    infer = schema_sub.add_parser(
        "infer", help="Profile a data file into a YAML schema"
    )
    infer.add_argument("source", help="Path to a CSV, Parquet, NDJSON or IPC file")
    infer.add_argument(
        "-o",
        "--output",
        required=True,
        help="File to write: .yaml/.yml, or .py for a FrameSpec subclass",
    )
    infer.add_argument("--name", help="Spec class name (default: derived from source)")
    infer.add_argument(
        "--weights",
        action="store_true",
        help="Record each category's observed frequency",
    )
    infer.add_argument(
        "--max-unique-enum",
        type=int,
        default=50,
        metavar="N",
        help="Max distinct values for a string column to become an Enum (default: 50)",
    )
    infer.add_argument(
        "--no-bounds",
        action="store_true",
        help="Skip computing numeric/temporal bounds and string lengths",
    )
    infer.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Profile only the first N rows",
    )
    infer.set_defaults(func=_cmd_schema_infer)

    new = schema_sub.add_parser("new", help="Write a blank FrameSpec to edit by hand")
    new.add_argument("name", help="Spec class name, e.g. Orders")
    new.add_argument("-o", "--output", required=True, help="Python file to write")
    new.set_defaults(func=_cmd_schema_new)

    test = subparsers.add_parser(
        "test", help="Generate a pytest round-trip test from a schema"
    )
    test.add_argument("source", help="A .yaml/.yml spec, or a .py file defining one")
    test.add_argument("-o", "--output", required=True, help="Test file to write")
    test.add_argument(
        "--rows",
        type=int,
        default=500,
        metavar="N",
        help="Rows to generate (default: 500)",
    )
    test.add_argument(
        "--seed", type=int, default=42, help="Generation seed (default: 42)"
    )
    test.add_argument(
        "--no-cartesian",
        action="store_true",
        help="Skip the coverage-guaranteeing cartesian test",
    )
    test.add_argument(
        "--class",
        dest="cls",
        metavar="NAME",
        help="Generate a test for only this class (a .py source may define several)",
    )
    test.set_defaults(func=_cmd_test)

    validate = subparsers.add_parser(
        "validate", help="Check a data file against a schema"
    )
    validate.add_argument(
        "spec",
        help="A .yaml/.yml spec, or a .py file defining one; with --all, a directory",
    )
    validate.add_argument(
        "data",
        help=(
            "Path to a CSV, Parquet, NDJSON or IPC file; with --all, a directory "
            "of files named after the specs"
        ),
    )
    validate.add_argument(
        "--all",
        action="store_true",
        help=(
            "Every spec found under SPEC against DATA/<name>.<suffix>, each "
            "seeing the others as parents"
        ),
    )
    validate.add_argument(
        "--references",
        action="append",
        metavar="NAME=PATH",
        help=(
            "Parent data for a foreign key to another spec, as the spec's name "
            "and a data file; repeat for several"
        ),
    )
    validate.add_argument(
        "--class",
        dest="cls",
        metavar="NAME",
        help="Validate against only this class (a .py source may define several)",
    )
    validate.add_argument(
        "--allow-extra",
        action="store_true",
        help="Do not report columns the spec does not declare",
    )
    validate.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not report declared columns the data lacks",
    )
    validate.add_argument(
        "--strict-dtypes",
        action="store_true",
        help="Require exact dtypes rather than compatible ones",
    )
    validate.add_argument(
        "--skip",
        action="append",
        choices=_SWITCHES,
        metavar="CHECK",
        help=(
            "Turn off one kind of check, as validate_CHECK=False does; repeat "
            f"for several. One of: {', '.join(_SWITCHES)}"
        ),
    )
    validate.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON instead of text",
    )
    validate.add_argument(
        "--output",
        metavar="PATH",
        help=(
            "Write the rows that passed, typed as the spec declares, to PATH "
            "(the format by its extension)"
        ),
    )
    validate.add_argument(
        "--failing",
        metavar="PATH",
        help="Write the rows that failed, with the finding each broke, to PATH",
    )
    validate.set_defaults(func=_cmd_validate)

    generate = subparsers.add_parser(
        "generate", help="Generate rows from a schema into a data file"
    )
    generate.add_argument(
        "spec",
        help="A .yaml/.yml spec, or a .py file defining one; with --all, a directory",
    )
    generate.add_argument(
        "--all",
        action="store_true",
        help=(
            "Every spec found under SPEC, parents first, to -o/--output as a "
            "directory of <name> files in --format"
        ),
    )
    generate.add_argument(
        "--format",
        default="parquet",
        metavar="EXT",
        help="With --all, the file format to write (default: parquet)",
    )
    generate.add_argument(
        "-n", "--rows", type=int, required=True, metavar="N", help="Rows to generate"
    )
    generate.add_argument(
        "-o",
        "--output",
        required=True,
        help="File to write; the extension picks the format (.parquet, .csv, ...)",
    )
    generate.add_argument("--seed", type=int, help="Generation seed (default: random)")
    generate.add_argument(
        "--method",
        choices=("random", "cartesian"),
        default="random",
        help="Sampling method (default: random)",
    )
    generate.add_argument(
        "--references",
        action="append",
        metavar="NAME=PATH",
        help="Parent data for a foreign key to another spec; repeat for several",
    )
    generate.add_argument(
        "--class",
        dest="cls",
        metavar="NAME",
        help="Generate from only this class (a .py source may define several)",
    )
    generate.set_defaults(func=_cmd_generate)

    def add_report_options(sub: argparse.ArgumentParser) -> None:
        output = sub.add_mutually_exclusive_group()
        output.add_argument(
            "--json", action="store_true", help="Print the report as JSON"
        )
        output.add_argument(
            "--markdown",
            action="store_true",
            help="Print the report as Markdown, for a pull-request comment",
        )
        sub.add_argument(
            "--fail-on",
            choices=_FAIL_ON,
            default="breaking",
            help=(
                "Which findings make the exit status 1: breaking (default), "
                "any, or none"
            ),
        )
        sub.add_argument(
            "--strict-dtypes",
            action="store_true",
            help="A dtype change is breaking unless the dtypes are identical",
        )
        sub.add_argument(
            "--class",
            dest="cls",
            metavar="NAME",
            help="Use only this class (a .py source may define several)",
        )

    diff_parser = subparsers.add_parser(
        "diff", help="What changed between two schemas, and whether it breaks"
    )
    diff_parser.add_argument("old", help="The earlier spec: .yaml/.yml, or a .py file")
    diff_parser.add_argument("new", help="The later spec")
    diff_parser.add_argument(
        "--rename",
        action="append",
        metavar="OLD=NEW",
        help="A column renamed between the two; repeat for several",
    )
    add_report_options(diff_parser)
    diff_parser.set_defaults(func=_cmd_diff)

    drift_parser = subparsers.add_parser(
        "drift", help="How a data file has moved relative to its schema"
    )
    drift_parser.add_argument(
        "spec",
        help="A .yaml/.yml spec, or a .py file defining one; with --all, a directory",
    )
    drift_parser.add_argument(
        "data",
        help=(
            "Path to a CSV, Parquet, NDJSON or IPC file; with --all, a directory "
            "of files named after the specs"
        ),
    )
    drift_parser.add_argument(
        "--all",
        action="store_true",
        help="Every spec found under SPEC against DATA/<name>.<suffix>",
    )
    drift_parser.add_argument(
        "--sample", type=int, metavar="N", help="Measure only the first N rows"
    )
    drift_parser.add_argument(
        "--null-rate-tolerance",
        type=float,
        default=0.05,
        metavar="F",
        help="How far the null rate may sit from null_probability (default: 0.05)",
    )
    drift_parser.add_argument(
        "--no-unseen",
        action="store_true",
        help="Do not report declared values the data never holds",
    )
    drift_parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        metavar="N",
        help="Offending values to carry per finding (default: 10)",
    )
    add_report_options(drift_parser)
    drift_parser.set_defaults(func=_cmd_drift)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PolspecError as exc:
        # The library's own complaints are the user's to fix, and their messages
        # already say how; a traceback would only bury that.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level CLI error boundary
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
