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
import contextlib
import importlib.util
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import polars as pl

from polspec import FrameSpec
from polspec.drift import DriftOptions, DriftReport, diff, drift
from polspec.errors import CliError, PolspecError

try:
    from importlib.metadata import version as _pkg_version

    _VERSION = _pkg_version("polspec")
except Exception:  # noqa: BLE001 - version detection must never break the CLI
    _VERSION = "unknown"

_DATA_READERS = {
    ".csv": pl.read_csv,
    ".tsv": lambda p: pl.read_csv(p, separator="\t"),
    ".parquet": pl.read_parquet,
    ".pq": pl.read_parquet,
    ".ndjson": pl.read_ndjson,
    ".jsonl": pl.read_ndjson,
    ".json": pl.read_json,
    ".arrow": pl.read_ipc,
    ".ipc": pl.read_ipc,
    ".feather": pl.read_ipc,
}


# One writer per reader, so what `generate` writes, `validate` and `drift`
# read back; a suffix in one map and not the other is a red test.
_DATA_WRITERS = {
    ".csv": pl.DataFrame.write_csv,
    ".tsv": lambda df, p: df.write_csv(p, separator="\t"),
    ".parquet": pl.DataFrame.write_parquet,
    ".pq": pl.DataFrame.write_parquet,
    ".ndjson": pl.DataFrame.write_ndjson,
    ".jsonl": pl.DataFrame.write_ndjson,
    ".json": pl.DataFrame.write_json,
    ".arrow": pl.DataFrame.write_ipc,
    ".ipc": pl.DataFrame.write_ipc,
    ".feather": pl.DataFrame.write_ipc,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _existing(path_text: str, *, what: str = "file") -> Path:
    path = Path(path_text)
    if not path.exists():
        raise CliError(f"no such {what}: {path}")
    return path


def _read_data_file(path: Path, sample: int | None) -> pl.DataFrame:
    reader = _DATA_READERS.get(path.suffix.lower())
    if reader is None:
        raise CliError(
            f"don't know how to read {path.suffix!r} files ({path}). "
            f"Supported: {', '.join(sorted(_DATA_READERS))}"
        )
    try:
        df = reader(path)
    except ImportError as exc:
        hint = ' Try: pip install "polspec[arrow]"' if "pyarrow" in str(exc) else ""
        raise CliError(f"could not read {path}: {exc}.{hint}") from exc
    except Exception as exc:
        raise CliError(f"could not read {path}: {exc}") from exc
    return df.head(sample) if sample is not None else df


def _write_data_file(df: pl.DataFrame, path: Path) -> None:
    writer = _DATA_WRITERS.get(path.suffix.lower())
    if writer is None:
        raise CliError(
            f"don't know how to write {path.suffix!r} files ({path}). "
            f"Supported: {', '.join(sorted(_DATA_WRITERS))}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer(df, path)
    except ImportError as exc:
        hint = ' Try: pip install "polspec[arrow]"' if "pyarrow" in str(exc) else ""
        raise CliError(f"could not write {path}: {exc}.{hint}") from exc


def _single_spec(source: Path, class_name: str | None) -> type[FrameSpec]:
    """The one FrameSpec a file defines, or the one `--class` picks out."""
    specs = _loaded_specs(source, class_name, source)
    if len(specs) != 1:
        names = ", ".join(name for name, _, _ in specs)
        raise CliError(
            f"{source} defines several specs ({names}); pick one with --class"
        )
    return specs[0][1]


def _references_from(items: list[str] | None) -> dict[str, pl.DataFrame] | None:
    """`--references NAME=PATH ...` as the mapping `references=` takes."""
    references: dict[str, pl.DataFrame] = {}
    for item in items or ():
        name, path = _parse_reference(item)
        if not path.exists():
            raise CliError(f"no such file for reference {name!r}: {path}")
        references[name] = _read_data_file(path, None)
    return references or None


def _class_name_from(text: str) -> str:
    """A PascalCase identifier from an arbitrary file stem or name."""
    words = re.findall(r"[A-Za-z0-9]+", text) or ["Spec"]
    name = "".join(w[:1].upper() + w[1:] for w in words)
    return name if name[0].isalpha() else f"Spec{name}"


def _snake_case(name: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
    return s.lower()


def _require_identifier(name: str, *, what: str) -> None:
    if not name.isidentifier():
        raise CliError(f"{what} {name!r} is not a valid Python identifier")


def _load_module_from_path(path: Path) -> object:
    module_name = f"_polspec_cli_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise CliError(f"could not load {path} as a Python module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise CliError(f"error importing {path}: {exc}") from exc
    return module


def _frame_specs_in_module(module: object) -> dict[str, type[FrameSpec]]:
    """FrameSpec subclasses this module itself defines, excluding imports."""
    return {
        name: value
        for name, value in vars(module).items()
        if isinstance(value, type)
        and issubclass(value, FrameSpec)
        and value is not FrameSpec
        and value.__module__ == module.__name__
    }


def _maybe_format(path: Path) -> None:
    """Runs ruff format on generated Python, best-effort.

    Not fatal if ruff is missing -- the file is already valid Python without
    it, just less consistently spaced.
    """
    if path.suffix != ".py":
        return
    with contextlib.suppress(OSError):
        subprocess.run(  # noqa: S603 - fixed argv, no shell, path came from our own writer
            [sys.executable, "-m", "ruff", "format", str(path)],
            check=False,
            capture_output=True,
        )


# ---------------------------------------------------------------------------
# schema infer
# ---------------------------------------------------------------------------


def _cmd_schema_infer(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if not source.exists():
        raise CliError(f"no such file: {source}")

    df = _read_data_file(source, args.sample)
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


# ---------------------------------------------------------------------------
# schema new
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _parse_reference(text: str) -> tuple[str, Path]:
    """`Customers=customers.parquet` -> ("Customers", Path("customers.parquet"))."""
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise CliError(
            f"--references expects NAME=PATH, got {text!r} "
            "(the spec name a foreign key points at, and a data file for it)"
        )
    return name, Path(path)


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


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# diff and drift
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------


def _loaded_specs(
    source: Path, class_name: str | None, output: Path
) -> list[tuple[str, type[FrameSpec], str]]:
    """The FrameSpec classes to generate tests for, and how to load each in
    the generated file.

    Returns (name, class, loader_snippet) triples, where the snippet is
    Python source that binds `name` in the generated test module. Loading
    happens exactly once here -- a .py source is only ever imported a single
    time, so any side effect its import causes only happens once.
    """
    rel = _relative_to_output(source, output)

    if source.suffix.lower() in (".yaml", ".yml"):
        spec_cls = FrameSpec.from_yaml(source)
        name = class_name or spec_cls.__name__
        loader = f"{name} = FrameSpec.from_yaml(Path(__file__).parent / {rel!r})"
        return [(name, spec_cls, loader)]

    if source.suffix.lower() == ".py":
        module = _load_module_from_path(source)
        found = _frame_specs_in_module(module)
        if class_name is not None:
            if class_name not in found:
                raise CliError(
                    f"no FrameSpec class {class_name!r} in {source} "
                    f"(found: {', '.join(sorted(found)) or 'none'})"
                )
            found = {class_name: found[class_name]}
        if not found:
            raise CliError(f"no FrameSpec subclasses defined in {source}")
        return [
            (
                name,
                spec_cls,
                f"{name} = _load_spec_module(Path(__file__).parent / {rel!r}).{name}",
            )
            for name, spec_cls in found.items()
        ]

    raise CliError(
        f"don't know how to load a spec from {source.suffix!r} files "
        f"({source}). Expected .yaml, .yml or .py"
    )


def _relative_to_output(path: Path, output: Path) -> str:
    """`path` relative to where `output` will live, else absolute.

    The generated test resolves this path against `Path(__file__).parent` at
    *its own* run time -- so the reference point has to be the output file's
    directory, not the current working directory the CLI happens to run
    from. Falls back to an absolute path when the two are on different
    drives, where no relative path exists.
    """
    try:
        return os.path.relpath(path.resolve(), start=output.resolve().parent)
    except ValueError:
        return str(path.resolve())


def _display_path(path: str) -> str:
    """Forward slashes, for a path shown inside a plain string body.

    A Windows path embedded raw between quotes turns a run like `\\U` into
    the start of a unicode escape, which fails to parse when Python
    re-reads the file it just wrote. Anywhere a path is quoted through
    `!r` this is not a concern -- `repr()` already escapes it -- but a
    docstring or comment inserts the text directly, so it needs to already
    be escape-free.
    """
    return path.replace("\\", "/")


def _skip_reasons(spec_cls: type[FrameSpec]) -> tuple[dict[str, bool], list[str]]:
    """validate() flags to disable, and why, so the generated test can pass.

    generate() does not attempt __checks__ or ColSpec.validators -- see
    docs/explanation/limitations.md. A round-trip test that did not account for
    this would simply fail on any spec using them, so the flag each
    constraint needs is disabled with a comment explaining why, rather than
    emitting a test the CLI already knows will not pass.

    `unique=True` and `__unique_together__` used to be on that list. They are
    generated now, so the generated test validates them like anything else.
    """
    flags: dict[str, bool] = {}
    reasons: list[str] = []

    if spec_cls.spec.checks:
        flags["validate_checks"] = False
        reasons.append(
            "__checks__ wraps arbitrary expressions that generation cannot "
            "be made to satisfy"
        )

    if any(c.validators for c in spec_cls.spec.columns.values()):
        flags["validate_validators"] = False
        reasons.append(
            "ColSpec.validators wraps arbitrary expressions that generation "
            "cannot be made to satisfy"
        )

    return flags, reasons


def _render_test_case(
    class_name: str,
    spec_cls: type[FrameSpec],
    *,
    rows: int,
    seed: int,
    cartesian: bool,
) -> str:
    fn_name = _snake_case(class_name)
    cross_spec_fks = [
        fk for fk in spec_cls.spec.foreign_keys if fk.references != "self"
    ]

    if cross_spec_fks:
        names = ", ".join(repr(fk.name) for fk in cross_spec_fks)
        return (
            f"@pytest.mark.skip(\n"
            f"    reason=(\n"
            f'        "{class_name} has foreign key(s) {names} referencing another "\n'
            f'        "FrameSpec. generate()/validate() need a parent DataFrame via "\n'
            f'        "references={{OtherSpec: parent_df}} -- see "\n'
            f'        "docs/how-to/constraints.md#referential-integrity-foreignkey."\n'
            f"    )\n"
            f")\n"
            f"def test_{fn_name}_roundtrip():\n"
            f"    pass\n"
        )

    flags, reasons = _skip_reasons(spec_cls)
    kwargs = "".join(f", {name}={value}" for name, value in flags.items())
    comment = "".join(f"    # {reason}\n" for reason in reasons)

    lines = [
        f"def test_{fn_name}_roundtrip():\n",
        (
            f"{comment}"
            f"    df = {class_name}.generate({rows}, seed={seed})\n"
            f"    {class_name}.validate(df{kwargs})\n"
        ),
    ]
    if cartesian and _supports_cartesian(spec_cls):
        lines.append(
            f"\n\ndef test_{fn_name}_cartesian_coverage():\n"
            f"{comment}"
            f'    df = {class_name}.generate({rows}, method="cartesian", seed={seed})\n'
            f"    {class_name}.validate(df{kwargs})\n"
        )
    elif cartesian:
        lines.append(
            f'\n\n# No cartesian-coverage test: method="cartesian" needs at least one\n'
            f"# Enum, Boolean, or bounded numeric column to build coverage from, and\n"
            f"# {class_name} has none.\n"
        )
    return "".join(lines)


def _supports_cartesian(spec_cls: type[FrameSpec]) -> bool:
    """Whether `method="cartesian"` has anything to build coverage from.

    Rather than re-deriving `_coverage_values`' eligibility rule here (and
    risk it silently drifting from the real one), this asks the engine
    directly with a throwaway single-row generation.
    """
    try:
        spec_cls.generate(1, method="cartesian", seed=0)
    except ValueError:
        return False
    return True


_TEST_MODULE_HEADER = '''"""Generated by `polspec test {source}`.

Regenerate with:

    polspec test {source} -o {output}

This file is only overwritten by running that command again -- edit freely.
"""

from pathlib import Path

from polspec import FrameSpec

'''

_PY_LOADER_HELPER = """

def _load_spec_module(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
"""


def _cmd_test(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if not source.exists():
        raise CliError(f"no such file: {source}")
    output = Path(args.output)

    specs = _loaded_specs(source, args.cls, output)

    parts = [
        _TEST_MODULE_HEADER.format(
            source=_display_path(_relative_to_output(source, output)),
            output=_display_path(str(output)),
        )
    ]
    if source.suffix.lower() == ".py":
        parts.append(_PY_LOADER_HELPER)
    if any(
        fk.references != "self"
        for _, spec_cls, _ in specs
        for fk in spec_cls.spec.foreign_keys
    ):
        parts.append("\nimport pytest\n")

    for class_name, spec_cls, loader_snippet in specs:
        parts.append(f"\n{loader_snippet}\n")
        parts.append(
            "\n\n"
            + _render_test_case(
                class_name,
                spec_cls,
                rows=args.rows,
                seed=args.seed,
                cartesian=not args.no_cartesian,
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts).rstrip() + "\n", encoding="utf-8")
    _maybe_format(output)

    print(f"Wrote {len(specs)} test spec(s) to {output}")
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


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
    validate.add_argument("spec", help="A .yaml/.yml spec, or a .py file defining one")
    validate.add_argument("data", help="Path to a CSV, Parquet, NDJSON or IPC file")
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
        "--json",
        action="store_true",
        help="Print the report as JSON instead of text",
    )
    validate.set_defaults(func=_cmd_validate)

    generate = subparsers.add_parser(
        "generate", help="Generate rows from a schema into a data file"
    )
    generate.add_argument("spec", help="A .yaml/.yml spec, or a .py file defining one")
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
        "spec", help="A .yaml/.yml spec, or a .py file defining one"
    )
    drift_parser.add_argument("data", help="Path to a CSV, Parquet, NDJSON or IPC file")
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
