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
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import polars as pl

from polspec import FrameSpec

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


class CliError(Exception):
    """A problem worth reporting as `error: ...` rather than a traceback."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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
    try:
        subprocess.run(
            [sys.executable, "-m", "ruff", "format", str(path)],
            check=False,
            capture_output=True,
        )
    except OSError:
        pass


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
        f"Inferred {len(spec_cls._columns)} column(s) from "
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

    generate() does not attempt unique=True, __unique_together__, __checks__
    or ColSpec.validators -- see docs/reference/limitations.md. A round-trip
    test that did not account for this would simply fail on any spec using
    them, so the flag each constraint needs is disabled with a comment
    explaining why, rather than emitting a test the CLI already knows will
    not pass.
    """
    flags: dict[str, bool] = {}
    reasons: list[str] = []

    has_unique = any(c.unique for c in spec_cls._columns.values())
    has_unique_together = bool(spec_cls._unique_together)
    if has_unique or has_unique_together:
        flags["validate_unique"] = False
        reasons.append(
            "unique=True / __unique_together__ is validated but not yet "
            "generated (see docs/reference/limitations.md)"
        )

    if spec_cls._checks:
        flags["validate_checks"] = False
        reasons.append(
            "__checks__ wraps arbitrary expressions that generation cannot "
            "be made to satisfy"
        )

    if any(c.validators for c in spec_cls._columns.values()):
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
    cross_spec_fks = [fk for fk in spec_cls._foreign_keys if fk.references != "self"]

    if cross_spec_fks:
        names = ", ".join(repr(fk.name) for fk in cross_spec_fks)
        return (
            f"@pytest.mark.skip(\n"
            f"    reason=(\n"
            f'        "{class_name} has foreign key(s) {names} referencing another "\n'
            f'        "FrameSpec. generate()/validate() need a parent DataFrame via "\n'
            f'        "references={{OtherSpec: parent_df}} -- see "\n'
            f'        "docs/guide/constraints.md#referential-integrity-foreignkey."\n'
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
        for fk in spec_cls._foreign_keys
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
        description="Generate a schema from data, or a test from a schema.",
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level CLI error boundary
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
