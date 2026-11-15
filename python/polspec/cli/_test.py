"""`polspec test`: a pytest round-trip test from a spec.

The one verb that writes code rather than data. What it emits is the
round-trip `tests/contracts/test_roundtrip.py` pins for polspec itself,
applied to the user's own spec: a frame from the `polspec_frames` fixture,
checked with `assert_valid`, with the checks generation cannot satisfy
switched off and a comment saying why. A spec with foreign keys into specs
loaded with it generates those first; one into a spec not loaded is skipped,
saying so.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from polspec import FrameSpec
from polspec.cli._io import (
    _display_path,
    _loaded_specs,
    _maybe_format,
    _relative_to_output,
    _snake_case,
)
from polspec.errors import CliError
from polspec.generation.engine import pattern_generation


def _skip_reasons(spec_cls: type[FrameSpec]) -> tuple[dict[str, bool], list[str]]:
    """validate() flags to disable, and why, so the generated test can pass.

    generate() does not attempt __checks__ or ColSpec.validators -- see
    docs/explanation/limitations.md. A round-trip test that did not account for
    this would simply fail on any spec using them, so the flag each
    constraint needs is disabled with a comment explaining why, rather than
    emitting a test the CLI already knows will not pass.

    `unique=True`, `__unique_together__` and `pattern` used to be on that
    list. They are generated now, so the generated test validates them like
    anything else -- except a pattern the generator cannot follow.
    """
    flags: dict[str, bool] = {}
    reasons: list[str] = []

    if spec_cls.spec.checks:
        flags["checks"] = False
        reasons.append(
            "__checks__ wraps arbitrary expressions that generation cannot "
            "be made to satisfy"
        )

    if spec_cls.spec.rows is not None:
        flags["rows"] = False
        reasons.append(
            "__rows__ describes the real table; the test generates its own "
            "number of rows"
        )

    if any(c.validators for c in spec_cls.spec.columns.values()):
        flags["validators"] = False
        reasons.append(
            "ColSpec.validators wraps arbitrary expressions that generation "
            "cannot be made to satisfy"
        )

    # A pattern is generated from since 1.4; only one out of the generator's
    # reach leaves text its validation refuses.
    unreachable = {
        name: reason
        for name, column in spec_cls.spec.columns.items()
        if column.pattern is not None
        for status, reason in [pattern_generation(column)]
        if status == "unreachable"
    }
    if unreachable:
        flags["pattern"] = False
        reasons.extend(
            f"the pattern of {name!r} is not generated from: {reason}"
            for name, reason in unreachable.items()
        )

    return flags, reasons


def _parents_in_order(
    spec_cls: type[FrameSpec], loaded: dict[str, tuple[str, type[FrameSpec]]]
) -> list[str] | None:
    """The specs `spec_cls`'s foreign keys reach, parents before children:
    the class names to generate first. None when one of them was not loaded
    with it, so the test has nothing to generate it from."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(cls: type[FrameSpec]) -> bool:
        for fk in cls.spec.foreign_keys:
            if fk.references == "self" or fk.references in seen:
                continue
            if fk.references not in loaded:
                return False
            seen.add(fk.references)
            parent_name, parent_cls = loaded[fk.references]
            if not visit(parent_cls):
                return False
            order.append(parent_name)
        return True

    return order if visit(spec_cls) else None


def _has_parents(spec_cls: type[FrameSpec]) -> bool:
    """Whether `spec_cls` has foreign keys into other specs."""
    return any(fk.references != "self" for fk in spec_cls.spec.foreign_keys)


def _render_test_case(
    class_name: str,
    spec_cls: type[FrameSpec],
    *,
    rows: int,
    seed: int | None,
    cartesian: bool,
    loaded: dict[str, tuple[str, type[FrameSpec]]],
) -> str:
    fn_name = _snake_case(class_name)
    parents = _parents_in_order(spec_cls, loaded)

    if parents is None:
        missing = sorted(
            {
                fk.references
                for fk in spec_cls.spec.foreign_keys
                if fk.references != "self" and fk.references not in loaded
            }
        )
        return (
            f"@pytest.mark.skip(\n"
            f"    reason=(\n"
            f'        "{class_name} has foreign key(s) into {", ".join(missing)}, "\n'
            f'        "which this file does not load. Generate the test from a file "\n'
            f'        "declaring it too, or pass its frame as "\n'
            f'        "references={{Spec: frame}} -- see docs/how-to/testing.md."\n'
            f"    )\n"
            f")\n"
            f"def test_{fn_name}_roundtrip():\n"
            f"    pass\n"
        )

    flags, reasons = _skip_reasons(spec_cls)
    kwargs = "".join(f", {name}={value}" for name, value in flags.items())
    comment = "".join(f"    # {reason}\n" for reason in reasons)
    seeded = "" if seed is None else f", seed={seed}"
    referenced = ", references=parents" if parents else ""
    by_class = dict(loaded.values())
    setup = ""
    if parents:
        setup = "    parents = {}\n" + "".join(
            f"    parents[{parent}] = polspec_frames({parent}, {rows}{seeded}"
            + (", references=parents" if _has_parents(by_class[parent]) else "")
            + ")\n"
            for parent in parents
        )

    lines = [
        f"def test_{fn_name}_roundtrip(polspec_frames):\n",
        (
            f"{comment}{setup}"
            f"    frame = polspec_frames({class_name}, {rows}{seeded}{referenced})\n"
            f"    assert_valid(frame, {class_name}{referenced}{kwargs})\n"
        ),
    ]
    if cartesian and not parents and _supports_cartesian(spec_cls):
        lines.append(
            f"\n\ndef test_{fn_name}_cartesian_coverage(polspec_frames):\n"
            f"{comment}"
            f'    frame = polspec_frames({class_name}, {rows}, method="cartesian"{seeded})\n'
            f"    assert_valid(frame, {class_name}{kwargs})\n"
        )
    elif cartesian and parents:
        lines.append(
            f"\n\n# No cartesian-coverage test: the foreign keys of {class_name} draw from\n"
            f"# their parents' rows, which a coverage set as large as its columns'\n"
            f"# combinations may outgrow.\n"
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

from polspec import FrameSpec, assert_valid

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
    loaded = {spec_cls.spec.name: (name, spec_cls) for name, spec_cls, _ in specs}
    if any(_parents_in_order(spec_cls, loaded) is None for _, spec_cls, _ in specs):
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
                loaded=loaded,
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts).rstrip() + "\n", encoding="utf-8")
    _maybe_format(output)

    print(f"Wrote {len(specs)} test spec(s) to {output}")
    return 0
