"""Draw specs with awkward names and check that what polspec writes for
people parses where people read it: every ER diagram with Mermaid 11 and
10, every data dictionary's table with GitHub-flavoured Markdown rules.

    cd scripts/deep/render && npm install   # once: jsdom, markdown-it, both Mermaids
    uv run python scripts/deep/render/run.py [N]          # N drawn specs, default 400

The specs come from the property tests' own names strategy
(`tests/contracts/test_properties.py`), so a name added there is checked
here too. Mermaid is run in batches of a few files per process: its parser
slows down in a long-lived one.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tests" / "contracts")]

import test_properties as properties  # noqa: E402
from hypothesis import HealthCheck, given, settings  # noqa: E402
from polspec.render import framespec_to_markdown, framespec_to_mermaid  # noqa: E402

BATCH = 10


def dump(out: Path, n: int) -> list[Path]:
    """`n` drawn specs' diagrams and tables, and their column names."""
    written: list[Path] = []

    @settings(
        max_examples=n,
        deadline=None,
        database=None,
        suppress_health_check=list(HealthCheck),
    )
    @given(spec=properties._NAMED)
    def draw(spec) -> None:
        stem = out / f"d{len(written)}"
        stem.with_suffix(".mmd").write_text(
            framespec_to_mermaid(spec), encoding="utf-8"
        )
        stem.with_suffix(".md").write_text(
            framespec_to_markdown(spec), encoding="utf-8"
        )
        names = "\n".join(" ".join(name.splitlines()) for name in spec.columns)
        stem.with_suffix(".names").write_text(names, encoding="utf-8")
        written.append(stem)

    draw()
    return written


def check(mode: str, files: list[Path]) -> list[str]:
    node = shutil.which("node")
    if node is None:
        raise SystemExit("node is not on PATH: the render check runs Node's parsers")
    failures = []
    for start in range(0, len(files), BATCH):
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell, our own files
            [node, "check.mjs", mode, *map(str, files[start : start + BATCH])],
            cwd=HERE,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        failures += [
            line for line in result.stdout.splitlines() if line.startswith("FAIL")
        ]
    return failures


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    with tempfile.TemporaryDirectory() as directory:
        stems = dump(Path(directory), n)
        report = {
            "mermaid 11": check("mermaid", [s.with_suffix(".mmd") for s in stems]),
            "mermaid 10": check("mermaid10", [s.with_suffix(".mmd") for s in stems]),
            "tables": check("tables", [s.with_suffix(".md") for s in stems]),
        }
    for label, failures in report.items():
        print(f"{label}: {len(stems) - len(failures)} of {len(stems)} parse")
        for line in failures[:10]:
            print("   ", line)
    return 1 if any(report.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
