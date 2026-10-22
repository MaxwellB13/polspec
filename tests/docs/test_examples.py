"""The worked examples in `examples/` run.

CI runs `examples/related_specs.py` as a smoke test of its own; this runs
every example inside the suite too, so an API change that leaves one behind
fails here first, rather than only in CI.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.parametrize("example", sorted(EXAMPLES.glob("*.py")), ids=lambda p: p.name)
def test_an_example_runs(example: Path, capsys, monkeypatch) -> None:
    # Run from the example's own directory, as `python examples/x.py` reads
    # files beside it.
    monkeypatch.chdir(example.parent)
    runpy.run_path(str(example), run_name="__main__")
    assert capsys.readouterr().out
