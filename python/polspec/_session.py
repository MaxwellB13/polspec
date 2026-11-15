"""The seed of the test running now, when the pytest plugin is running it.

The plugin (`polspec._pytest`) sets a `TestRun` for each test; `assert_valid`
names its seed in a failure, and `check_pipeline` generates from it, so a
failure says how to rerun it and two tests never share their inputs. With
no plugin running there is no run, and both behave as they would anywhere.

Kept apart from the plugin and from `asserting` so neither has to import the
other: the plugin loads whenever pytest runs, and imports nothing heavy.

Internal: not part of the public API.
"""

from __future__ import annotations

import hashlib
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(slots=True)
class TestRun:
    """One test, as the plugin runs it: the session's seed, the test's own
    derived from it, and whether polspec generated or checked anything --
    which decides whether a failure's report names the seed."""

    __test__ = False  # not a test class, whatever pytest makes of its name

    session_seed: int
    seed: int
    used: bool = False

    def rerun(self) -> str:
        """How to run this test again on the same frames."""
        return f"polspec seed {self.session_seed} -- rerun with --polspec-seed={self.session_seed}"


CURRENT: ContextVar[TestRun | None] = ContextVar("polspec_test_run", default=None)


def derived(*parts: object) -> int:
    """A seed from `parts`, the same in every process and run: what a test's
    seed is made from its session's and its node id, and a frame's from its
    test's and its spec's name."""
    digest = hashlib.sha256("\0".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def current() -> TestRun | None:
    """The test running now, marked as one polspec took part in."""
    run = CURRENT.get()
    if run is not None:
        run.used = True
    return run
