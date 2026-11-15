"""The pytest plugin: the `polspec_frames` fixture and `--polspec-seed`.

Registered under the `pytest11` entry point, so it loads whenever pytest
runs with polspec installed; `-p no:polspec` turns it off. It imports
pytest, which polspec itself never does.

Every test gets a seed of its own, derived from the session's seed and the
test's node id, so adding, removing or reordering tests never changes
another test's frames. The session seed is 0 unless `--polspec-seed` says
otherwise -- a number, or `random` for a new one each run -- and the report
header names it, as does the report of a failing test that polspec took
part in.

Internal: not part of the public API.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Generator
from typing import Any

import pytest

from polspec import _session

_SEED = pytest.StashKey[int]()
_RUN = pytest.StashKey[_session.TestRun]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("polspec")
    group.addoption(
        "--polspec-seed",
        default="0",
        metavar="N|random",
        help=(
            "Seed for the frames polspec generates in tests: a number, or "
            "'random' for a new one each run (default: 0)"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    given = str(config.getoption("polspec_seed", "0"))
    if given == "random":
        seed = random.SystemRandom().randrange(1 << 32)
    else:
        try:
            seed = int(given)
        except ValueError:
            raise pytest.UsageError(
                f"--polspec-seed takes a number or 'random', got {given!r}"
            ) from None
    config.stash[_SEED] = seed


def pytest_report_header(config: pytest.Config) -> str:
    seed = config.stash[_SEED]
    return f"polspec: seed {seed} (--polspec-seed={seed} repeats this run)"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: pytest.Item | None
) -> Generator[None, object, object]:
    session_seed = item.config.stash[_SEED]
    run = _session.TestRun(session_seed, _session.derived(session_seed, item.nodeid))
    item.stash[_RUN] = run
    token = _session.CURRENT.set(run)
    try:
        return (yield)
    finally:
        _session.CURRENT.reset(token)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    run = item.stash.get(_RUN, None)
    if report.failed and run is not None and run.used:
        report.sections.append(("polspec", run.rerun()))
    return report


@pytest.fixture
def polspec_frames(request: pytest.FixtureRequest) -> Callable[..., Any]:
    """Generates a frame of a spec for this test: `polspec_frames(Orders, 500,
    references={Customers: customers})` is `generate()` with a seed of the
    test's own.

    The seed is derived from the session's (`--polspec-seed`), the test's
    node id and the spec's name, and counts the calls for each spec: two
    frames of one spec in a test differ, and a frame does not change when
    another test, or another spec's frame in this one, is added. Pass `seed=`
    to choose one yourself.
    """
    from polspec import generate
    from polspec.tablespec import as_spec_name

    session_seed = request.config.stash[_SEED]
    test_seed = _session.derived(session_seed, request.node.nodeid)
    calls: Counter[str] = Counter()

    def frames(spec: Any, n: int, **options: Any) -> Any:
        _session.current()  # this test now uses polspec: a failure names the seed
        name = as_spec_name(spec)
        options.setdefault("seed", _session.derived(test_seed, name, calls[name]))
        calls[name] += 1
        return generate(spec, n, **options)

    return frames
