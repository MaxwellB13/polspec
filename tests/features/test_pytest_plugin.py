"""The pytest plugin: the `polspec_frames` fixture, each test's own seed from
`--polspec-seed`, and a failure that says how to rerun it -- tested by
running pytest on generated test files, through pytest's `pytester`."""

import re
import subprocess
import sys

import pytest

SPECS = """
import polars as pl
import polspec
from polspec import ColSpec, FrameSpec


class Customers(FrameSpec):
    id = ColSpec(pl.Int64, bounds=(1, 1_000_000), unique=True)


class Orders(FrameSpec):
    order_id = ColSpec(pl.Int64, bounds=(1, 1_000_000), unique=True)
    total = ColSpec(pl.Float64, bounds=(0, 100))


def show(label, frame):
    print(f"\\nSHOWN {label}={frame.hash_rows().sum()}")
"""


def _shown(result: pytest.RunResult) -> dict[str, str]:
    """What the generated tests printed with `show`: label to frame digest."""
    return dict(re.findall(r"^SHOWN (\w+)=(\d+)", result.stdout.str(), re.M))


def _run(pytester: pytest.Pytester, *args: str) -> pytest.RunResult:
    return pytester.runpytest("-s", "-p", "no:cacheprovider", *args)


def test_the_fixture_generates_from_a_seed_of_the_tests_own(pytester):
    pytester.makepyfile(
        test_frames=SPECS
        + """
def test_one(polspec_frames):
    show("one", polspec_frames(Customers, 100))


def test_two(polspec_frames):
    show("two", polspec_frames(Customers, 100))
"""
    )
    first, second = _shown(_run(pytester)), _shown(_run(pytester))
    assert first == second  # every run the same
    assert first["one"] != first["two"]  # but not one test's frames another's


def test_adding_a_test_changes_no_other_tests_frames(pytester):
    body = """
def test_kept(polspec_frames):
    show("kept", polspec_frames(Customers, 100))
"""
    pytester.makepyfile(test_frames=SPECS + body)
    before = _shown(_run(pytester))
    pytester.makepyfile(
        test_frames=SPECS
        + """
def test_added_first(polspec_frames):
    show("added", polspec_frames(Customers, 100))
"""
        + body
    )
    assert _shown(_run(pytester))["kept"] == before["kept"]


def test_calls_in_a_test_are_keyed_by_spec_and_count(pytester):
    """Two frames of one spec differ; a frame of another spec, added between
    them, changes neither."""
    pytester.makepyfile(
        test_frames=SPECS
        + """
def test_calls(polspec_frames):
    show("first", polspec_frames(Customers, 100))
    show("second", polspec_frames(Customers, 100))
"""
    )
    plain = _shown(_run(pytester))
    pytester.makepyfile(
        test_frames=SPECS
        + """
def test_calls(polspec_frames):
    show("first", polspec_frames(Customers, 100))
    polspec_frames(Orders, 100)
    show("second", polspec_frames(Customers, 100))
"""
    )
    assert plain["first"] != plain["second"]
    assert _shown(_run(pytester)) == plain


def test_the_session_seed_changes_every_frame_and_a_given_seed_wins(pytester):
    pytester.makepyfile(
        test_frames=SPECS
        + """
def test_seeded(polspec_frames):
    show("drawn", polspec_frames(Customers, 100))
    show("given", polspec_frames(Customers, 100, seed=7))
    assert polspec_frames(Orders, 10, seed=7).equals(polspec.generate(Orders, 10, seed=7))
"""
    )
    zero, other = _shown(_run(pytester)), _shown(_run(pytester, "--polspec-seed=5"))
    assert zero["drawn"] != other["drawn"]
    assert zero["given"] == other["given"]


def test_the_header_names_the_seed_and_random_draws_one(pytester):
    pytester.makepyfile(test_nothing="def test_nothing():\n    pass\n")
    assert (
        "polspec: seed 0 (--polspec-seed=0 repeats this run)"
        in _run(pytester).stdout.str()
    )
    drawn = re.search(
        r"polspec: seed (\d+)", _run(pytester, "--polspec-seed=random").stdout.str()
    )
    assert drawn is not None


def test_a_seed_that_is_not_a_number_is_a_usage_error(pytester):
    result = _run(pytester, "--polspec-seed=abc")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--polspec-seed takes a number or 'random'*"])


def test_a_failing_test_that_used_polspec_says_how_to_rerun_it(pytester):
    pytester.makepyfile(
        test_frames=SPECS
        + """
def test_uses_polspec(polspec_frames):
    assert polspec_frames(Customers, 10).height == 11


def test_does_not():
    assert 1 == 2
"""
    )
    result = _run(pytester, "--polspec-seed=42")
    result.assert_outcomes(failed=2)
    output = result.stdout.str()
    uses, does_not = output.split("_ test_does_not _")
    assert "polspec seed 42 -- rerun with --polspec-seed=42" in uses
    assert "polspec seed" not in does_not


def test_assert_valid_names_the_seed_under_pytest(pytester):
    pytester.makepyfile(
        test_frames=SPECS
        + """
def test_checked(polspec_frames):
    orders = polspec_frames(Orders, 50)
    polspec.assert_valid(Orders, orders.with_columns(total=pl.lit(500.0)))
"""
    )
    result = _run(pytester)
    result.stdout.fnmatch_lines(
        [
            "*Orders: 1 finding(s) in 50 row(s) (polspec seed 0 -- rerun with --polspec-seed=0)*"
        ]
    )


def test_check_pipeline_generates_from_the_tests_seed(pytester):
    """Two tests making the same call get inputs of their own."""
    pytester.makepyfile(
        test_frames=SPECS
        + """
def same_call():
    _, inputs = polspec.check_pipeline(
        lambda orders: orders, inputs={"orders": Orders}, output=Orders, n=100
    )
    return inputs["orders"]


def test_a():
    show("a", same_call())


def test_b():
    show("b", same_call())
"""
    )
    shown = _shown(_run(pytester))
    assert shown["a"] != shown["b"]
    assert _shown(_run(pytester)) == shown


def test_the_plugin_can_be_turned_off(pytester):
    pytester.makepyfile(test_frames="def test_frames(polspec_frames):\n    pass\n")
    result = _run(pytester, "-p", "no:polspec")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*fixture 'polspec_frames' not found*"])


def test_polspec_runs_without_pytest():
    """The plugin is the only module that imports pytest, and only pytest
    loads it."""
    probe = (
        "import sys; sys.modules['pytest'] = None; import polars as pl, polspec; "
        "spec = polspec.TableSpec('T', {'a': pl.Int64}); "
        "polspec.assert_valid(spec, polspec.generate(spec, 5, seed=1)); print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "ok"
