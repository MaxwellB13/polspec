"""The examples in the API reference: every public name's docstring examples
run, in a namespace holding Polars as `pl` and every public polspec name,
and print what they say they print -- so the reference cannot show a call
the API does not take, or an answer it does not give."""

import doctest

import polars as pl
import polspec
import pytest

# The public names with code of their own to show: every class and function.
# A constant, an exception or a code's `Literal` type has nothing to call.
CALLABLE = sorted(
    name
    for name in polspec.__all__
    if callable(getattr(polspec, name))
    and not (
        isinstance(getattr(polspec, name), type)
        and issubclass(getattr(polspec, name), Exception)
    )
    and name not in ("DriftCode", "FindingCode", "Severity")
)


def _namespace() -> dict:
    return {
        "pl": pl,
        "polspec": polspec,
        **{n: getattr(polspec, n) for n in polspec.__all__},
    }


@pytest.mark.parametrize("name", CALLABLE)
def test_every_public_callable_shows_an_example(name):
    examples = doctest.DocTestFinder(recurse=False).find(getattr(polspec, name), name)
    assert any(test.examples for test in examples), (
        f"{name}'s docstring has no example: add an Examples section"
    )


@pytest.mark.parametrize("name", CALLABLE)
def test_every_example_runs_as_written(name):
    finder = doctest.DocTestFinder(recurse=False)
    runner = doctest.DocTestRunner(
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
    )
    for test in finder.find(getattr(polspec, name), name, globs=_namespace()):
        runner.run(test)
    results = runner.summarize(verbose=False)
    assert results.failed == 0, f"{results.failed} example(s) in {name} failed"
