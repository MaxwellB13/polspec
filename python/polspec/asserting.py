"""`assert_valid` and `check_pipeline`: the check at the end of a pipeline
test, and the whole test in one call.

A test of a pipeline stage generates seeded inputs, runs the stage, and
validates what comes out. `assert_valid` is the last step as an assertion:
a failure is an `AssertionError` -- a failed test, in any runner, rather than
an error -- whose message names the spec, each finding, and the first rows
that fail. `check_pipeline` is the three steps at once, the inputs generated
together so a foreign key between them lines up.

Internal: not part of the public API.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Unpack, overload

import polars as pl

from polspec import _session
from polspec.errors import ValidationError
from polspec.tablespec import as_spec_name, as_table_spec
from polspec.validation import validate

if TYPE_CHECKING:
    from polspec.frames import References
    from polspec.tablespec import SpecLike
    from polspec.validation import (
        ValidationKeywords,
        ValidationOptions,
        ValidationReport,
    )

# Failing rows a message shows: enough to see the pattern, few enough to read.
SHOWN_ROWS = 10


def _failure(report: ValidationReport) -> str:
    """A failed report as an assertion message: the spec and how many rows,
    each finding, and the first rows that fail."""
    rows = report.frame.select(pl.len()).collect().item()
    run = _session.current()
    seed = "" if run is None else f" ({run.rerun()})"
    lines = [
        f"{report.spec_name}: {len(report.findings)} finding(s) in {rows:,} row(s)"
        + seed,
        *(f"  - {finding.key}: {finding.message}" for finding in report.findings),
    ]
    failing = report.failing_rows(once=True).head(SHOWN_ROWS).collect()
    if failing.height:
        with pl.Config(
            tbl_rows=SHOWN_ROWS,
            tbl_cols=-1,
            tbl_hide_dataframe_shape=True,
            tbl_hide_column_data_types=True,
            fmt_str_lengths=40,
        ):
            lines += ["First failing rows:", str(failing)]
    return "\n".join(lines)


@overload
def assert_valid(
    df: pl.DataFrame,
    spec: SpecLike,
    *,
    references: References = None,
    options: ValidationOptions | None = None,
    **keywords: Unpack[ValidationKeywords],
) -> pl.DataFrame: ...


@overload
def assert_valid(
    df: pl.LazyFrame,
    spec: SpecLike,
    *,
    references: References = None,
    options: ValidationOptions | None = None,
    **keywords: Unpack[ValidationKeywords],
) -> pl.LazyFrame: ...


def assert_valid(
    df: pl.DataFrame | pl.LazyFrame,
    spec: SpecLike,
    *,
    references: References = None,
    options: ValidationOptions | None = None,
    **keywords: Unpack[ValidationKeywords],
) -> pl.DataFrame | pl.LazyFrame:
    """Asserts that `df` meets `spec`, and returns the validated frame.

    `validate()` as an assertion, for the end of a test: it takes the same
    options, returns what `validate()` returns -- so it chains -- and on a
    failure raises `AssertionError`, which every test runner reports as a
    failed test rather than an error. The message names the spec, each
    finding by its key, and up to ten of the rows that fail, as a table.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        What the code under test produced. A `LazyFrame` comes back lazy.
    spec : TableSpec | FrameSpec class
        What it should meet.
    references : mapping, optional
        Parent frames for `spec`'s foreign keys, as `validate()` takes them.
    options : ValidationOptions, optional
        Every option at once; or pass them as keywords.

    Raises
    ------
    AssertionError
        When `df` does not meet `spec`.

    Examples
    --------
    >>> spec = TableSpec("Totals", {"total": ColSpec(pl.Float64, bounds=(0, 100))})
    >>> assert_valid(pl.DataFrame({"total": [10.0, 20.0]}), spec).height
    2
    >>> try:
    ...     assert_valid(pl.DataFrame({"total": [10.0, 250.0]}), spec)
    ... except AssertionError as failure:
    ...     print(failure)
    Totals: 1 finding(s) in 2 row(s)...
      - total__bounds: Column 'total': found 1 value(s) out of bounds [0, 100] ...
    First failing rows:
    ┌───────┬───────────────────┐
    │ total ┆ __polspec_finding │
    ╞═══════╪═══════════════════╡
    │ 250.0 ┆ total__bounds     │
    └───────┴───────────────────┘
    """
    try:
        return validate(spec, df, references=references, options=options, **keywords)
    except ValidationError as error:
        report = error.report
        message = str(error) if report is None else _failure(report)
        raise AssertionError(message) from None


def check_pipeline(
    fn: Callable[..., pl.DataFrame | pl.LazyFrame],
    inputs: Mapping[str, Any],
    output: SpecLike,
    *,
    n: int | Mapping[str, int] = 1_000,
    seed: int | None = None,
    references: References = None,
    options: ValidationOptions | None = None,
    **keywords: Unpack[ValidationKeywords],
) -> tuple[pl.DataFrame | pl.LazyFrame, dict[str, pl.DataFrame | pl.LazyFrame]]:
    """Generates `fn`'s inputs, runs it, and asserts its result meets `output`.

    Each entry of `inputs` names one of `fn`'s keyword arguments, and is a
    spec -- generated for the call -- or a frame, passed as it is. The specs
    are generated together, through a `Registry`, so a foreign key from one
    input to another lines up without being wired by hand, and each from its
    own seed derived from `seed` and its name. The result is checked with
    `assert_valid`, its foreign keys into the inputs resolved against the
    generated frames.

    Parameters
    ----------
    fn : callable
        The pipeline stage, called with the inputs as keyword arguments.
    inputs : mapping of str to spec or frame
        `fn`'s arguments: a spec to generate, or a frame to pass.
    output : TableSpec | FrameSpec class
        What `fn` must return.
    n : int or mapping of str to int, default 1,000
        Rows to generate: one count for every spec, or one per argument.
    seed : int, optional
        Seeds the inputs, so the same call is the same test every run.
        Under pytest, each test's own seed, from `--polspec-seed` and the
        test's name; elsewhere 0.
    references : mapping, optional
        Parent frames for foreign keys into specs that are not inputs.
    options : ValidationOptions, optional
        How `output` is validated; or pass the options as keywords.

    Returns
    -------
    tuple
        What `fn` returned, validated, and the inputs it was called with, by
        argument name -- for any assertion the spec cannot make.

    Raises
    ------
    AssertionError
        When the result does not meet `output`.

    Examples
    --------
    >>> customers = TableSpec("Customers", {"id": ColSpec(pl.Int64, bounds=(1, 500), unique=True)})
    >>> orders = TableSpec(
    ...     "Orders",
    ...     {"customer_id": ColSpec(pl.Int64, bounds=(1, 500)), "total": ColSpec(pl.Float64, bounds=(0, 100))},
    ...     foreign_keys=[ForeignKey("customer_id", references="Customers", ref_columns="id")],
    ... )
    >>> spend = TableSpec("Spend", {"customer_id": pl.Int64, "spend": ColSpec(pl.Float64, bounds=(0, None))})
    >>> def summarise(customers, orders):
    ...     return orders.group_by("customer_id").agg(spend=pl.col("total").sum())
    >>> result, inputs = check_pipeline(
    ...     summarise,
    ...     inputs={"customers": customers, "orders": orders},
    ...     output=spend,
    ...     n={"customers": 100, "orders": 1_000},
    ... )
    >>> inputs["orders"].height, set(result["customer_id"]) <= set(inputs["customers"]["id"])
    (1000, True)
    """
    from polspec.registry import Registry

    if seed is None:
        run = _session.current()
        seed = 0 if run is None else run.seed

    specs = {arg: value for arg, value in inputs.items() if not _is_frame(value)}
    given = {arg: value for arg, value in inputs.items() if _is_frame(value)}
    counts = _counts(n, specs)
    names = {arg: as_spec_name(spec) for arg, spec in specs.items()}
    registry = Registry(*dict.fromkeys(as_table_spec(spec) for spec in specs.values()))
    generated = (
        registry.generate_all(
            {names[arg]: count for arg, count in counts.items()},
            seed=seed,
            references=references,
        )
        if specs
        else {}
    )
    frames: dict[str, pl.DataFrame | pl.LazyFrame] = {
        arg: generated[names[arg]] for arg in specs
    } | given
    result = fn(**frames)
    parents: dict[Any, Any] = {**(references or {}), **generated}
    checked = assert_valid(
        result, output, references=parents, options=options, **keywords
    )
    return checked, frames


def _is_frame(value: object) -> bool:
    return isinstance(value, (pl.DataFrame, pl.LazyFrame))


def _counts(n: int | Mapping[str, int], specs: Mapping[str, Any]) -> dict[str, int]:
    """Each generated argument's row count, from one count or one per
    argument; a count for an argument that is not a spec to generate is a
    mistake worth naming."""
    if not isinstance(n, Mapping):
        return dict.fromkeys(specs, n)
    unknown = sorted(set(n) - set(specs))
    if unknown:
        raise TypeError(
            f"n names {unknown}, which are not inputs generated from a spec; "
            f"those are {sorted(specs)}"
        )
    missing = sorted(set(specs) - set(n))
    if missing:
        raise TypeError(f"n gives no row count for {missing}")
    return dict(n)
