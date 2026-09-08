"""Measuring polspec's generator, reproducibly.

Two jobs, one harness:

- ``compare`` reproduces the table in ``docs/explanation/comparison.md``: the
  same four-column frame built by polspec, by hand-vectorised NumPy, and by a
  pure-Python loop.
- ``record`` and ``check`` guard against regressions: every case below is
  measured and written to JSON, and a later run is compared against it.

Three things make the numbers worth trusting, each of them a lesson from a
measurement that lied:

- **Every measurement gets its own process.** Generating twenty million
  strings leaves the allocator and the page cache in a state that makes the
  *next* measurement in the same process read differently -- an ``int64``
  column measured 0.0100s after a string benchmark and 0.0060s on its own.
- **Every measurement is repeated**, and reported as a minimum alongside its
  spread. One timed run of the four-column case varies by around 25% between
  runs, which is wider than most changes worth detecting.
- **Every run records the machine and the build.** ``codegen-units = 1`` alone
  moves the ``unique`` path by 2.1x, so a number without its cargo profile
  beside it cannot be compared against a number from last week.

That last point is a warning as much as a feature: ``check`` compares against a
baseline recorded on *some* machine, and only a baseline from *this* machine,
at this build profile, means anything. It says so when they disagree.

A baseline is therefore local, not committed. The workflow is to record one
before touching anything and compare against it afterwards:

    uv run --group bench python benchmarks/bench.py record   # before
    ... make the change, rebuild ...
    uv run --group bench python benchmarks/bench.py check    # after

`check` exits non-zero if any case regressed by more than `--tolerance`.
While iterating, `--suite quick` covers one case per engine branch, and
`--only NAME ...` narrows further.

The published comparison table comes from:

    uv run --group bench python benchmarks/bench.py compare
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from polspec import Bound, ColRule, ColSpec, ForeignKey, FrameSpec, Hierarchy, col

ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).resolve().parent / "baseline.json"
SEED = 42

# A case has to regress by more than this before `check` calls it a regression.
# Generous on purpose: below it the harness is measuring the machine.
DEFAULT_TOLERANCE = 0.20

# A fast case keeps repeating until it has spent this long in total, so that
# its minimum settles rather than tracking whatever else the scheduler did.
MIN_SAMPLE_SECONDS = 0.5
MAX_REPEAT = 50


# ---------------------------------------------------------------------------
# The specs each case generates from
# ---------------------------------------------------------------------------

CATEGORIES = ["mammal", "reptile", "insect"]
NULL_P = 0.1
INT_LO, INT_HI = -100, 100
FLOAT_LO, FLOAT_HI = -2_000.0, 2_000.0
STR_MIN_LEN, STR_MAX_LEN = 5, 15
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


class DataSource(FrameSpec):
    """The four-column frame the published comparison table is built from."""

    string_1 = ColSpec(pl.String, nullable=False)
    enum_1 = ColSpec(pl.Enum(CATEGORIES), nullable=True, null_probability=NULL_P)
    int_1 = ColSpec(
        pl.Int64, bounds=Bound(INT_LO, INT_HI), nullable=True, null_probability=NULL_P
    )
    float_1 = ColSpec(
        pl.Float64,
        bounds=(FLOAT_LO, FLOAT_HI),
        nullable=True,
        null_probability=NULL_P,
    )


class OneString(FrameSpec):
    c = ColSpec(pl.String, string_length=(STR_MIN_LEN, STR_MAX_LEN))


class OneInt(FrameSpec):
    c = ColSpec(
        pl.Int64, bounds=Bound(INT_LO, INT_HI), nullable=True, null_probability=NULL_P
    )


class OneFloat(FrameSpec):
    c = ColSpec(
        pl.Float64,
        bounds=(FLOAT_LO, FLOAT_HI),
        nullable=True,
        null_probability=NULL_P,
    )


class OneEnum(FrameSpec):
    c = ColSpec(pl.Enum(CATEGORIES), nullable=True, null_probability=NULL_P)


class OneBool(FrameSpec):
    c = ColSpec(pl.Boolean, nullable=True, null_probability=NULL_P)


class OneDatetime(FrameSpec):
    """Temporal columns cross the boundary as integers and are cast back."""

    c = ColSpec(pl.Datetime("us"))


# Two unique cases, because the engine takes a different branch for each: a
# roomy domain draws and rejects, a crowded one uses Floyd's algorithm. A
# change that speeds one up and ruins the other is invisible with only one.
class UniqueRoomy(FrameSpec):
    c = ColSpec(pl.Int64, bounds=(0, 10_000_000_000), unique=True)


class UniqueCrowded(FrameSpec):
    c = ColSpec(pl.Int64, bounds=(0, 12_000_000), unique=True)


class UniqueString(FrameSpec):
    c = ColSpec(pl.String, string_length=(8, 12), unique=True)


class Wide(FrameSpec):
    __columns__ = {f"c{i}": ColSpec(pl.Int64, bounds=(0, 1000)) for i in range(200)}


class Coverage(FrameSpec):
    """Small enough that the cartesian product is padding, not an explosion."""

    status = ColSpec(pl.Enum(["NEW", "PAID", "SHIPPED"]))
    flag = ColSpec(pl.Boolean)
    amount = ColSpec(pl.Int64, bounds=(-50, 50), nullable=True)


class Ruled(FrameSpec):
    """A post-generation pass: rules rewrite the rows their condition matches."""

    region = ColSpec(pl.Enum(["UK", "US", "EU"]))
    carrier = ColSpec(
        pl.String,
        choices=["RM", "UPS", "DHL"],
        rules=(
            ColRule(when=col("region") == "UK", choices=["RM"]),
            ColRule(when=col("region").is_in(["US", "EU"]), choices=["UPS", "DHL"]),
        ),
    )


class SelfKeyed(FrameSpec):
    """A self-referencing foreign key, sampled from the frame as it stands."""

    id = ColSpec(pl.Int64, unique=True, bounds=(0, 100_000_000))
    manager_id = ColSpec(pl.Int64, nullable=True)
    __foreign_keys__ = [ForeignKey("manager_id", references="self", ref_columns="id")]


class Linked(FrameSpec):
    """A link table: one pool of references wired into a forest of depth 5."""

    PARENT_REF = ColSpec(pl.String)
    CHILD_REF = ColSpec(pl.String)
    __hierarchy__ = Hierarchy(child="CHILD_REF", parent="PARENT_REF", max_depth=5)


class Composite(FrameSpec):
    """A composite key, whose repair resamples the rows that repeat."""

    a = ColSpec(pl.Int64, bounds=(0, 5_000))
    b = ColSpec(pl.Int64, bounds=(0, 5_000))
    __unique_together__ = ["a", "b"]


# ---------------------------------------------------------------------------
# The reference implementations, for `compare`
# ---------------------------------------------------------------------------


def generate_numpy(n: int) -> pl.DataFrame:
    """Hand-vectorised NumPy.

    The string column uses a fixed width where polspec's is ragged 5-15:
    NumPy has no efficient way to vectorise per-row lengths, and writing the
    slow version would not be the fair comparison either. Strings dominate this
    frame's cost, so read the headline ratio with that in mind.
    """
    import numpy as np

    rng = np.random.default_rng(SEED)
    charset = np.frombuffer(CHARSET.encode("ascii"), dtype=np.uint8)

    idx = rng.integers(0, len(charset), size=(n, STR_MAX_LEN), dtype=np.uint8)
    string_1 = np.char.decode(charset[idx].view(f"S{STR_MAX_LEN}").reshape(-1), "ascii")

    enum_1 = rng.choice(np.array(CATEGORIES, dtype=object), size=n)
    enum_1[rng.random(n) < NULL_P] = None

    int_1 = rng.integers(INT_LO, INT_HI + 1, size=n).astype(np.float64)
    int_1[rng.random(n) < NULL_P] = np.nan

    float_1 = rng.uniform(FLOAT_LO, FLOAT_HI, size=n)
    float_1[rng.random(n) < NULL_P] = np.nan

    df = pl.DataFrame(
        {"string_1": string_1, "enum_1": enum_1, "int_1": int_1, "float_1": float_1}
    )
    return df.with_columns(
        pl.col("enum_1").cast(pl.Enum(CATEGORIES)),
        pl.col("int_1").cast(pl.Int64, strict=False),
    )


def generate_python(n: int) -> pl.DataFrame:
    """A pure-Python loop over `random`, the version most people would write."""
    import random

    rng = random.Random(SEED)
    string_1: list[str | None] = [None] * n
    enum_1: list[str | None] = [None] * n
    int_1: list[int | None] = [None] * n
    float_1: list[float | None] = [None] * n

    for i in range(n):
        length = rng.randint(STR_MIN_LEN, STR_MAX_LEN)
        string_1[i] = "".join(rng.choices(CHARSET, k=length))
        enum_1[i] = None if rng.random() < NULL_P else rng.choice(CATEGORIES)
        int_1[i] = None if rng.random() < NULL_P else rng.randint(INT_LO, INT_HI)
        float_1[i] = None if rng.random() < NULL_P else rng.uniform(FLOAT_LO, FLOAT_HI)

    df = pl.DataFrame(
        {
            "string_1": pl.Series("string_1", string_1, dtype=pl.String),
            "enum_1": pl.Series("enum_1", enum_1, dtype=pl.String),
            "int_1": pl.Series("int_1", int_1, dtype=pl.Int64),
            "float_1": pl.Series("float_1", float_1, dtype=pl.Float64),
        }
    )
    return df.with_columns(pl.col("enum_1").cast(pl.Enum(CATEGORIES)))


def _validate_cyclic_hierarchy(n: int) -> None:
    """Validating a frame built to contain cycles.

    Its own case because the walk is the part with a cliff in it: advancing a
    lazy frame along itself nests its plan, and an implementation that slipped
    back into doing so took half a minute over fifty thousand rows rather than
    the twenty milliseconds it takes now.
    """
    from polspec import inspect as inspect_frame

    inspect_frame(Linked.spec, Linked.generate(n, seed=SEED, cycles=10))


def _sink_parquet(n: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        DataSource.sink_parquet(Path(tmp) / "out.parquet", n, batch_size=100_000)


# ---------------------------------------------------------------------------
# The case registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One thing to measure, at one or more row counts.

    `suite` decides when it runs: "compare" cases build the published table,
    "regression" cases guard the engine. `run` takes a row count and does the
    work being timed, setup included -- if a caller has to pay for it, it is
    part of the measurement.
    """

    name: str
    suite: str
    sizes: tuple[int, ...]
    run: Callable[[int], Any]


def _polspec(spec: type[FrameSpec], **kwargs: Any) -> Callable[[int], Any]:
    return lambda n: spec.generate(n, seed=SEED, **kwargs)


CASES: tuple[Case, ...] = (
    # The published comparison: one frame, three implementations.
    Case(
        "compare:polspec",
        "compare",
        (1_000, 10_000, 100_000, 1_000_000, 5_000_000, 20_000_000),
        _polspec(DataSource),
    ),
    Case(
        "compare:numpy",
        "compare",
        (1_000, 10_000, 100_000, 1_000_000, 5_000_000, 20_000_000),
        generate_numpy,
    ),
    Case(
        "compare:python",
        "compare",
        (1_000, 10_000, 100_000, 1_000_000),
        generate_python,
    ),
    # One column at a time, so a regression names the kind that caused it.
    Case("string", "regression", (20_000_000,), _polspec(OneString)),
    Case("int64", "regression", (20_000_000,), _polspec(OneInt)),
    Case("float64", "regression", (20_000_000,), _polspec(OneFloat)),
    Case("enum", "regression", (20_000_000,), _polspec(OneEnum)),
    Case("bool", "regression", (20_000_000,), _polspec(OneBool)),
    Case("datetime", "regression", (20_000_000,), _polspec(OneDatetime)),
    # Whole frames.
    Case("mixed", "regression", (1_000_000, 20_000_000), _polspec(DataSource)),
    Case("wide_200_cols", "regression", (1_000, 100_000), _polspec(Wide)),
    # Draws without replacement: one case per branch of the engine.
    Case("unique_roomy", "regression", (1_000_000, 10_000_000), _polspec(UniqueRoomy)),
    Case(
        "unique_crowded", "regression", (1_000_000, 10_000_000), _polspec(UniqueCrowded)
    ),
    Case("unique_string", "regression", (1_000_000,), _polspec(UniqueString)),
    # The passes that rewrite a frame after the engine has filled it.
    Case(
        "cartesian", "regression", (1_000_000,), _polspec(Coverage, method="cartesian")
    ),
    Case("rules", "regression", (1_000_000,), _polspec(Ruled)),
    Case("foreign_key_self", "regression", (1_000_000,), _polspec(SelfKeyed)),
    Case("unique_together", "regression", (1_000_000,), _polspec(Composite)),
    Case("hierarchy", "regression", (100_000, 1_000_000), _polspec(Linked)),
    Case(
        "hierarchy_validate_cyclic",
        "regression",
        (100_000,),
        _validate_cyclic_hierarchy,
    ),
    # Streaming to disk, which no other case exercises.
    Case("sink_parquet", "regression", (1_000_000,), _sink_parquet),
)

BY_NAME = {case.name: case for case in CASES}

# `quick` is the subset worth running while iterating: one per engine branch.
QUICK = {"mixed", "string", "int64", "unique_roomy", "unique_crowded"}


# ---------------------------------------------------------------------------
# Measuring, in a process of its own
# ---------------------------------------------------------------------------


def peak_rss_mb() -> float | None:
    """Peak resident set of *this* process, which is why each case gets one."""
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes as wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        get_info = ctypes.windll.kernel32.K32GetProcessMemoryInfo
        get_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        get_info.restype = wintypes.BOOL
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not get_info(handle, ctypes.byref(counters), counters.cb):
            return None
        return counters.PeakWorkingSetSize / 1e6

    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS bytes.
    return peak / 1e3 if sys.platform.startswith("linux") else peak / 1e6


def measure(case: Case, n: int, repeat: int) -> dict[str, Any]:
    """One case at one size: a discarded warm-up, then repeated timed runs.

    The warm-up runs the real workload rather than a token one -- a
    thousand-row call warms the imports and the thread pool but not the
    allocation path a twenty-million-row call actually takes.

    `repeat` is a floor, not a count. A case that finishes in ten milliseconds
    has a spread of about 20% over three runs, which is wider than the
    threshold `check` is trying to enforce and would report a regression that
    is only the scheduler. So a fast case keeps running until it has spent
    `MIN_SAMPLE_SECONDS` in total, which costs almost nothing and brings the
    minimum somewhere near the true floor. A slow case is already stable and
    stops at `repeat`.
    """
    case.run(n)
    times = []
    total = 0.0
    while len(times) < repeat or (
        total < MIN_SAMPLE_SECONDS and len(times) < MAX_REPEAT
    ):
        start = time.perf_counter()
        case.run(n)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total += elapsed
    return {
        "case": case.name,
        "n": n,
        "runs": len(times),
        "min": min(times),
        "median": statistics.median(times),
        "max": max(times),
        "peak_rss_mb": peak_rss_mb(),
    }


# ---------------------------------------------------------------------------
# What was measured, and on what
# ---------------------------------------------------------------------------


def cargo_profile() -> dict[str, Any]:
    """The `[profile.release]` table the extension was (probably) built with.

    Read from `Cargo.toml` rather than from the binary, which does not carry
    it. So this records intent, not proof -- but a profile change is the
    single likeliest reason two runs disagree, and an unrecorded one is
    invisible.
    """
    try:
        manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return manifest.get("profile", {}).get("release", {})


def extension_fingerprint() -> dict[str, Any]:
    """Size and mtime of the built extension, to tell two builds apart."""
    try:
        from polspec import _polspec
    except ImportError:
        return {"built": False}
    path = Path(_polspec.__file__)
    stat = path.stat()
    return {
        "built": True,
        "bytes": stat.st_size,
        "modified": dt.datetime.fromtimestamp(stat.st_mtime, dt.UTC).isoformat(),
    }


def metadata() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as package_version

    try:
        polspec_version = package_version("polspec")
    except PackageNotFoundError:
        polspec_version = "unknown"

    return {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "polars": pl.__version__,
        "polspec": polspec_version,
        "polars_threads": pl.thread_pool_size(),
        "cargo_profile": cargo_profile(),
        "extension": extension_fingerprint(),
    }


def comparable(baseline: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """The metadata differences that make two runs not worth comparing."""
    interesting = ("processor", "polars_threads", "polars", "cargo_profile")
    return [
        f"{key}: baseline {baseline.get(key)!r} vs now {current.get(key)!r}"
        for key in interesting
        if baseline.get(key) != current.get(key)
    ]


# ---------------------------------------------------------------------------
# Driving it
# ---------------------------------------------------------------------------


def run_isolated(names: list[str], repeat: int) -> list[dict[str, Any]]:
    """Runs each (case, size) in a fresh interpreter and collects the JSON.

    The isolation is the point: a large case leaves the allocator and the page
    cache warm, and the next case in the same process reads faster or slower
    for reasons that have nothing to do with the code under test.
    """
    results: list[dict[str, Any]] = []
    for name in names:
        case = BY_NAME[name]
        for n in case.sizes:
            argv = [
                sys.executable,
                str(Path(__file__).resolve()),
                "_measure",
                "--case",
                name,
                "--size",
                str(n),
                "--repeat",
                str(repeat),
            ]
            completed = subprocess.run(  # noqa: S603 - fixed argv from this file's own case registry, no shell
                argv, capture_output=True, text=True, check=False
            )
            if completed.returncode != 0:
                print(f"  {name} n={n:,}: FAILED", file=sys.stderr)
                print(completed.stderr.strip()[-2000:], file=sys.stderr)
                continue
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            results.append(result)
            # How far the typical run sits above the floor, which is what says
            # whether the floor is well determined. Not max/min: over dozens of
            # samples that reports the worst thing the scheduler ever did.
            lift = (
                (result["median"] - result["min"]) / result["min"]
                if result["min"]
                else 0
            )
            print(
                f"  {name:<22} n={n:>12,}  {result['min']:>9.4f}s"
                f"  ({result['runs']:>2} runs, median +{lift * 100:.1f}%)"
            )
    return results


def select(suite: str, only: list[str] | None) -> list[str]:
    if only:
        unknown = [name for name in only if name not in BY_NAME]
        if unknown:
            raise SystemExit(f"unknown case(s): {unknown}. Known: {sorted(BY_NAME)}")
        return only
    if suite == "quick":
        return [c.name for c in CASES if c.name in QUICK]
    return [c.name for c in CASES if c.suite == suite]


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    """The markdown table `docs/explanation/comparison.md` publishes."""
    engines = ["polspec", "numpy", "python"]
    by_key = {(r["case"], r["n"]): r["min"] for r in results}
    sizes = sorted({r["n"] for r in results})

    print("\n| n_rows     | polspec (Rust) |     NumPy |    Python |")
    print("|-----------:|---------------:|----------:|----------:|")
    for n in sizes:
        cells = []
        for engine in engines:
            seconds = by_key.get((f"compare:{engine}", n))
            cells.append(f"{seconds:.4f}s" if seconds is not None else "skipped")
        print(f"| {n:>10,} | {cells[0]:>14} | {cells[1]:>9} | {cells[2]:>9} |")


def check(results: list[dict[str, Any]], tolerance: float) -> int:
    if not BASELINE.exists():
        raise SystemExit(
            f"no baseline at {BASELINE}. Record one on this machine first:\n"
            "    uv run --group bench python benchmarks/bench.py record"
        )
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    current_meta = metadata()

    differences = comparable(baseline["metadata"], current_meta)
    if differences:
        print("\nWARNING: the baseline was recorded under different conditions.")
        for line in differences:
            print(f"  - {line}")
        print("  Treat what follows as indicative only.\n")

    prior = {(r["case"], r["n"]): r["min"] for r in baseline["results"]}
    regressions: list[str] = []
    print(f"\n{'case':<22} {'n':>12} {'baseline':>10} {'now':>10} {'change':>9}")
    print("-" * 68)
    for result in results:
        was = prior.get((result["case"], result["n"]))
        if was is None:
            print(
                f"{result['case']:<22} {result['n']:>12,} {'-':>10} "
                f"{result['min']:>10.4f} {'new':>9}"
            )
            continue
        ratio = result["min"] / was
        flag = ""
        if ratio > 1 + tolerance:
            flag = "  REGRESSED"
            regressions.append(
                f"{result['case']} n={result['n']:,}: "
                f"{was:.4f}s -> {result['min']:.4f}s ({(ratio - 1) * 100:+.0f}%)"
            )
        elif ratio < 1 - tolerance:
            flag = "  improved"
        print(
            f"{result['case']:<22} {result['n']:>12,} {was:>10.4f} "
            f"{result['min']:>10.4f} {(ratio - 1) * 100:>8.0f}%{flag}"
        )

    if regressions:
        print(f"\n{len(regressions)} regression(s) beyond {tolerance:.0%}:")
        for line in regressions:
            print(f"  - {line}")
        return 1
    print(f"\nNo case regressed by more than {tolerance:.0%}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=["compare", "record", "check", "run", "_measure"],
        help=(
            "compare: the published table. record: write a baseline. "
            "check: compare against the baseline. run: measure without comparing."
        ),
    )
    parser.add_argument("--case", help="internal, or a single case to run")
    parser.add_argument("--size", type=int, help="internal: one row count")
    parser.add_argument("--repeat", type=int, default=3, help="timed runs per case")
    parser.add_argument(
        "--suite",
        default="regression",
        choices=["regression", "compare", "quick"],
        help="which cases to run (default: regression)",
    )
    parser.add_argument("--only", nargs="+", help="run only these named cases")
    parser.add_argument("--output", type=Path, help="write results as JSON here")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    # The child half of the isolation: measure one case and print one JSON line.
    if args.command == "_measure":
        print(json.dumps(measure(BY_NAME[args.case], args.size, args.repeat)))
        return 0

    suite = "compare" if args.command == "compare" else args.suite
    names = select(suite, args.only)
    print(
        f"Measuring {len(names)} case(s), {args.repeat} timed run(s) each, "
        "one process per measurement.\n"
    )
    results = run_isolated(names, args.repeat)
    payload = {"metadata": metadata(), "results": results}

    if args.command == "compare":
        print_comparison_table(results)
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
    if args.command == "record":
        BASELINE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nbaseline written to {BASELINE}")
    if args.command == "check":
        return check(results, args.tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
