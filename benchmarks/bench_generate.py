"""Benchmarks polspec's Rust generator against a numpy implementation and a
pure-Python (stdlib `random`) implementation of the same DataSource spec:

    string_1: pl.String,   nullable=False, random alphanumeric, length 5-15
    enum_1:   pl.Enum([...]), nullable=True,  null_probability=0.1
    int_1:    pl.Int64,    nullable=True,  bounds=(-100, 100)
    float_1:  pl.Float64,  nullable=True,  bounds=(-2000, 2000)

All three produce an equivalent polars DataFrame so the comparison is a fair
"how fast can each approach deliver a usable DataFrame", not just raw loop
speed. Run with: uv run --group bench python benchmarks/bench_generate.py
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
from polspec import Bound, ColSpec, FrameSpec

CATEGORIES = ["mammal", "reptile", "insect"]
NULL_P = 0.1
INT_LO, INT_HI = -100, 100
FLOAT_LO, FLOAT_HI = -2_000.0, 2_000.0
STR_MIN_LEN, STR_MAX_LEN = 5, 15
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
CHARSET_BYTES = np.frombuffer(CHARSET.encode("ascii"), dtype=np.uint8)


class DataSource(FrameSpec):
    string_1 = ColSpec(dtype=pl.String, nullable=False)
    enum_1 = ColSpec(dtype=pl.Enum(CATEGORIES), nullable=True, null_probability=NULL_P)
    int_1 = ColSpec(
        dtype=pl.Int64,
        bounds=Bound(INT_LO, INT_HI),
        nullable=True,
        null_probability=NULL_P,
    )
    float_1 = ColSpec(
        dtype=pl.Float64,
        bounds=(FLOAT_LO, FLOAT_HI),
        nullable=True,
        null_probability=NULL_P,
    )


def generate_rust(n: int, seed: int) -> pl.DataFrame:
    return DataSource.generate(n, seed=seed)


def generate_numpy(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)

    # string_1: not nullable. Vectorized via the classic "view as fixed-width
    # bytes" trick -- numpy has no efficient way to vectorize *ragged*
    # per-row lengths, so this uses a fixed width (max length) rather than
    # 5-15 like the other two implementations.
    idx = rng.integers(0, len(CHARSET_BYTES), size=(n, STR_MAX_LEN), dtype=np.uint8)
    chars = CHARSET_BYTES[idx]
    string_1 = np.char.decode(chars.view(f"S{STR_MAX_LEN}").reshape(-1), "ascii")

    # enum_1: nullable
    enum_1 = rng.choice(np.array(CATEGORIES, dtype=object), size=n)
    enum_1[rng.random(n) < NULL_P] = None

    # int_1: nullable, bounds -100..100. Generated as float64 + NaN so nulls
    # survive the numpy stage, then cast (non-strict) to Int64.
    int_1 = rng.integers(INT_LO, INT_HI + 1, size=n).astype(np.float64)
    int_1[rng.random(n) < NULL_P] = np.nan

    # float_1: nullable, bounds -2000..2000
    float_1 = rng.uniform(FLOAT_LO, FLOAT_HI, size=n)
    float_1[rng.random(n) < NULL_P] = np.nan

    df = pl.DataFrame(
        {"string_1": string_1, "enum_1": enum_1, "int_1": int_1, "float_1": float_1}
    )
    return df.with_columns(
        pl.col("enum_1").cast(pl.Enum(CATEGORIES)),
        pl.col("int_1").cast(pl.Int64, strict=False),
    )


def generate_python(n: int, seed: int) -> pl.DataFrame:
    rng = random.Random(seed)
    string_1: list[str] = [None] * n
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


@dataclass
class Result:
    name: str
    n_rows: int
    seconds: float | None  # None means skipped


IMPLEMENTATIONS = [
    ("rust", generate_rust),
    ("numpy", generate_numpy),
    ("python", generate_python),
]

# Sizes to run for every implementation, plus extra large sizes reserved for
# the fast implementations only (pure Python would take far too long there).
SIZES = [1_000, 10_000, 100_000, 1_000_000]
LARGE_SIZES_FAST_ONLY = [5_000_000, 20_000_000]
PYTHON_SLOW_CUTOFF_SECONDS = (
    5.0  # stop giving pure-python bigger sizes once it crosses this
)


def bench_one(fn, n: int, seed: int) -> tuple[float, pl.DataFrame]:
    start = time.perf_counter()
    df = fn(n, seed)
    elapsed = time.perf_counter() - start
    return elapsed, df


def run_benchmark() -> list[Result]:
    results: list[Result] = []
    python_skipped = False

    # Warm up each implementation once so first-call overhead (imports,
    # RNG init) doesn't pollute the smallest timed size.
    for _name, fn in IMPLEMENTATIONS:
        fn(10, seed=0)

    for n in SIZES + LARGE_SIZES_FAST_ONLY:
        for name, fn in IMPLEMENTATIONS:
            if n in LARGE_SIZES_FAST_ONLY and name == "python":
                results.append(Result(name, n, None))
                continue
            if name == "python" and python_skipped:
                results.append(Result(name, n, None))
                continue

            elapsed, df = bench_one(fn, n, seed=42)
            assert df.height == n
            results.append(Result(name, n, elapsed))
            print(
                f"{name:>8} n={n:>10,}  {elapsed:>10.4f}s  ({n / elapsed:,.0f} rows/s)"
            )

            if name == "python" and elapsed > PYTHON_SLOW_CUTOFF_SECONDS:
                python_skipped = True

    return results


def print_table(results: list[Result]) -> None:
    sizes = sorted({r.n_rows for r in results})
    names = [name for name, _ in IMPLEMENTATIONS]
    by_key = {(r.name, r.n_rows): r.seconds for r in results}

    header = f"{'n_rows':>12} | " + " | ".join(f"{n:>12}" for n in names)
    print("\n" + header)
    print("-" * len(header))
    for n in sizes:
        row = [f"{n:>12,}"]
        for name in names:
            seconds = by_key.get((name, n))
            row.append(
                f"{seconds:>10.4f}s" if seconds is not None else f"{'skipped':>11}"
            )
        print(" | ".join(row))


if __name__ == "__main__":
    results = run_benchmark()
    print_table(results)
