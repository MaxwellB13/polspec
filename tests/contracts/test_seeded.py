"""The same seed, the same data -- on every Polars polspec supports.

`seeded.json` beside this file holds a digest of what a fixed seed generates
for each case in the catalogue: every column case whole, batched and
scanned, every dtype, and the table-level features (rules, keys, a
hierarchy, a composite key, cartesian coverage). This test generates each
again, on whichever Polars is installed, and compares.

A digest is of the values, not of `DataFrame.hash_rows`, whose hash differs
between Polars majors: each row as Python objects, written as JSON. So a
Polars release that changes what a seed draws, or how a value comes back,
fails here, on the CI job that runs it -- the floor, the lock, the newest.

One kind of case is held to its own operating system: a column drawn from
a non-uniform distribution. Its sampler uses `exp`, `ln` and `pow` from the
platform's math library -- Polars enables `num-traits`' standard-library
math for the whole build, `rand_distr` included -- and those may round the
last bit differently on Windows, glibc and macOS; a rejection sampler
(gamma, beta) can then accept a different draw. Such a case's digest is
recorded per platform (`case@win32`) and skipped where none is recorded
(Known limitations).

A deliberate change to seeded output rewrites the file:

    POLSPEC_WRITE_SEEDED=1 uv run pytest tests/contracts/test_seeded.py

and says so in the changelog.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from cases import COLUMN_CASES, EVERY_DTYPE
from polspec import (
    ColRule,
    ColSpec,
    ForeignKey,
    Hierarchy,
    TableSpec,
    col,
    generate,
    generate_batches,
    scan,
)

REFERENCE = Path(__file__).with_name("seeded.json")
WRITE = os.environ.get("POLSPEC_WRITE_SEEDED") == "1"


# Column cases whose values come through the platform's math library.
PLATFORM_MATH = frozenset(
    name
    for name, column in COLUMN_CASES.items()
    if column.distribution not in (None, "uniform")
)


def _key(case: str) -> str:
    """The reference key a case is recorded under: per platform for one
    drawn through the platform's math library."""
    return f"{case}@{sys.platform}" if case.split("/")[0] in PLATFORM_MATH else case


def _digest(frame: pl.DataFrame) -> str:
    rows = json.dumps(frame.to_dicts(), default=str, sort_keys=True)
    return hashlib.sha256(rows.encode("utf-8")).hexdigest()


def _cases() -> dict[str, Callable[[], pl.DataFrame]]:
    cases: dict[str, Callable[[], pl.DataFrame]] = {}
    for name, column in COLUMN_CASES.items():
        spec = TableSpec("Case", {"c": column})
        cases[f"{name}/generate"] = lambda s=spec: generate(s, 300, seed=1)
        cases[f"{name}/batches"] = lambda s=spec: pl.concat(
            list(generate_batches(s, 300, batch_size=97, seed=3))
        )
        cases[f"{name}/scan"] = lambda s=spec: scan(
            s, 300, seed=5, batch_size=64
        ).collect()
    for dtype in EVERY_DTYPE:
        spec = TableSpec("Every", {"c": ColSpec(dtype)})
        cases[f"every/{dtype}"] = lambda s=spec: generate(s, 200, seed=11)

    d1, d2 = dt.date(2020, 1, 1), dt.date(2020, 1, 2)
    tables = {
        "rules": TableSpec(
            "R",
            {
                "d": ColSpec(pl.Date, choices=[d1, d2]),
                "k": ColSpec(pl.Enum(["a", "b", "c"])),
                "v": ColSpec(
                    pl.Int64,
                    choices=[1, 2, 3],
                    rules=[
                        ColRule(when=col("d").is_in([d1]), choices=[1]),
                        ColRule(when=col("k") == "b", choices=[2, 3]),
                    ],
                ),
            },
        ),
        "self_key": TableSpec(
            "N",
            {
                "ref": ColSpec(pl.String, unique=True, string_length=(8, 8)),
                "parent": ColSpec(pl.String, nullable=True, null_probability=0.2),
            },
            foreign_keys=[ForeignKey("parent", references="self", ref_columns="ref")],
        ),
        "hierarchy": TableSpec(
            "H",
            {"child": ColSpec(pl.String), "parent": ColSpec(pl.String)},
            hierarchy=Hierarchy(child="child", parent="parent", max_depth=4),
        ),
        "unique_together": TableSpec(
            "U",
            {
                "a": ColSpec(pl.Int8, bounds=(0, 50)),
                "b": ColSpec(pl.Int8, bounds=(0, 50)),
            },
            unique_together=[("a", "b")],
        ),
        "extras": TableSpec(
            "X",
            {
                "c": ColSpec(
                    pl.String, format="iso_country", extra_values={"UK (ISO)": 0.2}
                )
            },
        ),
    }
    for name, spec in tables.items():
        cases[f"table/{name}"] = lambda s=spec: generate(s, 400, seed=13)
    cartesian = TableSpec(
        "C",
        {
            "e": ColSpec(pl.Enum(["x", "y"])),
            "b": ColSpec(pl.Boolean),
            "n": ColSpec(pl.Int32, bounds=(-5, 5), nullable=True),
        },
    )
    cases["table/cartesian"] = lambda: generate(
        cartesian, 30, method="cartesian", seed=17
    )
    return cases


CASES = _cases()


def _reference() -> dict[str, str]:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


@pytest.mark.skipif(
    not WRITE, reason="set POLSPEC_WRITE_SEEDED=1 to rewrite seeded.json"
)
def test_write_the_reference():
    """Not a check: writes `seeded.json` from the installed Polars, keeping
    the digests of cases this Polars cannot generate (`Map` needs Polars 2)."""
    written = _reference() if REFERENCE.exists() else {}
    written.update({_key(name): _digest(make()) for name, make in CASES.items()})
    REFERENCE.write_text(
        json.dumps(written, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


@pytest.mark.skipif(WRITE, reason="rewriting the reference")
@pytest.mark.parametrize("case", sorted(CASES))
def test_a_seed_generates_the_same_data_on_every_polars(case):
    reference = _reference()
    key = _key(case)
    if key != case and key not in reference:
        pytest.skip(f"no {sys.platform} digest: a platform-math case (see above)")
    assert key in reference, f"{case} has no digest: rewrite seeded.json (see above)"
    assert _digest(CASES[case]()) == reference[key], (
        f"{case}: the seeded output changed on Polars {pl.__version__}"
    )


def test_every_reference_digest_has_a_case():
    """A case removed from the catalogue takes its digest with it -- except
    one this Polars cannot draw (`Map`, before Polars 2)."""
    stale = {key.split("@")[0] for key in _reference()} - set(CASES)
    assert all("Map" in name for name in stale), sorted(stale)
