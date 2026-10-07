"""The round-trip property through the streaming verbs.

`test_roundtrip.py` holds `generate()` to its own `validate()`. The frames
`generate_batches()`, `scan()` and the sinks produce are made window by
window, which is a different path -- and one that went unchecked long enough
for a nested column to repeat its first batch's elements in every batch
after it, and for a projected cartesian scan to change its values. So every
case in the shared catalogue (`cases.py`) is streamed here too, and held to
what the streaming verbs promise:

- what is streamed validates against its spec;
- a column no pass rewrites holds, window by window, what `generate()` puts
  there -- a nested column its lengths and nulls, its elements being drawn
  per batch;
- no batch repeats another where the whole frame's rows at those places
  differ;
- a scan is the batches, and projecting one cannot change a column's values,
  under either method.
"""

from __future__ import annotations

import itertools

import polars as pl
import pytest
from cases import COLUMN_CASES
from polspec import (
    ColRule,
    ColSpec,
    ForeignKey,
    TableSpec,
    col,
    generate,
    generate_batches,
    inspect,
    scan,
    validate,
)
from polspec.dtypes import map_entries

ROWS, BATCH, SEED = 300, 97, 11


def _batches(spec: TableSpec, references=None) -> list[pl.DataFrame]:
    return list(
        generate_batches(spec, ROWS, batch_size=BATCH, seed=SEED, references=references)
    )


def _is_nested(dtype: pl.DataType) -> bool:
    """Whether a column's elements are drawn per batch."""
    return isinstance(dtype, (pl.List, pl.Array)) or map_entries(dtype) is not None


def _shape(column: pl.Series) -> pl.DataFrame:
    """What a nested column holds as a window: where it is null, and how long
    each present value is."""
    lengths = (
        column.list.len()
        if isinstance(column.dtype, pl.List)
        else pl.Series([None] * len(column), dtype=pl.UInt32)
    )
    return pl.DataFrame({"null": column.is_null(), "len": lengths})


def _assert_no_batch_repeats_another(
    parts: list[pl.DataFrame], whole: pl.DataFrame
) -> None:
    """Two full batches are equal only where the whole frame is equal at the
    same rows -- a column that can only ever hold one value repeats by
    right, and one drawn fresh per batch must not."""
    full = [b for b in parts if b.height == BATCH]
    for i, j in itertools.combinations(range(len(full)), 2):
        same_rows = whole.slice(i * BATCH, BATCH).equals(whole.slice(j * BATCH, BATCH))
        if not same_rows:
            assert not full[i].equals(full[j]), (i, j)


# ---------------------------------------------------------------------------
# Every column case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(COLUMN_CASES))
def test_every_column_streams(case):
    spec = TableSpec(f"Stream_{case}", {"c": COLUMN_CASES[case]})
    parts = _batches(spec)
    streamed = pl.concat(parts)
    whole = generate(spec, ROWS, seed=SEED)

    assert streamed.height == ROWS
    assert streamed.schema == spec.schema()
    streamed.to_dicts()  # every value materialises
    validate(spec, streamed)

    if _is_nested(spec["c"].dtype):
        assert _shape(streamed["c"]).equals(_shape(whole["c"]))
    else:
        assert streamed.equals(whole)
    _assert_no_batch_repeats_another(parts, whole)

    rerun = pl.concat(_batches(spec))
    assert rerun.equals(streamed)
    assert scan(spec, ROWS, seed=SEED, batch_size=BATCH).collect().equals(streamed)


# ---------------------------------------------------------------------------
# Specs whose passes rewrite columns: rules, keys, composite keys
# ---------------------------------------------------------------------------

_PARENT = TableSpec("Parent", {"k": ColSpec(pl.Int64, bounds=(100, 200))})
_COMPOSITE_PARENT = TableSpec(
    "CompositeParent",
    {
        "k1": ColSpec(pl.Int64, bounds=(1, 20)),
        "k2": ColSpec(pl.String, choices=["a", "b"]),
    },
)
_UNIQUE_PARENT = TableSpec(
    "UniqueParent", {"id": ColSpec(pl.Int64, bounds=(1, 10_000), unique=True)}
)

STREAM_SPECS: dict[str, tuple[TableSpec, dict | None]] = {
    "rules": (
        TableSpec(
            "Rules",
            {
                "g": ColSpec(pl.Enum(["x", "y", "z"])),
                "v": ColSpec(
                    pl.String,
                    choices=["a", "b", "c"],
                    rules=[
                        ColRule(when=col("g") == "x", choices=["a"]),
                        ColRule(when=col("g").is_in(["y", "z"]), choices=["b"]),
                    ],
                ),
                "tags": ColSpec(pl.List(pl.Int64), list_length=(1, 3)),
            },
        ),
        None,
    ),
    "chained_rules": (
        TableSpec(
            "ChainedRules",
            {
                "a": ColSpec(pl.Enum(["x", "y"])),
                "b": ColSpec(
                    pl.Enum(["x", "y"]),
                    rules=[ColRule(when=col("a") == "x", choices=["y"])],
                ),
                "c": ColSpec(
                    pl.Enum(["x", "y"]),
                    rules=[ColRule(when=col("b") == "y", choices=["x"])],
                ),
            },
        ),
        None,
    ),
    "unique_together": (
        TableSpec(
            "UniqueTogether",
            {
                "a": ColSpec(pl.Enum(["x", "y"])),
                "b": ColSpec(pl.Int64, bounds=(1, 10_000)),
            },
            unique_together=[["a", "b"]],
        ),
        None,
    ),
    "self_reference": (
        TableSpec(
            "SelfRef",
            {
                "id": ColSpec(pl.Int64, bounds=(1, 50)),
                "mgr": ColSpec(pl.Int64, bounds=(1, 50), nullable=True),
            },
            foreign_keys=[ForeignKey("mgr", references="self", ref_columns="id")],
        ),
        None,
    ),
    "chained_keys": (
        TableSpec(
            "ChainedKeys",
            {
                "a": ColSpec(pl.Int64, bounds=(100, 200)),
                "b": ColSpec(pl.Int64, bounds=(100, 200)),
            },
            foreign_keys=[
                ForeignKey("a", references=_PARENT, ref_columns="k", name="fk_a"),
                ForeignKey("b", references="self", ref_columns="a", name="fk_b"),
            ],
        ),
        {"Parent": generate(_PARENT, 200, seed=1)},
    ),
    "composite_key": (
        TableSpec(
            "CompositeChild",
            {
                "f1": ColSpec(pl.Int64, bounds=(1, 20)),
                "f2": ColSpec(pl.String, choices=["a", "b"]),
            },
            foreign_keys=[
                ForeignKey(
                    ["f1", "f2"],
                    references=_COMPOSITE_PARENT,
                    ref_columns=["k1", "k2"],
                )
            ],
        ),
        {"CompositeParent": generate(_COMPOSITE_PARENT, 200, seed=1)},
    ),
    "unique_key": (
        TableSpec(
            "UniqueChild",
            {
                "id": ColSpec(pl.Int64, bounds=(1, 10_000), unique=True),
                "flag": ColSpec(pl.Boolean),
            },
            foreign_keys=[
                ForeignKey("id", references=_UNIQUE_PARENT, ref_columns="id")
            ],
        ),
        {"UniqueParent": generate(_UNIQUE_PARENT, 1_000, seed=1)},
    ),
}


@pytest.mark.parametrize("name", sorted(STREAM_SPECS))
def test_every_batch_of_a_rewritten_spec_validates(name):
    """Each batch satisfies every claim; the stream as a whole satisfies all
    but a composite key's, which is distinct within a batch only."""
    spec, references = STREAM_SPECS[name]
    parts = _batches(spec, references)
    for batch in parts:
        validate(spec, batch, references=references)
    streamed = pl.concat(parts)
    assert streamed.height == ROWS
    report = inspect(spec, streamed, references=references)
    assert {f.code for f in report} <= {"unique_together"}
    assert pl.concat(_batches(spec, references)).equals(streamed)
    _assert_no_batch_repeats_another(parts, generate(spec, ROWS, seed=SEED))


@pytest.mark.parametrize("method", ["random", "cartesian"])
@pytest.mark.parametrize("name", sorted(STREAM_SPECS))
def test_projecting_a_scan_cannot_change_what_a_column_holds(name, method):
    spec, references = STREAM_SPECS[name]
    lf = scan(
        spec,
        ROWS,
        seed=SEED,
        batch_size=BATCH,
        method=method,
        references=references,
    )
    whole = lf.collect()
    for size in range(1, len(spec.columns) + 1):
        for subset in itertools.combinations(spec.columns, size):
            projected = lf.select(subset).collect()
            assert projected.equals(whole.select(subset)), subset


def _chunked(series: pl.Series, sizes: list[int]) -> pl.Series:
    """`series` in chunks of `sizes` rows, each a Series of its own -- as a
    batch is -- rather than a slice of one buffer."""
    values, parts, start = series.to_list(), [], 0
    for size in sizes:
        parts.append(pl.Series(series.name, values[start : start + size], series.dtype))
        start += size
    return pl.concat(parts, rechunk=False)


def test_a_frame_concatenated_from_batches_validates_however_it_is_chunked():
    """Polars 2.0.0's streaming engine -- the default for a lazy query --
    panics on validation's aggregation over a list column of this frame, its
    columns chunked differently as 2.0.0's `concat` of batches leaves them.
    Found by the batched property on 2.0.0, about one draw in 25,000;
    validation runs on the in-memory engine."""
    spec = TableSpec(
        "Drawn",
        {
            "c0": ColSpec(
                pl.List(pl.Enum(["LhCB", "UX", "lfaz", "dMixyX", "f"])),
                nullable=True,
                null_probability=0.15793906434049723,
                list_length=(2, 3),
                weights=[1 / 3, 4.857979561927767, 3.2679425436102743, 1.9, 1 / 3],
            ),
            "c2": ColSpec(
                pl.Boolean,
                nullable=True,
                null_probability=3.8116806693526224e-15,
                weights=[6.713186804849229, 8.318356618249505],
            ),
        },
    )
    whole = pl.concat(list(generate_batches(spec, 28, batch_size=1, seed=95))).rechunk()
    # The chunk boundaries Polars 2.0.0's `concat` left; 1.x keeps columns
    # aligned there, so they are laid out by hand for every version.
    layout = {
        "c0": [1] * 8 + [2] + [1] * 18,
        "c2": [1, 1, 1, 4, 1, 1, 1, 4, 1, 2] + [1] * 9 + [2],
    }
    frame = pl.DataFrame(
        [_chunked(whole[name], sizes) for name, sizes in layout.items()]
    )
    assert [frame[name].chunk_lengths() for name in layout] == list(layout.values())
    report = inspect(spec, frame)
    assert not report
    assert report.passing_rows().collect().height == 28
    validate(spec, frame)


def test_a_list_of_categories_arriving_beside_a_null_column_validates():
    """The same Polars 2.0.0 panic on a two-row, single-chunk frame: a list
    of categories with declared choices beside a column of nulls -- found by
    the arrival property, where rechunking does not help."""
    spec = TableSpec(
        "D",
        {
            "c1": ColSpec(pl.Boolean, nullable=True, null_probability=0.9),
            "c2": ColSpec(
                pl.List(pl.Categorical),
                nullable=True,
                list_length=(0, 2),
                choices=["oGodMQ", "C", "easHk", "NpyYi"],
            ),
        },
    )
    frame = pl.DataFrame(
        {"c1": [None, None], "c2": [["easHk", "NpyYi"], []]},
        schema={"c1": pl.Boolean, "c2": pl.List(pl.Categorical)},
    )
    assert not inspect(spec, frame)
    validate(spec, frame)
