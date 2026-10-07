"""Polars 2.0.0: a lazy `select` that runs `list.eval` beside a `filter`
on the same list column panics on the streaming engine -- the default for a
lazy query in 2.0 -- with "assertion `left == right` failed"
(polars-expr/src/expressions/eval.rs). Here the frame's columns are chunked
differently (27 chunks against 20, as 2.0.0's `concat` of batches leaves
them); a two-row single-chunk frame of a list of categories beside a column
of nulls triggers it too, so chunking is a trigger, not the cause. Passes on
Polars 1.44.2, and on 2.0.0 eagerly or on the in-memory engine.

No polspec involved: this is the reproduction for an upstream report.
polspec runs validation on the in-memory engine (0.16.0).

    uv run --with "polars==2.0.0" python scripts/deep/polars2_eval_chunks_repro.py
"""

import polars as pl

L_VALUES = [
    ["UX", "dMixyX", "dMixyX"],
    ["UX", "dMixyX"],
    ["UX", "dMixyX", "dMixyX"],
    ["lfaz", "UX", "lfaz"],
    None,
    None,
    ["UX", "lfaz", "UX"],
    ["UX", "UX"],
    ["UX", "UX", "lfaz"],
    ["UX", "UX", "lfaz"],
    ["UX", "lfaz", "lfaz"],
    None,
    ["UX", "dMixyX", "lfaz"],
    ["f", "dMixyX", "UX"],
    ["UX", "UX", "lfaz"],
    None,
    ["lfaz", "UX"],
    ["UX", "UX", "dMixyX"],
    None,
    ["UX", "UX"],
    ["UX", "UX", "UX"],
    ["UX", "UX"],
    ["lfaz", "lfaz", "dMixyX"],
    ["lfaz", "dMixyX", "UX"],
    ["dMixyX", "dMixyX", "lfaz"],
    ["dMixyX", "UX"],
    ["UX", "lfaz"],
    None,
]
L_CHUNKS = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]
F_VALUES = [
    False,
    False,
    True,
    True,
    True,
    True,
    True,
    False,
    False,
    True,
    False,
    False,
    False,
    False,
    False,
    True,
    True,
    False,
    True,
    False,
    True,
    False,
    False,
    True,
    True,
    False,
    False,
    False,
]
F_CHUNKS = [1, 1, 1, 4, 1, 1, 1, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
E = pl.Enum(["LhCB", "UX", "lfaz", "dMixyX", "f"])


def chunked(name, values, lengths, dtype):
    parts, start = [], 0
    for n in lengths:
        parts.append(pl.Series(name, values[start : start + n], dtype=dtype))
        start += n
    return pl.concat(parts, rechunk=False)


df = pl.DataFrame(
    [
        chunked("l", L_VALUES, L_CHUNKS, pl.List(E)),
        chunked("f", F_VALUES, F_CHUNKS, pl.Boolean),
    ]
)
hit = (
    pl.col("l").is_not_null()
    & pl.col("l")
    .list.eval(
        pl.element().is_not_null()
        & ~pl.element().cast(pl.String).is_in(["LhCB", "UX", "f"])
    )
    .list.any()
)
query = [
    hit.sum().alias("n"),
    pl.col("l").filter(hit).unique(maintain_order=True).head(5).implode().alias("s"),
]

print(pl.__version__)
print("rechunked:", df.rechunk().lazy().select(query).collect().row(0))
print("eager:    ", df.select(query).row(0))
print("lazy:     ", df.lazy().select(query).collect().row(0))  # panics on 2.0.0
