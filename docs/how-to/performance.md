# Performance

polspec is fast by default: columns are filled in Rust, in parallel, and
validation is one Polars query. A few choices still decide whether a job
takes milliseconds or minutes, and whether it fits in memory. This page
gathers them, with what each costs.

The numbers are medians on an Intel 13900K with 64 GB of DDR5, Polars 2.1,
polspec 2.0, for a million rows unless a row says otherwise. Yours will
differ; the ratios matter more than the absolute times.
`benchmarks/bench.py` measures every case below.

## Generating

| A million rows of | Time |
|:--|--:|
| a four-column frame (string, enum, int, float) | 7 ms |
| the same, 20 million rows | 109 ms |
| an eleven-column frame of every kind a real table has | 40 ms |
| one `unique` integer key | 2 ms |
| one column with rules | 15 ms |
| a composite unique key (`__unique_together__`) | 23 ms |
| a hierarchy | 28 ms |

Plain columns cost almost nothing; the claims that relate rows to other
rows cost most. A composite key whose members all have a finite or
bounded space — the one above — is drawn as one permutation of their
combinations, at 143 MB for a million rows. One with a float or free-text
member is resampled until its combinations are distinct instead, and holds
several copies of the frame while it does, so it is the claim to leave out
of a spec that does not need it. *2.0: 1.7 took 95 ms and 850 MB for the
key above.*

### Memory: generate, batches, scan or a sink

`generate()` builds the whole frame before it returns: the 20-million-row
frame above peaks near 1 GB. For more than fits, stream it:

| A million rows | Holds | Four columns | Eleven columns |
|:--|:--|--:|--:|
| `generate(n)` | the whole frame | 7 ms | 40 ms |
| `generate_batches(n, batch_size=10_000)` | one batch | 7 ms | |
| `generate_batches(n, batch_size=1_000)` | one batch | 12 ms | |
| `scan(n)`, collected | what the plan asks for | | 68 ms |
| `sink_parquet(path, n)`, written | one batch | 25 ms | |

A small batch costs little: the columns are drawn many batches at a time
and sliced. `scan()` generates only the columns and rows a plan
asks for, so `.select("total").head(5)` makes five values of one column,
and a projection never changes what a column holds. See
[Lazy output](generating.md#lazy-output-scan) and
[Batching](generating.md#batching). `estimated_size(n)` says what a frame
will hold before anything is allocated, and `max_bytes=` refuses one too
large; see [Memory](generating.md#memory).

### Keeping checks

Generation keeps the rows every check passes and reads more of the frame
for the rest, reading only the rows it has not read, so its cost grows
with how many rows the checks reject. Checks are kept by default since
2.0:

| A million rows of the eleven columns | Time |
|:--|--:|
| `generate(n, checks=False)` | 40 ms |
| a check rejecting 10% of rows | 71 ms |
| a check rejecting 50% of rows | 125 ms |

A check most rows pass is cheap to keep; for one that rejects most of
them, narrow the columns it reads — bounds, choices, a rule — so fewer
rows fail. See
[Keeping checks](generating.md#keeping-checks).

## Validating

Every claim on every column is compiled into one Polars aggregation, so a
frame is read once however many claims there are:

| A million rows | Time |
|:--|--:|
| the eleven-column frame, validated | 41 ms |
| the same, as a `LazyFrame` validated and collected | 35 ms |
| one bounded column | 1.7 ms |

A `LazyFrame` is validated by running its query, and collecting what
`validate()` hands back runs it again; `report.failing_rows()` and
`passing_rows()` run it once more each. That costs nothing for a frame in
memory and a lot for a slow scan or join: collect once, then validate the
`DataFrame`.

<!-- docs: skip -->
```python
frame = pl.scan_parquet("s3://bucket/orders/*.parquet").collect()  # read once
report = Orders.inspect(frame)
clean = report.passing_rows().collect()
```

`streaming=True` runs validation on Polars' streaming engine, for a frame
larger than memory. See
[Streaming](validating.md#streaming).

## Profiling

Profiling reads every value of every column: the eleven-column frame
takes 180 ms at a million rows. `sample=` profiles a random sample instead,
seeded so it is the same every run:

| A million rows of the eleven columns | Time |
|:--|--:|
| `profile(frame)` | 180 ms |
| `profile(frame, sample=100_000)` | 55 ms |

A sample sees most of what the whole column holds, and misses its rare
values and extremes: bounds come from the sample, so a few real values may
fall outside them. Profile the whole frame for a contract, and a sample to
get started on a large one. Columns are profiled in parallel from 50,000
rows. `parse_text=True` reads text as numbers and dates first: a frame
whose numbers and dates arrive as text takes about three times as long
to profile as the same frame typed.

## Importing

`import polspec` takes about half a second cold, nearly all of it Polars
itself: polspec's own share is about 50 ms.
Generation, validation, drift and profiling load on first use, so a script
that only declares specs pays for none of them.
