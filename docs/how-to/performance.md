# Performance

polspec is fast by default: columns are filled in Rust, in parallel, and
validation is one Polars query. A few choices still decide whether a job
takes milliseconds or minutes, and whether it fits in memory. This page
gathers them, with what each costs.

The numbers are medians on an Intel 13900K with 64 GB of DDR5, Polars 1.44,
polspec 1.7, for a million rows unless a row says otherwise. Yours will
differ; the ratios matter more than the absolute times.
`benchmarks/bench.py` measures every case below.

## Generating

| A million rows of | Time |
|:--|--:|
| a four-column frame (string, enum, int, float) | 7 ms |
| the same, 20 million rows | 115 ms |
| an eleven-column frame of every kind a real table has | 49 ms |
| one `unique` integer key | 4 ms |
| one column with rules | 29 ms |
| a composite unique key (`__unique_together__`) | 95 ms |
| a hierarchy | 60 ms |

Plain columns cost almost nothing; the claims that relate rows to other
rows cost most. A composite key is resampled until its combinations are
distinct, and holds several copies of the frame while it does — about
850 MB at a million rows — so it is the claim to leave out of a spec that
does not need it.

### Memory: generate, batches, scan or a sink

`generate()` builds the whole frame before it returns: the 20-million-row
frame above peaks near 1 GB. For more than fits, stream it:

| A million rows | Holds | Four columns | Eleven columns |
|:--|:--|--:|--:|
| `generate(n)` | the whole frame | 7 ms | 49 ms |
| `generate_batches(n, batch_size=10_000)` | one batch | 8 ms | |
| `generate_batches(n, batch_size=1_000)` | one batch | 12 ms | |
| `scan(n)`, collected | what the plan asks for | | 76 ms |
| `sink_parquet(path, n)`, written | one batch | 27 ms | |

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
for the rest, so its cost grows with how many rows the checks reject.
These are 1.7's numbers; 2.0 reads only the rows it has not read, and
this page is re-measured for it:

| A million rows of the eleven columns | Time |
|:--|--:|
| `generate(n)` | 49 ms |
| `checks=True`, a check rejecting 10% of rows | 143 ms |
| `checks=True`, a check rejecting 50% of rows | 328 ms |

A check most rows pass is cheap to keep; for one that rejects most of
them, narrow the columns it reads — bounds, choices, a rule — so fewer
rows fail. See
[Keeping checks](generating.md#keeping-checks).

## Validating

Every claim on every column is compiled into one Polars aggregation, so a
frame is read once however many claims there are:

| A million rows | Time |
|:--|--:|
| the eleven-column frame, validated | 64 ms |
| the same, as a `LazyFrame` validated and collected | 61 ms |
| one bounded column | 1.5 ms |

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
takes 370 ms at a million rows. `sample=` profiles a random sample instead,
seeded so it is the same every run:

| A million rows of the eleven columns | Time |
|:--|--:|
| `profile(frame)` | 370 ms |
| `profile(frame, sample=100_000)` | 89 ms |

A sample sees most of what the whole column holds, and misses its rare
values and extremes: bounds come from the sample, so a few real values may
fall outside them. Profile the whole frame for a contract, and a sample to
get started on a large one. Columns are profiled in parallel from 50,000
rows. `parse_text=True` reads text as numbers and dates first: a frame
whose numbers and dates arrive as text takes about twice as long to
profile as the same frame typed.

## Importing

`import polspec` takes about half a second cold, most of it Polars itself.
Generation, validation, drift and profiling load on first use, so a script
that only declares specs pays for none of them.
