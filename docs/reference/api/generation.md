# Generation

Every function here takes a spec as its first argument -- a `TableSpec`,
or a `FrameSpec` class, which stands for its `.spec` -- and every one has a
`FrameSpec` classmethod that forwards to it with `cls.spec` -- see
[Generating data](../../how-to/generating.md) for what the options mean and
[Specs as values](../../how-to/tablespec.md) for when to reach for which.

## generate

::: polspec.generate

## generate_batches

::: polspec.generate_batches

## scan

::: polspec.scan

## estimated_size

::: polspec.estimated_size

## sink_parquet

::: polspec.sink_parquet

## sink_ipc

::: polspec.sink_ipc

## sink_csv

::: polspec.sink_csv

## sink_ndjson

::: polspec.sink_ndjson
