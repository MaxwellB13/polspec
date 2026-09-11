# API reference

Every name `polspec` exports, rendered from its own docstrings. A page here
cannot describe a signature the code does not have.

| Name | Page |
| --- | --- |
| [`ColSpec`][polspec.ColSpec] | [Columns](columns.md) |
| [`Bound`][polspec.Bound] | [Columns](columns.md) |
| [`ColRule`][polspec.ColRule] | [Columns](columns.md) |
| [`Check`][polspec.Check] | [Columns](columns.md) |
| [`col`][polspec.col] | [Predicates](predicates.md) |
| [`Pred`][polspec.Pred] | [Predicates](predicates.md) |
| [`TableSpec`][polspec.TableSpec] | [Specs](specs.md) |
| [`FrameSpec`][polspec.FrameSpec] | [Specs](specs.md) |
| [`ForeignKey`][polspec.ForeignKey] | [Specs](specs.md) |
| [`Hierarchy`][polspec.Hierarchy] | [Specs](specs.md) |
| [`Registry`][polspec.Registry] | [Registry](registry.md) |
| [`CatSpec`][polspec.CatSpec] | [Registry](registry.md) |
| [`generate`][polspec.generate] | [Generation](generation.md) |
| [`generate_batches`][polspec.generate_batches] | [Generation](generation.md) |
| [`sink_parquet`][polspec.sink_parquet] | [Generation](generation.md) |
| [`sink_ipc`][polspec.sink_ipc] | [Generation](generation.md) |
| [`sink_csv`][polspec.sink_csv] | [Generation](generation.md) |
| [`sink_ndjson`][polspec.sink_ndjson] | [Generation](generation.md) |
| [`inspect`][polspec.inspect] | [Validation](validation.md) |
| [`validate`][polspec.validate] | [Validation](validation.md) |
| [`ValidationOptions`][polspec.ValidationOptions] | [Validation](validation.md) |
| [`ValidationReport`][polspec.ValidationReport] | [Validation](validation.md) |
| [`Finding`][polspec.Finding] | [Validation](validation.md) |
| [`profile_dataframe`][polspec.profile_dataframe] | [Profiling](profiling.md) |
| [`PolspecError`][polspec.PolspecError] | [Errors](errors.md) |
| [`SpecError`][polspec.SpecError] | [Errors](errors.md) |
| [`GenerationError`][polspec.GenerationError] | [Errors](errors.md) |
| [`ValidationError`][polspec.ValidationError] | [Errors](errors.md) |
| [`MultiValidationError`][polspec.MultiValidationError] | [Errors](errors.md) |
| [`SerializationError`][polspec.SerializationError] | [Errors](errors.md) |
| [`RegistryError`][polspec.RegistryError] | [Errors](errors.md) |
| [`CliError`][polspec.CliError] | [Errors](errors.md) |

Anything not listed here is internal: it can change in a patch release without
a changelog entry.
