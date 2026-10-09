# Files and documents

Writing a spec to a file and reading it back, and rendering it for people
to read. Each takes a `TableSpec` or a `FrameSpec` class, and each has a
method on `FrameSpec` of the same name. See
[Specs as files](../../how-to/files.md) for the format and what survives a
round trip, and [Documenting a spec](../../how-to/documenting.md) for the
Markdown and the diagram. `polspec.serialization` also has the `CatSpec`
and `Registry` forms (`catspec_to_yaml`, `registry_to_yaml`, ...) and the
dict forms (`to_dict`, `from_dict`).

## to_yaml

::: polspec.to_yaml

## from_yaml

::: polspec.from_yaml

## to_python

::: polspec.to_python

## to_markdown

::: polspec.to_markdown

## to_mermaid

::: polspec.to_mermaid
