"""A List, Array, Struct or Map column: what its elements, fields or
entries may say, and where it may be said.

Internal: not part of the public API.
"""

from __future__ import annotations

import polars as pl

from polspec import spec as colspec_module
from polspec._colspec.common import distinct_values
from polspec.dtypes import (
    field_dtypes,
    map_parts,
)
from polspec.errors import SpecError


def validate_nesting(spec: colspec_module.ColSpec) -> None:
    """A `List` or `Array` needs an element dtype to describe."""
    if not isinstance(spec.dtype, (pl.List, pl.Array)):
        return
    if spec.dtype.inner == pl.Null:
        raise SpecError(
            f"ColSpec cannot describe {spec.dtype!r}: give the list an element "
            "dtype, e.g. pl.List(pl.Int64)."
        )


def validate_struct_fields(spec: colspec_module.ColSpec) -> None:
    """What `fields` may say, and where it may be said.

    The dtype is the schema; `fields` is what is claimed about the
    values in it. So a name the dtype does not declare is a typo worth
    refusing, a field spec whose dtype disagrees with the struct's is
    two answers to one question, and a claim that only makes sense
    about a column among columns has no meaning inside a value.
    """
    value_dtype = spec.value_dtype
    if spec.fields is None:
        return
    if not isinstance(value_dtype, pl.Struct):
        raise SpecError(
            f"ColSpec.fields is only supported for pl.Struct, got "
            f"{spec.dtype!r}. It describes the fields of a struct value."
        )
    declared = field_dtypes(value_dtype)
    for name, field in spec.fields.items():
        if not isinstance(field, colspec_module.ColSpec):
            raise SpecError(
                f"ColSpec.fields[{name!r}] must be a ColSpec, got {type(field).__name__}"
            )
        if name not in declared:
            raise SpecError(
                f"ColSpec.fields names {name!r}, which {value_dtype!r} does "
                f"not declare. Its fields are {', '.join(declared) or 'none'}."
            )
        if field.dtype != declared[name]:
            raise SpecError(
                f"ColSpec.fields[{name!r}] declares {field.dtype!r}, but the "
                f"struct declares {declared[name]!r}. The dtype is the "
                "schema; fields describes the values in it."
            )
        for claim in (
            "unique",
            "rules",
            "validators",
            "seed_name",
            "col_name",
            "optional",
            "default",
        ):
            value = getattr(field, claim)
            if value is not None and value is not False and value != ():
                raise SpecError(
                    f"ColSpec.fields[{name!r}] sets {claim}, which has no "
                    "meaning on a struct field: it is a value, not a column "
                    "among columns."
                )


def validate_list_fields(spec: colspec_module.ColSpec) -> None:
    """What a `List` column takes, and what only a scalar one can. A
    `Map` is a list of entries, so it takes what a list takes."""
    if not 0.0 <= spec.element_null_probability <= 1.0:
        raise SpecError("element_null_probability must be between 0 and 1")
    is_map = map_parts(spec.dtype) is not None
    if is_map and spec.element_null_probability:
        raise SpecError(
            f"ColSpec.element_null_probability has no meaning on {spec.dtype!r}: "
            "a map holds entries, never a null one. A value may be null -- "
            'declare fields={"value": ColSpec(..., nullable=True)}.'
        )
    if not isinstance(spec.dtype, (pl.List, pl.Array)) and not is_map:
        if spec.element_null_probability:
            raise SpecError(
                "ColSpec.element_null_probability is only supported for pl.List "
                f"and pl.Array, got {spec.dtype!r}. It is how often an element "
                "inside a list is null; null_probability is the column's own."
            )
        if spec.list_length is not None:
            raise SpecError(
                f"ColSpec.list_length is only supported for pl.List, got {spec.dtype!r}"
            )
        return
    if isinstance(spec.dtype, pl.Array) and spec.list_length is not None:
        raise SpecError(
            f"ColSpec.list_length has no meaning on {spec.dtype!r}: an Array's "
            "length is part of its dtype. Drop it, or declare a pl.List."
        )
    if spec.list_length is not None and spec.list_length.closed()[0] < 0:
        raise SpecError(
            f"ColSpec.list_length must be non-negative, got {spec.list_length}"
        )
    if spec.unique:
        raise SpecError(
            f"ColSpec cannot be unique=True on {spec.dtype!r}: a list is not "
            "drawn without replacement. Declare uniqueness on a scalar column."
        )
    if spec.rules:
        raise SpecError(
            f"ColSpec cannot carry rules on {spec.dtype!r}: a rule's choices "
            "are values, and a list value would be a list of lists."
        )


def validate_map(spec: colspec_module.ColSpec) -> None:
    """What only a `Map` promises: a key is never null, and no key repeats
    within one map -- so a map declared to hold up to `n` entries needs
    `n` keys to choose from. Checked here, where it is a sentence, rather
    than as a generator redrawing keys that are not there."""
    if map_parts(spec.dtype) is None:
        return
    for claim in ("choices", "weights"):
        if getattr(spec, claim) is not None:
            raise SpecError(
                f"ColSpec.{claim} has no meaning on {spec.dtype!r}: a map's "
                "entries are drawn key and value apart. Declare them on the "
                'key or the value, in fields={"key": ..., "value": ...}.'
            )
    key = colspec_module.field_of(spec, "key")
    if key.nullable:
        raise SpecError(
            f"ColSpec.fields['key'] is nullable, but a key of {spec.dtype!r} "
            "is never null: Polars refuses a map with one. A value may be "
            "null; a key may not."
        )
    if spec.list_length is None:
        return
    longest = spec.list_length.closed()[1]
    keys = distinct_values(key)
    if keys is not None and keys < longest:
        raise SpecError(
            f"ColSpec declares a map of up to {longest} entries, which needs "
            f"{longest} distinct keys, but its keys can take {keys}: no key "
            "repeats within a map. Widen the key's domain, or lower "
            "list_length."
        )
