"""A column's choices and weights, and an Enum's categories: one finite
domain, said consistently.

Internal: not part of the public API.
"""

from __future__ import annotations

import dataclasses
import decimal
from contextlib import suppress
from typing import Any

import polars as pl

from polspec import spec as colspec_module
from polspec.dtypes import (
    unholdable,
)
from polspec.errors import SpecError
from polspec.rules import reject_duplicate_choices


def normalize_choices_and_weights(spec: colspec_module.ColSpec) -> None:
    """Splits a `{choice: weight}` mapping into the two fields, and tuples both."""
    choices: Any = spec.choices
    if isinstance(choices, dict):
        if spec.weights is not None:
            raise SpecError(
                "Cannot specify both a dict for choices and an explicit weights parameter"
            )
        object.__setattr__(spec, "weights", tuple(float(w) for w in choices.values()))
        object.__setattr__(spec, "choices", tuple(choices.keys()))
    else:
        if spec.choices is not None:
            object.__setattr__(spec, "choices", tuple(spec.choices))
        if spec.weights is not None:
            object.__setattr__(spec, "weights", tuple(float(w) for w in spec.weights))

    if spec.choices is not None and not spec.choices:
        raise SpecError("ColSpec.choices must not be empty")
    enum_from_choices(spec)
    if isinstance(spec.value_dtype, pl.Decimal):
        read_decimal_choices(spec)
    # Before the duplicate check, which reads the choices through the
    # dtype and would otherwise report a truncated `1.5` as a repeated `1`.
    validate_choices_fit_dtype(spec)
    if spec.choices is not None:
        reject_duplicate_choices(spec.choices, "ColSpec.choices", spec.value_dtype)


def enum_from_choices(spec: colspec_module.ColSpec) -> None:
    """A bare `pl.Enum` takes its categories from `choices`, in the order
    written; and choices that are an Enum's categories, in its order,
    are no narrower domain than the Enum's own, so they are dropped.

    Either way the spec is the one `pl.Enum([...])` and `weights=` make,
    so the spellings compare equal and write the same file -- and they
    already generate the same frame, so no seed moves.
    """
    dtype: Any = spec.dtype
    bare = dtype is pl.Enum or (
        isinstance(dtype, pl.Enum) and dtype.categories.len() == 0
    )
    if bare:
        if spec.choices is None:
            raise SpecError(
                "A bare pl.Enum has no categories: list them, as "
                "pl.Enum(['a', 'b']), or give choices -- choices=['a', 'b'], "
                "or {'a': 0.7, 'b': 0.3} with weights -- to be its categories."
            )
        if not all(isinstance(c, str) for c in spec.choices):
            raise SpecError(
                f"ColSpec.choices {list(spec.choices)!r} cannot be the "
                "categories of a pl.Enum: an Enum's categories are text."
            )
        reject_duplicate_choices(spec.choices, "ColSpec.choices", pl.String())
        object.__setattr__(spec, "dtype", pl.Enum(list(spec.choices)))
    dtype = spec.dtype
    if (
        isinstance(dtype, pl.Enum)
        and spec.choices is not None
        and spec.choices == tuple(dtype.categories.to_list())
    ):
        object.__setattr__(spec, "choices", None)


def read_decimal_choices(spec: colspec_module.ColSpec) -> None:
    """A `Decimal` column's text choices -- the column's and each
    rule's -- as `decimal.Decimal`, read exactly.

    A spec file writes a Decimal as the exact string it prints as, since
    YAML has no decimal type and a float would not be the value
    declared; this is the reading half, as `_decimal_endpoints` is for
    bounds. A choice the text does not parse as is left for
    `_validate_choices_fit_dtype` to refuse by name.
    """

    def read(values: tuple[Any, ...]) -> tuple[Any, ...]:
        out = []
        for value in values:
            if isinstance(value, str):
                with suppress(decimal.InvalidOperation):
                    value = decimal.Decimal(value)
            out.append(value)
        return tuple(out)

    if spec.choices is not None:
        object.__setattr__(spec, "choices", read(spec.choices))
    if any(any(isinstance(c, str) for c in rule.choices) for rule in spec.rules):
        object.__setattr__(
            spec,
            "rules",
            tuple(
                dataclasses.replace(rule, choices=read(rule.choices))
                for rule in spec.rules
            ),
        )


def validate_choices_fit_dtype(spec: colspec_module.ColSpec) -> None:
    """Refuses a choice -- the column's or a rule's -- that the column's
    dtype cannot hold as written.

    Generation gathers choices from a Series of the column's own dtype,
    cast leniently, so a choice the dtype cannot hold does not fail: it
    arrives as a null, in a column that may not allow one, or as a
    different value -- `1000` on an `Int8` is a null, `1.5` is `1`. The
    spec would then fail its own validation, or generate a domain it
    never declared. An `Enum`'s choices are held to its categories by
    `_validate_choices_against_domain`, in words about categories.
    """
    if isinstance(spec.value_dtype, pl.Enum):
        return
    dtype = spec.value_dtype
    for label, choices in (
        ("ColSpec.choices", spec.choices),
        *(("ColRule.choices", rule.choices) for rule in spec.rules),
    ):
        if choices is None:
            continue
        offending = unholdable(choices, dtype)
        if offending:
            raise SpecError(
                f"{label} {offending} cannot be held by {dtype!r}: generated, "
                "each would come out as a null or as a different value. "
                f"Declare choices a {dtype!r} column holds exactly, or widen "
                "the dtype."
            )


def validate_weights(spec: colspec_module.ColSpec) -> None:
    """Checks weights against whatever defines this column's domain."""
    if spec.weights is None:
        return

    if spec.choices is not None:
        if len(spec.weights) != len(spec.choices):
            raise SpecError(
                f"Length of weights ({len(spec.weights)}) must match length of choices ({len(spec.choices)})"
            )
    elif isinstance(spec.value_dtype, pl.Enum):
        if len(spec.weights) != len(spec.value_dtype.categories):
            raise SpecError(
                f"Length of weights ({len(spec.weights)}) must match number of Enum categories ({len(spec.value_dtype.categories)})"
            )
    elif spec.value_dtype == pl.Boolean:
        if len(spec.weights) != 2:
            raise SpecError(
                "Boolean weights must be a 2-element sequence [p_false, p_true]"
            )
    else:
        raise SpecError(
            "ColSpec.weights requires 'choices', an Enum dtype, or a Boolean "
            f"dtype to define the domain weights apply to; got dtype={spec.value_dtype!r} "
            "with no choices"
        )

    if any(w < 0 for w in spec.weights):
        raise SpecError("Weights must all be non-negative")
    if sum(spec.weights) <= 0:
        raise SpecError("Sum of weights must be positive")


def validate_choices_against_domain(spec: colspec_module.ColSpec) -> None:
    """Checks this column's choices, and its rules', against its domain.

    A rule's choices are checked here too: a value a rule could assign but
    the column could never hold is a contradiction worth catching at
    declaration time rather than at generation.
    """
    if isinstance(spec.value_dtype, pl.Enum):
        categories = set(spec.value_dtype.categories.to_list())
        reject_choices(
            spec,
            lambda c: c not in categories,
            f"are not among this column's Enum categories {sorted(categories)}",
        )

    if spec.bounds is not None:
        low, high = spec.bounds.min, spec.bounds.max

        def outside(value: Any) -> bool:
            return (low is not None and value < low) or (
                high is not None and value > high
            )

        reject_choices(
            spec, outside, f"fall outside this column's bounds {spec.bounds}"
        )


def reject_choices(spec: colspec_module.ColSpec, offends, complaint: str) -> None:
    """Raises if any of this column's or its rules' choices `offends`."""
    if spec.choices is not None:
        offending = [c for c in spec.choices if offends(c)]
        if offending:
            raise SpecError(f"ColSpec.choices {offending} {complaint}")
    for rule in spec.rules:
        offending = [c for c in rule.choices if offends(c)]
        if offending:
            raise SpecError(f"ColRule.choices {offending} {complaint}")
