"""A column's names, dtype, tags, null and NaN rates, distribution and
validators: what it is before what it holds.

Internal: not part of the public API.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from contextlib import suppress
from typing import Any

import polars as pl

from polspec import spec as colspec_module
from polspec._colspec.common import is_dtype, not_a_dtype
from polspec.checks import Check
from polspec.constants import (
    DEFAULT_NULL_PROBABILITY,
)
from polspec.distributions import (
    canonicalize_params,
    normalize_distribution,
    validate_distribution_params,
)
from polspec.errors import SpecError
from polspec.expr import Pred


def validate_col_name(spec: colspec_module.ColSpec) -> None:
    if spec.col_name is not None and not spec.col_name:
        raise SpecError("ColSpec.col_name must not be an empty string")


def validate_seed_name(spec: colspec_module.ColSpec) -> None:
    if spec.seed_name is None:
        return
    if not isinstance(spec.seed_name, str) or not spec.seed_name:
        raise SpecError(
            f"ColSpec.seed_name must be a non-empty string, got {spec.seed_name!r}"
        )


def normalize_dtype(spec: colspec_module.ColSpec) -> None:
    """Instantiates a dtype passed as a class, so `pl.Int64` means `pl.Int64()`."""
    raw: Any = spec.dtype
    if not is_dtype(raw):
        raise SpecError(not_a_dtype(raw))
    if isinstance(raw, type) and issubclass(raw, pl.DataType):
        with suppress(TypeError):
            object.__setattr__(spec, "dtype", raw())
    # `pl.List(pl.Categorical)` keeps its element as the class; the frame
    # generated from it holds an instance, and polars compares the two as
    # different dtypes. Instantiate the element too.
    nested: Any = spec.dtype
    if isinstance(nested, (pl.List, pl.Array)) and isinstance(nested.inner, type):
        with suppress(TypeError):
            inner = nested.inner()
            rebuilt = (
                pl.List(inner)
                if isinstance(nested, pl.List)
                else pl.Array(inner, nested.size)
            )
            object.__setattr__(spec, "dtype", rebuilt)


def normalize_tags(spec: colspec_module.ColSpec) -> None:
    """Reduces `tags` to a tuple of distinct, non-empty strings.

    A `list` or `tuple` keeps the order it was written in -- that is the
    author's choice and it survives into the spec file. A `set` has no
    order to keep: Python salts string hashing per process, so the same
    declaration would yield a different tuple on every run, writing a
    different `tags:` line each time `to_yaml` is called and making two
    identically-written specs compare unequal across processes. Sorting is
    the only stable reading of an unordered input.
    """
    tags: Any = spec.tags
    if tags is None:
        object.__setattr__(spec, "tags", ())
    elif isinstance(tags, str):
        object.__setattr__(spec, "tags", (tags,) if tags else ())
    elif isinstance(tags, (list, tuple, set, Sequence)):
        raw = sorted(tags) if isinstance(tags, (set, frozenset)) else tags
        distinct: dict[str, None] = {}
        for tag in raw:
            text = str(tag)
            if text:
                distinct.setdefault(text, None)
        object.__setattr__(spec, "tags", tuple(distinct))
    else:
        raise SpecError(
            f"ColSpec.tags must be a string or sequence of strings, got {type(tags).__name__}"
        )


def normalize_validators(spec: colspec_module.ColSpec) -> None:
    given: Any = spec.validators
    if given is None:
        object.__setattr__(spec, "validators", ())
        return

    raw = [given] if isinstance(given, (pl.Expr, Pred, Check)) else list(given)

    normalized: list[Check] = []
    seen: dict[str, Check] = {}
    for v in raw:
        if isinstance(v, Check):
            chk = v
        elif isinstance(v, (pl.Expr, Pred)):
            chk = Check(v)
        else:
            raise SpecError(
                "ColSpec.validators items must be a polars Expr, a predicate "
                "built with col(), or a Check, "
                f"got {type(v).__name__}"
            )
        prior = seen.get(chk.name)
        if prior is not None:
            if prior != chk:
                raise SpecError(
                    f"Duplicate validator name {chk.name!r} on this ColSpec: "
                    f"{prior.expr!r} vs {chk.expr!r}. Give each validator a "
                    "distinct name."
                )
            # The same validator written twice is one claim, not two. Kept
            # once so it produces one finding, matching how TableSpec
            # de-duplicates identical checks and foreign keys.
            continue
        seen[chk.name] = chk
        normalized.append(chk)

    object.__setattr__(spec, "validators", tuple(normalized))


def canonicalize_distribution(spec: colspec_module.ColSpec) -> None:
    """Canonicalizes the distribution name and floats its parameters."""
    if spec.distribution is not None:
        if not (
            spec.value_dtype.is_integer()
            or spec.value_dtype.is_float()
            or spec.value_dtype.is_decimal()
            or spec.value_dtype.is_temporal()
        ):
            raise SpecError(
                f"ColSpec.distribution is only supported for numeric or temporal "
                f"dtypes, got {spec.value_dtype!r}"
            )
        name = normalize_distribution(spec.distribution)
        object.__setattr__(spec, "distribution", name)
        if spec.distribution_params is not None:
            params = canonicalize_params(
                name,
                {str(k): float(v) for k, v in spec.distribution_params.items()},
            )
            object.__setattr__(spec, "distribution_params", params)
            validate_distribution_params(name, params)

    # A Boolean column takes no distribution, but does accept `p`.
    if spec.value_dtype == pl.Boolean and spec.distribution_params is not None:
        params = {str(k): float(v) for k, v in spec.distribution_params.items()}
        object.__setattr__(spec, "distribution_params", params)
        if "p" in params and not 0.0 <= params["p"] <= 1.0:
            raise SpecError(
                f"Boolean distribution_params['p'] must be between 0 and 1, got {params['p']}"
            )


def validate_null(spec: colspec_module.ColSpec) -> None:
    """A `Null` column holds nothing but nulls: it must be declared
    nullable, and its null rate is 1 by definition -- the default rate
    becomes 1, and any other rate written for it is refused."""
    if spec.dtype != pl.Null:
        return
    if not spec.nullable:
        raise SpecError(
            "ColSpec(pl.Null) holds only nulls, so it must be declared nullable=True."
        )
    if spec.null_probability not in (DEFAULT_NULL_PROBABILITY, 1.0):
        raise SpecError(
            f"ColSpec(pl.Null) declares null_probability={spec.null_probability}"
            ", but every value of a Null column is null. Leave it out."
        )
    object.__setattr__(spec, "null_probability", 1.0)


def validate_probabilities(spec: colspec_module.ColSpec) -> None:
    if not 0.0 <= spec.null_probability <= 1.0:
        raise SpecError("null_probability must be between 0 and 1")
    warn_unused_null_probability(spec)


def validate_nan_probability(spec: colspec_module.ColSpec) -> None:
    """A NaN share belongs to a float value, and a unique column can hold
    none: every NaN is unequal to every other, so distinctness has
    nothing to say about one."""
    if not 0.0 <= spec.nan_probability <= 1.0:
        raise SpecError("nan_probability must be between 0 and 1")
    if not spec.nan_probability:
        return
    if not spec.value_dtype.is_float():
        raise SpecError(
            "ColSpec.nan_probability is only supported for a float value, got "
            f"{spec.dtype!r}. NaN is a float; a missing value of any other "
            "dtype is a null -- declare nullable=True."
        )
    if spec.unique:
        raise SpecError(
            "ColSpec cannot be unique=True and carry nan_probability: no NaN "
            "equals another, so a unique column has nothing to say about one."
        )


def warn_unused_null_probability(spec: colspec_module.ColSpec) -> None:
    """Warns about a null rate on a column that cannot hold a null.

    `nullable=False` wins and the rate is ignored, which is deliberate:
    turning nullability off should not also require deleting the rate
    beside it. But the same silence covers a genuine mistake -- asking for
    nulls and forgetting `nullable=True` -- where the column generates
    none and nothing says why.

    So this warns rather than raising, and only for a rate that cannot
    have been left behind by turning nullability off: the default is what
    every non-nullable column carries, and an explicit zero already agrees
    with `nullable=False`. Anything else was written on purpose and does
    not do what it says.
    """
    if spec.nullable or spec.null_probability in (
        0.0,
        DEFAULT_NULL_PROBABILITY,
    ):
        return
    warnings.warn(
        f"ColSpec declares null_probability={spec.null_probability} with "
        "nullable=False, so no nulls will be generated and validation will "
        "reject any it finds. Add nullable=True to get that rate, or drop "
        "null_probability to say the column holds no nulls.",
        stacklevel=4,
    )


def validate_unique_is_generatable(spec: colspec_module.ColSpec) -> None:
    """Rejects what `unique=True` cannot be combined with.

    A unique column is drawn *without* replacement, so anything that
    describes how often a value should recur has nothing left to say:
    weights bias a repeated draw, and a distribution shapes a pile of
    independent ones. A rule is worse than meaningless -- it overwrites
    values after the draw, from a fixed set of choices, which is how
    duplicates would get back in.
    """
    if not spec.unique:
        return
    if spec.weights is not None:
        raise SpecError(
            "ColSpec cannot be unique=True and carry weights: values are "
            "drawn without replacement, so a weight has no repeated draw "
            "to bias. Drop the weights, or the uniqueness."
        )
    if spec.distribution is not None and spec.distribution != "uniform":
        raise SpecError(
            f"ColSpec cannot be unique=True and carry "
            f"distribution={spec.distribution!r}: values are drawn without "
            "replacement from the column's domain, which no distribution "
            "shapes. Drop the distribution, or the uniqueness."
        )
    if spec.rules:
        raise SpecError(
            "ColSpec cannot be unique=True and carry rules: a rule "
            "overwrites matched rows with values from a fixed set, which "
            "would reintroduce the duplicates uniqueness rules out. Drop "
            "the rules, or the uniqueness."
        )
