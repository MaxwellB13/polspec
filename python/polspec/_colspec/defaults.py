"""A column's default and whether it may be absent.

Internal: not part of the public API.
"""

from __future__ import annotations

import decimal
import math
from contextlib import suppress

import polars as pl

from polspec import spec as colspec_module
from polspec.dtypes import (
    map_parts,
    unholdable,
)
from polspec.errors import SpecError


def validate_optional(spec: colspec_module.ColSpec) -> None:
    if not isinstance(spec.optional, bool):
        raise SpecError(
            f"ColSpec.optional must be True or False, got {spec.optional!r}"
        )


def validate_default(spec: colspec_module.ColSpec) -> None:
    """Refuses a default the column could not hold, or would itself
    report, and keeps the one it accepts as the column holds it.

    Filling with a value the column then fails would only trade a null
    for another finding, so the default is held to every claim about a
    value, on a one-row frame of it. A claim among rows (`unique`, a
    rule reading other columns) has nothing to say of one value, and is
    not asked. A test holds the answer to validation's on the same
    value, since a declaration may not import validation.
    """
    default = spec.default
    if default is None:
        return
    dtype = spec.dtype
    if (
        isinstance(dtype, (pl.List, pl.Array, pl.Struct))
        or map_parts(dtype) is not None
    ):
        raise SpecError(
            f"ColSpec.default is not supported on {dtype!r}: a default is "
            "one value, and only a scalar column takes one."
        )
    if dtype == pl.Null:
        raise SpecError(
            "ColSpec.default has no meaning on pl.Null: the column holds "
            "nothing but nulls, which a default would replace."
        )
    if isinstance(default, float) and math.isnan(default):
        raise SpecError(
            "ColSpec.default cannot be NaN: a default fills a null, and a "
            "NaN is a value, not a stand-in for one."
        )
    if isinstance(dtype, pl.Decimal) and isinstance(default, str):
        with suppress(decimal.InvalidOperation):
            default = decimal.Decimal(default)
    textual = dtype in (pl.String, pl.Utf8, pl.Categorical) or isinstance(
        dtype, (pl.Categorical, pl.Enum)
    )
    if textual and not isinstance(default, str):
        raise SpecError(
            f"ColSpec.default {default!r} is not text, and {dtype!r} holds "
            "text. Write it as a string."
        )
    if isinstance(dtype, pl.Enum):
        categories = dtype.categories.to_list()
        if default not in categories:
            raise SpecError(
                f"ColSpec.default {default!r} is not among this column's "
                f"Enum categories {categories}"
            )
    elif unholdable([default], dtype):
        raise SpecError(
            f"ColSpec.default {default!r} cannot be held by {dtype!r}: it "
            "would come out as a null or as a different value."
        )
    # A validator names its column, which a ColSpec does not know: the
    # one-row frame takes that name. Validators naming several columns
    # are a TableSpec's to refuse, and are not asked here.
    named = {n for v in spec.validators for n in v.expr.meta.root_names()}
    name = next(iter(named)) if len(named) == 1 else "default"
    held = pl.Series(name, [default], dtype=dtype, strict=False)
    broken = colspec_module.claims_refused(spec, held)
    if broken:
        raise SpecError(
            f"ColSpec.default {default!r} is not a value this column "
            f"accepts: it breaks its {', '.join(broken)}. A default fills a "
            "missing value, so it has to be one the column passes."
        )
    object.__setattr__(spec, "default", held[0])
