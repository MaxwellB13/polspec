"""The engine: a column's values drawn from its declaration.

Internal: not part of the public API.
"""

from __future__ import annotations

import dataclasses
import decimal
import functools
import random
import struct
import warnings
from typing import TYPE_CHECKING

import polars as pl

from polspec import frames
from polspec._ffi import (
    column_plan,
    decimal_from_physical,
    list_from_lengths,
    pattern_status,
)
from polspec._ffi import generate_dataframe as _generate_dataframe
from polspec._ffi import unique_values_at as _unique_values_at
from polspec.bound import Bound
from polspec.constants import (
    CATEGORICAL_PHYSICAL_CAPACITY,
    DEFAULT_FLOAT_BOUND,
    DEFAULT_LIST_LEN,
    DEFAULT_STRING_LEN,
    DEFAULT_WIDE_INT_BOUND,
    MAX_CARTESIAN_ROWS,
)
from polspec.domain import Domain
from polspec.dtypes import (
    TIME_UNIT_FACTORS,
    DtypeLike,
    bound_endpoint_to_physical,
    dtype_value_limits,
    field_dtypes,
    float16_inside,
    typed_values,
    wide_decimals,
)
from polspec.errors import GenerationError
from polspec.generation.seeds import pass_seed, stable_seed
from polspec.spec import ColSpec, column_kind, drawn_entries_of, element_of, field_of
from polspec.string_formats import lookup as _lookup_format

if TYPE_CHECKING:
    from collections.abc import Callable

    from polspec._polspec import ColumnPlan


def _resolve_bounded_categorical(name: str, spec: ColSpec, frame_seed: int) -> ColSpec:
    """Pins `choices` for a bare Categorical whose physical dtype caps how
    many distinct categories it can hold (UInt8/UInt16).

    Without `choices`, a Categorical generates unbounded-cardinality random
    strings, which will eventually try to insert more categories than a
    capacity-limited registry allows. Instead, generate a pool of
    representative string values up to that dtype's own maximum capacity
    once, then treat it exactly like a user-supplied `choices` list, so
    generation samples from (rather than overflows) the registry. The pool
    is seeded by the column's seed name, like everything else about it.
    """
    if spec.choices is not None or not isinstance(spec.dtype, pl.Categorical):
        return spec
    capacity = CATEGORICAL_PHYSICAL_CAPACITY.get(spec.dtype.categories.physical())
    if capacity is None:
        return spec
    seed = pass_seed(frame_seed, f"categorical:{spec.seed_name or name}")
    length = spec.string_length or Bound(*DEFAULT_STRING_LEN)
    cat_name = spec.dtype.categories.name()
    if cat_name:
        # A named pl.Categories() registry is meant to be shared across
        # ColSpecs/FrameSpecs (e.g. so two tables can be joined on physical
        # codes). Every one of them must draw its pool from the exact same
        # domain -- pinned to the registry's own identity, not this call's
        # seed -- otherwise their independently-chosen pools could still
        # jointly exceed the registry's capacity even though each one alone
        # stayed within it.
        pool_seed = stable_seed(
            cat_name,
            spec.dtype.categories.namespace(),
            str(length.min),
            str(length.max),
        )
    else:
        pool_seed = seed
    shortest, longest = length.closed()
    pool_plan = column_plan(
        "__pool", "string", str_min_len=int(shortest), str_max_len=int(longest)
    )
    pool_df = _generate_dataframe([pool_plan], capacity, pool_seed)
    # maintain_order=True: unique()'s default order isn't stable across calls,
    # which would make the same seed silently pick different pool[i] -> string
    # mappings and break reproducibility.
    domain = pool_df["__pool"].unique(maintain_order=True).to_list()
    return dataclasses.replace(spec, choices=domain, string_length=None)


def resolve_numeric_bounds(spec: ColSpec) -> tuple[float | int, float | int]:
    """Returns the (min, max) an int/float/temporal ColSpec generates within.

    Each end is the user's own bound where they gave one, and this dtype's
    default otherwise. An open end (`bounds=(0, None)`) therefore generates
    within the same default range it would have used with no bounds at all --
    generation cannot sample an unbounded range, so `validate()` is where an
    open end stays genuinely unconstrained.
    """
    lo, hi = _default_numeric_bounds(spec)
    if spec.bounds is None:
        return _representable(spec, lo, hi)

    if spec.bounds.min is not None:
        lo = bound_endpoint_to_physical(spec.bounds.min, spec.dtype)
    if spec.bounds.max is not None:
        hi = bound_endpoint_to_physical(spec.bounds.max, spec.dtype)

    # A closed end can sit outside the default range -- bounds=(2_000_000, None)
    # on Int64 leaves lo above the default hi of 1_000_000. Widen the open end
    # to the dtype's own limit rather than hand the engine an inverted range,
    # which it would silently swap.
    if lo > hi:
        limits = dtype_value_limits(_DRAWN_AS.get(spec.dtype, spec.dtype))
        if limits is not None:
            if spec.bounds.max is None:
                hi = limits[1]
            elif spec.bounds.min is None:
                lo = limits[0]
    return _representable(spec, lo, hi)


# A dtype the engine has no kind for is drawn as the nearest one it has and
# cast back in `_finish`, the way a Decimal is drawn as its physical integer:
# a half-precision float as a single. 128-bit integers have kinds of their
# own since 2.0, and are drawn at full width.
_DRAWN_AS: dict[DtypeLike, pl.DataType] = {
    pl.Float16: pl.Float32(),
}


def _representable(
    spec: ColSpec, lo: float | int, hi: float | int
) -> tuple[float | int, float | int]:
    """`[lo, hi]` narrowed to values the column can hold exactly at both ends.

    Only a `Float16` needs it. A value drawn as a `Float32` rounds to the
    nearest half when it is cast, and near an endpoint the half can land
    *outside* the bounds -- the nearest half to `0.1001` is above it. Drawn
    between the halves nearest each endpoint from inside, rounding can move
    a value only as far as a half it cannot pass.
    """
    if spec.dtype != pl.Float16:
        return lo, hi
    top = _FLOAT16_MAX
    return (
        float16_inside(max(float(lo), -top), up=True),
        float16_inside(min(float(hi), top), up=False),
    )


_FLOAT16_MAX = 65504.0


@functools.cache
def _every_float16() -> pl.Series:
    """Every finite half, ascending: 63,487 of them, distinct."""
    halves = {
        struct.unpack("<e", struct.pack("<H", bits))[0] for bits in range(1 << 16)
    }
    finite = sorted(h for h in halves if h == h and abs(h) != float("inf"))
    return pl.Series(finite, dtype=pl.Float16)


def _default_numeric_bounds(spec: ColSpec) -> tuple[float | int, float | int]:
    """The range polspec generates within when a ColSpec declares no bounds.

    A fixed-width int dtype defaults to its own range, temporal dtypes to a
    reasonable era or span, and anything else to the wide/default constants.

    "Its own range" is read from `dtype_value_limits` rather than kept as a
    second table here: what an Int16 may hold and what an Int16 generates
    within are the same nine words, and two copies of them is one copy that
    can go stale. The 64-bit types are the exception -- their full range is
    not a useful default -- so they fall through to the wide bound below.
    """
    kind = column_kind(spec.dtype)
    if kind == "int":
        if spec.dtype not in (pl.Int64, pl.UInt64, pl.Int128, pl.UInt128):
            limits = dtype_value_limits(spec.dtype)
            if limits is not None:
                return limits
        if spec.dtype.is_unsigned_integer():
            return 0, DEFAULT_WIDE_INT_BOUND
        return -DEFAULT_WIDE_INT_BOUND, DEFAULT_WIDE_INT_BOUND
    if kind == "float":
        return -DEFAULT_FLOAT_BOUND, DEFAULT_FLOAT_BOUND
    if kind == "decimal":
        # The float default, in physical units, unless the precision is
        # narrower. Drawn at full width, so any precision up to 38 holds.
        assert isinstance(spec.dtype, pl.Decimal)  # noqa: S101 - kind says so
        widest = min(
            10**spec.dtype.precision - 1,
            int(DEFAULT_FLOAT_BOUND) * 10**spec.dtype.scale,
        )
        return -widest, widest
    if kind == "temporal":
        if spec.dtype == pl.Date:
            return 0, 36525
        if spec.dtype == pl.Time:
            return 0, 86_399_999_999_999
        if isinstance(spec.dtype, pl.Datetime) or spec.dtype == pl.Datetime:
            factor = TIME_UNIT_FACTORS[getattr(spec.dtype, "time_unit", None) or "us"]
            return 0, 36525 * 86400 * factor
        if isinstance(spec.dtype, pl.Duration) or spec.dtype == pl.Duration:
            factor = TIME_UNIT_FACTORS[getattr(spec.dtype, "time_unit", None) or "us"]
            return 0, 365 * 86400 * factor
        return 0, DEFAULT_WIDE_INT_BOUND
    raise TypeError(f"{spec.dtype!r} is not a numeric or temporal dtype")


# The name the Rust engine knows each fixed-width dtype by. Anything absent --
# String, Binary, Boolean, Enum, Categorical -- keeps the kind `column_kind`
# already worked out.
_ENGINE_KINDS: dict[DtypeLike, str] = {
    pl.Int8: "int8",
    pl.Int16: "int16",
    pl.Int32: "int32",
    pl.Int64: "int64",
    pl.UInt8: "uint8",
    pl.UInt16: "uint16",
    pl.UInt32: "uint32",
    pl.UInt64: "uint64",
    pl.Int128: "int128",
    pl.UInt128: "uint128",
    pl.Float32: "float32",
    pl.Float64: "float64",
}


def _domain(spec: ColSpec) -> pl.Series | None:
    """The finite, typed domain a column draws from, if it has one: the
    values `Domain.of` declares it to hold -- `choices`, an Enum's
    categories, or a finite `format`'s list and its extras -- the one
    definition validation checks against too.

    The engine samples *indices* into this and the values are gathered back
    on the Python side, so a choice keeps its type -- a `datetime`, a
    `bytes`, a `True` -- with no string round-trip.
    """
    dtype = spec.value_dtype
    if spec.choices is None and spec.unique and dtype == pl.Float16:
        # Distinct singles can round to the same half, so a unique half is
        # drawn from the halves themselves: the finite set between the
        # bounds, sampled without replacement like any other domain.
        lo, hi = resolve_numeric_bounds(spec)
        halves = _every_float16()
        return halves.filter(halves.is_between(lo, hi))
    if spec.choices is None and spec.format is None and not isinstance(dtype, pl.Enum):
        # `Domain.of` finds a finite domain only in these three, and is not
        # asked for every plain column a frame draws: at 200 columns of 1,000
        # rows it cost 13% of the frame.
        return None
    values = Domain.of(spec).values
    if values is None:
        return None
    # A finite format's extras are among the values, weighted by
    # `_extended_weights`.
    return typed_values(values, dtype)


def joint_space(spec: ColSpec) -> tuple[int, Callable[[pl.Series], pl.Series]] | None:
    """How many values a composite key's member can take, and how to turn
    an index in `[0, size)` into one -- for drawing a `__unique_together__`
    group as one permutation of its members' combinations. None for a member
    that cannot be drawn so: one a rule or a derivation writes, one with no
    finite or bounded space (a float, a free string), and one whose space is
    past what a 64-bit index and 64-bit arithmetic reach.

    The index is uniform, so a member drawn this way takes each value of its
    space equally often: its weights and distribution are not kept.
    """
    if spec.rules or spec.derive is not None:
        return None
    dtype = spec.value_dtype
    domain = _domain(spec)
    if domain is not None:
        return len(domain), domain.gather
    if dtype == pl.Boolean:
        return 2, lambda index: (index == 1).alias(index.name)
    kind = column_kind(dtype)
    if kind not in ("int", "temporal") or dtype in (pl.Int128, pl.UInt128):
        return None
    lo, hi = (int(v) for v in resolve_numeric_bounds(spec))
    size = hi - lo + 1
    if lo >= 0 and size <= 2**64 - 1:

        def decode(index: pl.Series) -> pl.Series:
            return (index + lo).cast(_physical(dtype)).cast(dtype)

        return size, decode
    if lo < 0 and size <= 2**63:

        def decode(index: pl.Series) -> pl.Series:
            return (index.cast(pl.Int64) + lo).cast(_physical(dtype)).cast(dtype)

        return size, decode
    return None


def _physical(dtype: pl.DataType) -> pl.DataType:
    """The integer type a temporal dtype stores, or the integer dtype itself."""
    if dtype == pl.Date:
        return pl.Int32()
    if dtype.is_temporal():
        return pl.Int64()
    return dtype


def _extended_weights(listed: int, shares: tuple[float, ...]) -> list[float]:
    """Index weights for a finite format of `listed` values with extras of
    these `shares` after it: each extra its share, and the format's own
    values an even split of what the extras leave."""
    rest = max(0.0, 1.0 - sum(shares))
    return [rest / listed] * listed + list(shares)


def overlays_extras(spec: ColSpec) -> bool:
    """Whether `spec` is a template format with extra values, which the
    engine cannot draw in one column: the template's values, and a pick of
    which rows take an extra instead."""
    return (
        spec.extra_values is not None
        and spec.format is not None
        and not _lookup_format(spec.format).is_finite
    )


def plan_column(name: str, spec: ColSpec) -> tuple[ColumnPlan, pl.Series | None]:
    """The engine's instructions for one column, and the domain to gather
    from when it samples indices.
    """
    kind = column_kind(spec.dtype)
    weights = [float(w) for w in spec.weights] if spec.weights is not None else None
    options: dict = {
        "nullable": spec.nullable,
        "null_probability": spec.null_probability if spec.nullable else 0.0,
        # The engine draws a unique column without replacement, and refuses --
        # naming the column -- a domain too small to cover `n`. That check
        # needs the row count, which only the engine has, so it is not
        # duplicated here.
        "unique": spec.unique,
        # A renamed column that must keep its data is seeded as its old name.
        "seed_name": spec.seed_name,
    }

    domain = _domain(spec)
    if domain is not None and spec.format is not None and spec.extra_values:
        weights = _extended_weights(
            len(domain) - len(spec.extra_values), tuple(spec.extra_values.values())
        )
    if domain is not None:
        plan = column_plan(
            name, "index", n_categories=len(domain), weights=weights, **options
        )
        return plan, domain

    if kind in ("int", "float", "temporal", "decimal"):
        return _numeric_plan(name, spec, kind, options), None

    if kind == "bool":
        if spec.distribution_params:
            options["params"] = {
                k: float(v) for k, v in spec.distribution_params.items()
            }
        return column_plan(name, "bool", weights=weights, **options), None

    return _string_plan(name, spec, options)


def _numeric_plan(name: str, spec: ColSpec, kind: str, options: dict) -> ColumnPlan:
    """An integer, float, temporal or Decimal column's plan: its physical
    kind, its range, and the distribution it is drawn from."""
    engine_kind = _ENGINE_KINDS.get(_DRAWN_AS.get(spec.dtype, spec.dtype), kind)
    if kind == "decimal":
        # A Decimal is filled as its physical integer, at full width, and
        # scaled back in `_finish`.
        engine_kind = "int128"
    if spec.dtype.is_temporal():
        # Temporal columns cross as their physical integer: Date is an
        # i32 day count, everything else an i64 in its own time unit.
        engine_kind = "int32" if spec.dtype == pl.Date else "int64"
    if (
        spec.bounds is not None
        or spec.distribution is None
        or spec.distribution == "uniform"
    ):
        options["min"], options["max"] = resolve_numeric_bounds(spec)
    elif spec.dtype == pl.Float16:
        # A distribution with no bounds keeps its own shape, but not past
        # what a half holds: beyond it the cast is an infinity.
        options["min"], options["max"] = -_FLOAT16_MAX, _FLOAT16_MAX
    elif spec.dtype.is_temporal():
        # A non-uniform distribution with no explicit bounds is left to
        # its own shape rather than squeezed into polspec's default
        # range -- but a temporal dtype's own domain is not optional:
        # it reaches the engine as a bare integer kind, and without this
        # clamp could produce values the dtype cannot represent.
        limits = dtype_value_limits(spec.dtype)
        if limits is None:  # pragma: no cover - every temporal dtype has limits
            raise GenerationError(f"no value limits for {spec.dtype!r}")
        options["min"], options["max"] = limits
    options["distribution"] = spec.distribution
    if spec.distribution_params:
        options["params"] = {k: float(v) for k, v in spec.distribution_params.items()}
    return column_plan(name, engine_kind, **options)


def _string_plan(name: str, spec: ColSpec, options: dict) -> tuple[ColumnPlan, None]:
    """A text column's plan: a format's template, a pattern, or free strings
    of the declared lengths."""
    # A shaped string: the engine fills each value from the format's template.
    if spec.format is not None:
        template = list(_lookup_format(spec.format).template or ())
        return column_plan(name, "template", template=template, **options), None

    # Free strings: String, Binary, and a Categorical with no pinned domain.
    length = spec.string_length or Bound(*DEFAULT_STRING_LEN)
    shortest, longest = length.closed()

    # A pattern: each value a full match of it, walked through the DFA
    # validation's engine builds -- unless every free string below already
    # passes it, whose output a seed keeps, or it is out of the generator's
    # reach, when the column is the free string it was before 1.4.
    if spec.pattern is not None:
        status, reason = pattern_generation(spec)
        if status == "generate":
            if spec.string_length is not None:
                options["str_min_len"], options["str_max_len"] = (
                    int(shortest),
                    int(longest),
                )
            return column_plan(name, "pattern", pattern=spec.pattern, **options), None
        if status == "unreachable":
            warnings.warn(
                f"Column {name!r} is not generated from its pattern "
                f"{spec.pattern!r}: {reason}. It is filled with random text, as "
                "before 1.4, which validation will not accept; validate with "
                "pattern=False, or pin it with fixed=.",
                stacklevel=3,
            )

    options["str_min_len"], options["str_max_len"] = int(shortest), int(longest)
    return column_plan(name, "string", **options), None


def pattern_generation(spec: ColSpec) -> tuple[str, str | None]:
    """How a column with a `pattern` is generated: `("generate", None)` from
    the pattern; `("keep", None)` as plain random text, every value of which
    already passes it; or `("unreachable", reason)`, as plain text that may
    not -- a Unicode word boundary, an automaton too large, no match within
    the declared lengths."""
    if spec.pattern is None:
        raise ValueError("pattern_generation needs a column with a pattern")
    length = spec.string_length or Bound(*DEFAULT_STRING_LEN)
    shortest, longest = (int(v) for v in length.closed())
    declared = (shortest, longest) if spec.string_length is not None else None
    return pattern_status(spec.pattern, declared, (shortest, longest), spec.unique)


_ENGINE_NATIVE: frozenset = frozenset(
    {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Boolean,
        pl.String,
    }
)


def as_decimal(raw: pl.Series, dtype: pl.Decimal) -> pl.Series:
    """An integer column as the Decimal it is the physical form of.

    Polars casts an integer to a Decimal *value*, not to its physical, so
    the engine builds it: the integer `v` is the decimal `v / 10**scale`,
    with nothing divided.
    """
    precision = dtype.precision if dtype.precision is not None else 38
    return decimal_from_physical(raw, precision, dtype.scale or 0).alias(raw.name)


def _finish(raw: pl.Series, spec: ColSpec, domain: pl.Series | None) -> pl.Series:
    """A raw engine column as the declared dtype: gathered from its domain,
    cast from its physical form, or as it came.
    """
    if domain is not None:
        return domain.gather(raw).alias(raw.name)
    if spec.dtype in _ENGINE_NATIVE:
        return raw
    if isinstance(spec.dtype, pl.Decimal):
        return as_decimal(raw, spec.dtype)
    # Temporal from its physical integer; Binary from strings; a Categorical
    # cast to spec.dtype itself so a named pl.Categories() registry survives.
    return raw.cast(spec.dtype)


@wide_decimals
def _coverage_values(spec: ColSpec, seed: int) -> list | None:
    """The finite set of representative values a coverage dimension takes.

    Enum/Boolean contribute their whole domain; numeric columns contribute
    one representative each from the negative/zero/positive partitions their
    bounds actually reach (e.g. bounds of 5..100 only reaches "positive").
    Nullable columns also get a `None` entry. Returns None for columns with
    no natural finite domain (String, bare Categorical) -- those are filled
    in with ordinary random generation instead.
    """
    if spec.choices is not None:
        values = list(spec.choices)
        if spec.nullable:
            values.append(None)
        return values

    kind = column_kind(spec.dtype)
    values: list = []
    rng = random.Random(seed)

    if isinstance(spec.dtype, pl.Enum):
        values = list(spec.dtype.categories.to_list())
    elif kind == "bool":
        values = [True, False]
    elif kind in ("int", "temporal", "decimal"):
        lo, hi = (int(v) for v in resolve_numeric_bounds(spec))
        if lo < 0:
            values.append(rng.randint(lo, min(hi, -1)))
        if lo <= 0 <= hi:
            values.append(0)
        if hi > 0:
            values.append(rng.randint(max(lo, 1), hi))
        if isinstance(spec.dtype, pl.Decimal):
            # Drawn in physical units; a coverage value is the Decimal itself.
            scale = spec.dtype.scale
            values = [decimal.Decimal(v).scaleb(-scale) for v in values]
    elif kind == "float":
        lo, hi = (float(v) for v in resolve_numeric_bounds(spec))
        if lo < 0:
            upper = hi if hi < 0 else (lo / 2.0 if lo > -2e-9 else -1e-9)
            values.append(rng.uniform(lo, upper))
        if lo <= 0.0 <= hi:
            values.append(0.0)
        if hi > 0:
            lower = lo if lo > 0 else (hi / 2.0 if hi < 2e-9 else 1e-9)
            values.append(rng.uniform(lower, hi))
    else:
        return None

    if spec.nullable:
        values.append(None)
    return values


def generate_random(
    columns: dict[str, ColSpec], n: int, seed: int | None, row_offset: int = 0
) -> pl.DataFrame:
    """`n` rows drawn per column -- rows `[row_offset, row_offset + n)` of the
    frame `seed` describes, so a batch is a window onto one frame. A List
    column's lengths are a window too; its elements are drawn per call,
    and so are the fields of a struct."""
    if not columns:
        return pl.DataFrame()
    frame_seed = seed if seed is not None else random.randrange(2**63)
    columns = {
        name: _resolve_bounded_categorical(name, spec, frame_seed)
        for name, spec in columns.items()
    }
    # `column_kind` refuses what the engine cannot fill, so every column is
    # classified before any is generated. The scalar ones share a single
    # engine call; a nested one is built from the columns it contains.
    scalars = {
        n_: s
        for n_, s in columns.items()
        if column_kind(s.dtype) not in ("list", "struct", "map", "null")
        and not overlays_extras(s)
    }
    plans: list[ColumnPlan] = []
    domains: dict[str, pl.Series | None] = {}
    for name, spec in scalars.items():
        plan, domain = plan_column(name, spec)
        plans.append(plan)
        domains[name] = domain
    # Each raw column is dropped as it is finished, so a column the finish
    # copies (a temporal cast, a gathered domain) never exists twice.
    raw = _generate_dataframe(plans, n, seed, row_offset).to_dict() if plans else {}
    finished: dict[str, pl.Series] = {}
    for name, spec in columns.items():
        if name in scalars:
            values = raw.pop(name)
            if frames.is_pattern(name):
                # Finished under a plain name -- see `frames.plain` -- and
                # seeded by its own, so its NaNs are the ones it would draw.
                values = frames.plain(values)
                spec = dataclasses.replace(spec, seed_name=spec.seed_name or name)
            finished[name] = _with_nans(
                _finish(values, spec, domains[name]), spec, n, seed, row_offset
            ).alias(name)
        else:
            finished[name] = _generate_column(name, spec, n, seed, row_offset)
    # From a list, which is several times cheaper for a wide frame than a
    # mapping; but Polars renames a series called "" `column_<i>` there, so
    # the names are put back when that happened.
    frame = pl.DataFrame([finished[name] for name in columns])
    names = list(columns)
    if frame.columns != names:
        frame.columns = names
    return frame


def readable_at(spec: ColSpec) -> bool:
    """Whether `unique_at` can read `spec`'s column at given rows: a unique
    scalar column with no nulls, no NaNs and no format extras, whose value at
    a row depends on that row alone."""
    return (
        spec.unique
        and not spec.nullable
        and not spec.nan_probability
        and column_kind(spec.dtype) not in ("list", "struct", "map", "null")
        and not overlays_extras(spec)
    )


def unique_at(name: str, spec: ColSpec, seed: int, rows: pl.Series) -> pl.Series:
    """What `generate_random({name: spec}, n, seed)[name]` holds at `rows`, for
    rows anywhere in the frame and in any order, without drawing the rows
    between: a unique scalar column with no nulls and no NaNs, read where a
    reference points.

    A column with nulls or NaNs, a nested one, or a format whose extras are
    drawn beside it is refused by name: where such a row's value sits in
    the column's space depends on the rows before it.
    """
    if not readable_at(spec):
        raise GenerationError(
            f"Column {name!r} cannot be read at given rows: only a unique "
            "scalar column with no nulls, no NaNs and no format extras can."
        )
    if frames.is_pattern(name):
        spec = dataclasses.replace(spec, seed_name=spec.seed_name or name)
    spec = _resolve_bounded_categorical(name, spec, seed)
    plan, domain = plan_column(name, spec)
    return _finish(_unique_values_at(plan, seed, rows), spec, domain).alias(name)


def _generate_column(
    name: str, spec: ColSpec, n: int, seed: int | None, row_offset: int = 0
) -> pl.Series:
    """One column of `n` values, whatever its dtype nests.

    The scalar path is a plan handed to the engine; a `List` wraps the
    column its elements make, a `Map` is the list of entries it is, and a
    `Struct` gathers the columns its fields make -- each by calling back here, so a dtype nests as deeply as it
    likes and every value is drawn by the code that draws a column of its
    own type.
    """
    if frames.is_pattern(name):
        # Many Series methods select their Series by name, so one named as a
        # pattern loses itself. Drawn under a plain name and seeded by its
        # own, it holds exactly the values it would have.
        seeded = dataclasses.replace(spec, seed_name=spec.seed_name or name)
        return _generate_column(frames.PLAIN, seeded, n, seed, row_offset).alias(name)
    kind = column_kind(spec.dtype)
    if kind == "null":
        # Nothing to draw: every value of a Null column is null.
        return pl.Series(name, [None] * n, dtype=pl.Null)
    if kind == "list":
        return _generate_list_column(name, spec, n, seed, row_offset)
    if kind == "map":
        return _generate_map_column(name, spec, n, seed, row_offset)
    if kind == "struct":
        return _generate_struct_column(name, spec, n, seed, row_offset)
    if overlays_extras(spec):
        return _generate_with_extras(name, spec, n, seed, row_offset)
    plan, domain = plan_column(name, spec)
    return _with_nans(
        _finish(_generate_dataframe([plan], n, seed, row_offset)[name], spec, domain),
        spec,
        n,
        seed,
        row_offset,
    )


def _with_nans(
    values: pl.Series, spec: ColSpec, n: int, seed: int | None, row_offset: int = 0
) -> pl.Series:
    """`values` with NaN on `nan_probability` of its present rows.

    One engine column of coin flips, a window onto the frame like the values
    it overlays, seeded as a name no user column can carry: so a batch holds
    the NaNs the whole frame holds there, and a column that declares no NaN
    draws nothing more -- its seeded values are what they always were. A
    null row stays null; the share is of the present rows.
    """
    if not spec.nan_probability:
        return values
    name = values.name
    coin = column_plan(
        name,
        "bool",
        weights=[1.0 - spec.nan_probability, spec.nan_probability],
        seed_name=f"{spec.seed_name or name}\x00nan",
    )
    hit = _generate_dataframe([coin], n, seed, row_offset)[name]
    nans = pl.repeat(float("nan"), n, dtype=values.dtype, eager=True)
    return values.zip_with(~hit | values.is_null(), nans)


def _generate_with_extras(
    name: str, spec: ColSpec, n: int, seed: int | None, row_offset: int = 0
) -> pl.Series:
    """A template format with extra values: the template's values, and on
    each extra's share of the present rows that extra instead.

    Two engine columns, both windows onto the frame: the template's, drawn
    as the column always was, and one pick per row among the extras and
    "keep the template", seeded as a name no user column can carry. A null
    row stays null -- the shares are of the present rows.
    """
    assert spec.extra_values is not None  # noqa: S101 - the caller checked
    base_spec = dataclasses.replace(spec, extra_values=None)
    plan, domain = plan_column(name, base_spec)
    base = _finish(
        _generate_dataframe([plan], n, seed, row_offset)[name], base_spec, domain
    )
    extras = list(spec.extra_values)
    shares = list(spec.extra_values.values())
    pick_plan = column_plan(
        name,
        "index",
        n_categories=len(extras) + 1,
        weights=[*shares, max(0.0, 1.0 - sum(shares))],
        seed_name=f"{spec.seed_name or name}\x00extra",
    )
    pick = _generate_dataframe([pick_plan], n, seed, row_offset)[name]
    last = len(extras) - 1
    replacement = pl.Series(name, extras, dtype=pl.String).gather(
        pick.clip(upper_bound=last)
    )
    keep = (pick > last) | base.is_null()
    return base.zip_with(keep, replacement)


def _per_spec(build):
    """Caches `build(name, spec, seed_key)` per spec object, not per value:
    a stream hands the same `ColSpec` to every window, and a spec need not
    be hashable. The spec is held beside its value, so its id is not reused
    while the entry lives."""
    cache: dict[tuple[int, str, str], tuple[ColSpec, object]] = {}

    @functools.wraps(build)
    def cached(name: str, spec: ColSpec, seed_key: str):
        key = (id(spec), name, seed_key)
        hit = cache.get(key)
        if hit is not None and hit[0] is spec:
            return hit[1]
        if len(cache) >= 256:
            cache.clear()
        value = build(name, spec, seed_key)
        cache[key] = (spec, value)
        return value

    return cached


def _field_spec(spec: ColSpec, name: str, seed_key: str) -> ColSpec:
    """The declaration one struct field is generated from.

    What `fields` says about it, or its dtype alone when `fields` says
    nothing -- `fields` is partial by design, and a field is null only
    where its own declaration says so. It is seeded under its parent so
    that renaming the column moves every field with it and adding a field
    beside one moves nothing.
    """
    return _field_spec_of(name, spec, seed_key)


@_per_spec
def _field_spec_of(name: str, spec: ColSpec, seed_key: str) -> ColSpec:
    return dataclasses.replace(field_of(spec, name), seed_name=f"{seed_key}.{name}")


def _generate_struct_column(
    name: str, spec: ColSpec, n: int, seed: int | None, row_offset: int = 0
) -> pl.Series:
    """A `Struct` column: one column per field, gathered into a struct.

    The cell's own nullability rides on a separate draw, so a null struct
    is a null cell rather than a struct of nulls.
    """
    dtype = spec.value_dtype
    assert isinstance(dtype, pl.Struct)  # noqa: S101 - the caller checked the kind
    seed_key = spec.seed_name or name

    fields = [
        _generate_column(
            field_name,
            _field_spec(spec, field_name, seed_key),
            n,
            seed,
            row_offset,
        )
        for field_name in field_dtypes(dtype)
    ]
    # A struct with no fields is still a struct: a frame of `n` rows and no
    # columns gathers into one, where a list of no series cannot say how
    # many rows it has.
    cells = (pl.DataFrame(fields) if fields else pl.DataFrame(height=n)).to_struct(name)
    if not spec.nullable:
        return cells
    return cells.zip_with(_present(name, spec, n, seed, row_offset), pl.Series([None]))


def _present(
    name: str, spec: ColSpec, n: int, seed: int | None, row_offset: int
) -> pl.Series:
    """A mask of the rows a nullable cell is present on, drawn at the
    declared rate from a name no user column can carry."""
    plan = column_plan(
        name,
        "int64",
        nullable=True,
        null_probability=spec.null_probability,
        seed_name=f"{spec.seed_name or name}\x00null",
    )
    return _generate_dataframe([plan], n, seed, row_offset)[name].is_not_null()


def _generate_list_column(
    name: str, spec: ColSpec, n: int, seed: int | None, row_offset: int = 0
) -> pl.Series:
    """A `List` or `Array` column: the lengths as one engine column, the
    elements as another of the inner dtype, wrapped by the lengths.

    Both are seeded from the column's own seed key -- the elements *as* it,
    the lengths as a name no user column can carry -- so a List column keeps
    its data across a rename like any other, and its elements are drawn by
    the same code as a scalar column of the inner dtype would be.
    """
    lengths, elements, _ = _list_parts(name, spec, n, seed, row_offset)
    return _wrap_list(name, spec, lengths, elements)


def _list_parts(
    name: str, spec: ColSpec, n: int, seed: int | None, row_offset: int
) -> tuple[pl.Series, pl.Series, int]:
    """A `List` column before it is wrapped: each cell's length, null for a
    null cell, every element of every cell in one flat column, and where
    the first of those elements sits in the whole frame's element column.

    Both are windows onto the whole frame's. The lengths are an engine
    column; the elements are the frame's element column from the element
    the window's first row starts at -- the number the rows before it hold
    -- so a batch or a scan holds exactly the lists `generate()` holds there.
    """
    assert isinstance(spec.dtype, (pl.List, pl.Array))  # noqa: S101 - the caller checked
    seed_key = spec.seed_name or name
    lengths_plan, key = _lengths_plan(name, spec, seed_key)
    lengths = _generate_dataframe([lengths_plan], n, seed, row_offset)[name]
    total = int(lengths.fill_null(0).sum())
    start = 0
    if seed is not None:
        start = _elements_before(lengths_plan, key, seed, row_offset)
        # The next window of a stream starts where this one ends.
        _remember(seed, key, row_offset + n, start + total)
    # The element is a column in its own right, so a list of structs -- or
    # of lists -- is the same recursion one level down, a window onto its
    # own element column.
    elements = _generate_column(
        name, _element_spec(name, spec, seed_key), total, seed, start
    )
    return lengths, elements, start


@_per_spec
def _lengths_plan(name: str, spec: ColSpec, seed_key: str) -> tuple[ColumnPlan, tuple]:
    """The engine column a list's lengths are, and a key naming everything
    the lengths depend on. Built once per column, not once per window."""
    if isinstance(spec.dtype, pl.Array):
        width = spec.dtype.size
        bounds: tuple[int, int] = (width, width)
    else:
        bounds = spec.list_length.closed() if spec.list_length else DEFAULT_LIST_LEN
    rate = spec.null_probability if spec.nullable else 0.0
    # The lengths carry the list's own nullability: a null length is a null
    # cell. Seeded as a name no user column can carry.
    plan = column_plan(
        name,
        "int64",
        min=bounds[0],
        max=bounds[1],
        nullable=spec.nullable,
        null_probability=rate,
        seed_name=f"{seed_key}\x00len",
    )
    return plan, (seed_key, bounds, spec.nullable, rate)


@_per_spec
def _element_spec(_name: str, spec: ColSpec, seed_key: str) -> ColSpec:
    """The declaration a list's elements are drawn from, built once per
    column. A scalar element is seeded as the column itself, so a list keeps
    its data across a rename like any other column; a nested one under a
    name of its own, so an inner list's lengths are not the outer's. A
    column named "" seeds its elements by that name, as `None` does: a
    `seed_name` cannot be empty."""
    element = element_of(spec)
    nested = column_kind(element.dtype) in ("list", "map")
    key = f"{seed_key}\x00item" if nested else (seed_key or None)
    return dataclasses.replace(element, seed_name=key)


# How many elements the rows before a window hold, per list column and seed:
# the last offset each was asked at. A stream asks for its windows in order,
# so each extends the last by one batch's lengths, not the frame's.
_ELEMENTS_BEFORE: dict[tuple, tuple[int, int]] = {}
_MAX_REMEMBERED = 256
# Rows of lengths summed at a time, when the gap to a window is long.
_LENGTHS_PER_STEP = 1 << 22


def _elements_before(plan: ColumnPlan, key: tuple, seed: int, row_offset: int) -> int:
    """The elements rows `[0, row_offset)` of a list column hold."""
    if row_offset == 0:
        return 0
    remembered = (seed, key)
    at, total = _ELEMENTS_BEFORE.get(remembered, (0, 0))
    if at > row_offset:
        at, total = 0, 0
    while at < row_offset:
        rows = min(row_offset - at, _LENGTHS_PER_STEP)
        lengths = _generate_dataframe([plan], rows, seed, at).to_series()
        total += int(lengths.fill_null(0).sum())
        at += rows
    _remember(seed, key, row_offset, total)
    return total


def _remember(seed: int, key: tuple, row_offset: int, elements: int) -> None:
    """That rows `[0, row_offset)` of the list column `key` names hold
    `elements` elements under `seed`."""
    if len(_ELEMENTS_BEFORE) >= _MAX_REMEMBERED:
        _ELEMENTS_BEFORE.clear()
    _ELEMENTS_BEFORE[(seed, key)] = (row_offset, elements)


def _row_of_element(lengths: pl.Series) -> pl.Series:
    """The cell each element of a flat list column belongs to."""
    counts = lengths.fill_null(0)
    return (
        pl.int_range(0, len(lengths), eager=True)
        .repeat_by(counts)
        .explode(empty_as_null=False)
    )


def _wrap_list(
    name: str, spec: ColSpec, lengths: pl.Series, elements: pl.Series
) -> pl.Series:
    """The `List` (or `Array`) column `_list_parts` describes: the flat
    elements, cut into cells by the lengths; a null length is a null cell.
    Built from the lengths as offsets, by the engine, with no element
    copied."""
    cells = list_from_lengths(name, lengths, elements)
    if cells.dtype != pl.List(elements.dtype):
        # A dtype the engine's Polars does not know -- Polars 2's `Map` --
        # reaches it as its physical form, the list of entries it is, and
        # comes back so. The values are the same; the dtype is put back.
        cells = cells.cast(pl.List(elements.dtype))
    if isinstance(spec.dtype, pl.Array):
        return cells.list.to_array(spec.dtype.size)
    return cells


# A map whose keys still repeat after this many rounds of redrawing the
# repeats has keys too few for its length -- the same bound, and the same
# reasoning, as a composite key's (`generation.passes.unique_together.MAX_ROUNDS`).
_MAX_KEY_ROUNDS = 50


def _generate_map_column(
    name: str, spec: ColSpec, n: int, seed: int | None, row_offset: int = 0
) -> pl.Series:
    """A `Map` column: the list of `{key, value}` entries it is, drawn as a
    list of structs, its repeated keys redrawn, and cast to the map.

    The cast would not refuse a repeated key -- Polars folds the two entries
    into one -- so a map drawn with one would come out shorter than its
    `list_length`. The repeats are separated first, and only they move.
    """
    entries = drawn_entries_of(spec)
    lengths, elements, start = _list_parts(name, entries, n, seed, row_offset)
    elements = _separate_repeated_keys(name, entries, lengths, elements, seed, start)
    return _wrap_list(name, entries, lengths, elements).cast(spec.dtype)


def _separate_repeated_keys(
    name: str,
    entries: ColSpec,
    lengths: pl.Series,
    elements: pl.Series,
    seed: int | None,
    start: int,
) -> pl.Series:
    """The flat entries of a map column with no key repeated within a map.

    As `generation.passes.unique_together.apply_unique_together` separates a composite
    key: the first entry to use a key keeps it, every later one in the same
    map takes a fresh key, and that repeats until none are left. Its value
    stays -- only the key was the problem. Round `r`'s fresh key for an entry
    is that entry's value in a key column of round `r`'s own, a window from
    `start`, so it depends on where the entry sits in the whole frame and a
    batch separates its keys as the frame does.
    """
    if elements.len() == 0:
        return elements
    frame = pl.DataFrame(
        {"__row": _row_of_element(lengths), "key": elements.struct.field("key")}
    )
    repeats = ~pl.struct("__row", "key").is_first_distinct()

    def repeated(frame: pl.DataFrame) -> pl.Series:
        return frame.select(repeats).to_series().arg_true()

    rows = repeated(frame)
    if rows.len() == 0:
        return elements
    seed_key = entries.seed_name or name
    key = field_of(entries, "key")
    for round_ in range(_MAX_KEY_ROUNDS):
        spec = dataclasses.replace(key, seed_name=f"{seed_key}\x00key{round_}")
        fresh = _generate_column(name, spec, elements.len(), seed, start).gather(rows)
        frame = frame.with_columns(frame["key"].scatter(rows, fresh))
        rows = repeated(frame)
        if rows.len() == 0:
            value = elements.struct.field("value")
            return pl.DataFrame([frame["key"], value]).to_struct(name)
    raise GenerationError(
        f"Map column '{name}' still repeats a key within a map on {rows.len()} "
        f"entr{'y' if rows.len() == 1 else 'ies'} after {_MAX_KEY_ROUNDS} rounds "
        "of redrawing them. Its keys are too few for its length -- widen the "
        "key's domain, or lower list_length."
    )


def generate_cartesian(
    columns: dict[str, ColSpec], n: int, seed: int | None
) -> pl.DataFrame:
    """Guarantees coverage: the cartesian product of every Enum/Boolean's
    categories crossed with the negative/zero/positive/null partitions of
    every bounded numeric column, padded with ordinary random rows up to
    `n` if the coverage set doesn't already reach it.

    A `unique` column takes no part in the product. Crossing a column with
    anything repeats it, and concatenating the padding rows would repeat it
    again, so unique columns are held back and drawn once over the finished
    frame -- where "once" is what makes them distinct.
    """
    if n == 0:
        # Nothing was asked for, so nothing is covered: the same empty, typed
        # frame `generate(0)` and the sinks produce.
        return generate_random(columns, 0, seed)

    frame_seed = seed if seed is not None else random.randrange(2**63)
    rng = random.Random(frame_seed)

    coverage_values: dict[str, list] = {}
    filler_columns: dict[str, ColSpec] = {}
    unique_columns: dict[str, ColSpec] = {}
    for name, spec in columns.items():
        if spec.unique:
            unique_columns[name] = spec
            continue
        if spec.derive is not None:
            # Its inputs decide it: no dimension of its own to cover.
            filler_columns[name] = spec
            continue
        # Keyed by the column, so an inserted dimension leaves its
        # neighbours' representatives alone.
        values = _coverage_values(
            spec, pass_seed(frame_seed, f"coverage:{spec.seed_name or name}")
        )
        if values is None:
            filler_columns[name] = spec
        else:
            coverage_values[name] = values

    if not coverage_values:
        raise GenerationError(
            "method='cartesian' needs at least one non-unique Enum, Boolean, "
            "or bounded numeric column to build coverage from"
        )

    coverage_size = 1
    for values in coverage_values.values():
        coverage_size *= len(values)
    if coverage_size > MAX_CARTESIAN_ROWS:
        breakdown = ", ".join(f"{name}={len(v)}" for name, v in coverage_values.items())
        raise GenerationError(
            f"cartesian coverage would need {coverage_size:,} rows ({breakdown}), "
            f"which exceeds the {MAX_CARTESIAN_ROWS:,}-row safety cap"
        )

    dims = [
        pl.DataFrame({name: values}, schema={name: columns[name].dtype})
        for name, values in coverage_values.items()
    ]
    coverage_df = dims[0]
    for dim_df in dims[1:]:
        coverage_df = coverage_df.join(dim_df, how="cross")

    coverage_n = coverage_df.height

    if filler_columns:
        filler_df = generate_random(filler_columns, coverage_n, rng.randrange(2**63))
        # Drawn at the coverage frame's height, so the two line up row for
        # row.
        coverage_df = coverage_df.hstack(filler_df)

    spread = [name for name in columns if name not in unique_columns]
    coverage_df = coverage_df.select(frames.columns(spread))

    if coverage_n < n:
        topup_df = generate_random(
            {name: columns[name] for name in spread},
            n - coverage_n,
            rng.randrange(2**63),
        )
        coverage_df = pl.concat([coverage_df, topup_df], how="vertical")

    if unique_columns:
        # One draw over the whole frame, after any padding: a unique column
        # is only distinct if nothing else ever appends to it.
        distinct_df = generate_random(
            unique_columns, coverage_df.height, rng.randrange(2**63)
        )
        coverage_df = coverage_df.hstack(distinct_df)

    return coverage_df.select(frames.columns(columns))
