from __future__ import annotations

import dataclasses
import decimal
import hashlib
import random
from typing import TYPE_CHECKING

import polars as pl

from polspec._ffi import column_plan
from polspec._ffi import generate_dataframe as _generate_dataframe
from polspec.bound import Bound
from polspec.constants import (
    _CATEGORICAL_PHYSICAL_CAPACITY,
    _DEFAULT_FLOAT_BOUND,
    _DEFAULT_LIST_LEN,
    _DEFAULT_STRING_LEN,
    _DEFAULT_WIDE_INT_BOUND,
    _I64_MAX,
    _MAX_CARTESIAN_ROWS,
)
from polspec.dtypes import (
    _TIME_UNIT_FACTORS,
    DtypeLike,
    _bound_endpoint_to_physical,
    _dtype_value_limits,
    _typed_values,
    field_dtypes,
)
from polspec.errors import GenerationError
from polspec.formats import lookup as _lookup_format
from polspec.generation.seeds import pass_seed
from polspec.spec import ColSpec, _column_kind

if TYPE_CHECKING:
    from polspec._polspec import ColumnPlan


def _stable_seed(*parts: str) -> int:
    """A seed derived only from `parts`, stable across processes and runs
    (unlike `hash()`, which is salted per-process for strings).
    """
    digest = hashlib.sha256("\0".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


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
    capacity = _CATEGORICAL_PHYSICAL_CAPACITY.get(spec.dtype.categories.physical())
    if capacity is None:
        return spec
    seed = pass_seed(frame_seed, f"categorical:{spec.seed_name or name}")
    length = spec.string_length or Bound(*_DEFAULT_STRING_LEN)
    cat_name = spec.dtype.categories.name()
    if cat_name:
        # A named pl.Categories() registry is meant to be shared across
        # ColSpecs/FrameSpecs (e.g. so two tables can be joined on physical
        # codes). Every one of them must draw its pool from the exact same
        # domain -- pinned to the registry's own identity, not this call's
        # seed -- otherwise their independently-chosen pools could still
        # jointly exceed the registry's capacity even though each one alone
        # stayed within it.
        pool_seed = _stable_seed(
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


def _resolve_numeric_bounds(spec: ColSpec) -> tuple[float | int, float | int]:
    """Returns the (min, max) an int/float/temporal ColSpec generates within.

    Each end is the user's own bound where they gave one, and this dtype's
    default otherwise. An open end (`bounds=(0, None)`) therefore generates
    within the same default range it would have used with no bounds at all --
    generation cannot sample an unbounded range, so `validate()` is where an
    open end stays genuinely unconstrained.
    """
    lo, hi = _default_numeric_bounds(spec)
    if spec.bounds is None:
        return lo, hi

    if spec.bounds.min is not None:
        lo = _bound_endpoint_to_physical(spec.bounds.min, spec.dtype)
    if spec.bounds.max is not None:
        hi = _bound_endpoint_to_physical(spec.bounds.max, spec.dtype)

    # A closed end can sit outside the default range -- bounds=(2_000_000, None)
    # on Int64 leaves lo above the default hi of 1_000_000. Widen the open end
    # to the dtype's own limit rather than hand the engine an inverted range,
    # which it would silently swap.
    if lo > hi:
        limits = _dtype_value_limits(spec.dtype)
        if limits is not None:
            if spec.bounds.max is None:
                hi = limits[1]
            elif spec.bounds.min is None:
                lo = limits[0]
    return lo, hi


def _default_numeric_bounds(spec: ColSpec) -> tuple[float | int, float | int]:
    """The range polspec generates within when a ColSpec declares no bounds.

    A fixed-width int dtype defaults to its own range, temporal dtypes to a
    reasonable era or span, and anything else to the wide/default constants.

    "Its own range" is read from `_dtype_value_limits` rather than kept as a
    second table here: what an Int16 may hold and what an Int16 generates
    within are the same nine words, and two copies of them is one copy that
    can go stale. The 64-bit types are the exception -- their full range is
    not a useful default -- so they fall through to the wide bound below.
    """
    kind = _column_kind(spec.dtype)
    if kind == "int":
        if spec.dtype not in (pl.Int64, pl.UInt64):
            limits = _dtype_value_limits(spec.dtype)
            if limits is not None:
                return limits
        if spec.dtype.is_unsigned_integer():
            return 0, _DEFAULT_WIDE_INT_BOUND
        return -_DEFAULT_WIDE_INT_BOUND, _DEFAULT_WIDE_INT_BOUND
    if kind == "float":
        return -_DEFAULT_FLOAT_BOUND, _DEFAULT_FLOAT_BOUND
    if kind == "decimal":
        # The float default, in physical units, unless the precision is
        # narrower -- and never past what the engine's 64-bit draw can hold.
        assert isinstance(spec.dtype, pl.Decimal)  # noqa: S101 - kind says so
        widest = min(
            10**spec.dtype.precision - 1,
            int(_DEFAULT_FLOAT_BOUND) * 10**spec.dtype.scale,
            _I64_MAX,
        )
        return -widest, widest
    if kind == "temporal":
        if spec.dtype == pl.Date:
            return 0, 36525
        if spec.dtype == pl.Time:
            return 0, 86_399_999_999_999
        if isinstance(spec.dtype, pl.Datetime) or spec.dtype == pl.Datetime:
            factor = _TIME_UNIT_FACTORS[getattr(spec.dtype, "time_unit", None) or "us"]
            return 0, 36525 * 86400 * factor
        if isinstance(spec.dtype, pl.Duration) or spec.dtype == pl.Duration:
            factor = _TIME_UNIT_FACTORS[getattr(spec.dtype, "time_unit", None) or "us"]
            return 0, 365 * 86400 * factor
        return 0, _DEFAULT_WIDE_INT_BOUND
    raise TypeError(f"{spec.dtype!r} is not a numeric or temporal dtype")


# The name the Rust engine knows each fixed-width dtype by. Anything absent --
# String, Binary, Boolean, Enum, Categorical -- keeps the kind `_column_kind`
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
    pl.Float32: "float32",
    pl.Float64: "float64",
}


def _domain(spec: ColSpec) -> pl.Series | None:
    """The finite, typed domain a column draws from, if it has one.

    Declared `choices`, an Enum's categories, or a finite `format`'s list.
    The engine samples *indices* into this and the values are gathered back
    on the Python side, so a choice keeps its type -- a `datetime`, a
    `bytes`, a `True` -- with no string round-trip.
    """
    dtype = spec.value_dtype
    if spec.choices is not None:
        return _typed_values(spec.choices, dtype)
    if isinstance(dtype, pl.Enum):
        return pl.Series(dtype.categories, dtype=dtype)
    if spec.format is not None:
        fmt = _lookup_format(spec.format)
        if fmt.is_finite:
            return _typed_values(fmt.values, dtype)
    return None


def _plan_column(name: str, spec: ColSpec) -> tuple[ColumnPlan, pl.Series | None]:
    """The engine's instructions for one column, and the domain to gather
    from when it samples indices.
    """
    kind = _column_kind(spec.dtype)
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
    if domain is not None:
        plan = column_plan(
            name, "index", n_categories=len(domain), weights=weights, **options
        )
        return plan, domain

    if kind in ("int", "float", "temporal", "decimal"):
        engine_kind = _ENGINE_KINDS.get(spec.dtype, kind)
        if kind == "decimal":
            # A Decimal is filled as its physical integer and scaled back in
            # `_finish`; the draw is 64-bit, whatever the declared precision.
            engine_kind = "int64"
        if spec.dtype.is_temporal():
            # Temporal columns cross as their physical integer: Date is an
            # i32 day count, everything else an i64 in its own time unit.
            engine_kind = "int32" if spec.dtype == pl.Date else "int64"
        if (
            spec.bounds is not None
            or spec.distribution is None
            or spec.distribution == "uniform"
        ):
            options["min"], options["max"] = _resolve_numeric_bounds(spec)
            if kind == "decimal":
                _check_decimal_fits_the_draw(name, spec, options["min"], options["max"])
        elif spec.dtype.is_temporal():
            # A non-uniform distribution with no explicit bounds is left to
            # its own shape rather than squeezed into polspec's default
            # range -- but a temporal dtype's own domain is not optional:
            # it reaches the engine as a bare integer kind, and without this
            # clamp could produce values the dtype cannot represent.
            limits = _dtype_value_limits(spec.dtype)
            if limits is None:  # pragma: no cover - every temporal dtype has limits
                raise GenerationError(f"no value limits for {spec.dtype!r}")
            options["min"], options["max"] = limits
        options["distribution"] = spec.distribution
        if spec.distribution_params:
            options["params"] = {
                k: float(v) for k, v in spec.distribution_params.items()
            }
        return column_plan(name, engine_kind, **options), None

    if kind == "bool":
        if spec.distribution_params:
            options["params"] = {
                k: float(v) for k, v in spec.distribution_params.items()
            }
        return column_plan(name, "bool", weights=weights, **options), None

    # A shaped string: the engine fills each value from the format's template.
    if spec.format is not None:
        template = list(_lookup_format(spec.format).template or ())
        return column_plan(name, "template", template=template, **options), None

    # Free strings: String, Binary, and a Categorical with no pinned domain.
    length = spec.string_length or Bound(*_DEFAULT_STRING_LEN)
    shortest, longest = length.closed()
    options["str_min_len"], options["str_max_len"] = int(shortest), int(longest)
    return column_plan(name, "string", **options), None


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


def _check_decimal_fits_the_draw(
    name: str, spec: ColSpec, lo: float | int, hi: float | int
) -> None:
    if lo >= -_I64_MAX - 1 and hi <= _I64_MAX:
        return
    raise GenerationError(
        f"Column {name!r}: bounds {spec.bounds} on {spec.dtype!r} need more "
        "than 18 significant digits, which generation cannot draw. Validation "
        "checks the full precision; narrow the bounds to generate."
    )


def _decimal_from_physical(raw: pl.Series, dtype: pl.Decimal) -> pl.Series:
    """An integer column as the Decimal it is the physical form of.

    Polars casts an integer to a Decimal *value*, not to its physical, so
    the scale is applied as a division in the widest Decimal and the result
    narrowed to the declared precision.
    """
    one = pl.lit(10**dtype.scale).cast(pl.Decimal(38, dtype.scale))
    scaled = pl.select(pl.lit(raw).cast(pl.Decimal(38, 0)).truediv(one)).to_series()
    return scaled.cast(dtype).alias(raw.name)


def _finish(raw: pl.Series, spec: ColSpec, domain: pl.Series | None) -> pl.Series:
    """A raw engine column as the declared dtype: gathered from its domain,
    cast from its physical form, or as it came.
    """
    if domain is not None:
        return domain.gather(raw).alias(raw.name)
    if spec.dtype in _ENGINE_NATIVE:
        return raw
    if isinstance(spec.dtype, pl.Decimal):
        return _decimal_from_physical(raw, spec.dtype)
    # Temporal from its physical integer; Binary from strings; a Categorical
    # cast to spec.dtype itself so a named pl.Categories() registry survives.
    return raw.cast(spec.dtype)


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

    kind = _column_kind(spec.dtype)
    values: list = []
    rng = random.Random(seed)

    if isinstance(spec.dtype, pl.Enum):
        values = list(spec.dtype.categories.to_list())
    elif kind == "bool":
        values = [True, False]
    elif kind in ("int", "temporal", "decimal"):
        lo, hi = (int(v) for v in _resolve_numeric_bounds(spec))
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
        lo, hi = (float(v) for v in _resolve_numeric_bounds(spec))
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


def _generate_random(
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
    # `_column_kind` refuses what the engine cannot fill, so every column is
    # classified before any is generated. The scalar ones share a single
    # engine call; a nested one is built from the columns it contains.
    scalars = {
        n_: s
        for n_, s in columns.items()
        if _column_kind(s.dtype) not in ("list", "struct")
    }
    plans: list[ColumnPlan] = []
    domains: dict[str, pl.Series | None] = {}
    for name, spec in scalars.items():
        plan, domain = _plan_column(name, spec)
        plans.append(plan)
        domains[name] = domain
    # Each raw column is dropped as it is finished, so a column the finish
    # copies (a temporal cast, a gathered domain) never exists twice.
    raw = _generate_dataframe(plans, n, seed, row_offset).to_dict() if plans else {}
    finished: dict[str, pl.Series] = {}
    for name, spec in columns.items():
        if name in scalars:
            finished[name] = _finish(raw.pop(name), spec, domains[name])
        else:
            finished[name] = _generate_column(name, spec, n, seed, row_offset)
    return pl.DataFrame([finished[name] for name in columns])


def _generate_column(
    name: str, spec: ColSpec, n: int, seed: int | None, row_offset: int = 0
) -> pl.Series:
    """One column of `n` values, whatever its dtype nests.

    The scalar path is a plan handed to the engine; a `List` wraps the
    column its elements make, and a `Struct` gathers the columns its fields
    make -- each by calling back here, so a dtype nests as deeply as it
    likes and every value is drawn by the code that draws a column of its
    own type.
    """
    kind = _column_kind(spec.dtype)
    if kind == "list":
        return _generate_list_column(name, spec, n, seed, row_offset)
    if kind == "struct":
        return _generate_struct_column(name, spec, n, seed, row_offset)
    plan, domain = _plan_column(name, spec)
    return _finish(_generate_dataframe([plan], n, seed, row_offset)[name], spec, domain)


def _field_spec(spec: ColSpec, name: str, seed_key: str) -> ColSpec:
    """The declaration one struct field is generated from.

    What `fields` says about it, or its dtype alone when `fields` says
    nothing -- `fields` is partial by design, and a field is null only
    where its own declaration says so. It is seeded under its parent so
    that renaming the column moves every field with it and adding a field
    beside one moves nothing.
    """
    return dataclasses.replace(spec._field(name), seed_name=f"{seed_key}.{name}")


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
    assert isinstance(spec.dtype, (pl.List, pl.Array))  # noqa: S101 - the caller checked
    seed_key = spec.seed_name or name

    if isinstance(spec.dtype, pl.Array):
        width = spec.dtype.size
        length_bounds: tuple[int, int] = (width, width)
    else:
        length_bounds = (
            spec.list_length.closed() if spec.list_length else _DEFAULT_LIST_LEN
        )

    # The lengths carry the list's own nullability: a null length is a null cell.
    lengths_plan = column_plan(
        name,
        "int64",
        min=length_bounds[0],
        max=length_bounds[1],
        nullable=spec.nullable,
        null_probability=spec.null_probability if spec.nullable else 0.0,
        seed_name=f"{seed_key}\x00len",
    )
    lengths = _generate_dataframe([lengths_plan], n, seed, row_offset)[name]
    counts = lengths.fill_null(0)
    total = int(counts.sum())

    element_spec = dataclasses.replace(spec._element(), seed_name=seed_key)
    # The element is a column in its own right, so a list of structs -- or
    # of lists -- is the same recursion one level down.
    elements = _generate_column(name, element_spec, total, seed)

    row_of_element = (
        pl.int_range(0, n, eager=True).repeat_by(counts).explode(empty_as_null=False)
    )
    grouped = (
        pl.DataFrame({"__row": row_of_element, name: elements})
        .group_by("__row", maintain_order=True)
        .agg(pl.col(name))
    )
    cells = (
        pl.DataFrame({"__row": pl.int_range(0, n, eager=True)})
        .join(grouped, on="__row", how="left", maintain_order="left")
        .with_columns(
            pl.when(pl.lit(lengths).is_null())
            .then(None)
            .otherwise(
                pl.col(name).fill_null(pl.lit([], dtype=pl.List(spec.value_dtype)))
            )
            .alias(name)
        )[name]
    )
    if isinstance(spec.dtype, pl.Array):
        return cells.list.to_array(spec.dtype.size)
    return cells


def _generate_cartesian(
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
        return _generate_random(columns, 0, seed)

    frame_seed = seed if seed is not None else random.randrange(2**63)
    rng = random.Random(frame_seed)

    coverage_values: dict[str, list] = {}
    filler_columns: dict[str, ColSpec] = {}
    unique_columns: dict[str, ColSpec] = {}
    for name, spec in columns.items():
        if spec.unique:
            unique_columns[name] = spec
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
    if coverage_size > _MAX_CARTESIAN_ROWS:
        breakdown = ", ".join(f"{name}={len(v)}" for name, v in coverage_values.items())
        raise GenerationError(
            f"cartesian coverage would need {coverage_size:,} rows ({breakdown}), "
            f"which exceeds the {_MAX_CARTESIAN_ROWS:,}-row safety cap"
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
        filler_df = _generate_random(filler_columns, coverage_n, rng.randrange(2**63))
        coverage_df = pl.concat([coverage_df, filler_df], how="horizontal_extend")

    spread = [name for name in columns if name not in unique_columns]
    coverage_df = coverage_df.select(spread)

    if coverage_n < n:
        topup_df = _generate_random(
            {name: columns[name] for name in spread},
            n - coverage_n,
            rng.randrange(2**63),
        )
        coverage_df = pl.concat([coverage_df, topup_df], how="vertical")

    if unique_columns:
        # One draw over the whole frame, after any padding: a unique column
        # is only distinct if nothing else ever appends to it.
        distinct_df = _generate_random(
            unique_columns, coverage_df.height, rng.randrange(2**63)
        )
        coverage_df = pl.concat([coverage_df, distinct_df], how="horizontal_extend")

    return coverage_df.select(list(columns.keys()))
