"""What one value must be: present, in the domain, within bounds, of a
length, of a format, matching a pattern, distinct -- built against the column
for a scalar, against each field for a Struct, and lifted through `list.eval`
for the elements of a List, recursing as deeply as the dtype nests.
"""

from __future__ import annotations

import dataclasses
import decimal
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec.domain import is_textual
from polspec.dtypes import (
    dtype_value_limits,
    element_dtype,
    field_dtypes,
    map_entries,
)
from polspec.formats import lookup as _lookup_format
from polspec.validation.report import FindingCode

if TYPE_CHECKING:
    from polspec.bound import Bound
    from polspec.check import Check
    from polspec.formats import Format
    from polspec.spec import ColSpec
    from polspec.validation import ValidationOptions

from polspec.validation.constraints._base import (
    Constraint,
    as_strings,
    in_values,
    sample_source,
)
from polspec.validation.constraints._rules import rule_constraints


@dataclass(kw_only=True)
class _ColumnConstraint(Constraint):
    """A claim about one column's values -- or about a field inside them.

    `where` names the value the claim is about: the column, or a path into
    it such as `point.lat`. The rows are always the column's, so a finding
    is involved with the column alone and `rows()` locates it there.
    """

    column: str
    where: str = ""

    def __post_init__(self) -> None:
        if not self.where:
            self.where = self.column

    def involved(self) -> tuple[str, ...]:
        return (self.column,)


@dataclass(kw_only=True)
class _Nullability(_ColumnConstraint):
    sample_expr: None = None
    code: FindingCode = "nullability"

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        kind = "column" if self.where == self.column else "field"
        return (
            f"Column '{self.where}': non-nullable {kind} contains {count} null value(s)"
        )


@dataclass(kw_only=True)
class _Unholdable(_ColumnConstraint):
    """A value the declared dtype cannot hold, in a column whose own dtype
    can: an `Int64` of 1000 where an `Int8` is declared. The permissive
    dtype check lets the wider dtype stand in; this is what keeps a frame
    that passes castable to the spec it passed."""

    declared: pl.DataType
    holds: str
    code: FindingCode = "dtype"

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"expected": str(self.declared), "holds": self.holds}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.where}': found {count} value(s) that "
            f"{self.declared} cannot hold ({self.holds}). Samples: {samples}"
        )


@dataclass(kw_only=True)
class _NotANumber(_ColumnConstraint):
    """A NaN in a float value that declares none: `nan_probability=0`, the
    default, says the column holds numbers."""

    code: FindingCode = "nan"

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.where}': found {count} NaN value(s); declare "
            "nan_probability for a column that holds them"
        )


@dataclass(kw_only=True)
class _AllowedValues(_ColumnConstraint):
    allowed: list[Any]
    code: FindingCode = "choices"

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"allowed": list(self.allowed)}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.where}': found {count} invalid value(s) not in "
            f"allowed choices/categories {self.allowed}. Invalid samples: {samples}"
        )


@dataclass(kw_only=True)
class _Bounds(_ColumnConstraint):
    bounds: Bound[Any]
    # The values the extremes are measured over: the column, a field, or for
    # a List the elements, however deep.
    values: pl.Expr
    unique_samples: bool = False
    code: FindingCode = "bounds"

    def aggregations(self) -> list[pl.Expr]:
        # The observed extremes make an out-of-bounds report actionable, so
        # they are gathered alongside the violating samples.
        return [
            *super().aggregations(),
            self.values.min().alias(self._alias("min")),
            self.values.max().alias(self._alias("max")),
        ]

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {
            "bounds": [self.bounds.min, self.bounds.max] if self.bounds else None,
            "min_found": stats[self._alias("min")][0],
            "max_found": stats[self._alias("max")][0],
        }

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        found_min = stats[self._alias("min")][0]
        found_max = stats[self._alias("max")][0]
        return (
            f"Column '{self.where}': found {count} value(s) out of bounds "
            f"{self.bounds} (min found: {found_min}, max found: {found_max}). "
            f"Out of bounds samples: {samples}"
        )


@dataclass(kw_only=True)
class _StringLength(_ColumnConstraint):
    length: Bound[int]
    unique_samples: bool = False
    code: FindingCode = "string_length"

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {
            "string_length": [self.length.min, self.length.max] if self.length else None
        }

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.where}': found {count} value(s) with string length "
            f"outside [{self.length.min}, {self.length.max}]. "
            f"Invalid samples: {samples}"
        )


@dataclass(kw_only=True)
class _ListLength(_ColumnConstraint):
    length: Bound[int]
    unique_samples: bool = False
    code: FindingCode = "list_length"

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"list_length": [self.length.min, self.length.max]}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.where}': found {count} list(s) with a length "
            f"outside [{self.length.min}, {self.length.max}]. "
            f"Invalid samples: {samples}"
        )


@dataclass(kw_only=True)
class _ListElementNull(_ColumnConstraint):
    """A null *inside* a list whose `element_null_probability` is 0 -- the
    default, which says elements are never null -- reported under the
    nullability code like a null in a non-nullable column."""

    code: FindingCode = "nullability"

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.where}': found {count} list(s) containing a null "
            f"element. Samples: {samples}"
        )


@dataclass(kw_only=True)
class _Format(_ColumnConstraint):
    format: Format
    extras: tuple[str, ...] = ()
    code: FindingCode = "format"

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        found: dict[str, Any] = {"format": self.format.name}
        if self.extras:
            found["extra_values"] = list(self.extras)
        return found

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        also = f" or one of its extra values {list(self.extras)}" if self.extras else ""
        return (
            f"Column '{self.where}': found {count} value(s) that are not "
            f"{self.format}{also}. Invalid samples: {samples}"
        )


@dataclass(kw_only=True)
class _Pattern(_ColumnConstraint):
    pattern: str
    code: FindingCode = "pattern"

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"pattern": self.pattern}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.where}': found {count} value(s) not matching "
            f"pattern {self.pattern!r}. Invalid samples: {samples}"
        )


@dataclass(kw_only=True)
class _ColumnValidator(_ColumnConstraint):
    validator: Check
    code: FindingCode = "validator"

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"validator": self.validator.name, "condition": str(self.validator.expr)}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        described = (
            f" ({self.validator.description})" if self.validator.description else ""
        )
        return (
            f"Column '{self.column}': validator '{self.validator.name}' failed: "
            f"found {count} row(s) violating condition {self.validator.expr}{described}"
        )


@dataclass(kw_only=True)
class _UniqueValues(_ColumnConstraint):
    code: FindingCode = "unique"

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': unique column contains {count} duplicate "
            f"value(s). Duplicate samples: {samples}"
        )


@dataclass(kw_only=True)
class _MapKeyRepeats(_ColumnConstraint):
    """A map holding one key twice. Polars folds a repeat away when it casts
    to a `Map`, but not when one arrives from Arrow, so a map read in can
    carry one."""

    code: FindingCode = "unique"

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.where}': found {count} map(s) holding a key more "
            f"than once. Samples: {samples}"
        )


# ---------------------------------------------------------------------------
# Building constraints from a spec
# ---------------------------------------------------------------------------


def column_constraints(
    name: str,
    spec: ColSpec,
    actual_dtype: pl.DataType,
    *,
    compatible: bool,
    options: ValidationOptions,
    df_col_names: Sequence[str],
) -> list[Constraint]:
    """The constraints one declared column contributes to the single pass.

    Only nullability survives an incompatible dtype. Everything else compares
    values against something typed -- bounds, choices, a rule's operands -- and
    against the wrong type that is at best noise on top of the dtype finding
    the caller will already see, and at worst an expression Polars refuses to
    compile at all. For the same reason `df_col_names` is the columns a claim
    may *read*: present, and of a compatible dtype if declared.
    """
    column = pl.col(name)
    constraints: list[Constraint] = []

    if not spec.nullable:
        constraints.append(
            _Nullability(key=f"{name}__null", mask=column.is_null(), column=name)
        )

    # Nothing below this line can be asked of a column whose dtype is already
    # wrong. `is_in` against choices of another type does not merely produce a
    # noisy finding -- Polars refuses to compile it -- so the dtype finding the
    # caller will already see is the whole answer for this column.
    if not compatible:
        return constraints

    constraints.extend(_value_tree(name, name, spec, actual_dtype, column, options))

    if options.rules and spec.rules:
        constraints.extend(rule_constraints(name, spec, actual_dtype, df_col_names))

    if options.validators and spec.validators:
        constraints.extend(
            _ColumnValidator(
                key=f"{name}__validator_{index}",
                mask=validator._failure_mask(),
                sample_expr=sample_source(column, actual_dtype),
                column=name,
                validator=validator,
            )
            for index, validator in enumerate(spec.validators)
        )

    if options.unique and spec.unique:
        constraints.append(
            _UniqueValues(
                key=f"{name}__unique",
                mask=column.is_not_null() & column.is_duplicated(),
                sample_expr=column,
                column=name,
            )
        )

    return constraints


def _value_tree(
    name: str,
    where: str,
    spec: ColSpec,
    actual_dtype: pl.DataType,
    column: pl.Expr,
    options: ValidationOptions,
) -> list[Constraint]:
    """Every constraint on the values `column` holds, recursing on kind.

    `column` is the expression for the value at this depth -- the column,
    a field of a struct, or `pl.element()` inside a list -- and every mask
    returned is aligned with it, so the level above can lift it: a struct
    reads its fields in place, and a list runs its elements' masks through
    `list.eval`. `where` names the value, so a finding says which field.
    """
    if isinstance(actual_dtype, (pl.List, pl.Array)):
        return _list_constraints(name, where, spec, actual_dtype, column, options)
    if (entries := map_entries(actual_dtype)) is not None:
        return _map_constraints(name, where, spec, entries, column, options)
    if isinstance(actual_dtype, pl.Struct):
        return _struct_constraints(name, where, spec, actual_dtype, column, options)
    return _value_constraints(name, where, spec, actual_dtype, column, options)


def _value_constraints(
    name: str,
    where: str,
    spec: ColSpec,
    actual_dtype: pl.DataType,
    column: pl.Expr,
    options: ValidationOptions,
) -> list[Constraint]:
    """The constraints on one scalar *value* -- its domain, bounds, length,
    format, pattern -- with masks over `column`.
    """
    present = column.is_not_null()
    dtype = spec.value_dtype
    constraints: list[Constraint] = []

    # NaN is not a value of a float column in any sense `bounds` or `choices`
    # speak of -- it sits in no range and equals nothing -- so it is judged
    # once, by the NaN claim, and the value claims read the numbers alone.
    nan_able = dtype.is_float() and actual_dtype.is_float()
    valued = present & column.is_not_nan() if nan_able else present
    if nan_able and not spec.nan_probability:
        constraints.append(
            _NotANumber(
                key=f"{where}__nan",
                mask=present & column.is_nan(),
                sample_expr=None,
                column=name,
                where=where,
            )
        )

    # The dtype's claim, not the bounds' -- but a side a checked bound
    # closes is the bound's to report: see `_unholdable`.
    unfit = _unholdable(
        column, dtype, actual_dtype, spec.bounds if options.bounds else None
    )
    if unfit is not None:
        mask, holds = unfit
        constraints.append(
            _Unholdable(
                key=f"{where}__dtype_range",
                mask=present & mask,
                sample_expr=column,
                column=name,
                where=where,
                declared=dtype,
                holds=holds,
            )
        )

    allowed = _allowed_values(spec)
    if allowed is not None:
        if is_textual(actual_dtype):
            in_domain = column.cast(pl.String).is_in(as_strings(allowed, dtype))
            sample_expr = column.cast(pl.String)
        else:
            in_domain = in_values(column, allowed, dtype, actual_dtype)
            sample_expr = column
        constraints.append(
            _AllowedValues(
                key=f"{where}__choices",
                mask=valued & ~in_domain,
                sample_expr=sample_expr,
                column=name,
                where=where,
                allowed=allowed,
            )
        )

    if options.bounds and spec.bounds is not None and not spec.bounds.is_open_both:
        # A temporal standing in for another is measured as `cast=True` would
        # hand it back, as choices are (`in_values`): in its own unit or kind a
        # bound can round -- a microsecond past midnight, on a `Date`, is that
        # day -- and pass a value the cast then puts outside it. A number is
        # measured as it is: a cast could only lose an out-of-range value.
        measured, measured_dtype = column, actual_dtype
        if actual_dtype.is_temporal() and actual_dtype != dtype:
            measured, measured_dtype = column.cast(dtype, strict=False), dtype
        constraints.append(
            _Bounds(
                key=f"{where}__bounds",
                mask=valued & _out_of_bounds(measured, spec.bounds, measured_dtype),
                sample_expr=column,
                column=name,
                where=where,
                bounds=spec.bounds,
                values=measured,
            )
        )

    if spec.string_length is not None:
        measured = _measure_length(column, actual_dtype)
        if measured is not None:
            too_short = measured < spec.string_length.min
            too_long = measured > spec.string_length.max
            constraints.append(
                _StringLength(
                    key=f"{where}__len",
                    mask=present & (too_short | too_long),
                    sample_expr=column,
                    column=name,
                    where=where,
                    length=spec.string_length,
                )
            )

    if spec.format is not None:
        fmt = _lookup_format(spec.format)
        extras = tuple(spec.extra_values or ())
        accepted = fmt.check(column)
        if extras:
            accepted = accepted | column.is_in(list(extras))
        constraints.append(
            _Format(
                key=f"{where}__format",
                mask=present & ~accepted,
                sample_expr=column,
                column=name,
                where=where,
                format=fmt,
                extras=extras,
            )
        )

    if options.pattern and spec.pattern is not None:
        constraints.append(
            _Pattern(
                key=f"{where}__pattern",
                mask=present & ~column.str.contains(spec.pattern),
                sample_expr=column,
                column=name,
                where=where,
                pattern=spec.pattern,
            )
        )

    return constraints


def _struct_constraints(
    name: str,
    where: str,
    spec: ColSpec,
    actual_dtype: pl.Struct,
    column: pl.Expr,
    options: ValidationOptions,
) -> list[Constraint]:
    """A Struct's constraints: each field's, read in place.

    A field is addressable, so its constraints are built with
    `column.struct.field(f)` as their column and need no lifting -- each
    mask is already one per struct. A null field inside a present struct is
    reported unless the field is declared nullable, and a null struct has
    no fields to check.
    """
    declared_dtype = spec.value_dtype
    assert isinstance(declared_dtype, pl.Struct)  # noqa: S101 - compatible with one
    present = column.is_not_null()
    actual_fields = field_dtypes(actual_dtype)
    constraints: list[Constraint] = []
    for field in field_dtypes(declared_dtype):
        if field not in actual_fields:
            continue  # a dtype finding already says so
        declared = spec._field(field)
        value = column.struct.field(field)
        path = f"{where}.{field}"
        if not declared.nullable:
            constraints.append(
                _Nullability(
                    key=f"{path}__null",
                    mask=present & value.is_null(),
                    column=name,
                    where=path,
                )
            )
        constraints.extend(
            _value_tree(name, path, declared, actual_fields[field], value, options)
        )
    return constraints


def _list_constraints(
    name: str,
    where: str,
    spec: ColSpec,
    actual_dtype: pl.List | pl.Array,
    column: pl.Expr,
    options: ValidationOptions,
) -> list[Constraint]:
    """A List's constraints: its length, no null elements, and every
    constraint its elements carry, lifted.

    The elements' constraints are built with `pl.element()` as their column
    -- recursing, so an element may be a struct or a list itself -- then
    each mask is run inside `list.eval` and a list fails where *any* element
    does. The samples are the offending lists.
    """
    present = column.is_not_null()
    sample_expr = sample_source(column, actual_dtype)
    constraints: list[Constraint] = []

    if spec.list_length is not None and isinstance(actual_dtype, pl.List):
        length = column.list.len()
        constraints.append(
            _ListLength(
                key=f"{where}__list_len",
                mask=present
                & ~length.is_between(spec.list_length.min, spec.list_length.max),
                sample_expr=column,
                column=name,
                where=where,
                length=spec.list_length,
            )
        )

    is_array = isinstance(actual_dtype, pl.Array)

    def any_element(mask: pl.Expr) -> pl.Expr:
        # `arr.eval` hands back an Array of booleans, which has its own `any`.
        return (
            column.arr.eval(mask).arr.any()
            if is_array
            else column.list.eval(mask).list.any()
        )

    elements = column.arr if is_array else column.list
    if not spec.element_null_probability:
        # A declared element null rate says nulls belong inside the list.
        constraints.append(
            _ListElementNull(
                key=f"{where}__element_null",
                mask=present & any_element(pl.element().is_null()),
                sample_expr=sample_expr,
                column=name,
                where=where,
            )
        )

    inner = element_dtype(actual_dtype)
    # A list of lists -- or of maps, which are lists of entries -- has two
    # levels of list claims under one name; `[]` tells the inner one's apart.
    # Every other element keeps its list's name, so a List column's findings
    # read as they always have.
    nested = isinstance(inner, (pl.List, pl.Array)) or map_entries(inner) is not None
    inner_where = f"{where}[]" if nested else where
    for constraint in _value_tree(
        name, inner_where, spec._element(), inner, pl.element(), options
    ):
        lifted: dict[str, Any] = {
            "mask": present & any_element(constraint.mask),
            "sample_expr": sample_expr,
        }
        if isinstance(constraint, _Bounds):
            lifted["values"] = elements.eval(constraint.values).explode(
                empty_as_null=False
            )
        constraints.append(dataclasses.replace(constraint, **lifted))
    return constraints


def _map_constraints(
    name: str,
    where: str,
    spec: ColSpec,
    entries: pl.List,
    column: pl.Expr,
    options: ValidationOptions,
) -> list[Constraint]:
    """A Map's constraints: those of the list of entries it is -- its
    length, and each entry's key and value, found under `m.key` and
    `m.value` -- and no key twice in one map.

    The map is cast to that list, which Polars does both ways, so every
    claim is checked by the code that checks a list of structs.
    """
    as_list = column.cast(entries)
    constraints = _list_constraints(
        name, where, spec._as_list(), entries, as_list, options
    )
    keys = as_list.list.eval(pl.element().struct.field("key"))
    constraints.append(
        _MapKeyRepeats(
            key=f"{where}.key__unique",
            mask=column.is_not_null() & (as_list.list.len() != keys.list.n_unique()),
            sample_expr=column,
            column=name,
            where=f"{where}.key",
        )
    )
    return constraints


def _allowed_values(spec: ColSpec) -> list[Any] | None:
    """The closed domain this column's values must fall in, if it has one."""
    if isinstance(spec.value_dtype, pl.Enum):
        categories = spec.value_dtype.categories.to_list()
        if spec.choices is not None:
            return [c for c in spec.choices if c in categories]
        return categories
    if spec.choices is not None:
        return list(spec.choices)
    return None


def _unholdable(
    column: pl.Expr,
    declared: pl.DataType,
    actual: pl.DataType,
    bounds: Bound | None = None,
) -> tuple[pl.Expr, str] | None:
    """A mask for the values `actual` holds and `declared` cannot, and the
    range `declared` holds in words; None when every value fits.

    Only a declared integer or `Decimal` has a range a compatible dtype can
    exceed and a cast then refuse -- and a declared `Null`, which any dtype
    stands in for and which holds no value at all. A float declared narrower
    than it arrives is left alone: past its range a cast makes an infinity,
    not an error. Each literal is typed as the column is, and each side is
    tested only where `actual` reaches past it, so every literal is one
    `actual` holds.

    `bounds` are the declared bounds being checked, if they are. A bound
    must fit its dtype to be declared, so a value past the dtype's range on
    a bounded side is past the bound too and already a `bounds` finding:
    that side is left to it. Most of the time -- a CSV's `Int64` for a
    bounded `Int32` -- that is both sides, and the check costs nothing.
    """
    if declared == pl.Null:
        return None if actual == pl.Null else (column.is_not_null(), "only nulls")
    low: pl.Expr | None = None
    high: pl.Expr | None = None

    if declared.is_integer() and actual.is_integer():
        declared_limits = dtype_value_limits(declared)
        actual_limits = dtype_value_limits(actual)
        if declared_limits is None or actual_limits is None:
            return None
        lo, hi = (int(v) for v in declared_limits)
        holds = f"{lo}..{hi}"
        if actual_limits[0] < lo:
            low = column < pl.lit(lo, dtype=actual)
        if actual_limits[1] > hi:
            high = column > pl.lit(hi, dtype=actual)
    elif isinstance(declared, pl.Decimal):
        precision = declared.precision or 38
        scale = declared.scale or 0
        # A Decimal(p, s) holds magnitudes under 10**(p - s); a value is
        # rounded to s places before that is asked, so half a unit below it
        # is too many.
        limit = 10 ** (precision - scale)
        holds = f"magnitude under {limit}"
        rounding = decimal.Decimal(5).scaleb(-(scale + 1))
        if actual.is_float():
            threshold = float(limit - rounding)
            # A NaN compares greater than any number, so it is caught on the
            # high side: a Decimal cannot hold one either.
            low, high = column <= -threshold, column >= threshold
        elif actual.is_integer():
            actual_limits = dtype_value_limits(actual)
            if actual_limits is None:
                return None
            if actual_limits[0] <= -limit:
                low = column <= pl.lit(-limit, dtype=actual)
            if actual_limits[1] >= limit:
                high = column >= pl.lit(limit, dtype=actual)
        elif isinstance(actual, pl.Decimal):
            actual_scale = actual.scale or 0
            if (actual.precision or 38) - actual_scale <= precision - scale:
                return None
            edge = decimal.Decimal(limit) - (rounding if actual_scale > scale else 0)
            low = column <= pl.lit(-edge, dtype=actual)
            high = column >= pl.lit(edge, dtype=actual)
        else:
            return None
    else:
        return None

    if bounds is not None and bounds.min is not None:
        low = None
    if bounds is not None and bounds.max is not None:
        high = None
    return _either([side for side in (low, high) if side is not None], holds)


def _either(sides: list[pl.Expr], holds: str) -> tuple[pl.Expr, str] | None:
    """`sides` as one mask, with `holds`; None when there is no side to test."""
    if not sides:
        return None
    mask = sides[0]
    for side in sides[1:]:
        mask = mask | side
    return mask, holds


def _out_of_bounds(
    column: pl.Expr, bounds: Bound, actual_dtype: pl.DataType
) -> pl.Expr:
    """A mask for values outside `bounds`, testing only the constrained sides.

    An open end is genuinely unconstrained here, unlike at generation time
    where it falls back to a default (see `ColSpec.bounds`).
    """

    def limit(value: Any) -> Any:
        return pl.lit(value).cast(actual_dtype) if actual_dtype.is_temporal() else value

    violations = []
    if bounds.min is not None:
        violations.append(column < limit(bounds.min))
    if bounds.max is not None:
        violations.append(column > limit(bounds.max))
    return violations[0] if len(violations) == 1 else violations[0] | violations[1]


def _measure_length(column: pl.Expr, actual_dtype: pl.DataType) -> pl.Expr | None:
    """Length of each value, for the dtypes where that is meaningful."""
    if actual_dtype in (pl.String, pl.Utf8):
        return column.str.len_chars()
    if actual_dtype == pl.Binary:
        return column.bin.size()
    return None
