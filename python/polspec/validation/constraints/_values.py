"""What one value must be: present, in the domain, within bounds, of a
length, of a format, matching a pattern, distinct -- built against the column
for a scalar, against each field for a Struct, and lifted through `list.eval`
for the elements of a List, recursing as deeply as the dtype nests.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec import frames
from polspec.domain import Domain, is_textual
from polspec.dtypes import (
    element_dtype,
    field_dtypes,
    map_entries,
)
from polspec.string_formats import lookup as _lookup_format
from polspec.validation.report import FindingCode

if TYPE_CHECKING:
    from polspec.bound import Bound
    from polspec.checks import Check
    from polspec.spec import ColSpec
    from polspec.string_formats import Format
    from polspec.validation import ValidationOptions

from polspec._colspec.masks import (
    in_values,
    measure_length,
    out_of_bounds,
    unholdable_mask,
)
from polspec.checks import failure_mask
from polspec.spec import element_of, entries_of, field_of
from polspec.validation.constraints._base import (
    Constraint,
    as_strings,
)
from polspec.validation.constraints._derived import derived_constraint
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
    column = frames.column(name)
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

    if options.derived and spec.derive is not None:
        constraints.extend(derived_constraint(name, spec, actual_dtype, df_col_names))

    if options.validators and spec.validators:
        constraints.extend(
            _ColumnValidator(
                key=f"{name}__validator_{index}",
                mask=failure_mask(validator),
                sample_expr=column,
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


@dataclass(frozen=True, slots=True)
class _Value:
    """One scalar value a claim is made of: the column, where in it, as the
    spec declares it and as the frame holds it, and its two masks."""

    name: str
    where: str
    spec: ColSpec
    actual_dtype: pl.DataType
    column: pl.Expr
    options: ValidationOptions
    # Not null; and not null nor NaN, for the claims that read numbers.
    present: pl.Expr
    valued: pl.Expr


def _value_constraints(
    name: str,
    where: str,
    spec: ColSpec,
    actual_dtype: pl.DataType,
    column: pl.Expr,
    options: ValidationOptions,
) -> list[Constraint]:
    """The constraints on one scalar *value* -- its domain, bounds, length,
    format, pattern -- with masks over `column`, one builder a claim, in the
    order findings report them.
    """
    present = column.is_not_null()
    # NaN is not a value of a float column in any sense `bounds` or `choices`
    # speak of -- it sits in no range and equals nothing -- so it is judged
    # once, by the NaN claim, and the value claims read the numbers alone.
    nan_able = spec.value_dtype.is_float() and actual_dtype.is_float()
    value = _Value(
        name=name,
        where=where,
        spec=spec,
        actual_dtype=actual_dtype,
        column=column,
        options=options,
        present=present,
        valued=present & column.is_not_nan() if nan_able else present,
    )
    claims = (
        _nan_claim(value, nan_able=nan_able),
        _dtype_range_claim(value),
        _choices_claim(value),
        _bounds_claim(value),
        _length_claim(value),
        _format_claim(value),
        _pattern_claim(value),
    )
    return [claim for claim in claims if claim is not None]


def _nan_claim(value: _Value, *, nan_able: bool) -> Constraint | None:
    # A derived column's NaN is its expression's, which `derived` judges.
    if not nan_able or value.spec.nan_probability or value.spec.derive is not None:
        return None
    return _NotANumber(
        key=f"{value.where}__nan",
        mask=value.present & value.column.is_nan(),
        sample_expr=None,
        column=value.name,
        where=value.where,
    )


def _dtype_range_claim(value: _Value) -> Constraint | None:
    # The dtype's claim, not the bounds' -- but a side a checked bound
    # closes is the bound's to report: see `_unholdable`.
    spec = value.spec
    unfit = unholdable_mask(
        value.column,
        spec.value_dtype,
        value.actual_dtype,
        spec.bounds if value.options.bounds else None,
    )
    if unfit is None:
        return None
    mask, holds = unfit
    return _Unholdable(
        key=f"{value.where}__dtype_range",
        mask=value.present & mask,
        sample_expr=value.column,
        column=value.name,
        where=value.where,
        declared=spec.value_dtype,
        holds=holds,
    )


def _choices_claim(value: _Value) -> Constraint | None:
    allowed = _allowed_values(value.spec)
    if allowed is None:
        return None
    declared = value.spec.value_dtype
    if (
        isinstance(declared, pl.Enum)
        and value.actual_dtype == declared
        and len(allowed) == declared.categories.len()
    ):
        # An Enum column holds only its categories, so values of the very
        # Enum declared -- with no choices narrowing it -- are all allowed.
        # Testing each against the categories is a pass that cannot fail.
        return None
    column, dtype = value.column, value.spec.value_dtype
    if is_textual(value.actual_dtype):
        in_domain = column.cast(pl.String).is_in(as_strings(allowed, dtype))
        sample_expr = column.cast(pl.String)
    else:
        in_domain = in_values(column, allowed, dtype, value.actual_dtype)
        sample_expr = column
    return _AllowedValues(
        key=f"{value.where}__choices",
        mask=value.valued & ~in_domain,
        sample_expr=sample_expr,
        column=value.name,
        where=value.where,
        allowed=allowed,
    )


def _bounds_claim(value: _Value) -> Constraint | None:
    bounds = value.spec.bounds
    if not value.options.bounds or bounds is None or bounds.is_open_both:
        return None
    # A temporal standing in for another is measured as `cast=True` would
    # hand it back, as choices are (`in_values`): in its own unit or kind a
    # bound can round -- a microsecond past midnight, on a `Date`, is that
    # day -- and pass a value the cast then puts outside it. A number is
    # measured as it is: a cast could only lose an out-of-range value.
    dtype = value.spec.value_dtype
    measured, measured_dtype = value.column, value.actual_dtype
    if value.actual_dtype.is_temporal() and value.actual_dtype != dtype:
        measured, measured_dtype = value.column.cast(dtype, strict=False), dtype
    return _Bounds(
        key=f"{value.where}__bounds",
        mask=value.valued & out_of_bounds(measured, bounds, measured_dtype),
        sample_expr=value.column,
        column=value.name,
        where=value.where,
        bounds=bounds,
        values=measured,
    )


def _length_claim(value: _Value) -> Constraint | None:
    length = value.spec.string_length
    if length is None:
        return None
    measured = measure_length(value.column, value.actual_dtype)
    if measured is None:
        return None
    return _StringLength(
        key=f"{value.where}__len",
        mask=value.present & ((measured < length.min) | (measured > length.max)),
        sample_expr=value.column,
        column=value.name,
        where=value.where,
        length=length,
    )


def _format_claim(value: _Value) -> Constraint | None:
    if value.spec.format is None:
        return None
    fmt = _lookup_format(value.spec.format)
    extras = tuple(value.spec.extra_values or ())
    accepted = fmt.check(value.column)
    if extras:
        accepted = accepted | value.column.is_in(list(extras))
    return _Format(
        key=f"{value.where}__format",
        mask=value.present & ~accepted,
        sample_expr=value.column,
        column=value.name,
        where=value.where,
        format=fmt,
        extras=extras,
    )


def _pattern_claim(value: _Value) -> Constraint | None:
    pattern = value.spec.pattern
    if not value.options.pattern or pattern is None:
        return None
    return _Pattern(
        key=f"{value.where}__pattern",
        mask=value.present & ~value.column.str.contains(pattern),
        sample_expr=value.column,
        column=value.name,
        where=value.where,
        pattern=pattern,
    )


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
        declared = field_of(spec, field)
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
    sample_expr = column
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
        if is_array:
            # An Array of booleans, which has its own `any`.
            return column.arr.eval(mask).arr.any()
        return column.list.eval(mask).list.any()

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
        name, inner_where, element_of(spec), inner, pl.element(), options
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
        name, where, entries_of(spec), entries, as_list, options
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
    """The closed domain this column's values must fall in, if it has one:
    its choices or an Enum's categories, as `Domain` defines them for
    generation too. A finite format's list is its `_format_claim`'s."""
    if spec.choices is None and not isinstance(spec.value_dtype, pl.Enum):
        return None
    return list(Domain.of(spec).values or ())
