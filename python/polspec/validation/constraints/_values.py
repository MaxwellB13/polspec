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

from polspec.constraints import is_textual as _is_textual
from polspec.dtypes import _typed_values, element_dtype, field_dtypes
from polspec.formats import lookup as _lookup_format
from polspec.validation.report import FindingCode

if TYPE_CHECKING:
    from polspec.bound import Bound
    from polspec.check import Check
    from polspec.formats import Format
    from polspec.spec import ColSpec
    from polspec.validation import ValidationOptions

from polspec.validation.constraints._base import (
    _as_strings,
    _Constraint,
)
from polspec.validation.constraints._rules import _rule_constraints


@dataclass(kw_only=True)
class _ColumnConstraint(_Constraint):
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
    """A null *inside* a list. Generation never makes one, and no field on
    a ColSpec can ask for one, so it is reported under the nullability code
    like a null in a non-nullable column."""

    code: FindingCode = "nullability"

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.where}': found {count} list(s) containing a null "
            f"element. Samples: {samples}"
        )


@dataclass(kw_only=True)
class _Format(_ColumnConstraint):
    format: Format
    code: FindingCode = "format"

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"format": self.format.name}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.where}': found {count} value(s) that are not "
            f"{self.format}. Invalid samples: {samples}"
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
class _CompositeUnique(_Constraint):
    columns: tuple[str, ...] = ()
    code: FindingCode = "unique_together"

    def involved(self) -> tuple[str, ...]:
        return self.columns

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Composite unique key {list(self.columns)} violated: found {count} "
            f"duplicate row(s). Duplicate samples: {samples}"
        )


@dataclass(kw_only=True)
class _FrameCheck(_Constraint):
    check: Check
    code: FindingCode = "check"

    def involved(self) -> tuple[str, ...]:
        return tuple(self.check.expr.meta.root_names())

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"check": self.check.name, "condition": str(self.check.expr)}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        described = f" ({self.check.description})" if self.check.description else ""
        return (
            f"Check '{self.check.name}' failed: found {count} row(s) violating "
            f"condition {self.check.expr}{described}"
        )


# ---------------------------------------------------------------------------
# Building constraints from a spec
# ---------------------------------------------------------------------------


def _column_constraints(
    name: str,
    spec: ColSpec,
    actual_dtype: pl.DataType,
    *,
    compatible: bool,
    options: ValidationOptions,
    df_col_names: Sequence[str],
) -> list[_Constraint]:
    """The constraints one declared column contributes to the single pass.

    Only nullability survives an incompatible dtype. Everything else compares
    values against something typed -- bounds, choices, a rule's operands -- and
    against the wrong type that is at best noise on top of the dtype finding
    the caller will already see, and at worst an expression Polars refuses to
    compile at all.
    """
    column = pl.col(name)
    constraints: list[_Constraint] = []

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
        constraints.extend(_rule_constraints(name, spec, actual_dtype, df_col_names))

    if options.validators and spec.validators:
        constraints.extend(
            _ColumnValidator(
                key=f"{name}__validator_{index}",
                mask=validator._failure_mask(),
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
) -> list[_Constraint]:
    """Every constraint on the values `column` holds, recursing on kind.

    `column` is the expression for the value at this depth -- the column,
    a field of a struct, or `pl.element()` inside a list -- and every mask
    returned is aligned with it, so the level above can lift it: a struct
    reads its fields in place, and a list runs its elements' masks through
    `list.eval`. `where` names the value, so a finding says which field.
    """
    if isinstance(actual_dtype, (pl.List, pl.Array)):
        return _list_constraints(name, where, spec, actual_dtype, column, options)
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
) -> list[_Constraint]:
    """The constraints on one scalar *value* -- its domain, bounds, length,
    format, pattern -- with masks over `column`.
    """
    present = column.is_not_null()
    dtype = spec.value_dtype
    constraints: list[_Constraint] = []

    allowed = _allowed_values(spec)
    if allowed is not None:
        if _is_textual(actual_dtype):
            in_domain = column.cast(pl.String).is_in(_as_strings(allowed, dtype))
            sample_expr = column.cast(pl.String)
        elif isinstance(dtype, pl.Decimal):
            # A Python list of Decimals reaches Polars at the widest precision,
            # which it refuses to compare; a Series of the column's own type,
            # imploded so Polars reads it as one set rather than row by row,
            # is compared as values.
            in_domain = column.is_in(_typed_values(allowed, dtype).implode())
            sample_expr = column
        else:
            in_domain = column.is_in(allowed)
            sample_expr = column
        constraints.append(
            _AllowedValues(
                key=f"{where}__choices",
                mask=present & ~in_domain,
                sample_expr=sample_expr,
                column=name,
                where=where,
                allowed=allowed,
            )
        )

    if spec.bounds is not None and not spec.bounds.is_open_both:
        constraints.append(
            _Bounds(
                key=f"{where}__bounds",
                mask=present & _out_of_bounds(column, spec.bounds, actual_dtype),
                sample_expr=column,
                column=name,
                where=where,
                bounds=spec.bounds,
                values=column,
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
        constraints.append(
            _Format(
                key=f"{where}__format",
                mask=present & ~fmt.check(column),
                sample_expr=column,
                column=name,
                where=where,
                format=fmt,
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
) -> list[_Constraint]:
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
    constraints: list[_Constraint] = []
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
) -> list[_Constraint]:
    """A List's constraints: its length, no null elements, and every
    constraint its elements carry, lifted.

    The elements' constraints are built with `pl.element()` as their column
    -- recursing, so an element may be a struct or a list itself -- then
    each mask is run inside `list.eval` and a list fails where *any* element
    does. The samples are the offending lists.
    """
    present = column.is_not_null()
    constraints: list[_Constraint] = []

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
    constraints.append(
        _ListElementNull(
            key=f"{where}__element_null",
            mask=present & any_element(pl.element().is_null()),
            sample_expr=column,
            column=name,
            where=where,
        )
    )

    inner = element_dtype(actual_dtype)
    # A list of lists has two levels of list claims under one name; `[]`
    # tells the inner one's apart. Every other element keeps its list's name,
    # so a List column's findings read as they always have.
    inner_where = f"{where}[]" if isinstance(inner, (pl.List, pl.Array)) else where
    for constraint in _value_tree(
        name, inner_where, spec._element(), inner, pl.element(), options
    ):
        lifted: dict[str, Any] = {
            "mask": present & any_element(constraint.mask),
            "sample_expr": column,
        }
        if isinstance(constraint, _Bounds):
            lifted["values"] = elements.eval(constraint.values).explode(
                empty_as_null=False
            )
        constraints.append(dataclasses.replace(constraint, **lifted))
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
