"""Every claim a spec makes, as a `_Constraint` that produces a `Finding`.

A constraint contributes aggregation expressions to one pass over the frame,
then turns the results back into a `Finding`: a count, a few samples, the
facts that make the message actionable, and a way to locate the offending
rows later. All constraints are collected first and evaluated together, so
validating fifty columns costs one scan rather than fifty. Foreign keys are
the exception: each needs its own anti-join.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec.constraints import is_textual as _is_textual
from polspec.dtypes import _typed_values, element_dtype
from polspec.formats import lookup as _lookup_format
from polspec.validation.report import Finding, FindingCode

if TYPE_CHECKING:
    from polspec.bound import Bound
    from polspec.check import Check
    from polspec.foreign_key import ForeignKey
    from polspec.formats import Format
    from polspec.hierarchy import Hierarchy
    from polspec.rules import ColRule
    from polspec.spec import ColSpec
    from polspec.validation import ValidationOptions

MAX_SAMPLES = 5


@dataclass(kw_only=True)
class _Constraint:
    """One checkable claim, measured by counting the rows that violate a mask.

    Subclasses supply the mask, the wording and the details; the aliases
    tying the two halves together stay private to the instance.
    """

    key: str
    mask: pl.Expr
    sample_expr: pl.Expr | None = None
    unique_samples: bool = True
    code: FindingCode

    def _alias(self, suffix: str) -> str:
        return f"__val__{self.key}__{suffix}"

    def aggregations(self) -> list[pl.Expr]:
        exprs = [self.mask.sum().alias(self._alias("cnt"))]
        if self.sample_expr is not None:
            samples = self.sample_expr.filter(self.mask)
            if self.unique_samples:
                samples = samples.unique(maintain_order=True)
            exprs.append(
                samples.head(MAX_SAMPLES).implode().alias(self._alias("samples"))
            )
        return exprs

    def failure(self, stats: dict[str, list]) -> Finding | None:
        count = stats[self._alias("cnt")][0]
        if not count:
            return None
        samples = self._samples(stats)
        mask = self.mask
        return Finding(
            code=self.code,
            key=self.key,
            message=self.message(count, samples, stats),
            columns=self.involved(),
            count=int(count),
            samples=tuple(samples),
            details=self.details(stats),
            _locate=lambda lf: lf.filter(mask),
        )

    def _samples(self, stats: dict[str, list]) -> list:
        if self.sample_expr is None:
            return []
        raw = stats[self._alias("samples")][0]
        return list(raw) if raw is not None else []

    def involved(self) -> tuple[str, ...]:
        return ()

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        raise NotImplementedError


@dataclass(kw_only=True)
class _Nullability(_Constraint):
    column: str
    sample_expr: None = None
    code: FindingCode = "nullability"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': non-nullable column contains "
            f"{count} null value(s)"
        )


@dataclass(kw_only=True)
class _AllowedValues(_Constraint):
    column: str
    allowed: list[Any]
    code: FindingCode = "choices"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"allowed": list(self.allowed)}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': found {count} invalid value(s) not in "
            f"allowed choices/categories {self.allowed}. Invalid samples: {samples}"
        )


@dataclass(kw_only=True)
class _Bounds(_Constraint):
    column: str
    bounds: Bound[Any]
    # The values the extremes are measured over: the column, or for a List
    # column its elements.
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

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

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
            f"Column '{self.column}': found {count} value(s) out of bounds "
            f"{self.bounds} (min found: {found_min}, max found: {found_max}). "
            f"Out of bounds samples: {samples}"
        )


@dataclass(kw_only=True)
class _StringLength(_Constraint):
    column: str
    length: Bound[int]
    unique_samples: bool = False
    code: FindingCode = "string_length"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {
            "string_length": [self.length.min, self.length.max] if self.length else None
        }

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': found {count} value(s) with string length "
            f"outside [{self.length.min}, {self.length.max}]. "
            f"Invalid samples: {samples}"
        )


@dataclass(kw_only=True)
class _ListLength(_Constraint):
    column: str
    length: Bound[int]
    unique_samples: bool = False
    code: FindingCode = "list_length"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"list_length": [self.length.min, self.length.max]}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': found {count} list(s) with a length "
            f"outside [{self.length.min}, {self.length.max}]. "
            f"Invalid samples: {samples}"
        )


@dataclass(kw_only=True)
class _ListElementNull(_Constraint):
    """A null *inside* a list. Generation never makes one, and no field on
    a ColSpec can ask for one, so it is reported under the nullability code
    like a null in a non-nullable column."""

    column: str
    code: FindingCode = "nullability"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': found {count} list(s) containing a null "
            f"element. Samples: {samples}"
        )


@dataclass(kw_only=True)
class _Format(_Constraint):
    column: str
    format: Format
    code: FindingCode = "format"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"format": self.format.name}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': found {count} value(s) that are not "
            f"{self.format}. Invalid samples: {samples}"
        )


@dataclass(kw_only=True)
class _Pattern(_Constraint):
    column: str
    pattern: str
    code: FindingCode = "pattern"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"pattern": self.pattern}

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': found {count} value(s) not matching "
            f"pattern {self.pattern!r}. Invalid samples: {samples}"
        )


@dataclass(kw_only=True)
class _RuleHolds(_Constraint):
    column: str
    rule: ColRule
    code: FindingCode = "rule"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {
            "when": repr(self.rule.when),
            "choices": list(self.rule.choices),
        }

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': found {count} value(s) violating "
            f"ColRule(when={self.rule.when}, choices={self.rule.choices}). "
            f"Violating samples: {samples}"
        )


@dataclass(kw_only=True)
class _ColumnValidator(_Constraint):
    column: str
    validator: Check
    code: FindingCode = "validator"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

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
class _UniqueValues(_Constraint):
    column: str
    code: FindingCode = "unique"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

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
# Dtype compatibility
# ---------------------------------------------------------------------------


def _is_dtype_compatible(
    expected: pl.DataType, actual: pl.DataType, *, strict: bool
) -> bool:
    """Whether `actual` can stand in for the declared `expected` dtype.

    Strict mode demands the same dtype, with String/Utf8 treated as one. The
    default is permissive about widening and about textual columns arriving as
    String rather than their declared Enum/Categorical, since that is how they
    come back from CSV and JSON.
    """
    if strict:
        if expected in (pl.String, pl.Utf8):
            return actual in (pl.String, pl.Utf8)
        return actual == expected

    if isinstance(expected, pl.Enum):
        return isinstance(actual, pl.Enum) or actual in (
            pl.String,
            pl.Utf8,
            pl.Categorical,
        )
    if isinstance(expected, pl.Categorical) or expected == pl.Categorical:
        return actual in (pl.String, pl.Utf8, pl.Categorical) or isinstance(
            actual, (pl.Enum, pl.Categorical)
        )
    if expected in (pl.String, pl.Utf8):
        return actual in (pl.String, pl.Utf8)
    if expected.is_integer():
        return actual.is_integer()
    if expected.is_float():
        return actual.is_float() or actual.is_integer()
    if expected.is_decimal():
        # A Decimal comes back from CSV and JSON as a float or an integer;
        # another Decimal stands in whatever its precision, since the
        # bounds are checked on the values either way.
        return actual.is_decimal() or actual.is_float() or actual.is_integer()
    if expected.is_temporal():
        return actual.is_temporal()
    if isinstance(expected, pl.List):
        return isinstance(actual, pl.List) and _is_dtype_compatible(
            element_dtype(expected), element_dtype(actual), strict=strict
        )
    if isinstance(expected, pl.Array):
        return (
            isinstance(actual, pl.Array)
            and actual.size == expected.size
            and _is_dtype_compatible(
                element_dtype(expected), element_dtype(actual), strict=strict
            )
        )
    return actual == expected


def _as_strings(values: Sequence[Any], dtype: pl.DataType) -> list[str]:
    """`values` as the strings a column of `dtype` holds them as, so a choice
    of `True` on a String column compares as `"true"` -- the same form
    generation produces -- rather than as Python's `str(True)`.
    """
    try:
        return _typed_values(values, dtype).cast(pl.String).to_list()
    except Exception:  # noqa: BLE001 - values the dtype cannot hold fall back to str()
        return [str(v) for v in values]


def _struct_of(names: Sequence[str]) -> pl.Expr | None:
    """A struct of the named columns, for sampling multi-column claims."""
    return pl.struct([pl.col(n) for n in names]) if names else None


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

    if isinstance(actual_dtype, (pl.List, pl.Array)):
        constraints.extend(_list_constraints(name, spec, actual_dtype, options))
    else:
        constraints.extend(
            _value_constraints(name, spec, actual_dtype, column, options)
        )

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


def _value_constraints(
    name: str,
    spec: ColSpec,
    actual_dtype: pl.DataType,
    column: pl.Expr,
    options: ValidationOptions,
) -> list[_Constraint]:
    """The constraints on one *value* -- its domain, bounds, length, format,
    pattern -- with masks over `column`, which is the column itself for a
    scalar and `pl.element()` for the elements of a List.
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
                key=f"{name}__choices",
                mask=present & ~in_domain,
                sample_expr=sample_expr,
                column=name,
                allowed=allowed,
            )
        )

    if spec.bounds is not None and not spec.bounds.is_open_both:
        constraints.append(
            _Bounds(
                key=f"{name}__bounds",
                mask=present & _out_of_bounds(column, spec.bounds, actual_dtype),
                sample_expr=column,
                column=name,
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
                    key=f"{name}__len",
                    mask=present & (too_short | too_long),
                    sample_expr=column,
                    column=name,
                    length=spec.string_length,
                )
            )

    if spec.format is not None:
        fmt = _lookup_format(spec.format)
        constraints.append(
            _Format(
                key=f"{name}__format",
                mask=present & ~fmt.check(column),
                sample_expr=column,
                column=name,
                format=fmt,
            )
        )

    if options.pattern and spec.pattern is not None:
        constraints.append(
            _Pattern(
                key=f"{name}__pattern",
                mask=present & ~column.str.contains(spec.pattern),
                sample_expr=column,
                column=name,
                pattern=spec.pattern,
            )
        )

    return constraints


def _list_constraints(
    name: str,
    spec: ColSpec,
    actual_dtype: pl.List | pl.Array,
    options: ValidationOptions,
) -> list[_Constraint]:
    """A List column's constraints: its length, no null elements, and every
    value constraint lifted over its elements.

    Each value constraint is built with `pl.element()` as its column, then
    its mask is run inside `list.eval` and a list fails where *any* element
    does. The samples and the located rows are the offending lists.
    """
    column = pl.col(name)
    present = column.is_not_null()
    constraints: list[_Constraint] = []

    if spec.list_length is not None and isinstance(actual_dtype, pl.List):
        length = column.list.len()
        constraints.append(
            _ListLength(
                key=f"{name}__list_len",
                mask=present
                & ~length.is_between(spec.list_length.min, spec.list_length.max),
                sample_expr=column,
                column=name,
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
            key=f"{name}__element_null",
            mask=present & any_element(pl.element().is_null()),
            sample_expr=column,
            column=name,
        )
    )

    for constraint in _value_constraints(
        name, spec, element_dtype(actual_dtype), pl.element(), options
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


def _rule_constraints(
    name: str,
    spec: ColSpec,
    actual_dtype: pl.DataType,
    df_col_names: Sequence[str],
) -> list[_Constraint]:
    """One constraint per ColRule, respecting first-match-wins ordering.

    Each rule only governs the rows no earlier rule already claimed, matching
    how `_apply_column_rules` assigns them at generation time -- including how
    it reads a null condition. A `when` that evaluates to null on a row does
    not match there, so generation folds it to False before both testing it and
    accumulating it into `claimed`. Doing anything else here lets a null
    propagate through `~claimed` and silently excuse every later rule on that
    row, which is a row generation did rewrite and validation would not check.
    """
    column = pl.col(name)
    constraints: list[_Constraint] = []
    claimed = pl.lit(False)

    for index, rule in enumerate(spec.rules):
        if not rule.when.root_names() <= set(df_col_names):
            continue  # reported through missing_cols instead
        matches = rule._expr().fill_null(False)
        applies = matches & ~claimed
        claimed = claimed | matches

        if _is_textual(actual_dtype):
            in_choices = column.cast(pl.String).is_in(
                _as_strings(rule.choices, spec.dtype)
            )
            sample_expr = column.cast(pl.String)
        else:
            in_choices = column.is_in(list(rule.choices))
            sample_expr = column

        constraints.append(
            _RuleHolds(
                key=f"{name}__rule_{index}",
                mask=applies & column.is_not_null() & ~in_choices,
                sample_expr=sample_expr,
                column=name,
                rule=rule,
            )
        )
    return constraints


def _frame_constraints(
    unique_together: Sequence[Sequence[str]] | None,
    checks: Sequence[Check] | None,
    df_col_names: Sequence[str],
) -> list[_Constraint]:
    """Constraints spanning several columns rather than belonging to one."""
    constraints: list[_Constraint] = []

    for index, group in enumerate(unique_together or ()):
        columns = tuple(group)
        if not all(c in df_col_names for c in columns):
            continue
        key_struct = pl.struct([pl.col(c) for c in columns])
        all_present = pl.all_horizontal([pl.col(c).is_not_null() for c in columns])
        constraints.append(
            _CompositeUnique(
                key=f"composite_{index}",
                mask=all_present & key_struct.is_duplicated(),
                sample_expr=key_struct,
                columns=columns,
            )
        )

    for check in checks or ():
        involved = [c for c in check.expr.meta.root_names() if c in df_col_names]
        constraints.append(
            _FrameCheck(
                key=f"check:{check.name}",
                mask=check._failure_mask(),
                sample_expr=_struct_of(involved),
                check=check,
            )
        )
    return constraints


# ---------------------------------------------------------------------------
# Foreign keys -- each needs its own join, so they cannot share the pass above
# ---------------------------------------------------------------------------


def _foreign_key_findings(
    lf: pl.LazyFrame,
    schema_name: str,
    foreign_keys: Sequence[tuple[ForeignKey, pl.LazyFrame | None]],
    df_col_names: Sequence[str],
    collect_kwargs: dict[str, Any],
) -> list[Finding]:
    findings: list[Finding] = []
    pending: list[tuple[ForeignKey, pl.LazyFrame | None, pl.LazyFrame, Any]] = []
    local_schema = lf.collect_schema()

    for fk, target_lf in foreign_keys:
        local_cols = list(fk.columns)
        ref_cols = list(fk.ref_columns)
        if not all(c in df_col_names for c in local_cols):
            continue  # already reported through missing_cols handling

        parent_lf = target_lf if target_lf is not None else lf
        parent_schema = parent_lf.collect_schema()
        parent_names = parent_schema.names()
        missing_ref = [c for c in ref_cols if c not in parent_names]
        if missing_ref:
            findings.append(
                Finding(
                    code="foreign_key",
                    key=f"fk:{fk.name}",
                    message=(
                        f"ForeignKey '{fk.name}' on {schema_name!r} references columns "
                        f"{missing_ref} not present in the referenced DataFrame"
                    ),
                    columns=tuple(local_cols),
                    details={
                        "target": fk.references,
                        "missing_ref_columns": missing_ref,
                    },
                )
            )
            continue

        key_expr = (
            pl.col(local_cols[0]) if len(local_cols) == 1 else pl.struct(local_cols)
        )
        present = (
            pl.col(local_cols[0]).is_not_null()
            if len(local_cols) == 1
            else pl.all_horizontal([pl.col(c).is_not_null() for c in local_cols])
        )
        # A key may legitimately span dtypes -- a String column referencing
        # an Enum primary key, which declaration allows and generation
        # handles -- and a join across the two would raise instead. Casting
        # the *parent's* keys to the local dtype settles it without touching
        # the frame under validation, so `rows()` returns it as it was. A
        # parent value the local dtype cannot hold becomes null and matches
        # nothing, which is right: the column could never have held it.
        parent_keys = parent_lf.select(
            [
                pl.col(ref_col).cast(local_schema[local_col], strict=False)
                if parent_schema[ref_col] != local_schema[local_col]
                else pl.col(ref_col)
                for local_col, ref_col in zip(local_cols, ref_cols, strict=True)
            ]
        ).unique()

        def orphans_of(
            frame: pl.LazyFrame, _p=present, _k=parent_keys, _l=local_cols, _r=ref_cols
        ) -> pl.LazyFrame:
            return frame.filter(_p).join(_k, left_on=_l, right_on=_r, how="anti")

        stats = orphans_of(lf.select(local_cols)).select(
            pl.len().alias("cnt"),
            key_expr.unique(maintain_order=True)
            .head(MAX_SAMPLES)
            .implode()
            .alias("samples"),
        )
        pending.append((fk, target_lf, stats, orphans_of))

    if pending:
        results = pl.collect_all([p[2] for p in pending], **collect_kwargs)
        for (fk, target_lf, _, orphans_of), stats in zip(pending, results, strict=True):
            count = stats["cnt"][0]
            if not count:
                continue
            raw_samples = stats["samples"][0]
            samples = list(raw_samples) if raw_samples is not None else []
            target_label = "self" if target_lf is None else fk.references
            local_cols = list(fk.columns)
            ref_cols = list(fk.ref_columns)
            findings.append(
                Finding(
                    code="foreign_key",
                    key=f"fk:{fk.name}",
                    message=(
                        f"ForeignKey '{fk.name}' violated ({local_cols} -> "
                        f"{target_label}.{ref_cols}): found {count} row(s) with no "
                        f"matching parent record. Violating samples: {samples}"
                    ),
                    columns=tuple(local_cols),
                    count=int(count),
                    samples=tuple(samples),
                    details={"target": target_label, "ref_columns": ref_cols},
                    _locate=orphans_of,
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Hierarchies -- pointer-chasing, so like foreign keys they run their own joins
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class _SingleParent(_Constraint):
    """Every reference points at one parent, which is what makes the walk
    terminate at a single ultimate parent."""

    column: str
    code: FindingCode = "hierarchy_multi_parent"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': the hierarchy gives each reference one "
            f"parent, but {count} row(s) repeat a reference that already has "
            f"one. Repeated samples: {samples}"
        )


def _hierarchy_constraints(
    hierarchy: Hierarchy, df_col_names: Sequence[str]
) -> list[_Constraint]:
    """The part of a hierarchy that fits the single aggregation pass."""
    if hierarchy.child not in df_col_names:
        return []
    column = pl.col(hierarchy.child)
    return [
        _SingleParent(
            key=f"{hierarchy.child}__single_parent",
            mask=column.is_not_null() & column.is_duplicated(),
            sample_expr=column,
            column=hierarchy.child,
        )
    ]


def _step(walk: pl.DataFrame, by: pl.DataFrame) -> pl.DataFrame:
    """`walk` advanced one hop along `by`, dropping a chain as it ends.

    One column pair throughout: `node` is where the walk started and `anc`
    where it has got to.

    Eager, and deliberately so. Advancing a *lazy* frame along itself nests
    that frame's own plan inside itself, so the doubling below would describe a
    query with two-to-the-rounds join nodes and spend half a minute planning a
    walk over fifty thousand rows. Collecting each round keeps the work
    proportional to the rows still walking, which is the point of the
    algorithm.
    """
    return walk.join(
        by.rename({"node": "_next", "anc": "_anc"}),
        left_on="anc",
        right_on="_next",
        how="inner",
    ).select("node", pl.col("_anc").alias("anc"))


def _deeper_than(edges: pl.DataFrame, hops: int) -> pl.Series:
    """The references whose chain of parents is longer than `hops` edges."""
    walk = edges
    for _ in range(hops):
        if walk.is_empty():
            break
        walk = _step(walk, edges)
    return walk["node"].unique()


def _endless(edges: pl.DataFrame, rounds: int) -> pl.Series:
    """The references whose chain never reaches an ultimate parent.

    Pointer doubling: each round carries `anc` twice as far up, so `rounds`
    of them cover a chain of two-to-the-rounds edges. A chain that reaches a
    root drops out along the way, and whatever outlasts any chain the frame
    could hold is exactly what loops -- a reference on a cycle, or one hanging
    below one.
    """
    ptr = edges
    for _ in range(rounds):
        if ptr.is_empty():
            break
        ptr = _step(ptr, ptr)
    return ptr["node"].unique()


def _hierarchy_findings(
    lf: pl.LazyFrame,
    schema_name: str,
    hierarchy: Hierarchy,
    df_col_names: Sequence[str],
    collect_kwargs: dict[str, Any],
) -> list[Finding]:
    """Cycles and over-deep chains, each found by a bounded walk.

    Bounded is the point. The data this runs against is data someone generated
    *in order* to contain cycles, so a validator that walks until it reaches a
    root is a validator that hangs on its own test fixtures. Depth costs
    `max_depth` joins; the cycle check costs a logarithmic number, because
    pointer doubling covers a chain of length `n` in `log2(n)` steps.
    """
    child, parent = hierarchy.child, hierarchy.parent
    if not all(c in df_col_names for c in (child, parent)):
        return []  # reported through missing_cols instead

    edges = (
        lf.select(pl.col(child).alias("node"), pl.col(parent).alias("anc"))
        .filter(pl.col("node").is_not_null() & pl.col("anc").is_not_null())
        .collect(**collect_kwargs)
    )
    if edges.is_empty():
        return []
    rounds = max(1, math.ceil(math.log2(max(edges.height, 2))) + 1)

    endless = _endless(edges, rounds)
    deep = _deeper_than(edges, hierarchy.max_depth)
    # A reference in a cycle also outruns any depth, so it is reported once,
    # as the cycle it is. `implode` because comparing two Series of one dtype
    # with `is_in` is ambiguous and deprecated: the right-hand side has to say
    # it is one collection rather than a column of values to match row-wise.
    over_deep = deep.filter(~deep.is_in(endless.implode()))

    findings: list[Finding] = []
    for values, code, describe in (
        (endless, "hierarchy_cycle", _cycle_message),
        (over_deep, "hierarchy_depth", _depth_message),
    ):
        if values.is_empty():
            continue
        offenders = values.to_list()
        mask = pl.col(child).is_in(offenders)
        count = int(lf.select(mask.sum()).collect(**collect_kwargs).item())
        samples = offenders[:MAX_SAMPLES]
        findings.append(
            Finding(
                code=code,  # type: ignore[arg-type]
                key=f"hierarchy:{code}",
                message=describe(schema_name, hierarchy, count, samples),
                columns=(child, parent),
                count=count,
                samples=tuple(samples),
                details={
                    "child": child,
                    "parent": parent,
                    "max_depth": hierarchy.max_depth,
                },
                _locate=lambda frame, _m=mask: frame.filter(_m),
            )
        )
    return findings


def _cycle_message(
    schema_name: str, hierarchy: Hierarchy, count: int, samples: list
) -> str:
    return (
        f"Hierarchy on {schema_name!r} ({hierarchy.child} -> {hierarchy.parent}): "
        f"{count} row(s) never reach an ultimate parent, because their chain of "
        f"parents forms a loop. Walking them without a visited set will not "
        f"terminate. Samples: {samples}"
    )


def _depth_message(
    schema_name: str, hierarchy: Hierarchy, count: int, samples: list
) -> str:
    return (
        f"Hierarchy on {schema_name!r} ({hierarchy.child} -> {hierarchy.parent}): "
        f"{count} row(s) sit deeper than the declared max_depth of "
        f"{hierarchy.max_depth}. Samples: {samples}"
    )
