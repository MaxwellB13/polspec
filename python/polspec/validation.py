"""Checking a frame against the claims a spec makes about it.

Every claim is a `_Constraint`: it contributes aggregation expressions, and it
turns the results back into an error message. All of them are collected first
and evaluated in a single pass over the frame, so validating fifty columns
costs one scan rather than fifty.

This replaces a pair of long loops that communicated through hand-built alias
strings (`f"__val__{name}__oob_cnt"`) built in one and looked up in the other,
two hundred lines apart. Each constraint now owns its own aliases, and adding
a new kind of check is a new class rather than an edit in two distant places.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

if TYPE_CHECKING:
    from polspec.bound import Bound
    from polspec.check import Check
    from polspec.foreign_key import ForeignKey
    from polspec.rules import ColRule
    from polspec.spec import ColSpec

_MAX_SAMPLES = 5


class ValidationError(ValueError):
    """Raised when a DataFrame or LazyFrame fails validation against a FrameSpec/FrameSchema."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


@dataclass(frozen=True, slots=True)
class _Options:
    """Which checks to run and what to do about structural mismatches.

    Grouped rather than passed as ten separate arguments, so adding a check
    does not widen every signature between here and `FrameSpec.validate`.
    """

    extra_cols: Literal["drop", "allow", "raise"] = "raise"
    missing_cols: Literal["add", "allow", "raise"] = "raise"
    strict_dtypes: bool = False
    rules: bool = True
    validators: bool = True
    unique: bool = True
    checks: bool = True
    foreign_keys: bool = True
    cast: bool = False
    streaming: bool = False

    def __post_init__(self) -> None:
        if self.extra_cols not in ("drop", "allow", "raise"):
            raise ValueError(
                f"extra_cols must be one of 'drop', 'allow', 'raise'; got {self.extra_cols!r}"
            )
        if self.missing_cols not in ("add", "allow", "raise"):
            raise ValueError(
                f"missing_cols must be one of 'add', 'allow', 'raise'; got {self.missing_cols!r}"
            )


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


@dataclass
class _Constraint:
    """One checkable claim, measured by counting the rows that violate a mask.

    Subclasses supply the mask and the wording; the aliases tying the two
    halves together stay private to the instance.
    """

    key: str
    mask: pl.Expr
    sample_expr: pl.Expr | None = None
    unique_samples: bool = True

    def _alias(self, suffix: str) -> str:
        return f"__val__{self.key}__{suffix}"

    def aggregations(self) -> list[pl.Expr]:
        exprs = [self.mask.sum().alias(self._alias("cnt"))]
        if self.sample_expr is not None:
            samples = self.sample_expr.filter(self.mask)
            if self.unique_samples:
                samples = samples.unique()
            exprs.append(
                samples.head(_MAX_SAMPLES).implode().alias(self._alias("samples"))
            )
        return exprs

    def failure(self, stats: dict[str, list]) -> str | None:
        count = stats[self._alias("cnt")][0]
        if not count:
            return None
        return self.message(count, self._samples(stats), stats)

    def _samples(self, stats: dict[str, list]) -> list:
        if self.sample_expr is None:
            return []
        raw = stats[self._alias("samples")][0]
        return list(raw) if raw is not None else []

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        raise NotImplementedError


@dataclass
class _Nullability(_Constraint):
    column: str = ""
    sample_expr: None = None

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': non-nullable column contains "
            f"{count} null value(s)"
        )


@dataclass
class _AllowedValues(_Constraint):
    column: str = ""
    allowed: list[Any] = field(default_factory=list)

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': found {count} invalid value(s) not in "
            f"allowed choices/categories {self.allowed}. Invalid samples: {samples}"
        )


@dataclass
class _Bounds(_Constraint):
    column: str = ""
    bounds: Bound | None = None
    unique_samples: bool = False

    def aggregations(self) -> list[pl.Expr]:
        # The observed extremes make an out-of-bounds report actionable, so
        # they are gathered alongside the violating samples.
        return super().aggregations() + [
            pl.col(self.column).min().alias(self._alias("min")),
            pl.col(self.column).max().alias(self._alias("max")),
        ]

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        found_min = stats[self._alias("min")][0]
        found_max = stats[self._alias("max")][0]
        return (
            f"Column '{self.column}': found {count} value(s) out of bounds "
            f"{self.bounds} (min found: {found_min}, max found: {found_max}). "
            f"Out of bounds samples: {samples}"
        )


@dataclass
class _StringLength(_Constraint):
    column: str = ""
    length: Bound | None = None
    unique_samples: bool = False

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': found {count} value(s) with string length "
            f"outside [{self.length.min}, {self.length.max}]. "
            f"Invalid samples: {samples}"
        )


@dataclass
class _RuleHolds(_Constraint):
    column: str = ""
    rule: ColRule | None = None

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': found {count} value(s) violating "
            f"ColRule(when={self.rule.when}, choices={self.rule.choices}). "
            f"Violating samples: {samples}"
        )


@dataclass
class _ColumnValidator(_Constraint):
    column: str = ""
    validator: Check | None = None
    sample_expr: None = None

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        described = (
            f" ({self.validator.description})" if self.validator.description else ""
        )
        return (
            f"Column '{self.column}': validator '{self.validator.name}' failed: "
            f"found {count} row(s) violating condition {self.validator.expr}{described}"
        )


@dataclass
class _UniqueValues(_Constraint):
    column: str = ""

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': unique column contains {count} duplicate "
            f"value(s). Duplicate samples: {samples}"
        )


@dataclass
class _CompositeUnique(_Constraint):
    columns: tuple[str, ...] = ()

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Composite unique key {list(self.columns)} violated: found {count} "
            f"duplicate row(s). Duplicate samples: {samples}"
        )


@dataclass
class _FrameCheck(_Constraint):
    check: Check | None = None
    sample_expr: None = None

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
    if expected.is_temporal():
        return actual.is_temporal()
    return actual == expected


def _is_textual(dtype: pl.DataType) -> bool:
    """Whether values of `dtype` are compared to `choices` by their string form."""
    return dtype in (pl.String, pl.Utf8, pl.Categorical) or isinstance(
        dtype, (pl.Enum, pl.Categorical)
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
    options: _Options,
    df_col_names: Sequence[str],
) -> list[_Constraint]:
    """The constraints one declared column contributes to the single pass.

    Most are skipped when the dtype is already wrong: comparing values against
    bounds or choices of an incompatible type produces noise on top of the
    dtype error the caller will already see.
    """
    column = pl.col(name)
    present = column.is_not_null()
    constraints: list[_Constraint] = []

    if not spec.nullable:
        constraints.append(
            _Nullability(key=f"{name}__null", mask=column.is_null(), column=name)
        )

    allowed = _allowed_values(spec)
    if allowed is not None:
        if _is_textual(actual_dtype):
            in_domain = column.cast(pl.String).is_in([str(c) for c in allowed])
            sample_expr = column.cast(pl.String)
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

    if not compatible:
        return constraints

    if spec.bounds is not None and not spec.bounds.is_open_both:
        constraints.append(
            _Bounds(
                key=f"{name}__bounds",
                mask=present & _out_of_bounds(name, spec.bounds, actual_dtype),
                sample_expr=column,
                column=name,
                bounds=spec.bounds,
            )
        )

    if spec.string_length is not None:
        measured = _measure_length(name, actual_dtype)
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

    if options.rules and spec.rules:
        constraints.extend(_rule_constraints(name, spec, actual_dtype, df_col_names))

    if options.validators and spec.validators:
        constraints.extend(
            _ColumnValidator(
                key=f"{name}__validator_{index}",
                mask=validator._failure_mask(),
                column=name,
                validator=validator,
            )
            for index, validator in enumerate(spec.validators)
        )

    if options.unique and spec.unique:
        constraints.append(
            _UniqueValues(
                key=f"{name}__unique",
                mask=present & column.is_duplicated(),
                sample_expr=column,
                column=name,
            )
        )

    return constraints


def _allowed_values(spec: ColSpec) -> list[Any] | None:
    """The closed domain this column's values must fall in, if it has one."""
    if isinstance(spec.dtype, pl.Enum):
        categories = spec.dtype.categories.to_list()
        if spec.choices is not None:
            return [c for c in spec.choices if c in categories]
        return categories
    if spec.choices is not None:
        return list(spec.choices)
    return None


def _out_of_bounds(name: str, bounds: Bound, actual_dtype: pl.DataType) -> pl.Expr:
    """A mask for values outside `bounds`, testing only the constrained sides.

    An open end is genuinely unconstrained here, unlike at generation time
    where it falls back to a default (see `ColSpec.bounds`).
    """
    column = pl.col(name)

    def limit(value: Any) -> Any:
        return pl.lit(value).cast(actual_dtype) if actual_dtype.is_temporal() else value

    violations = []
    if bounds.min is not None:
        violations.append(column < limit(bounds.min))
    if bounds.max is not None:
        violations.append(column > limit(bounds.max))
    return violations[0] if len(violations) == 1 else violations[0] | violations[1]


def _measure_length(name: str, actual_dtype: pl.DataType) -> pl.Expr | None:
    """Length of each value, for the dtypes where that is meaningful."""
    if actual_dtype in (pl.String, pl.Utf8):
        return pl.col(name).str.len_chars()
    if actual_dtype == pl.Binary:
        return pl.col(name).bin.size()
    return None


def _rule_constraints(
    name: str,
    spec: ColSpec,
    actual_dtype: pl.DataType,
    df_col_names: Sequence[str],
) -> list[_Constraint]:
    """One constraint per ColRule, respecting first-match-wins ordering.

    Each rule only governs the rows no earlier rule already claimed, matching
    how `_apply_rules` assigns them at generation time.
    """
    column = pl.col(name)
    constraints: list[_Constraint] = []
    claimed = pl.lit(False)

    for index, rule in enumerate(spec.rules):
        if rule.when.get("column") not in df_col_names:
            continue  # reported through missing_cols instead
        applies = rule._expr() & ~claimed
        claimed = claimed | rule._expr()

        if _is_textual(actual_dtype):
            in_choices = column.cast(pl.String).is_in([str(c) for c in rule.choices])
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

    constraints.extend(
        _FrameCheck(key=f"check_{index}", mask=check._failure_mask(), check=check)
        for index, check in enumerate(checks or ())
    )
    return constraints


# ---------------------------------------------------------------------------
# Foreign keys -- each needs its own join, so they cannot share the pass above
# ---------------------------------------------------------------------------


def _foreign_key_errors(
    lf: pl.LazyFrame,
    schema_name: str,
    foreign_keys: Sequence[tuple[ForeignKey, pl.LazyFrame | None]],
    df_col_names: Sequence[str],
    collect_kwargs: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for fk, target_lf in foreign_keys:
        local_cols = list(fk.columns)
        ref_cols = list(fk.ref_columns)
        if not all(c in df_col_names for c in local_cols):
            continue  # already reported through missing_cols handling

        parent_lf = target_lf if target_lf is not None else lf
        missing_ref = [
            c for c in ref_cols if c not in parent_lf.collect_schema().names()
        ]
        if missing_ref:
            raise ValueError(
                f"ForeignKey '{fk.name}' on {schema_name!r} references columns "
                f"{missing_ref} not present in the referenced DataFrame"
            )

        key_expr = (
            pl.col(local_cols[0]) if len(local_cols) == 1 else pl.struct(local_cols)
        )
        present = (
            pl.col(local_cols[0]).is_not_null()
            if len(local_cols) == 1
            else pl.all_horizontal([pl.col(c).is_not_null() for c in local_cols])
        )
        orphans = (
            lf.select(local_cols)
            .filter(present)
            .join(
                parent_lf.select(ref_cols).unique(),
                left_on=local_cols,
                right_on=ref_cols,
                how="anti",
            )
        )
        stats = orphans.select(
            pl.len().alias("cnt"),
            key_expr.unique().head(_MAX_SAMPLES).implode().alias("samples"),
        ).collect(**collect_kwargs)

        count = stats["cnt"][0]
        if not count:
            continue
        raw_samples = stats["samples"][0]
        target_label = (
            "self"
            if target_lf is None
            else getattr(fk.references, "__name__", str(fk.references))
        )
        errors.append(
            f"ForeignKey '{fk.name}' violated ({local_cols} -> "
            f"{target_label}.{ref_cols}): found {count} row(s) with no "
            f"matching parent record. Violating samples: "
            f"{list(raw_samples) if raw_samples is not None else []}"
        )
    return errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _validate_dataframe(
    columns: dict[str, ColSpec],
    schema_name: str,
    df: pl.DataFrame | pl.LazyFrame,
    options: _Options,
    *,
    checks: Sequence[Check] | None = None,
    unique_together: Sequence[Sequence[str]] | None = None,
    foreign_keys: Sequence[tuple[ForeignKey, pl.LazyFrame | None]] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Validates a DataFrame or LazyFrame against declared ColSpecs, checks, and unique constraints."""
    is_lazy = isinstance(df, pl.LazyFrame)
    lf = df if is_lazy else df.lazy()

    df_schema = lf.collect_schema()
    df_col_names = df_schema.names()

    extra = [col for col in df_col_names if col not in columns]
    missing = [col for col in columns if col not in df_col_names]

    errors: list[str] = []
    if extra and options.extra_cols == "raise":
        errors.append(f"Extra columns found that are not in schema: {extra}")
    if missing and options.missing_cols == "raise":
        errors.append(f"Missing required columns in DataFrame: {missing}")

    constraints: list[_Constraint] = []
    for name in (c for c in columns if c in df_col_names):
        spec = columns[name]
        actual_dtype = df_schema[name]
        compatible = _is_dtype_compatible(
            spec.dtype, actual_dtype, strict=options.strict_dtypes
        )
        if not compatible:
            errors.append(
                f"Column '{name}': expected dtype {spec.dtype}, got {actual_dtype}"
            )
        constraints.extend(
            _column_constraints(
                name,
                spec,
                actual_dtype,
                compatible=compatible,
                options=options,
                df_col_names=df_col_names,
            )
        )

    constraints.extend(
        _frame_constraints(
            unique_together if options.unique else None,
            checks if options.checks else None,
            df_col_names,
        )
    )

    collect_kwargs: dict[str, Any] = (
        {"engine": "streaming"} if options.streaming else {}
    )

    if constraints:
        aggregations = [expr for c in constraints for expr in c.aggregations()]
        stats = (
            lf.select(aggregations).collect(**collect_kwargs).to_dict(as_series=False)
        )
        errors.extend(
            message
            for message in (c.failure(stats) for c in constraints)
            if message is not None
        )

    if options.foreign_keys and foreign_keys:
        errors.extend(
            _foreign_key_errors(
                lf, schema_name, foreign_keys, df_col_names, collect_kwargs
            )
        )

    if errors:
        raise ValidationError(
            "\n".join(
                [
                    (
                        f"Validation failed for DataFrame against '{schema_name}' "
                        f"({len(errors)} error(s) found):"
                    ),
                    *(f"  - {err}" for err in errors),
                ]
            ),
            errors=errors,
        )

    result = _apply_transformations(lf, columns, extra, missing, options)
    return result if is_lazy else result.collect()


def _apply_transformations(
    lf: pl.LazyFrame,
    columns: dict[str, ColSpec],
    extra: list[str],
    missing: list[str],
    options: _Options,
) -> pl.LazyFrame:
    """Drops, adds, casts and reorders once validation has passed."""
    if extra and options.extra_cols == "drop":
        lf = lf.drop(extra)

    if missing and options.missing_cols == "add":
        lf = lf.with_columns(
            pl.lit(None, dtype=columns[c].dtype).alias(c) for c in missing
        )

    if options.cast:
        schema = lf.collect_schema()
        cast_exprs = [
            pl.col(name).cast(spec.dtype)
            for name, spec in columns.items()
            if name in schema and schema[name] != spec.dtype
        ]
        if cast_exprs:
            lf = lf.with_columns(cast_exprs)

    # Declared columns first, in declaration order, then anything extra that
    # survived.
    present = lf.collect_schema().names()
    declared = [c for c in columns if c in present]
    return lf.select(declared + [c for c in present if c not in columns])
