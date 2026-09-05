"""Every claim a spec makes, as a `_Constraint` that produces a `Finding`.

A constraint contributes aggregation expressions to one pass over the frame,
then turns the results back into a `Finding`: a count, a few samples, the
facts that make the message actionable, and a way to locate the offending
rows later. All constraints are collected first and evaluated together, so
validating fifty columns costs one scan rather than fifty. Foreign keys are
the exception: each needs its own anti-join.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec.dtypes import _typed_values
from polspec.validation.report import Finding

if TYPE_CHECKING:
    from polspec.bound import Bound
    from polspec.check import Check
    from polspec.foreign_key import ForeignKey
    from polspec.rules import ColRule
    from polspec.spec import ColSpec
    from polspec.validation import ValidationOptions

MAX_SAMPLES = 5


@dataclass
class _Constraint:
    """One checkable claim, measured by counting the rows that violate a mask.

    Subclasses supply the mask, the wording and the details; the aliases
    tying the two halves together stay private to the instance.
    """

    key: str
    mask: pl.Expr
    sample_expr: pl.Expr | None = None
    unique_samples: bool = True
    code: str = ""

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
            code=self.code,  # type: ignore[arg-type]
            key=self.key,
            message=self.message(count, samples, stats),
            columns=self.involved(),
            count=int(count),
            samples=samples,
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


@dataclass
class _Nullability(_Constraint):
    column: str = ""
    sample_expr: None = None
    code: str = "nullability"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': non-nullable column contains "
            f"{count} null value(s)"
        )


@dataclass
class _AllowedValues(_Constraint):
    column: str = ""
    allowed: list[Any] = field(default_factory=list)
    code: str = "choices"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

    def details(self, stats: dict[str, list]) -> dict[str, Any]:
        return {"allowed": list(self.allowed)}

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
    code: str = "bounds"

    def aggregations(self) -> list[pl.Expr]:
        # The observed extremes make an out-of-bounds report actionable, so
        # they are gathered alongside the violating samples.
        return [
            *super().aggregations(),
            pl.col(self.column).min().alias(self._alias("min")),
            pl.col(self.column).max().alias(self._alias("max")),
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


@dataclass
class _StringLength(_Constraint):
    column: str = ""
    length: Bound | None = None
    unique_samples: bool = False
    code: str = "string_length"

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


@dataclass
class _RuleHolds(_Constraint):
    column: str = ""
    rule: ColRule | None = None
    code: str = "rule"

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


@dataclass
class _ColumnValidator(_Constraint):
    column: str = ""
    validator: Check | None = None
    code: str = "validator"

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


@dataclass
class _UniqueValues(_Constraint):
    column: str = ""
    code: str = "unique"

    def involved(self) -> tuple[str, ...]:
        return (self.column,)

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Column '{self.column}': unique column contains {count} duplicate "
            f"value(s). Duplicate samples: {samples}"
        )


@dataclass
class _CompositeUnique(_Constraint):
    columns: tuple[str, ...] = ()
    code: str = "unique_together"

    def involved(self) -> tuple[str, ...]:
        return self.columns

    def message(self, count: int, samples: list, stats: dict[str, list]) -> str:
        return (
            f"Composite unique key {list(self.columns)} violated: found {count} "
            f"duplicate row(s). Duplicate samples: {samples}"
        )


@dataclass
class _FrameCheck(_Constraint):
    check: Check | None = None
    code: str = "check"

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
    if expected.is_temporal():
        return actual.is_temporal()
    return actual == expected


def _is_textual(dtype: pl.DataType) -> bool:
    """Whether values of `dtype` are compared to `choices` by their string form."""
    return dtype in (pl.String, pl.Utf8, pl.Categorical) or isinstance(
        dtype, (pl.Enum, pl.Categorical)
    )


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

    Most are skipped when the dtype is already wrong: comparing values against
    bounds or choices of an incompatible type produces noise on top of the
    dtype finding the caller will already see.
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
            in_domain = column.cast(pl.String).is_in(_as_strings(allowed, spec.dtype))
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
        if not rule.when.root_names() <= set(df_col_names):
            continue  # reported through missing_cols instead
        applies = rule._expr() & ~claimed
        claimed = claimed | rule._expr()

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

    for fk, target_lf in foreign_keys:
        local_cols = list(fk.columns)
        ref_cols = list(fk.ref_columns)
        if not all(c in df_col_names for c in local_cols):
            continue  # already reported through missing_cols handling

        parent_lf = target_lf if target_lf is not None else lf
        parent_names = parent_lf.collect_schema().names()
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
        parent_keys = parent_lf.select(ref_cols).unique()

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
                    samples=samples,
                    details={"target": target_label, "ref_columns": ref_cols},
                    _locate=orphans_of,
                )
            )
    return findings
