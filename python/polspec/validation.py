from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

if TYPE_CHECKING:
    from polspec.check import Check
    from polspec.foreign_key import ForeignKey
    from polspec.spec import ColSpec


class ValidationError(ValueError):
    """Raised when a DataFrame or LazyFrame fails validation against a FrameSpec/FrameSchema."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


def _validate_dataframe(
    columns: dict[str, ColSpec],
    schema_name: str,
    df: pl.DataFrame | pl.LazyFrame,
    *,
    extra_cols: Literal["drop", "allow", "raise"] = "raise",
    missing_cols: Literal["add", "allow", "raise"] = "raise",
    strict_dtypes: bool = False,
    validate_rules: bool = True,
    validate_unique: bool = True,
    validate_checks: bool = True,
    validate_foreign_keys: bool = True,
    checks: Sequence[Check] | None = None,
    unique_together: Sequence[Sequence[str]] | None = None,
    foreign_keys: Sequence[tuple[ForeignKey, pl.LazyFrame | None]] | None = None,
    cast: bool = False,
    streaming: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Validates a DataFrame or LazyFrame against declared ColSpecs, checks, and unique constraints."""
    if extra_cols not in ("drop", "allow", "raise"):
        raise ValueError(
            f"extra_cols must be one of 'drop', 'allow', 'raise'; got {extra_cols!r}"
        )
    if missing_cols not in ("add", "allow", "raise"):
        raise ValueError(
            f"missing_cols must be one of 'add', 'allow', 'raise'; got {missing_cols!r}"
        )

    is_lazy = isinstance(df, pl.LazyFrame)
    lf = df if is_lazy else df.lazy()

    df_schema = lf.collect_schema()
    df_col_names = df_schema.names()
    spec_col_names = list(columns.keys())

    extra = [col for col in df_col_names if col not in columns]
    missing = [col for col in spec_col_names if col not in df_col_names]

    errors: list[str] = []

    # 1. Check Extra Columns
    if extra and extra_cols == "raise":
        errors.append(f"Extra columns found that are not in schema: {extra}")

    # 2. Check Missing Columns
    if missing and missing_cols == "raise":
        errors.append(f"Missing required columns in DataFrame: {missing}")

    # 3. Present columns to validate
    present_cols = [col for col in spec_col_names if col in df_col_names]

    # Check dtypes and build aggregation expressions for data validation
    agg_exprs: list[pl.Expr] = []
    expr_metadata: list[dict[str, Any]] = []

    for name in present_cols:
        spec = columns[name]
        actual_dtype = df_schema[name]
        expected_dtype = spec.dtype

        # Type compatibility check
        is_type_compatible = True
        if strict_dtypes:
            if expected_dtype in (pl.String, pl.Utf8):
                if actual_dtype not in (pl.String, pl.Utf8):
                    is_type_compatible = False
            elif actual_dtype != expected_dtype:
                is_type_compatible = False
        else:
            if isinstance(expected_dtype, pl.Enum):
                if not (
                    isinstance(actual_dtype, pl.Enum)
                    or actual_dtype in (pl.String, pl.Utf8, pl.Categorical)
                ):
                    is_type_compatible = False
            elif (
                isinstance(expected_dtype, pl.Categorical)
                or expected_dtype == pl.Categorical
            ):
                if not (
                    actual_dtype in (pl.String, pl.Utf8, pl.Categorical)
                    or isinstance(actual_dtype, (pl.Enum, pl.Categorical))
                ):
                    is_type_compatible = False
            elif expected_dtype in (pl.String, pl.Utf8):
                if actual_dtype not in (pl.String, pl.Utf8):
                    is_type_compatible = False
            elif expected_dtype.is_integer():
                if not actual_dtype.is_integer():
                    is_type_compatible = False
            elif expected_dtype.is_float():
                if not (actual_dtype.is_float() or actual_dtype.is_integer()):
                    is_type_compatible = False
            elif expected_dtype.is_temporal():
                if not actual_dtype.is_temporal():
                    is_type_compatible = False
            elif actual_dtype != expected_dtype:
                is_type_compatible = False

        if not is_type_compatible:
            errors.append(
                f"Column '{name}': expected dtype {expected_dtype}, got {actual_dtype}"
            )

        # Nullability validation
        if not spec.nullable:
            null_cnt_alias = f"__val__{name}__null_cnt"
            agg_exprs.append(pl.col(name).null_count().alias(null_cnt_alias))
            expr_metadata.append(
                {
                    "type": "nullability",
                    "column": name,
                    "alias": null_cnt_alias,
                }
            )

        # Enum & Choices validation
        allowed_choices: list[Any] | None = None
        if isinstance(spec.dtype, pl.Enum):
            allowed_choices = spec.dtype.categories.to_list()
            if spec.choices is not None:
                allowed_choices = [c for c in spec.choices if c in allowed_choices]
        elif spec.choices is not None:
            allowed_choices = list(spec.choices)

        if allowed_choices is not None:
            invalid_alias_cnt = f"__val__{name}__choice_invalid_cnt"
            invalid_alias_samples = f"__val__{name}__choice_samples"

            # Use string representation comparison for Enum/Categorical/String
            if actual_dtype in (pl.String, pl.Utf8, pl.Categorical) or isinstance(
                actual_dtype, (pl.Enum, pl.Categorical)
            ):
                str_allowed = [str(c) for c in allowed_choices]
                invalid_mask = pl.col(name).is_not_null() & (
                    ~pl.col(name).cast(pl.String).is_in(str_allowed)
                )
                sample_expr = pl.col(name).cast(pl.String)
            else:
                invalid_mask = pl.col(name).is_not_null() & (
                    ~pl.col(name).is_in(allowed_choices)
                )
                sample_expr = pl.col(name)

            agg_exprs.append(invalid_mask.sum().alias(invalid_alias_cnt))
            agg_exprs.append(
                sample_expr.filter(invalid_mask)
                .unique()
                .head(5)
                .implode()
                .alias(invalid_alias_samples)
            )
            expr_metadata.append(
                {
                    "type": "choices",
                    "column": name,
                    "allowed": allowed_choices,
                    "cnt_alias": invalid_alias_cnt,
                    "samples_alias": invalid_alias_samples,
                }
            )

        # Bounds validation (numeric / temporal)
        if spec.bounds is not None and is_type_compatible:
            b_min = spec.bounds.min
            b_max = spec.bounds.max
            oob_cnt_alias = f"__val__{name}__oob_cnt"
            oob_samples_alias = f"__val__{name}__oob_samples"
            min_alias = f"__val__{name}__min_val"
            max_alias = f"__val__{name}__max_val"

            if actual_dtype.is_temporal():
                lit_min = pl.lit(b_min).cast(actual_dtype)
                lit_max = pl.lit(b_max).cast(actual_dtype)
                oob_mask = pl.col(name).is_not_null() & (
                    (pl.col(name) < lit_min) | (pl.col(name) > lit_max)
                )
            else:
                oob_mask = pl.col(name).is_not_null() & (
                    (pl.col(name) < b_min) | (pl.col(name) > b_max)
                )
            agg_exprs.append(oob_mask.sum().alias(oob_cnt_alias))
            agg_exprs.append(
                pl.col(name).filter(oob_mask).head(5).implode().alias(oob_samples_alias)
            )
            agg_exprs.append(pl.col(name).min().alias(min_alias))
            agg_exprs.append(pl.col(name).max().alias(max_alias))
            expr_metadata.append(
                {
                    "type": "bounds",
                    "column": name,
                    "min": b_min,
                    "max": b_max,
                    "cnt_alias": oob_cnt_alias,
                    "samples_alias": oob_samples_alias,
                    "min_alias": min_alias,
                    "max_alias": max_alias,
                }
            )

        # String / Binary length validation
        if spec.string_length is not None and is_type_compatible:
            len_min = spec.string_length.min
            len_max = spec.string_length.max
            len_cnt_alias = f"__val__{name}__len_cnt"
            len_samples_alias = f"__val__{name}__len_samples"

            if actual_dtype in (pl.String, pl.Utf8):
                len_col = pl.col(name).str.len_chars()
            elif actual_dtype == pl.Binary:
                len_col = pl.col(name).bin.size()
            else:
                len_col = None

            if len_col is not None:
                len_mask = pl.col(name).is_not_null() & (
                    (len_col < len_min) | (len_col > len_max)
                )
                agg_exprs.append(len_mask.sum().alias(len_cnt_alias))
                agg_exprs.append(
                    pl.col(name)
                    .filter(len_mask)
                    .head(5)
                    .implode()
                    .alias(len_samples_alias)
                )
                expr_metadata.append(
                    {
                        "type": "string_length",
                        "column": name,
                        "min": len_min,
                        "max": len_max,
                        "cnt_alias": len_cnt_alias,
                        "samples_alias": len_samples_alias,
                    }
                )

        # Rules validation
        if validate_rules and spec.rules and is_type_compatible:
            matched_prior = pl.lit(False)
            for r_idx, rule in enumerate(spec.rules):
                ref_col = rule.when.get("column")
                if ref_col not in df_col_names:
                    continue

                r_cnt_alias = f"__val__{name}__rule_{r_idx}_cnt"
                r_samples_alias = f"__val__{name}__rule_{r_idx}_samples"

                condition_expr = rule._expr() & (~matched_prior)
                matched_prior = matched_prior | rule._expr()

                if actual_dtype in (pl.String, pl.Utf8, pl.Categorical) or isinstance(
                    actual_dtype, (pl.Enum, pl.Categorical)
                ):
                    str_rule_choices = [str(c) for c in rule.choices]
                    rule_viol_mask = (
                        condition_expr
                        & pl.col(name).is_not_null()
                        & (~pl.col(name).cast(pl.String).is_in(str_rule_choices))
                    )
                    r_sample_expr = pl.col(name).cast(pl.String)
                else:
                    rule_viol_mask = (
                        condition_expr
                        & pl.col(name).is_not_null()
                        & (~pl.col(name).is_in(list(rule.choices)))
                    )
                    r_sample_expr = pl.col(name)

                agg_exprs.append(rule_viol_mask.sum().alias(r_cnt_alias))
                agg_exprs.append(
                    r_sample_expr.filter(rule_viol_mask)
                    .unique()
                    .head(5)
                    .implode()
                    .alias(r_samples_alias)
                )
                expr_metadata.append(
                    {
                        "type": "rule",
                        "column": name,
                        "rule_idx": r_idx,
                        "rule": rule,
                        "cnt_alias": r_cnt_alias,
                        "samples_alias": r_samples_alias,
                    }
                )

        # Uniqueness validation (single-column)
        if validate_unique and spec.unique and is_type_compatible:
            uniq_cnt_alias = f"__val__{name}__uniq_cnt"
            uniq_samples_alias = f"__val__{name}__uniq_samples"
            uniq_mask = pl.col(name).is_not_null() & pl.col(name).is_duplicated()
            agg_exprs.append(uniq_mask.sum().alias(uniq_cnt_alias))
            agg_exprs.append(
                pl.col(name)
                .filter(uniq_mask)
                .unique()
                .head(5)
                .implode()
                .alias(uniq_samples_alias)
            )
            expr_metadata.append(
                {
                    "type": "unique",
                    "column": name,
                    "cnt_alias": uniq_cnt_alias,
                    "samples_alias": uniq_samples_alias,
                }
            )

    # Composite uniqueness validation (__unique_together__)
    if validate_unique and unique_together:
        for u_idx, comp_cols in enumerate(unique_together):
            comp_cols_tuple = tuple(comp_cols)
            # Only validate if all constituent columns are present in df
            if all(c in df_col_names for c in comp_cols_tuple):
                comp_cnt_alias = f"__val__comp_unique_{u_idx}__cnt"
                comp_samples_alias = f"__val__comp_unique_{u_idx}__samples"
                struct_col = pl.struct([pl.col(c) for c in comp_cols_tuple])
                not_null_mask = pl.all_horizontal(
                    [pl.col(c).is_not_null() for c in comp_cols_tuple]
                )
                comp_mask = not_null_mask & struct_col.is_duplicated()
                agg_exprs.append(comp_mask.sum().alias(comp_cnt_alias))
                agg_exprs.append(
                    struct_col.filter(comp_mask)
                    .unique()
                    .head(5)
                    .implode()
                    .alias(comp_samples_alias)
                )
                expr_metadata.append(
                    {
                        "type": "composite_unique",
                        "columns": comp_cols_tuple,
                        "cnt_alias": comp_cnt_alias,
                        "samples_alias": comp_samples_alias,
                    }
                )

    # Multi-column Check constraints
    if validate_checks and checks:
        for c_idx, check in enumerate(checks):
            chk_cnt_alias = f"__val__check_{c_idx}__cnt"
            fail_mask = check._failure_mask()
            agg_exprs.append(fail_mask.sum().alias(chk_cnt_alias))
            expr_metadata.append(
                {
                    "type": "check",
                    "check": check,
                    "check_idx": c_idx,
                    "cnt_alias": chk_cnt_alias,
                }
            )

    # Execute data aggregations if any
    if agg_exprs:
        collect_kwargs = {"engine": "streaming"} if streaming else {}
        stats_df = lf.select(agg_exprs).collect(**collect_kwargs)
        stats_row = stats_df.to_dict(as_series=False)

        for meta in expr_metadata:
            col_name = meta.get("column", "")
            m_type = meta["type"]

            if m_type == "nullability":
                cnt = stats_row[meta["alias"]][0]
                if cnt and cnt > 0:
                    errors.append(
                        f"Column '{col_name}': non-nullable column contains {cnt} null value(s)"
                    )

            elif m_type == "choices":
                cnt = stats_row[meta["cnt_alias"]][0]
                if cnt and cnt > 0:
                    raw_samples = stats_row[meta["samples_alias"]][0]
                    samples = list(raw_samples) if raw_samples is not None else []
                    errors.append(
                        f"Column '{col_name}': found {cnt} invalid value(s) not in allowed choices/categories "
                        f"{meta['allowed']}. Invalid samples: {samples}"
                    )

            elif m_type == "bounds":
                cnt = stats_row[meta["cnt_alias"]][0]
                if cnt and cnt > 0:
                    raw_samples = stats_row[meta["samples_alias"]][0]
                    samples = list(raw_samples) if raw_samples is not None else []
                    min_f = stats_row[meta["min_alias"]][0]
                    max_f = stats_row[meta["max_alias"]][0]
                    errors.append(
                        f"Column '{col_name}': found {cnt} value(s) out of bounds [{meta['min']}, {meta['max']}] "
                        f"(min found: {min_f}, max found: {max_f}). Out of bounds samples: {samples}"
                    )

            elif m_type == "string_length":
                cnt = stats_row[meta["cnt_alias"]][0]
                if cnt and cnt > 0:
                    raw_samples = stats_row[meta["samples_alias"]][0]
                    samples = list(raw_samples) if raw_samples is not None else []
                    errors.append(
                        f"Column '{col_name}': found {cnt} value(s) with string length outside "
                        f"[{meta['min']}, {meta['max']}]. Invalid samples: {samples}"
                    )

            elif m_type == "rule":
                cnt = stats_row[meta["cnt_alias"]][0]
                if cnt and cnt > 0:
                    raw_samples = stats_row[meta["samples_alias"]][0]
                    samples = list(raw_samples) if raw_samples is not None else []
                    rule_obj = meta["rule"]
                    errors.append(
                        f"Column '{col_name}': found {cnt} value(s) violating ColRule(when={rule_obj.when}, "
                        f"choices={rule_obj.choices}). Violating samples: {samples}"
                    )

            elif m_type == "unique":
                cnt = stats_row[meta["cnt_alias"]][0]
                if cnt and cnt > 0:
                    raw_samples = stats_row[meta["samples_alias"]][0]
                    samples = list(raw_samples) if raw_samples is not None else []
                    errors.append(
                        f"Column '{col_name}': unique column contains {cnt} duplicate value(s). "
                        f"Duplicate samples: {samples}"
                    )

            elif m_type == "composite_unique":
                cnt = stats_row[meta["cnt_alias"]][0]
                if cnt and cnt > 0:
                    raw_samples = stats_row[meta["samples_alias"]][0]
                    samples = list(raw_samples) if raw_samples is not None else []
                    errors.append(
                        f"Composite unique key {list(meta['columns'])} violated: found {cnt} duplicate row(s). "
                        f"Duplicate samples: {samples}"
                    )

            elif m_type == "check":
                cnt = stats_row[meta["cnt_alias"]][0]
                if cnt and cnt > 0:
                    chk = meta["check"]
                    chk_desc = f" ({chk.description})" if chk.description else ""
                    errors.append(
                        f"Check '{chk.name}' failed: found {cnt} row(s) violating condition "
                        f"{chk.expr}{chk_desc}"
                    )

    # Foreign key (referential integrity) validation. Each one needs its own
    # join against a (possibly external) parent frame, so it can't share the
    # single scalar-aggregation pass above.
    if validate_foreign_keys and foreign_keys:
        collect_kwargs = {"engine": "streaming"} if streaming else {}
        for fk, target_lf in foreign_keys:
            local_cols = list(fk.columns)
            ref_cols = list(fk.ref_columns)
            if not all(c in df_col_names for c in local_cols):
                continue  # missing columns already reported via missing_cols handling

            parent_lf = target_lf if target_lf is not None else lf
            parent_schema_names = parent_lf.collect_schema().names()
            missing_ref = [c for c in ref_cols if c not in parent_schema_names]
            if missing_ref:
                raise ValueError(
                    f"ForeignKey '{fk.name}' on {schema_name!r} references columns "
                    f"{missing_ref} not present in the referenced DataFrame"
                )

            key_expr = (
                pl.col(local_cols[0]) if len(local_cols) == 1 else pl.struct(local_cols)
            )
            not_null_expr = (
                pl.col(local_cols[0]).is_not_null()
                if len(local_cols) == 1
                else pl.all_horizontal([pl.col(c).is_not_null() for c in local_cols])
            )

            parent_keys = parent_lf.select(ref_cols).unique()
            candidates = lf.select(local_cols).filter(not_null_expr)
            orphans_lf = candidates.join(
                parent_keys, left_on=local_cols, right_on=ref_cols, how="anti"
            )
            fk_stats = orphans_lf.select(
                pl.len().alias("cnt"),
                key_expr.unique().head(5).implode().alias("samples"),
            ).collect(**collect_kwargs)

            cnt = fk_stats["cnt"][0]
            if cnt and cnt > 0:
                raw_samples = fk_stats["samples"][0]
                samples = list(raw_samples) if raw_samples is not None else []
                target_label = (
                    "self"
                    if target_lf is None
                    else getattr(fk.references, "__name__", str(fk.references))
                )
                errors.append(
                    f"ForeignKey '{fk.name}' violated ({local_cols} -> "
                    f"{target_label}.{ref_cols}): found {cnt} row(s) with no "
                    f"matching parent record. Violating samples: {samples}"
                )

    if errors:
        msg_lines = [
            f"Validation failed for DataFrame against '{schema_name}' ({len(errors)} error(s) found):"
        ]
        for err in errors:
            msg_lines.append(f"  - {err}")
        raise ValidationError("\n".join(msg_lines), errors=errors)

    # Transformation if validation succeeded
    result_lf = lf

    if extra and extra_cols == "drop":
        keep_cols = [c for c in df_col_names if c not in extra]
        result_lf = result_lf.select(keep_cols)

    if missing and missing_cols == "add":
        add_exprs = [pl.lit(None, dtype=columns[c].dtype).alias(c) for c in missing]
        result_lf = result_lf.with_columns(add_exprs)

    if cast:
        cast_exprs = []
        current_schema = result_lf.collect_schema()
        for name, spec in columns.items():
            if name in current_schema and current_schema[name] != spec.dtype:
                cast_exprs.append(pl.col(name).cast(spec.dtype))
        if cast_exprs:
            result_lf = result_lf.with_columns(cast_exprs)

    # Reorder columns to match FrameSpec declaration order for present declared columns,
    # followed by any extra columns (if extra_cols == 'allow')
    spec_order = [c for c in columns if c in result_lf.collect_schema().names()]
    extra_order = [c for c in result_lf.collect_schema().names() if c not in columns]
    result_lf = result_lf.select(spec_order + extra_order)

    return result_lf if is_lazy else result_lf.collect()
