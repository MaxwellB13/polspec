from __future__ import annotations

import polars as pl

from polspec.bound import Bound
from polspec.constants import _DEFAULT_NULL_PROBABILITY
from polspec.spec import ColSpec, _is_categorical_dtype


def profile_dataframe(
    df: pl.DataFrame,
    *,
    weights: bool = False,
    max_unique_enum: int = 50,
    max_unique: int | None = None,
    calculate_bounds: bool = True,
    bounds: bool | None = None,
) -> dict[str, ColSpec]:
    """Infers ColSpec column definitions by profiling an existing DataFrame."""
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame, got {type(df).__name__}")

    if max_unique is not None:
        max_unique_enum = max_unique
    if bounds is not None:
        calculate_bounds = bounds

    total_rows = df.height
    columns: dict[str, ColSpec] = {}

    for col_name in df.columns:
        s = df[col_name]
        dtype = s.dtype
        null_count = s.null_count()
        nullable = null_count > 0
        null_probability = (
            float(null_count / total_rows)
            if total_rows > 0 and null_count > 0
            else (_DEFAULT_NULL_PROBABILITY if nullable else 0.0)
        )
        non_null = s.drop_nulls()
        num_valid = len(non_null)

        # Profiling categorical / enum / string types
        if isinstance(dtype, pl.Enum):
            categories = dtype.categories.to_list()
            col_weights: tuple[float, ...] | None = None
            if weights and num_valid > 0:
                vc = non_null.value_counts()
                counts_map = dict(
                    zip(vc[col_name].to_list(), vc[vc.columns[1]].to_list())
                )
                tot = float(num_valid)
                col_weights = tuple(counts_map.get(cat, 0) / tot for cat in categories)
            columns[col_name] = ColSpec(
                dtype=dtype,
                nullable=nullable,
                null_probability=null_probability,
                weights=col_weights,
            )
        elif dtype in (pl.String, pl.Utf8) or _is_categorical_dtype(dtype):
            n_unique = non_null.n_unique()
            if 0 < n_unique <= max_unique_enum:
                cats = non_null.unique().sort().to_list()
                enum_dtype = pl.Enum(cats)
                col_weights = None
                if weights and num_valid > 0:
                    vc = non_null.value_counts()
                    counts_map = dict(
                        zip(vc[col_name].to_list(), vc[vc.columns[1]].to_list())
                    )
                    tot = float(num_valid)
                    col_weights = tuple(counts_map.get(cat, 0) / tot for cat in cats)
                columns[col_name] = ColSpec(
                    dtype=enum_dtype,
                    nullable=nullable,
                    null_probability=null_probability,
                    weights=col_weights,
                )
            else:
                if _is_categorical_dtype(dtype):
                    columns[col_name] = ColSpec(
                        dtype=pl.Categorical(),
                        nullable=nullable,
                        null_probability=null_probability,
                    )
                else:
                    str_len: Bound | None = None
                    if calculate_bounds and num_valid > 0:
                        lens = non_null.str.len_bytes()
                        str_len = Bound(int(lens.min()), int(lens.max()))
                    columns[col_name] = ColSpec(
                        dtype=pl.String,
                        nullable=nullable,
                        null_probability=null_probability,
                        string_length=str_len,
                    )
        elif dtype == pl.Boolean:
            col_weights = None
            if weights and num_valid > 0:
                n_false = int((non_null == False).sum())
                n_true = int((non_null == True).sum())
                tot_b = n_false + n_true
                col_weights = (
                    (n_false / tot_b, n_true / tot_b) if tot_b > 0 else (0.5, 0.5)
                )
            columns[col_name] = ColSpec(
                dtype=pl.Boolean,
                nullable=nullable,
                null_probability=null_probability,
                weights=col_weights,
            )
        elif dtype.is_integer():
            num_bounds: Bound | None = None
            if calculate_bounds and num_valid > 0:
                num_bounds = Bound(int(non_null.min()), int(non_null.max()))
            columns[col_name] = ColSpec(
                dtype=dtype,
                nullable=nullable,
                null_probability=null_probability,
                bounds=num_bounds,
            )
        elif dtype.is_float():
            num_bounds = None
            if calculate_bounds and num_valid > 0:
                num_bounds = Bound(float(non_null.min()), float(non_null.max()))
            columns[col_name] = ColSpec(
                dtype=dtype,
                nullable=nullable,
                null_probability=null_probability,
                bounds=num_bounds,
            )
        elif dtype.is_temporal():
            temp_bounds: Bound | None = None
            if calculate_bounds and num_valid > 0:
                phys = non_null.to_physical()
                temp_bounds = Bound(int(phys.min()), int(phys.max()))
            columns[col_name] = ColSpec(
                dtype=dtype,
                nullable=nullable,
                null_probability=null_probability,
                bounds=temp_bounds,
            )
        elif dtype == pl.Binary:
            bin_len: Bound | None = None
            if calculate_bounds and num_valid > 0:
                lens = non_null.bin.size()
                bin_len = Bound(int(lens.min()), int(lens.max()))
            columns[col_name] = ColSpec(
                dtype=pl.Binary,
                nullable=nullable,
                null_probability=null_probability,
                string_length=bin_len,
            )
        else:
            columns[col_name] = ColSpec(
                dtype=dtype,
                nullable=nullable,
                null_probability=null_probability,
            )

    return columns
