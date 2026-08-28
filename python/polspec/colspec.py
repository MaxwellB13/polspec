from __future__ import annotations

import random
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import polars as pl
import yaml

from polspec import _polspec

# Bounds baked into the fixed-width integer dtypes themselves. Used as the
# default generation range when a ColSpec doesn't supply its own bounds, so
# generated values always survive the cast back to the requested dtype.
_INT_DTYPE_BOUNDS: dict[pl.DataType, tuple[int, int]] = {
    pl.Int8: (-128, 127),
    pl.Int16: (-32_768, 32_767),
    pl.Int32: (-2_147_483_648, 2_147_483_647),
    pl.UInt8: (0, 255),
    pl.UInt16: (0, 65_535),
    pl.UInt32: (0, 4_294_967_295),
}
# Int64/UInt64 have no default here: their true range loses precision once
# round-tripped through f64, so ColSpec.bounds must be set explicitly for
# ranges beyond the default below.
_DEFAULT_WIDE_INT_BOUND = 1_000_000
_DEFAULT_FLOAT_BOUND = 1_000_000.0
_DEFAULT_STRING_LEN = (5, 15)
_DEFAULT_NULL_PROBABILITY = 0.1
# Safety cap on method="cartesian": the cross-joined coverage set grows as
# the product of every dimension's cardinality, so a handful of wide enums
# can explode into an unreasonable row count by accident.
_MAX_CARTESIAN_ROWS = 50_000_000


@dataclass(frozen=True, slots=True)
class Bound:
    """An inclusive [min, max] range, used for numeric bounds and string lengths."""

    min: float | int
    max: float | int

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise ValueError(f"Bound min ({self.min}) must be <= max ({self.max})")

    @classmethod
    def _coerce(
        cls, value: Bound | tuple[float | int, float | int] | None
    ) -> Bound | None:
        if value is None or isinstance(value, Bound):
            return value
        lo, hi = value
        return cls(lo, hi)


# Condition operations ColRule.when accepts.
_CONDITION_OPS = (
    "equals",
    "not_equals",
    "in",
    "not_in",
    "lt",
    "lte",
    "le",
    "gt",
    "gte",
    "ge",
    "between",
    "is_null",
    "is_not_null",
)


def _validate_condition(condition: dict) -> None:
    if not isinstance(condition, dict) or "column" not in condition:
        raise TypeError(
            "ColRule.when must be a dict like {'column': 'enum_1', 'in': ['A', 'B']} "
            f"(supported condition keys: {', '.join(_CONDITION_OPS)})"
        )
    ops_present = [op for op in _CONDITION_OPS if op in condition]
    if len(ops_present) != 1:
        raise ValueError(
            f"ColRule.when for column {condition['column']!r} must have exactly one of "
            f"{_CONDITION_OPS}, got {ops_present}"
        )
    if "between" in condition:
        b = condition["between"]
        if not (isinstance(b, (list, tuple)) and len(b) == 2 and b[0] <= b[1]):
            raise ValueError(
                f"ColRule.when 'between' condition requires a 2-element sequence [min, max] where min <= max, got {b!r}"
            )
    if "in" in condition and not isinstance(condition["in"], (list, tuple, set)):
        raise TypeError(
            f"ColRule.when 'in' condition requires a collection, got {type(condition['in']).__name__}"
        )
    if "not_in" in condition and not isinstance(
        condition["not_in"], (list, tuple, set)
    ):
        raise TypeError(
            f"ColRule.when 'not_in' condition requires a collection, got {type(condition['not_in']).__name__}"
        )


def _condition_to_expr(condition: dict) -> pl.Expr:
    column = pl.col(condition["column"])
    if "equals" in condition:
        return column == condition["equals"]
    if "not_equals" in condition:
        return column != condition["not_equals"]
    if "in" in condition:
        return column.is_in(list(condition["in"]))
    if "not_in" in condition:
        return ~column.is_in(list(condition["not_in"]))
    if "lt" in condition:
        return column < condition["lt"]
    if "lte" in condition or "le" in condition:
        val = condition.get("lte", condition.get("le"))
        return column <= val
    if "gt" in condition:
        return column > condition["gt"]
    if "gte" in condition or "ge" in condition:
        val = condition.get("gte", condition.get("ge"))
        return column >= val
    if "between" in condition:
        lo, hi = condition["between"]
        return column.is_between(lo, hi)
    if "is_null" in condition:
        return column.is_null() if condition["is_null"] else column.is_not_null()
    if "is_not_null" in condition:
        return column.is_not_null() if condition["is_not_null"] else column.is_null()
    raise ValueError(f"Unrecognized condition: {condition}")


@dataclass(frozen=True, slots=True)
class ColRule:
    """Restricts a column's generated values on rows where `when` matches.

    Applied as a final vectorized pass after normal generation: rows where
    `when` matches get a value resampled uniformly from `choices` instead of
    whatever was freely generated for them. `when` is evaluated against the
    fully-generated, freely-sampled DataFrame -- never against another
    rule's output -- so rules on different columns are independent of
    declaration order. Multiple rules on the *same* column are checked in
    declaration order, first match wins (like SQL CASE/WHEN).

    `when` is a small dict, not an arbitrary polars expression, so that
    every rule can round-trip through DfSpec.to_yaml/from_yaml:
        {"column": "enum_1", "equals": "A"}
        {"column": "enum_1", "not_equals": "A"}
        {"column": "enum_1", "in": ["A", "B"]}
        {"column": "enum_1", "not_in": ["A", "B"]}

    Example: ColRule(when={"column": "enum_1", "in": ["A", "B"]}, choices=["X", "Y"])
    """

    when: dict
    choices: tuple

    def __post_init__(self) -> None:
        _validate_condition(self.when)
        object.__setattr__(self, "when", dict(self.when))
        object.__setattr__(self, "choices", tuple(self.choices))
        if not self.choices:
            raise ValueError("ColRule.choices must not be empty")

    def _expr(self) -> pl.Expr:
        return _condition_to_expr(self.when)


@dataclass(frozen=True, slots=True)
class ColSpec:
    """Specification for a single generated column."""

    dtype: pl.DataType
    nullable: bool = False
    bounds: Bound | tuple[float, float] | None = None
    category: str = ""
    null_probability: float = _DEFAULT_NULL_PROBABILITY
    string_length: Bound | tuple[int, int] | None = None
    rules: tuple[ColRule, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.dtype, type) and issubclass(self.dtype, pl.DataType):
            with suppress(TypeError):
                object.__setattr__(self, "dtype", self.dtype())
        object.__setattr__(self, "bounds", Bound._coerce(self.bounds))
        object.__setattr__(self, "string_length", Bound._coerce(self.string_length))
        object.__setattr__(self, "rules", tuple(self.rules))
        if not 0.0 <= self.null_probability <= 1.0:
            raise ValueError("null_probability must be between 0 and 1")
        if isinstance(self.dtype, pl.Enum):
            valid = set(self.dtype.categories.to_list())
            for rule in self.rules:
                unknown = [c for c in rule.choices if c not in valid]
                if unknown:
                    raise ValueError(
                        f"ColRule.choices {unknown} are not among this column's "
                        f"Enum categories {sorted(valid)}"
                    )


def _column_kind(dtype: pl.DataType) -> str:
    if dtype.is_integer():
        return "int"
    if dtype.is_float():
        return "float"
    if dtype.is_temporal():
        return "temporal"
    if dtype == pl.Boolean:
        return "bool"
    if dtype in (pl.String, pl.Utf8):
        return "string"
    if dtype == pl.Binary:
        return "binary"
    if isinstance(dtype, pl.Enum):
        return "enum"
    if (
        isinstance(dtype, pl.Categorical)
        or dtype == pl.Categorical
        or (isinstance(dtype, type) and issubclass(dtype, pl.Categorical))
    ):
        return "categorical"
    raise TypeError(f"polspec cannot generate data for dtype {dtype!r}")


# Every dtype polspec can generate that has a fixed, unparametrized identity
# -- Enum, Datetime, Duration, and (parametrized) Categorical are handled separately
# since they carry their own metadata.
_YAML_DTYPES: dict[pl.DataType, str] = {
    pl.String: "String",
    pl.Boolean: "Boolean",
    pl.Int8: "Int8",
    pl.Int16: "Int16",
    pl.Int32: "Int32",
    pl.Int64: "Int64",
    pl.UInt8: "UInt8",
    pl.UInt16: "UInt16",
    pl.UInt32: "UInt32",
    pl.UInt64: "UInt64",
    pl.Float32: "Float32",
    pl.Float64: "Float64",
    pl.Date: "Date",
    pl.Time: "Time",
    pl.Datetime: "Datetime",
    pl.Duration: "Duration",
    pl.Binary: "Binary",
}
_YAML_NAME_TO_DTYPE = {name: dtype for dtype, name in _YAML_DTYPES.items()}


def _dtype_to_yaml(dtype: pl.DataType) -> str | dict:
    if isinstance(dtype, pl.Enum):
        return {"Enum": dtype.categories.to_list()}
    if isinstance(dtype, pl.Datetime):
        res: dict = {"time_unit": dtype.time_unit}
        if dtype.time_zone is not None:
            res["time_zone"] = dtype.time_zone
        return {"Datetime": res}
    if isinstance(dtype, pl.Duration):
        return {"Duration": {"time_unit": dtype.time_unit}}
    if (
        isinstance(dtype, pl.Categorical)
        or dtype == pl.Categorical
        or (isinstance(dtype, type) and issubclass(dtype, pl.Categorical))
    ):
        return "Categorical"
    name = _YAML_DTYPES.get(dtype)
    if name is None:
        raise TypeError(f"polspec cannot write dtype {dtype!r} to YAML")
    return name


def _dtype_from_yaml(value: str | dict) -> pl.DataType:
    if isinstance(value, dict):
        if "Enum" in value:
            return pl.Enum(value["Enum"])
        if "Datetime" in value:
            dt_info = value["Datetime"]
            return pl.Datetime(
                time_unit=dt_info.get("time_unit", "us"),
                time_zone=dt_info.get("time_zone"),
            )
        if "Duration" in value:
            dur_info = value["Duration"]
            return pl.Duration(time_unit=dur_info.get("time_unit", "us"))
        raise ValueError(f"Unrecognized dtype mapping in YAML: {value!r}")
    if value == "Categorical":
        return pl.Categorical()
    dtype = _YAML_NAME_TO_DTYPE.get(value)
    if dtype is None:
        raise ValueError(f"Unrecognized dtype name in YAML: {value!r}")
    return dtype


def _colspec_to_yaml(spec: ColSpec) -> dict:
    data: dict = {"dtype": _dtype_to_yaml(spec.dtype)}
    if spec.nullable:
        data["nullable"] = True
    if spec.bounds is not None:
        data["bounds"] = [spec.bounds.min, spec.bounds.max]
    if spec.category:
        data["category"] = spec.category
    if spec.null_probability != _DEFAULT_NULL_PROBABILITY:
        data["null_probability"] = spec.null_probability
    if spec.string_length is not None:
        data["string_length"] = [spec.string_length.min, spec.string_length.max]
    if spec.rules:
        data["rules"] = [
            {"when": dict(rule.when), "choices": list(rule.choices)}
            for rule in spec.rules
        ]
    return data


def _colspec_from_yaml(data: dict) -> ColSpec:
    kwargs: dict = {}
    if "nullable" in data:
        kwargs["nullable"] = data["nullable"]
    if "bounds" in data:
        kwargs["bounds"] = tuple(data["bounds"])
    if "category" in data:
        kwargs["category"] = data["category"]
    if "null_probability" in data:
        kwargs["null_probability"] = data["null_probability"]
    if "string_length" in data:
        kwargs["string_length"] = tuple(data["string_length"])
    if "rules" in data:
        kwargs["rules"] = tuple(
            ColRule(when=rule["when"], choices=rule["choices"])
            for rule in data["rules"]
        )
    return ColSpec(dtype=_dtype_from_yaml(data["dtype"]), **kwargs)


def _resolve_numeric_bounds(spec: ColSpec) -> tuple[float | int, float | int]:
    """Returns the (min, max) an int/float/temporal ColSpec generates within.

    Mirrors the defaulting rule used at generation time: explicit
    `spec.bounds` wins, otherwise a fixed-width int dtype defaults to its own
    range, and anything else falls back to the wide/default bound constants.
    """
    kind = _column_kind(spec.dtype)
    if spec.bounds is not None:
        return spec.bounds.min, spec.bounds.max
    if kind == "int":
        if spec.dtype in _INT_DTYPE_BOUNDS:
            lo, hi = _INT_DTYPE_BOUNDS[spec.dtype]
            return lo, hi
        return -_DEFAULT_WIDE_INT_BOUND, _DEFAULT_WIDE_INT_BOUND
    if kind == "float":
        return -_DEFAULT_FLOAT_BOUND, _DEFAULT_FLOAT_BOUND
    if kind == "temporal":
        if spec.dtype == pl.Date:
            return 0, 36525
        if spec.dtype == pl.Time:
            return 0, 86_399_999_999_999
        if isinstance(spec.dtype, pl.Datetime) or spec.dtype == pl.Datetime:
            return 0, 36525 * 86400 * 1_000_000
        if isinstance(spec.dtype, pl.Duration) or spec.dtype == pl.Duration:
            return 0, 365 * 86400 * 1_000_000
        return 0, _DEFAULT_WIDE_INT_BOUND
    raise TypeError(f"{spec.dtype!r} is not a numeric or temporal dtype")


def _to_rust_spec(name: str, spec: ColSpec) -> tuple:
    """Builds the tuple the Rust extension expects for one column.

    Layout: (name, kind, nullable, null_probability, min, max, categories,
    str_min_len, str_max_len).
    """
    kind = _column_kind(spec.dtype)
    null_probability = spec.null_probability if spec.nullable else 0.0

    min_bound: float | None = None
    max_bound: float | None = None
    categories: list[str] | None = None
    str_min_len: int | None = None
    str_max_len: int | None = None

    if spec.dtype == pl.Int8:
        kind = "int8"
    elif spec.dtype == pl.Int16:
        kind = "int16"
    elif spec.dtype == pl.Int32:
        kind = "int32"
    elif spec.dtype == pl.Int64:
        kind = "int64"
    elif spec.dtype == pl.UInt8:
        kind = "uint8"
    elif spec.dtype == pl.UInt16:
        kind = "uint16"
    elif spec.dtype == pl.UInt32:
        kind = "uint32"
    elif spec.dtype == pl.UInt64:
        kind = "uint64"
    elif spec.dtype == pl.Float32:
        kind = "float32"
    elif spec.dtype == pl.Float64:
        kind = "float64"
    elif spec.dtype == pl.Date:
        kind = "int32"
    elif spec.dtype == pl.Time or isinstance(spec.dtype, pl.Datetime) or spec.dtype == pl.Datetime or isinstance(spec.dtype, pl.Duration) or spec.dtype == pl.Duration:
        kind = "int64"

    if spec.dtype.is_integer() or spec.dtype.is_float() or spec.dtype.is_temporal():
        lo, hi = _resolve_numeric_bounds(spec)
        min_bound, max_bound = float(lo), float(hi)
    elif kind in ("string", "binary"):
        length = spec.string_length or Bound(*_DEFAULT_STRING_LEN)
        str_min_len, str_max_len = int(length.min), int(length.max)
        kind = "string"
    elif kind in ("enum", "categorical"):
        if kind == "enum":
            categories = spec.dtype.categories.to_list()
        kind = "string"

    return (
        name,
        kind,
        spec.nullable,
        null_probability,
        min_bound,
        max_bound,
        categories,
        str_min_len,
        str_max_len,
    )


def _cast_expr(name: str, spec: ColSpec) -> pl.Expr:
    kind = _column_kind(spec.dtype)
    if (
        isinstance(spec.dtype, pl.Categorical)
        or spec.dtype == pl.Categorical
        or (isinstance(spec.dtype, type) and issubclass(spec.dtype, pl.Categorical))
    ):
        return pl.col(name).cast(pl.Categorical)
    if kind == "string" and spec.dtype in (pl.String, pl.Utf8):
        return pl.col(name)
    if spec.dtype in (
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
    ):
        return pl.col(name)
    return pl.col(name).cast(spec.dtype)


def _coverage_values(spec: ColSpec, rng: random.Random) -> list | None:
    """The finite set of representative values a coverage dimension takes.

    Enum/Boolean contribute their whole domain; numeric columns contribute
    one representative each from the negative/zero/positive partitions their
    bounds actually reach (e.g. bounds of 5..100 only reaches "positive").
    Nullable columns also get a `None` entry. Returns None for columns with
    no natural finite domain (String, bare Categorical) -- those are filled
    in with ordinary random generation instead.
    """
    kind = _column_kind(spec.dtype)
    values: list = []

    if kind == "enum":
        values = list(spec.dtype.categories.to_list())
    elif kind == "bool":
        values = [True, False]
    elif kind in ("int", "temporal"):
        lo, hi = (int(v) for v in _resolve_numeric_bounds(spec))
        if lo < 0:
            values.append(rng.randint(lo, min(hi, -1)))
        if lo <= 0 <= hi:
            values.append(0)
        if hi > 0:
            values.append(rng.randint(max(lo, 1), hi))
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
    columns: dict[str, ColSpec], n: int, seed: int | None
) -> pl.DataFrame:
    if not columns:
        return pl.DataFrame()
    rust_specs = [_to_rust_spec(name, spec) for name, spec in columns.items()]
    raw_df = _polspec.generate_dataframe(rust_specs, n, seed)
    cast_exprs = [_cast_expr(name, spec) for name, spec in columns.items()]
    return raw_df.select(cast_exprs)


def _sample_choices(choices: tuple, n: int, seed: int) -> pl.Series:
    """n values drawn uniformly (with replacement) from `choices`.

    Generates a random index column through the fast Rust generator, then
    gathers `choices` by those indices.
    """
    idx_spec = (
        "__idx",
        "uint32",
        False,
        0.0,
        0.0,
        float(len(choices) - 1),
        None,
        None,
        None,
    )
    idx_df = _polspec.generate_dataframe([idx_spec], n, seed)
    return pl.Series(list(choices)).gather(idx_df["__idx"])


def _apply_rules(
    df: pl.DataFrame, columns: dict[str, ColSpec], seed: int | None
) -> pl.DataFrame:
    """Overwrites values on rows matched by each column's ColRules.

    A single vectorized pass over the already-generated DataFrame: for a
    column with rules, rows matching a rule's `when` get a value resampled
    from that rule's `choices` (first matching rule wins); everything else
    keeps its freely-generated value. `when` expressions always see the
    original freely-generated values, never another rule's output, so rules
    on different columns never need dependency ordering.
    """
    if df.height == 0:
        return df
    rng = random.Random(seed)
    exprs = []
    for name, spec in columns.items():
        if not spec.rules:
            continue
        chain = None
        for rule in spec.rules:
            condition = rule._expr()
            if len(rule.choices) == 1:
                fill_expr = pl.lit(rule.choices[0], dtype=spec.dtype)
            else:
                fill_series = _sample_choices(
                    rule.choices, df.height, rng.randrange(2**63)
                ).cast(spec.dtype)
                fill_expr = pl.lit(fill_series)
            if chain is None:
                chain = pl.when(condition).then(fill_expr)
            else:
                chain = chain.when(condition).then(fill_expr)
        if chain is not None:
            exprs.append(chain.otherwise(pl.col(name)).alias(name))
    return df.with_columns(exprs) if exprs else df


def _generate_cartesian(
    columns: dict[str, ColSpec], n: int, seed: int | None
) -> pl.DataFrame:
    """Guarantees coverage: the cartesian product of every Enum/Boolean's
    categories crossed with the negative/zero/positive/null partitions of
    every bounded numeric column, padded with ordinary random rows up to
    `n` if the coverage set doesn't already reach it.
    """
    rng = random.Random(seed)

    coverage_values: dict[str, list] = {}
    filler_columns: dict[str, ColSpec] = {}
    for name, spec in columns.items():
        values = _coverage_values(spec, rng)
        if values is None:
            filler_columns[name] = spec
        else:
            coverage_values[name] = values

    if not coverage_values:
        raise ValueError(
            "method='cartesian' needs at least one Enum, Boolean, or bounded "
            "numeric column to build coverage from"
        )

    coverage_size = 1
    for values in coverage_values.values():
        coverage_size *= len(values)
    if coverage_size > _MAX_CARTESIAN_ROWS:
        breakdown = ", ".join(f"{name}={len(v)}" for name, v in coverage_values.items())
        raise ValueError(
            f"cartesian coverage would need {coverage_size:,} rows ({breakdown}), "
            f"which exceeds the {_MAX_CARTESIAN_ROWS:,}-row safety cap"
        )

    coverage_df: pl.DataFrame | None = None
    for name, values in coverage_values.items():
        dim_df = pl.DataFrame({name: values}, schema={name: columns[name].dtype})
        coverage_df = (
            dim_df if coverage_df is None else coverage_df.join(dim_df, how="cross")
        )

    coverage_n = coverage_df.height

    if filler_columns:
        filler_df = _generate_random(filler_columns, coverage_n, rng.randrange(2**63))
        coverage_df = pl.concat([coverage_df, filler_df], how="horizontal_extend")

    coverage_df = coverage_df.select(list(columns.keys()))

    if coverage_n >= n:
        return coverage_df

    topup_df = _generate_random(columns, n - coverage_n, rng.randrange(2**63))
    return pl.concat([coverage_df, topup_df], how="vertical")


class DfSpec:
    """Base class for declaring a DataFrame specification.

    Subclass it and assign `ColSpec` instances as class attributes, in the
    order columns should appear:

        class DataSource(DfSpec):
            string_1 = ColSpec(dtype=pl.String, nullable=False)
            enum_1 = ColSpec(dtype=pl.Enum(["mammal", "reptile", "insect"]), nullable=True)
            int_1 = ColSpec(dtype=pl.Int64, bounds=Bound(-100, 100), nullable=True)

        df = DataSource.generate(1_000_000, seed=42)
        df = DataSource.generate(n=1_000_000, method="cartesian", seed=42)
    """

    _columns: ClassVar[dict[str, ColSpec]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        columns: dict[str, ColSpec] = {}
        for base in reversed(cls.__mro__):
            for name, value in vars(base).items():
                if name.startswith("_"):
                    continue
                if isinstance(value, ColSpec):
                    columns[name] = value
                elif name in columns:
                    del columns[name]
        cls._columns = columns
        cls._validate_rules()

    @classmethod
    def _validate_rules(cls) -> None:
        for col_name, spec in cls._columns.items():
            for rule in spec.rules:
                ref_col = rule.when.get("column")
                if ref_col not in cls._columns:
                    raise ValueError(
                        f"ColRule on column {col_name!r} references unknown column {ref_col!r}"
                    )

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema({name: spec.dtype for name, spec in cls._columns.items()})

    @classmethod
    def to_yaml(cls, source: str | Path) -> None:
        """Writes this spec's columns to a human-readable YAML file at `source`."""
        if not cls._columns:
            raise ValueError(f"{cls.__name__} declares no ColSpec columns")
        data = {
            "name": cls.__name__,
            "columns": {
                name: _colspec_to_yaml(spec) for name, spec in cls._columns.items()
            },
        }
        Path(source).write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def from_yaml(cls, source: str | Path) -> type[DfSpec]:
        """Builds a new DfSpec subclass from a YAML file written by `to_yaml`.

        DataSource = DfSpec.from_yaml(source="spec.yaml")
        df = DataSource.generate(1_000, seed=42)
        """
        data = yaml.safe_load(Path(source).read_text())
        columns_data = data.get("columns") or {}
        if not columns_data:
            raise ValueError(f"{source} declares no columns")
        columns = {
            name: _colspec_from_yaml(col_data)
            for name, col_data in columns_data.items()
        }
        return type(data.get("name", "LoadedDfSpec"), (DfSpec,), columns)

    @classmethod
    def generate(
        cls,
        n: int,
        *,
        method: Literal["random", "cartesian"] = "random",
        seed: int | None = None,
    ) -> pl.DataFrame:
        """Generates a DataFrame matching this spec.

        method="random" (default): `n` rows, each column drawn independently.

        method="cartesian": guarantees a minimum level of coverage. Builds
        the cartesian product of every Enum/Boolean column's full set of
        values, crossed with the negative/zero/positive/null partitions of
        every bounded numeric column -- so every enum combination appears
        alongside every numeric sign/null case. `n` is then a *minimum*: if
        that coverage set has fewer than `n` rows, it's padded with ordinary
        random rows up to `n`; if it already has more, all of it is kept.

        Any ColSpec.rules are applied last, as a vectorized overwrite pass
        over the fully-generated DataFrame (see ColRule), regardless of
        method.
        """
        if not cls._columns:
            raise ValueError(f"{cls.__name__} declares no ColSpec columns")
        if n < 0:
            raise ValueError("n must be >= 0")

        rng = random.Random(seed)
        gen_seed = rng.randrange(2**63)
        if method == "random":
            df = _generate_random(cls._columns, n, gen_seed)
        elif method == "cartesian":
            df = _generate_cartesian(cls._columns, n, gen_seed)
        else:
            raise ValueError(
                f"Unknown method {method!r}; expected 'random' or 'cartesian'"
            )

        return _apply_rules(df, cls._columns, rng.randrange(2**63))
