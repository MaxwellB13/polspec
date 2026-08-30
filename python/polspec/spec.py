from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import polars as pl

from polspec.bound import Bound
from polspec.check import Check
from polspec.constants import _DEFAULT_NULL_PROBABILITY
from polspec.dtypes import _bound_endpoint_to_physical, _dtype_value_limits
from polspec.rules import ColRule, _reject_colliding_choices


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
    if _is_categorical_dtype(dtype):
        return "categorical"
    raise TypeError(f"polspec cannot generate data for dtype {dtype!r}")


def _is_categorical_dtype(dtype: pl.DataType) -> bool:
    return (
        isinstance(dtype, pl.Categorical)
        or dtype == pl.Categorical
        or (isinstance(dtype, type) and issubclass(dtype, pl.Categorical))
    )


@dataclass(frozen=True, slots=True)
class ColSpec:
    """
    Represents the specification of a column in a dataset, including its data type,
    nullability, value constraints, and distribution properties.

    This class is designed to encapsulate all related metadata and validations
    required for defining the structure and properties of a dataset column.

    :ivar dtype: The data type of the column.
    :ivar nullable: Whether the column allows null values.
    :ivar bounds: The inclusive range of values allowed in the column. Only
        supported for numeric and temporal data types.
    :ivar tags: Optional tag or sequence of tags to classify the column.
    :ivar unique: Whether values in the column must be unique (distinct).
    :ivar null_probability: Probability of a value being null in the column. Must
        be between 0 and 1.
    :ivar string_length: The inclusive range of string lengths for the column, if
        applicable.
    :ivar distribution: The name of the probability distribution for the column's
        values (e.g., "uniform", "normal").
    :ivar distribution_params: Parameters specific to the specified probability
        distribution.
    :ivar choices: A predefined set of allowed values for the column.
    :ivar weights: Weights associated with the `choices`, used to bias
        selection probabilities.
    :ivar rules: A sequence of rules (`ColRule`) applied to constrain or validate
        the column's values.
    :ivar validators: A single-column business rule, or sequence of them, each
        either a `pl.Expr` boolean predicate (referencing only this column) or
        a `Check` (for a custom name/description/null handling). Unlike
        `FrameSpec.__checks__`, these travel with the column's own declaration.
    """

    dtype: pl.DataType
    nullable: bool = False
    bounds: Bound | tuple[Any, Any] | list[Any] | None = None
    tags: str | Sequence[str] = ()
    unique: bool = False
    null_probability: float = _DEFAULT_NULL_PROBABILITY
    string_length: Bound | tuple[int, int] | list[int] | None = None
    distribution: str | None = None
    distribution_params: dict[str, float] | None = None
    choices: tuple | list | dict | None = None
    weights: tuple[float, ...] | list[float] | None = None
    rules: tuple[ColRule, ...] = ()
    validators: Check | pl.Expr | Sequence[Check | pl.Expr] | None = ()

    def __post_init__(self) -> None:
        if isinstance(self.dtype, type) and issubclass(self.dtype, pl.DataType):
            with suppress(TypeError):
                object.__setattr__(self, "dtype", self.dtype())
        object.__setattr__(self, "bounds", Bound._coerce(self.bounds))
        object.__setattr__(self, "string_length", Bound._coerce(self.string_length))
        object.__setattr__(self, "rules", tuple(self.rules))
        self._normalize_validators()
        if self.tags is None:
            object.__setattr__(self, "tags", ())
        elif isinstance(self.tags, str):
            if not self.tags:
                object.__setattr__(self, "tags", ())
            else:
                object.__setattr__(self, "tags", (self.tags,))
        elif isinstance(self.tags, (list, tuple, set, Sequence)):
            seen = set()
            cleaned = []
            for t in self.tags:
                s = str(t)
                if s and s not in seen:
                    seen.add(s)
                    cleaned.append(s)
            object.__setattr__(self, "tags", tuple(cleaned))
        else:
            raise TypeError(
                f"ColSpec.tags must be a string or sequence of strings, got {type(self.tags).__name__}"
            )
        if not 0.0 <= self.null_probability <= 1.0:
            raise ValueError("null_probability must be between 0 and 1")

        if self.bounds is not None and not (
            self.dtype.is_integer() or self.dtype.is_float() or self.dtype.is_temporal()
        ):
            raise ValueError(
                f"ColSpec.bounds is only supported for numeric or temporal "
                f"dtypes, got {self.dtype!r}"
            )

        self._validate_bounds_fit_dtype()

        # Process choices & weights
        if isinstance(self.choices, dict):
            if self.weights is not None:
                raise ValueError(
                    "Cannot specify both a dict for choices and an explicit weights parameter"
                )
            object.__setattr__(
                self, "weights", tuple(float(w) for w in self.choices.values())
            )
            object.__setattr__(self, "choices", tuple(self.choices.keys()))
        elif self.choices is not None:
            object.__setattr__(self, "choices", tuple(self.choices))
            if self.weights is not None:
                object.__setattr__(
                    self, "weights", tuple(float(w) for w in self.weights)
                )
        elif self.weights is not None:
            object.__setattr__(self, "weights", tuple(float(w) for w in self.weights))

        if self.choices is not None:
            if not self.choices:
                raise ValueError("ColSpec.choices must not be empty")
            _reject_colliding_choices(self.choices, "ColSpec.choices")

        if self.weights is not None:
            if self.choices is not None:
                if len(self.weights) != len(self.choices):
                    raise ValueError(
                        f"Length of weights ({len(self.weights)}) must match length of choices ({len(self.choices)})"
                    )
            elif isinstance(self.dtype, pl.Enum):
                if len(self.weights) != len(self.dtype.categories):
                    raise ValueError(
                        f"Length of weights ({len(self.weights)}) must match number of Enum categories ({len(self.dtype.categories)})"
                    )
            elif self.dtype == pl.Boolean:
                if len(self.weights) != 2:
                    raise ValueError(
                        "Boolean weights must be a 2-element sequence [p_false, p_true]"
                    )
            else:
                raise ValueError(
                    "ColSpec.weights requires 'choices', an Enum dtype, or a Boolean "
                    f"dtype to define the domain weights apply to; got dtype={self.dtype!r} "
                    "with no choices"
                )
            if any(w < 0 for w in self.weights):
                raise ValueError("Weights must all be non-negative")
            if sum(self.weights) <= 0:
                raise ValueError("Sum of weights must be positive")

        if self.distribution is not None:
            if not (
                self.dtype.is_integer()
                or self.dtype.is_float()
                or self.dtype.is_temporal()
            ):
                raise ValueError(
                    f"ColSpec.distribution is only supported for numeric or temporal "
                    f"dtypes, got {self.dtype!r}"
                )
            dist = self.distribution.lower()
            valid_dists = {
                "uniform",
                "normal",
                "lognormal",
                "exponential",
                "exp",
                "poisson",
                "gamma",
                "beta",
            }
            if dist not in valid_dists:
                raise ValueError(
                    f"Unsupported distribution '{self.distribution}'. Supported: {sorted(valid_dists)}"
                )
            if self.distribution_params is not None:
                params = {str(k): float(v) for k, v in self.distribution_params.items()}
                object.__setattr__(self, "distribution_params", params)
                if dist == "normal":
                    std = params.get(
                        "std", params.get("sigma", params.get("scale", 1.0))
                    )
                    if std <= 0:
                        raise ValueError(
                            f"Normal distribution std must be positive, got {std}"
                        )
                elif dist == "lognormal":
                    std = params.get(
                        "std", params.get("sigma", params.get("sdlog", 1.0))
                    )
                    if std <= 0:
                        raise ValueError(
                            f"LogNormal distribution std must be positive, got {std}"
                        )
                elif dist in ("exponential", "exp"):
                    rate = params.get(
                        "rate", params.get("lambda", params.get("lambda_", 1.0))
                    )
                    scale = params.get("scale", 1.0)
                    if "scale" in params and scale <= 0:
                        raise ValueError(
                            f"Exponential distribution scale must be positive, got {scale}"
                        )
                    if "scale" not in params and rate <= 0:
                        raise ValueError(
                            f"Exponential distribution rate must be positive, got {rate}"
                        )
                elif dist == "poisson":
                    lam = params.get(
                        "lambda",
                        params.get(
                            "lambda_", params.get("rate", params.get("mean", 1.0))
                        ),
                    )
                    if lam <= 0:
                        raise ValueError(
                            f"Poisson distribution lambda must be positive, got {lam}"
                        )
                elif dist == "gamma":
                    shape = params.get(
                        "shape", params.get("alpha", params.get("k", 1.0))
                    )
                    scale = params.get(
                        "scale", params.get("beta", params.get("theta", 1.0))
                    )
                    if shape <= 0 or scale <= 0:
                        raise ValueError(
                            f"Gamma distribution shape and scale must be positive, got shape={shape}, scale={scale}"
                        )
                elif dist == "beta":
                    alpha = params.get(
                        "alpha", params.get("a", params.get("shape1", 1.0))
                    )
                    beta = params.get(
                        "beta", params.get("b", params.get("shape2", 1.0))
                    )
                    if alpha <= 0 or beta <= 0:
                        raise ValueError(
                            f"Beta distribution alpha and beta must be positive, got alpha={alpha}, beta={beta}"
                        )

        if self.dtype == pl.Boolean and self.distribution_params is not None:
            params = {str(k): float(v) for k, v in self.distribution_params.items()}
            object.__setattr__(self, "distribution_params", params)
            if "p" in params and not 0.0 <= params["p"] <= 1.0:
                raise ValueError(
                    f"Boolean distribution_params['p'] must be between 0 and 1, got {params['p']}"
                )

        if isinstance(self.dtype, pl.Enum):
            valid = set(self.dtype.categories.to_list())
            if self.choices is not None:
                unknown = [c for c in self.choices if c not in valid]
                if unknown:
                    raise ValueError(
                        f"ColSpec.choices {unknown} are not among this column's Enum categories {sorted(valid)}"
                    )
            for rule in self.rules:
                unknown = [c for c in rule.choices if c not in valid]
                if unknown:
                    raise ValueError(
                        f"ColRule.choices {unknown} are not among this column's "
                        f"Enum categories {sorted(valid)}"
                    )

        if self.bounds is not None:
            b_min, b_max = self.bounds.min, self.bounds.max
            if self.choices is not None:
                out_of_bounds = [c for c in self.choices if not (b_min <= c <= b_max)]
                if out_of_bounds:
                    raise ValueError(
                        f"ColSpec.choices {out_of_bounds} fall outside this column's "
                        f"bounds [{b_min}, {b_max}]"
                    )
            for rule in self.rules:
                out_of_bounds = [c for c in rule.choices if not (b_min <= c <= b_max)]
                if out_of_bounds:
                    raise ValueError(
                        f"ColRule.choices {out_of_bounds} fall outside this column's "
                        f"bounds [{b_min}, {b_max}]"
                    )

    def _validate_bounds_fit_dtype(self) -> None:
        """Rejects bounds the dtype cannot represent.

        Generation clamps to these endpoints, so an endpoint outside the
        dtype's domain has no valid interpretation. Caught here rather than at
        generation time because the Rust engine reaches them as a saturated
        cast -- an out-of-range float becomes an infinity, and building a
        distribution over a non-finite range aborts the process.
        """
        if self.bounds is None:
            return
        limits = _dtype_value_limits(self.dtype)
        if limits is None:
            return
        lo_limit, hi_limit = limits
        for label, endpoint in (("min", self.bounds.min), ("max", self.bounds.max)):
            physical = _bound_endpoint_to_physical(endpoint, self.dtype)
            if not math.isfinite(physical):
                raise ValueError(
                    f"ColSpec.bounds {label} must be a finite value, got {endpoint!r}"
                )
            if not lo_limit <= physical <= hi_limit:
                raise ValueError(
                    f"ColSpec.bounds {label} ({endpoint!r}) is outside the range "
                    f"{self.dtype!r} can represent [{lo_limit}, {hi_limit}]"
                )

    def _normalize_validators(self) -> None:
        if self.validators is None:
            object.__setattr__(self, "validators", ())
            return

        raw = (
            [self.validators]
            if isinstance(self.validators, (pl.Expr, Check))
            else list(self.validators)
        )

        normalized: list[Check] = []
        seen: dict[str, Check] = {}
        for v in raw:
            if isinstance(v, Check):
                chk = v
            elif isinstance(v, pl.Expr):
                chk = Check(v)
            else:
                raise TypeError(
                    "ColSpec.validators items must be a polars Expr or Check, "
                    f"got {type(v).__name__}"
                )
            prior = seen.get(chk.name)
            if prior is not None and prior != chk:
                raise ValueError(
                    f"Duplicate validator name {chk.name!r} on this ColSpec: "
                    f"{prior.expr!r} vs {chk.expr!r}. Give each validator a "
                    "distinct name."
                )
            seen[chk.name] = chk
            normalized.append(chk)

        object.__setattr__(self, "validators", tuple(normalized))
