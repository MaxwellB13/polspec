from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

import polars as pl

from polspec.bound import Bound
from polspec.constants import _DEFAULT_NULL_PROBABILITY
from polspec.rules import ColRule


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
    """Specification for a single generated column."""

    dtype: pl.DataType
    nullable: bool = False
    bounds: Bound | tuple[float, float] | None = None
    category: str = ""
    null_probability: float = _DEFAULT_NULL_PROBABILITY
    string_length: Bound | tuple[int, int] | None = None
    distribution: str | None = None
    distribution_params: dict[str, float] | None = None
    choices: tuple | list | dict | None = None
    weights: tuple[float, ...] | list[float] | None = None
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

        if self.bounds is not None and not (
            self.dtype.is_integer() or self.dtype.is_float() or self.dtype.is_temporal()
        ):
            raise ValueError(
                f"ColSpec.bounds is only supported for numeric or temporal "
                f"dtypes, got {self.dtype!r}"
            )

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

        if self.choices is not None and not self.choices:
            raise ValueError("ColSpec.choices must not be empty")

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
