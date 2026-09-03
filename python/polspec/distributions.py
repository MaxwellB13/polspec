"""The distributions polspec can sample, and the parameter names each accepts.

The Rust engine resolves the same aliases in `DistKind::from_spec`
(`src/lib.rs`), so these two tables must agree. Keeping the Python side as a
table rather than a hundred lines of nested branching is what makes the pair
comparable at a glance -- the previous form hid, for example, that `normal`
accepts `scale` as a synonym for `std` while `lognormal` accepts `sdlog`
instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from polspec.errors import SpecError

# Distribution name (as written by the caller, lowercased) -> display name used
# in error messages. Several names are aliases for one distribution.
DISTRIBUTION_NAMES: dict[str, str] = {
    "uniform": "Uniform",
    "normal": "Normal",
    "lognormal": "LogNormal",
    "exponential": "Exponential",
    "exp": "Exponential",
    "poisson": "Poisson",
    "gamma": "Gamma",
    "beta": "Beta",
}


@dataclass(frozen=True, slots=True)
class _PositiveParam:
    """A distribution parameter that must be > 0, and the names it answers to."""

    label: str
    aliases: tuple[str, ...]
    default: float = 1.0
    # Exponential is parameterised by *either* a scale or a rate. Whichever the
    # caller actually supplied is the one that has to be positive; the other
    # falls back to a default that is never used.
    only_if_present: str | None = None
    only_if_absent: str | None = None

    def applies_to(self, params: dict[str, float]) -> bool:
        if self.only_if_present is not None and self.only_if_present not in params:
            return False
        return not (self.only_if_absent is not None and self.only_if_absent in params)

    def resolve(self, params: dict[str, float]) -> float:
        for alias in self.aliases:
            if alias in params:
                return params[alias]
        return self.default


_POSITIVE_PARAMS: dict[str, tuple[_PositiveParam, ...]] = {
    "uniform": (),
    "normal": (_PositiveParam("std", ("std", "sigma", "scale")),),
    "lognormal": (_PositiveParam("std", ("std", "sigma", "sdlog")),),
    "exponential": (
        _PositiveParam("scale", ("scale",), only_if_present="scale"),
        _PositiveParam("rate", ("rate", "lambda", "lambda_"), only_if_absent="scale"),
    ),
    "poisson": (_PositiveParam("lambda", ("lambda", "lambda_", "rate", "mean")),),
    "gamma": (
        _PositiveParam("shape", ("shape", "alpha", "k")),
        _PositiveParam("scale", ("scale", "beta", "theta")),
    ),
    "beta": (
        _PositiveParam("alpha", ("alpha", "a", "shape1")),
        _PositiveParam("beta", ("beta", "b", "shape2")),
    ),
}
_POSITIVE_PARAMS["exp"] = _POSITIVE_PARAMS["exponential"]


def normalize_distribution(distribution: str) -> str:
    """Returns the canonical lowercase name, or raises if polspec cannot sample it."""
    name = distribution.lower()
    if name not in DISTRIBUTION_NAMES:
        raise SpecError(
            f"Unsupported distribution '{distribution}'. "
            f"Supported: {sorted(DISTRIBUTION_NAMES)}"
        )
    return name


def validate_distribution_params(name: str, params: dict[str, float]) -> None:
    """Raises when a parameter that has to be positive is not.

    `name` must already be normalized. Unknown keys are left alone: the engine
    ignores them, and rejecting them here would break callers who pass one
    parameter dict to several columns.
    """
    display = DISTRIBUTION_NAMES[name]
    for param in _POSITIVE_PARAMS[name]:
        if not param.applies_to(params):
            continue
        value = param.resolve(params)
        if value <= 0:
            raise SpecError(
                f"{display} distribution {param.label} must be positive, got {value}"
            )
