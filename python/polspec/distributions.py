"""The distributions polspec can sample, and the parameter names each accepts.

Aliases are resolved here, at declaration, and only here: the Rust engine
(`src/dist.rs`) reads the canonical names exactly and exports its own table as
`distribution_params()`, which `tests/test_engine.py` compares with
`DISTRIBUTIONS` so the two sides cannot drift.
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
class Param:
    """One distribution parameter: its canonical name, its aliases, its default."""

    name: str
    aliases: tuple[str, ...]
    default: float
    positive: bool = False


# Canonical distribution name -> its parameters. The Rust engine accepts the
# canonical names only (`src/dist.rs`); Python is the only place aliases are
# resolved, at declaration, so a spec file is always canonical.
DISTRIBUTIONS: dict[str, tuple[Param, ...]] = {
    "uniform": (),
    "normal": (
        Param("mean", ("mean", "mu", "loc"), 0.0),
        Param("std", ("std", "sigma", "scale"), 1.0, positive=True),
    ),
    "lognormal": (
        Param("mean", ("mean", "mu", "meanlog"), 0.0),
        Param("std", ("std", "sigma", "sdlog"), 1.0, positive=True),
    ),
    # Exponential is parameterised by *either* a scale or a rate. Whichever
    # the caller gives is kept; the engine derives the other.
    "exponential": (
        Param("scale", ("scale",), 1.0, positive=True),
        Param("rate", ("rate", "lambda", "lambda_"), 1.0, positive=True),
    ),
    "poisson": (
        Param("lambda", ("lambda", "lambda_", "rate", "mean"), 1.0, positive=True),
    ),
    "gamma": (
        Param("shape", ("shape", "alpha", "k"), 1.0, positive=True),
        Param("scale", ("scale", "beta", "theta"), 1.0, positive=True),
    ),
    "beta": (
        Param("alpha", ("alpha", "a", "shape1"), 1.0, positive=True),
        Param("beta", ("beta", "b", "shape2"), 1.0, positive=True),
    ),
}

_CANONICAL_NAME: dict[str, str] = {"exp": "exponential"}


def normalize_distribution(distribution: str) -> str:
    """Returns the canonical lowercase name, or raises if polspec cannot sample it."""
    name = distribution.strip().lower()
    if name not in DISTRIBUTION_NAMES:
        raise SpecError(
            f"Unsupported distribution '{distribution}'. "
            f"Supported: {sorted(DISTRIBUTION_NAMES)}"
        )
    return _CANONICAL_NAME.get(name, name)


def canonicalize_params(name: str, params: dict[str, float]) -> dict[str, float]:
    """Rewrites parameter keys to their canonical names.

    `name` must already be canonical. The first alias present wins; a key no
    parameter answers to is kept as written, since the engine ignores it and
    rejecting it would break callers sharing one parameter dict across
    columns of different distributions.
    """
    out: dict[str, float] = {}
    claimed: set[str] = set()
    for param in DISTRIBUTIONS[name]:
        for alias in param.aliases:
            if alias in params:
                out[param.name] = params[alias]
                claimed.update(param.aliases)
                break
    for key, value in params.items():
        if key not in claimed and key not in out:
            out[key] = value
    return out


def validate_distribution_params(name: str, params: dict[str, float]) -> None:
    """Raises when a parameter that has to be positive is not.

    `name` and `params` must already be canonical.
    """
    display = DISTRIBUTION_NAMES[name]
    for param in DISTRIBUTIONS[name]:
        if not param.positive or param.name not in params:
            continue
        if name == "exponential" and param.name == "rate" and "scale" in params:
            continue  # scale takes precedence; rate is then unused
        value = params[param.name]
        if value <= 0:
            raise SpecError(
                f"{display} distribution {param.name} must be positive, got {value}"
            )
