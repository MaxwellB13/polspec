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
from polspec.distributions import (
    canonicalize_params,
    normalize_distribution,
    validate_distribution_params,
)
from polspec.dtypes import _bound_endpoint_to_physical, _dtype_value_limits
from polspec.errors import SpecError
from polspec.expr import Pred
from polspec.rules import ColRule, _reject_duplicate_choices


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
    raise SpecError(f"polspec cannot generate data for dtype {dtype!r}")


def _is_categorical_dtype(dtype: pl.DataType) -> bool:
    return (
        isinstance(dtype, pl.Categorical)
        or dtype == pl.Categorical
        or (isinstance(dtype, type) and issubclass(dtype, pl.Categorical))
    )


@dataclass(frozen=True, slots=True)
class ColSpec:
    """One column's declaration: its type, and every claim made about its values.

    A `ColSpec` is what `generate()` samples from and what `validate()` checks
    against, so each field below is a claim both sides read.

    Parameters
    ----------
    dtype : pl.DataType | type[pl.DataType]
        The data type of the column.
    col_name : str | None, optional
        Overrides the column's name in the generated/validated DataFrame.
        Declaring columns as class attributes on a `FrameSpec` requires a
        valid Python identifier, which cannot contain spaces or other special
        characters -- `col_name` lets the attribute keep a clean Python name
        (`unit_price`) while the actual column is named whatever the data uses
        (`"Unit Price"`). Everything else that refers to this column by name --
        `ColRule`, `unique_together`, tags lookups, `validate()` -- uses
        `col_name`, not the attribute name.
    nullable : bool, optional
        Whether the column allows null values.
    bounds : Bound | tuple | list | None, optional
        The inclusive range of values allowed in the column, as a `Bound` or a
        2-sequence. Only supported for numeric and temporal data types. Either
        endpoint may be None to leave that side unconstrained --
        `bounds=(0, None)` for a non-negative column, `bounds=(None, 0)` for a
        non-positive one.

        An open end means different things to the two consumers of this field,
        deliberately. `validate()` treats it as genuinely unconstrained and
        omits that half of the check. `generate()` cannot sample an unbounded
        range, so it falls back to the same default it would use with no bounds
        at all -- `bounds=(0, None)` on Int64 generates 0..1,000,000 while
        validating any value >= 0. This mirrors how `bounds=None` already
        behaves rather than adding a third rule.
    tags : str | Sequence[str], optional
        Tag or tags classifying the column, for later selection.
    unique : bool, optional
        Whether values in the column must be distinct. Generation draws the
        column without replacement; nulls are exempt. Cannot be combined with
        `weights`, a non-uniform `distribution`, or `rules`, none of which
        survive a draw without replacement.
    null_probability : float, optional
        Probability of a value being null. Must be between 0 and 1.
    string_length : Bound | tuple[int, int] | list[int] | None, optional
        The inclusive range of string lengths, where that applies.
    distribution : str | None, optional
        The name of the probability distribution for the column's values
        (e.g. `"uniform"`, `"normal"`).
    distribution_params : dict[str, float] | None, optional
        Parameters specific to the chosen distribution.
    choices : tuple | list | dict | None, optional
        A finite set of allowed values. A dict maps each choice to its weight.
    weights : tuple[float, ...] | list[float] | None, optional
        Weights associated with `choices`, biasing selection probabilities.
    rules : tuple[ColRule, ...], optional
        Rules (`ColRule`) that overwrite the column's values on the rows their
        condition matches.
    validators : Check | pl.Expr | Pred | Sequence[...] | None, optional
        A single-column business rule, or several: each either a `pl.Expr`
        boolean predicate (referencing only this column) or a `Check` (for a
        custom name, description or null handling). Unlike
        `FrameSpec.__checks__`, these travel with the column's own declaration.

    Examples
    --------
    >>> ColSpec(pl.Int64, bounds=(1, 100), nullable=True, null_probability=0.1)
    >>> ColSpec(pl.String, choices=["NEW", "PAID"], weights=[3.0, 1.0])
    """

    dtype: pl.DataType | type[pl.DataType]
    col_name: str | None = None
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
    validators: Check | pl.Expr | Pred | Sequence[Check | pl.Expr | Pred] | None = ()

    def __post_init__(self) -> None:
        # Order matters: normalization first, so every check below sees the
        # canonical form; then the checks that need only one field; then the
        # ones that compare fields against each other.
        self._validate_col_name()
        self._normalize_dtype()
        self._normalize_ranges()
        self._normalize_tags()
        object.__setattr__(self, "rules", tuple(self.rules))
        self._normalize_validators()
        self._normalize_choices_and_weights()
        self._normalize_distribution()

        self._validate_probabilities()
        self._validate_bounds_dtype_support()
        self._validate_bounds_fit_dtype()
        self._validate_weights()
        self._validate_choices_against_domain()
        self._validate_unique_is_generatable()

    def _validate_unique_is_generatable(self) -> None:
        """Rejects what `unique=True` cannot be combined with.

        A unique column is drawn *without* replacement, so anything that
        describes how often a value should recur has nothing left to say:
        weights bias a repeated draw, and a distribution shapes a pile of
        independent ones. A rule is worse than meaningless -- it overwrites
        values after the draw, from a fixed set of choices, which is how
        duplicates would get back in.
        """
        if not self.unique:
            return
        if self.weights is not None:
            raise SpecError(
                "ColSpec cannot be unique=True and carry weights: values are "
                "drawn without replacement, so a weight has no repeated draw "
                "to bias. Drop the weights, or the uniqueness."
            )
        if self.distribution is not None and self.distribution != "uniform":
            raise SpecError(
                f"ColSpec cannot be unique=True and carry "
                f"distribution={self.distribution!r}: values are drawn without "
                "replacement from the column's domain, which no distribution "
                "shapes. Drop the distribution, or the uniqueness."
            )
        if self.rules:
            raise SpecError(
                "ColSpec cannot be unique=True and carry rules: a rule "
                "overwrites matched rows with values from a fixed set, which "
                "would reintroduce the duplicates uniqueness rules out. Drop "
                "the rules, or the uniqueness."
            )

    def _validate_col_name(self) -> None:
        if self.col_name is not None and not self.col_name:
            raise SpecError("ColSpec.col_name must not be an empty string")

    def _normalize_dtype(self) -> None:
        """Instantiates a dtype passed as a class, so `pl.Int64` means `pl.Int64()`."""
        if isinstance(self.dtype, type) and issubclass(self.dtype, pl.DataType):
            with suppress(TypeError):
                object.__setattr__(self, "dtype", self.dtype())

    def _normalize_ranges(self) -> None:
        """Coerces `bounds` and `string_length` to `Bound`, rejecting open lengths."""
        object.__setattr__(self, "bounds", Bound._coerce(self.bounds))
        object.__setattr__(self, "string_length", Bound._coerce(self.string_length))
        # One internal representation for "unconstrained", so every downstream
        # `if spec.bounds is not None` guard keeps meaning what it says.
        if self.bounds is not None and self.bounds.is_open_both:
            object.__setattr__(self, "bounds", None)
        if self.string_length is not None and self.string_length.is_open:
            raise SpecError(
                "ColSpec.string_length requires both endpoints, got "
                f"{self.string_length!r}. An open end (None) is supported on "
                "ColSpec.bounds only."
            )

    def _normalize_tags(self) -> None:
        """Reduces `tags` to a tuple of distinct, non-empty strings."""
        if self.tags is None:
            object.__setattr__(self, "tags", ())
        elif isinstance(self.tags, str):
            object.__setattr__(self, "tags", (self.tags,) if self.tags else ())
        elif isinstance(self.tags, (list, tuple, set, Sequence)):
            distinct: dict[str, None] = {}
            for tag in self.tags:
                text = str(tag)
                if text:
                    distinct.setdefault(text, None)
            object.__setattr__(self, "tags", tuple(distinct))
        else:
            raise SpecError(
                f"ColSpec.tags must be a string or sequence of strings, got {type(self.tags).__name__}"
            )

    def _normalize_choices_and_weights(self) -> None:
        """Splits a `{choice: weight}` mapping into the two fields, and tuples both."""
        if isinstance(self.choices, dict):
            if self.weights is not None:
                raise SpecError(
                    "Cannot specify both a dict for choices and an explicit weights parameter"
                )
            object.__setattr__(
                self, "weights", tuple(float(w) for w in self.choices.values())
            )
            object.__setattr__(self, "choices", tuple(self.choices.keys()))
        else:
            if self.choices is not None:
                object.__setattr__(self, "choices", tuple(self.choices))
            if self.weights is not None:
                object.__setattr__(
                    self, "weights", tuple(float(w) for w in self.weights)
                )

        if self.choices is not None:
            if not self.choices:
                raise SpecError("ColSpec.choices must not be empty")
            _reject_duplicate_choices(self.choices, "ColSpec.choices", self.dtype)

    def _normalize_distribution(self) -> None:
        """Canonicalizes the distribution name and floats its parameters."""
        if self.distribution is not None:
            if not (
                self.dtype.is_integer()
                or self.dtype.is_float()
                or self.dtype.is_temporal()
            ):
                raise SpecError(
                    f"ColSpec.distribution is only supported for numeric or temporal "
                    f"dtypes, got {self.dtype!r}"
                )
            name = normalize_distribution(self.distribution)
            object.__setattr__(self, "distribution", name)
            if self.distribution_params is not None:
                params = canonicalize_params(
                    name,
                    {str(k): float(v) for k, v in self.distribution_params.items()},
                )
                object.__setattr__(self, "distribution_params", params)
                validate_distribution_params(name, params)

        # A Boolean column takes no distribution, but does accept `p`.
        if self.dtype == pl.Boolean and self.distribution_params is not None:
            params = {str(k): float(v) for k, v in self.distribution_params.items()}
            object.__setattr__(self, "distribution_params", params)
            if "p" in params and not 0.0 <= params["p"] <= 1.0:
                raise SpecError(
                    f"Boolean distribution_params['p'] must be between 0 and 1, got {params['p']}"
                )

    def _validate_probabilities(self) -> None:
        if not 0.0 <= self.null_probability <= 1.0:
            raise SpecError("null_probability must be between 0 and 1")

    def _validate_bounds_dtype_support(self) -> None:
        if self.bounds is not None and not (
            self.dtype.is_integer() or self.dtype.is_float() or self.dtype.is_temporal()
        ):
            raise SpecError(
                f"ColSpec.bounds is only supported for numeric or temporal "
                f"dtypes, got {self.dtype!r}"
            )

    def _validate_weights(self) -> None:
        """Checks weights against whatever defines this column's domain."""
        if self.weights is None:
            return

        if self.choices is not None:
            if len(self.weights) != len(self.choices):
                raise SpecError(
                    f"Length of weights ({len(self.weights)}) must match length of choices ({len(self.choices)})"
                )
        elif isinstance(self.dtype, pl.Enum):
            if len(self.weights) != len(self.dtype.categories):
                raise SpecError(
                    f"Length of weights ({len(self.weights)}) must match number of Enum categories ({len(self.dtype.categories)})"
                )
        elif self.dtype == pl.Boolean:
            if len(self.weights) != 2:
                raise SpecError(
                    "Boolean weights must be a 2-element sequence [p_false, p_true]"
                )
        else:
            raise SpecError(
                "ColSpec.weights requires 'choices', an Enum dtype, or a Boolean "
                f"dtype to define the domain weights apply to; got dtype={self.dtype!r} "
                "with no choices"
            )

        if any(w < 0 for w in self.weights):
            raise SpecError("Weights must all be non-negative")
        if sum(self.weights) <= 0:
            raise SpecError("Sum of weights must be positive")

    def _validate_choices_against_domain(self) -> None:
        """Checks this column's choices, and its rules', against its domain.

        A rule's choices are checked here too: a value a rule could assign but
        the column could never hold is a contradiction worth catching at
        declaration time rather than at generation.
        """
        if isinstance(self.dtype, pl.Enum):
            categories = set(self.dtype.categories.to_list())
            self._reject_choices(
                lambda c: c not in categories,
                f"are not among this column's Enum categories {sorted(categories)}",
            )

        if self.bounds is not None:
            low, high = self.bounds.min, self.bounds.max

            def outside(value: Any) -> bool:
                return (low is not None and value < low) or (
                    high is not None and value > high
                )

            self._reject_choices(
                outside, f"fall outside this column's bounds {self.bounds}"
            )

    def _reject_choices(self, offends, complaint: str) -> None:
        """Raises if any of this column's or its rules' choices `offends`."""
        if self.choices is not None:
            offending = [c for c in self.choices if offends(c)]
            if offending:
                raise SpecError(f"ColSpec.choices {offending} {complaint}")
        for rule in self.rules:
            offending = [c for c in rule.choices if offends(c)]
            if offending:
                raise SpecError(f"ColRule.choices {offending} {complaint}")

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
            if endpoint is None:
                continue  # unconstrained on this side; nothing to fit
            physical = _bound_endpoint_to_physical(endpoint, self.dtype)
            if not math.isfinite(physical):
                raise SpecError(
                    f"ColSpec.bounds {label} must be a finite value, got {endpoint!r}"
                )
            if not lo_limit <= physical <= hi_limit:
                raise SpecError(
                    f"ColSpec.bounds {label} ({endpoint!r}) is outside the range "
                    f"{self.dtype!r} can represent [{lo_limit}, {hi_limit}]"
                )

    def _normalize_validators(self) -> None:
        if self.validators is None:
            object.__setattr__(self, "validators", ())
            return

        raw = (
            [self.validators]
            if isinstance(self.validators, (pl.Expr, Pred, Check))
            else list(self.validators)
        )

        normalized: list[Check] = []
        seen: dict[str, Check] = {}
        for v in raw:
            if isinstance(v, Check):
                chk = v
            elif isinstance(v, (pl.Expr, Pred)):
                chk = Check(v)
            else:
                raise SpecError(
                    "ColSpec.validators items must be a polars Expr, a predicate "
                    "built with col(), or a Check, "
                    f"got {type(v).__name__}"
                )
            prior = seen.get(chk.name)
            if prior is not None and prior != chk:
                raise SpecError(
                    f"Duplicate validator name {chk.name!r} on this ColSpec: "
                    f"{prior.expr!r} vs {chk.expr!r}. Give each validator a "
                    "distinct name."
                )
            seen[chk.name] = chk
            normalized.append(chk)

        object.__setattr__(self, "validators", tuple(normalized))
