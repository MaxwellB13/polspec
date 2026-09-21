from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
from polspec.formats import lookup as _lookup_format
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
    seed_name : str | None, optional
        The name the column's seed is derived from, when it is not the
        column's own. Generation seeds each column from the frame seed and
        the column *name*, so renaming a column changes the values it
        produces; a column declared with `seed_name="old"` keeps producing
        the data it did as `"old"`. That is all it does: a column inserted
        ahead of one carrying rules, a foreign key or a hierarchy still
        reshuffles it, and nothing is promised across polspec versions.
        Two columns of one spec cannot share a seed name, nor may one name
        another column: they would draw identical values.
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
        Probability of a value being null. Must be between 0 and 1, and only
        has effect alongside `nullable=True` -- so that turning nullability
        off does not also require deleting the rate beside it. Declaring a
        rate of your own without `nullable=True` warns, since that reads as
        asking for nulls rather than as a leftover.
    string_length : Bound | tuple[int, int] | list[int] | None, optional
        The inclusive range of string lengths, where that applies.
    format : str | None, optional
        The shape a `String` column's values take, by name: `"uuid4"`,
        `"email"`, `"ipv4"`, `"ipv6"`, `"mac"`, `"hostname"`,
        `"iso_country"` or `"iso_currency"`. Generation fills the column
        from that format's own sampler and validation checks every value
        against it, so a column carrying `format="email"` is generated to
        satisfy its own spec. A format owns the column's whole domain:
        it cannot be combined with `choices` or `string_length`, and only a
        `String` column can carry one. What it promises is syntax -- an
        address that is well-formed, not one that is deliverable.
    pattern : str | None, optional
        A regular expression every value of a `String` column must match,
        checked by validation only. Generation does not read it: a `String`
        column with a pattern is filled with ordinary random text, and the
        round trip that holds for every other field does not hold here --
        the same boundary as `validators`. Prefer `format` for a shape
        polspec can generate; use `pattern` for a shape it cannot. Cannot
        be combined with `format`, which already is a pattern with a
        sampler. Polars regex syntax; a pattern that does not compile is
        refused at declaration.
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

    # The fields are annotated with what a constructed ColSpec *holds*, after
    # `__post_init__` has normalised each one: a dtype instance, a `Bound`, a
    # tuple. What the constructor *accepts* is wider -- `pl.Int64` or
    # `pl.Int64()`, a tuple or a `Bound`, one validator or several -- and is
    # spelled out in the `__init__` below, which exists only for type
    # checkers (the dataclass generates the real one). A test holds the two
    # to the same parameters, so a field added to one is a red test in the
    # other.
    dtype: pl.DataType
    col_name: str | None = None
    seed_name: str | None = None
    nullable: bool = False
    bounds: Bound[Any] | None = None
    tags: tuple[str, ...] = ()
    unique: bool = False
    null_probability: float = _DEFAULT_NULL_PROBABILITY
    string_length: Bound[int] | None = None
    format: str | None = None
    pattern: str | None = None
    distribution: str | None = None
    distribution_params: dict[str, float] | None = None
    choices: tuple[Any, ...] | None = None
    weights: tuple[float, ...] | None = None
    rules: tuple[ColRule, ...] = ()
    validators: tuple[Check, ...] = ()

    if TYPE_CHECKING:

        def __init__(
            self,
            dtype: pl.DataType | type[pl.DataType],
            col_name: str | None = None,
            seed_name: str | None = None,
            nullable: bool = False,
            bounds: Bound[Any] | tuple[Any, Any] | list[Any] | None = None,
            tags: str | Sequence[str] | None = (),
            unique: bool = False,
            null_probability: float = _DEFAULT_NULL_PROBABILITY,
            string_length: Bound[int] | tuple[int, int] | list[int] | None = None,
            format: str | None = None,
            pattern: str | None = None,
            distribution: str | None = None,
            distribution_params: dict[str, float] | None = None,
            choices: Sequence[Any] | dict[Any, float] | None = None,
            weights: Sequence[float] | None = None,
            rules: Sequence[ColRule] = (),
            validators: Check
            | pl.Expr
            | Pred
            | Sequence[Check | pl.Expr | Pred]
            | None = (),
        ) -> None: ...

    def __post_init__(self) -> None:
        # Order matters: normalization first, so every check below sees the
        # canonical form; then the checks that need only one field; then the
        # ones that compare fields against each other.
        self._validate_col_name()
        self._validate_seed_name()
        self._normalize_dtype()
        self._normalize_ranges()
        self._normalize_tags()
        object.__setattr__(self, "rules", tuple(self.rules))
        self._normalize_validators()
        self._normalize_choices_and_weights()
        self._normalize_distribution()

        self._validate_probabilities()
        self._validate_format()
        self._validate_pattern()
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

    def _validate_seed_name(self) -> None:
        if self.seed_name is None:
            return
        if not isinstance(self.seed_name, str) or not self.seed_name:
            raise SpecError(
                f"ColSpec.seed_name must be a non-empty string, got {self.seed_name!r}"
            )

    def _normalize_dtype(self) -> None:
        """Instantiates a dtype passed as a class, so `pl.Int64` means `pl.Int64()`."""
        raw: Any = self.dtype
        if isinstance(raw, type) and issubclass(raw, pl.DataType):
            with suppress(TypeError):
                object.__setattr__(self, "dtype", raw())

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
        """Reduces `tags` to a tuple of distinct, non-empty strings.

        A `list` or `tuple` keeps the order it was written in -- that is the
        author's choice and it survives into the spec file. A `set` has no
        order to keep: Python salts string hashing per process, so the same
        declaration would yield a different tuple on every run, writing a
        different `tags:` line each time `to_yaml` is called and making two
        identically-written specs compare unequal across processes. Sorting is
        the only stable reading of an unordered input.
        """
        tags: Any = self.tags
        if tags is None:
            object.__setattr__(self, "tags", ())
        elif isinstance(tags, str):
            object.__setattr__(self, "tags", (tags,) if tags else ())
        elif isinstance(tags, (list, tuple, set, Sequence)):
            raw = sorted(tags) if isinstance(tags, (set, frozenset)) else tags
            distinct: dict[str, None] = {}
            for tag in raw:
                text = str(tag)
                if text:
                    distinct.setdefault(text, None)
            object.__setattr__(self, "tags", tuple(distinct))
        else:
            raise SpecError(
                f"ColSpec.tags must be a string or sequence of strings, got {type(tags).__name__}"
            )

    def _normalize_choices_and_weights(self) -> None:
        """Splits a `{choice: weight}` mapping into the two fields, and tuples both."""
        choices: Any = self.choices
        if isinstance(choices, dict):
            if self.weights is not None:
                raise SpecError(
                    "Cannot specify both a dict for choices and an explicit weights parameter"
                )
            object.__setattr__(
                self, "weights", tuple(float(w) for w in choices.values())
            )
            object.__setattr__(self, "choices", tuple(choices.keys()))
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
        self._warn_unused_null_probability()

    def _warn_unused_null_probability(self) -> None:
        """Warns about a null rate on a column that cannot hold a null.

        `nullable=False` wins and the rate is ignored, which is deliberate:
        turning nullability off should not also require deleting the rate
        beside it. But the same silence covers a genuine mistake -- asking for
        nulls and forgetting `nullable=True` -- where the column generates
        none and nothing says why.

        So this warns rather than raising, and only for a rate that cannot
        have been left behind by turning nullability off: the default is what
        every non-nullable column carries, and an explicit zero already agrees
        with `nullable=False`. Anything else was written on purpose and does
        not do what it says.
        """
        if self.nullable or self.null_probability in (
            0.0,
            _DEFAULT_NULL_PROBABILITY,
        ):
            return
        warnings.warn(
            f"ColSpec declares null_probability={self.null_probability} with "
            "nullable=False, so no nulls will be generated and validation will "
            "reject any it finds. Add nullable=True to get that rate, or drop "
            "null_probability to say the column holds no nulls.",
            stacklevel=4,
        )

    def _validate_format(self) -> None:
        """Rejects a format on a column it cannot describe, or beside a
        second definition of the same domain.

        A format is the whole story of what a `String` value looks like, so
        anything else that says what the values are -- `choices`, or the
        length a `uuid4` already fixes at 36 -- is a contradiction of the
        kind `_validate_choices_against_domain` already refuses, and is
        refused here in the same voice: drop one or the other.
        """
        if self.format is None:
            return
        fmt = _lookup_format(self.format)  # raises, naming the nearest format
        object.__setattr__(self, "format", fmt.name)
        if self.dtype not in (pl.String, pl.Utf8):
            raise SpecError(
                f"ColSpec.format is only supported for pl.String, got "
                f"{self.dtype!r}. A format describes the text a value is "
                "written as, which no other dtype holds."
            )
        if self.choices is not None:
            raise SpecError(
                f"ColSpec cannot carry both format={fmt.name!r} and choices: "
                "each is a complete description of the column's domain, and "
                "they cannot both hold. Drop the choices, or the format."
            )
        if self.string_length is not None:
            raise SpecError(
                f"ColSpec cannot carry both format={fmt.name!r} and "
                "string_length: the format already fixes how long a value is. "
                "Drop the string_length, or the format."
            )

    def _validate_pattern(self) -> None:
        """Refuses a pattern on a column it cannot describe, beside a format,
        or one Polars cannot compile.

        Compiled by Polars itself rather than Python's `re`: the two dialects
        differ (look-around, for one), and the engine that will run the
        check is the one whose opinion counts.
        """
        if self.pattern is None:
            return
        if not isinstance(self.pattern, str) or not self.pattern:
            raise SpecError(
                f"ColSpec.pattern must be a non-empty string, got {self.pattern!r}"
            )
        if self.dtype not in (pl.String, pl.Utf8):
            raise SpecError(
                f"ColSpec.pattern is only supported for pl.String, got "
                f"{self.dtype!r}. A pattern describes the text a value is "
                "written as, which no other dtype holds."
            )
        if self.format is not None:
            raise SpecError(
                f"ColSpec cannot carry both format={self.format!r} and pattern: "
                "a format is a pattern polspec can also generate. Drop the "
                "pattern, or the format."
            )
        try:
            pl.select(pl.lit("").str.contains(self.pattern))
        except Exception as exc:
            raise SpecError(
                f"ColSpec.pattern {self.pattern!r} is not a valid regular "
                f"expression: {' '.join(str(exc).split())}"
            ) from exc

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
        given: Any = self.validators
        if given is None:
            object.__setattr__(self, "validators", ())
            return

        raw = [given] if isinstance(given, (pl.Expr, Pred, Check)) else list(given)

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
            if prior is not None:
                if prior != chk:
                    raise SpecError(
                        f"Duplicate validator name {chk.name!r} on this ColSpec: "
                        f"{prior.expr!r} vs {chk.expr!r}. Give each validator a "
                        "distinct name."
                    )
                # The same validator written twice is one claim, not two. Kept
                # once so it produces one finding, matching how TableSpec
                # de-duplicates identical checks and foreign keys.
                continue
            seen[chk.name] = chk
            normalized.append(chk)

        object.__setattr__(self, "validators", tuple(normalized))
