"""`ColSpec` -- one column, declared.

Internal: not part of the public API.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from polspec import frames
from polspec._colspec import basics, choices, defaults, derived, nested, ranges, text

# Defined beside the checks, which read them too, and imported from here by
# the rest of the package: each `as` re-exports the name.
from polspec._colspec.common import check_description as check_description
from polspec._colspec.common import column_kind as column_kind
from polspec._colspec.common import distinct_values, value_dtype_of
from polspec._colspec.common import is_categorical_dtype as is_categorical_dtype
from polspec._colspec.common import is_dtype as is_dtype
from polspec._colspec.masks import in_values, measure_length, out_of_bounds
from polspec.bound import Bound
from polspec.checks import Check, failure_mask
from polspec.constants import (
    DEFAULT_LIST_LEN,
    DEFAULT_NULL_PROBABILITY,
)
from polspec.dtypes import (
    field_dtypes,
    map_entries,
)
from polspec.expr import Pred
from polspec.formats import lookup as _lookup_format
from polspec.rules import ColRule


@dataclass(frozen=True, slots=True)
class ColSpec:
    """One column's declaration: its type, and every claim made about its values.

    A `ColSpec` is what `generate()` samples from and what `validate()` checks
    against, so each field below is a claim both sides read.

    Parameters
    ----------
    dtype : pl.DataType | type[pl.DataType]
        The data type of the column. A bare `pl.Enum` takes its categories
        from `choices`, in the order written: `ColSpec(pl.Enum,
        choices={"A": 0.5, "B": 0.5})` is `pl.Enum(["A", "B"])` with those
        weights.
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
        the data it did as `"old"`, its rules included. Nothing is promised
        across polspec versions. Two columns of one spec cannot share a
        seed name, nor may one name another column: they would draw
        identical values.
    nullable : bool, optional
        Whether the column allows null values.
    optional : bool, optional
        Whether the column may be absent from a frame. An absent optional
        column is no `missing_columns` finding, and none of its other claims
        is checked; present, it is held to every one. Generation makes it
        like any other column.
    default : Any, optional
        The value that stands for a missing one: `validate(fill_defaults=True)`
        fills the column's nulls with it, and adds the column holding it when
        it is absent; `missing_cols="add"` adds it holding it too. It must be
        a value the column's own claims accept -- its dtype, bounds, choices,
        format, pattern, lengths and validators -- and is refused at
        declaration if not. `None`, the default, is no default. A `List`,
        `Array`, `Map` or `Struct` column takes none. Generation does not
        read it.
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
    description : str, optional
        What the column means, for a reader: `to_markdown` shows it, and
        neither generation nor validation reads it.
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
    nan_probability : float, optional
        For a float column -- or a list's float elements, or a struct's float
        field -- how often a present value is NaN, from 0 (the default: a
        NaN is never generated, and validation reports one) to 1. NaN is not
        a value in any other sense: `bounds` and `choices` say nothing about
        it. Refused with `unique=True`, since no NaN equals another.
    string_length : Bound | tuple[int, int] | list[int] | None, optional
        The inclusive range of string lengths, where that applies.
    list_length : Bound | tuple[int, int] | list[int] | None, optional
        For a `List` column, the inclusive range of elements a value holds;
        generation defaults to 0..5. On a `List` or `Array` column every
        other field that describes a value -- `bounds`, `choices`, `weights`,
        `format`, `pattern`, `string_length`, `distribution` -- describes
        each *element*; `nullable` and `null_probability` describe the list
        itself. An `Array` takes its length from the dtype and refuses
        `list_length`. For a `Map`, it is the range of entries a map holds;
        undeclared, 0..5 or as many as its keys can take, if fewer.
    element_null_probability : float, optional
        For a `List` or `Array` column, how often an element is null, from 0
        (the default: elements are never null, and validation reports one)
        to 1. The list's own `nullable` is a null cell; this is a null
        inside a present one.
    fields : Mapping[str, ColSpec] | None, optional
        For a `Struct` column, what is claimed about each field's values:
        a `ColSpec` per field, keyed by name. The dtype is the schema --
        every field's name and type comes from it -- and `fields` is what
        is claimed about the values within it, so a struct of twenty
        fields where one needs bounds spells one field. A field the
        mapping omits is generated from its dtype alone.

        A field is a value, not a column: `unique`, `rules`, `validators`,
        `seed_name` and `col_name` are refused on one, since each is a
        statement about a column among columns. Everything that describes a
        value is allowed -- `nullable` included, which says whether the
        field may be null inside a struct that is present -- and `fields`
        too, so a struct may nest to any depth. A finding about a field
        names it: `point.lat`.

        A `List` or `Array` of a `Struct` takes `fields` too: it describes
        the element, as `bounds` and `format` already do. So does a `Map`,
        which is a list of `{key, value}` entries: `fields` may say what
        its `key` and its `value` are. A key is never null and never
        repeats within a map, so a nullable key is refused, and so is a
        `list_length` longer than the keys can fill.
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
    extra_values : Sequence[str] | Mapping[str, float] | None, optional
        Values a `format` column holds beside the format's own: ISO
        countries plus `"UK (ISO)"`, IPv4 addresses or `"NOT AVAILABLE"`.
        Validation accepts a value that has the format or is one of these.
        Generation draws each on its share of the present rows -- a list
        gives each 1%, a mapping sets each share -- and the format's own
        values on the rest. Only a column with a `format` takes them; a
        value the format already accepts, a repeated one, and shares
        summing past 1 are refused, and so is `unique=True`.
    pattern : str | None, optional
        A regular expression every value of a `String` column must match.
        Generation fills the column with full matches of it, drawn through
        the automaton Polars' own regex engine builds, so the values pass
        validation; ASCII wherever the pattern allows, within
        `string_length` when it is declared, and an unbounded repeat runs
        to 16 characters past the pattern's shortest match. A pattern that
        every plain random string already passes keeps those strings, and
        one the generator cannot follow (a Unicode word boundary, an
        automaton too large) is filled with them, as before 1.4. Cannot be
        combined with `format`, which names a shape polspec knows. Polars
        regex syntax; a pattern that does not compile is refused at
        declaration. *Generated from since 1.4.*
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
    derive : Pred | None, optional
        A `col()` expression computing the column from others, such as
        `col("qty") * col("price")`. Generation computes it after every
        column it reads, and casts it to the column's dtype; validation holds
        the column to it -- equal, or for a float within a relative 1e-9 --
        as a `derived` finding. A derived value is null exactly where the
        expression is. The expression decides the values, so a derived column
        declares none of `bounds`, `choices`, `rules`, `distribution`,
        `format`, `pattern`, `string_length`, `unique` or `default`, and is
        `nullable` when an input is. *Added in 1.4.*

    Examples
    --------
    >>> ColSpec(pl.Int64, bounds=(1, 100), nullable=True, null_probability=0.2)
    ColSpec(Int64, nullable=True, bounds=(1, 100), null_probability=0.2)
    >>> ColSpec(pl.String, choices={"NEW": 0.75, "PAID": 0.25})
    ColSpec(String, choices={'NEW': 0.75, 'PAID': 0.25})
    >>> ColSpec(pl.Enum, choices={"NEW": 0.75, "PAID": 0.25}).dtype
    Enum(categories=['NEW', 'PAID'])
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
    # Everything after this is keyword-only: a positional call would freeze
    # the order of the fields, and say nothing about what each argument is.
    _: KW_ONLY
    col_name: str | None = None
    seed_name: str | None = None
    nullable: bool = False
    optional: bool = False
    default: Any = None
    bounds: Bound[Any] | None = None
    tags: tuple[str, ...] = ()
    description: str | None = None
    unique: bool = False
    null_probability: float = DEFAULT_NULL_PROBABILITY
    nan_probability: float = 0.0
    string_length: Bound[int] | None = None
    list_length: Bound[int] | None = None
    element_null_probability: float = 0.0
    fields: Mapping[str, ColSpec] | None = None
    format: str | None = None
    extra_values: Mapping[str, float] | None = None
    pattern: str | None = None
    distribution: str | None = None
    distribution_params: dict[str, float] | None = None
    choices: tuple[Any, ...] | None = None
    weights: tuple[float, ...] | None = None
    rules: tuple[ColRule, ...] = ()
    validators: tuple[Check, ...] = ()
    derive: Pred | None = None
    text_format: str | None = None

    if TYPE_CHECKING:

        def __init__(
            self,
            dtype: pl.DataType | type[pl.DataType],
            *,
            col_name: str | None = None,
            seed_name: str | None = None,
            nullable: bool = False,
            optional: bool = False,
            default: Any = None,
            bounds: Bound[Any] | tuple[Any, Any] | list[Any] | None = None,
            tags: str | Sequence[str] | None = (),
            description: str | None = None,
            unique: bool = False,
            null_probability: float = DEFAULT_NULL_PROBABILITY,
            nan_probability: float = 0.0,
            string_length: Bound[int] | tuple[int, int] | list[int] | None = None,
            list_length: Bound[int] | tuple[int, int] | list[int] | None = None,
            element_null_probability: float = 0.0,
            fields: Mapping[str, ColSpec] | None = None,
            format: str | None = None,
            extra_values: Sequence[str] | Mapping[str, float] | None = None,
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
            derive: Pred | None = None,
            text_format: str | None = None,
        ) -> None: ...

    def __post_init__(self) -> None:
        # Order matters: normalization first, so every check below sees the
        # canonical form; then the checks that need only one field; then the
        # ones that compare fields against each other.
        basics.validate_col_name(self)
        basics.validate_seed_name(self)
        check_description(self.description, "ColSpec")
        basics.normalize_dtype(self)
        nested.validate_nesting(self)
        ranges.normalize_ranges(self)
        basics.normalize_tags(self)
        object.__setattr__(self, "rules", tuple(self.rules))
        basics.normalize_validators(self)
        choices.normalize_choices_and_weights(self)
        basics.canonicalize_distribution(self)

        basics.validate_null(self)
        basics.validate_probabilities(self)
        basics.validate_nan_probability(self)
        text.validate_format(self)
        text.validate_extra_values(self)
        text.validate_pattern(self)
        text.validate_text_format(self)
        ranges.validate_bounds_dtype_support(self)
        ranges.validate_bounds_fit_dtype(self)
        choices.validate_weights(self)
        choices.validate_choices_against_domain(self)
        basics.validate_unique_is_generatable(self)
        nested.validate_list_fields(self)
        nested.validate_struct_fields(self)
        nested.validate_map(self)
        defaults.validate_optional(self)
        defaults.validate_default(self)
        derived.validate_derive(self)

    def __repr__(self) -> str:
        """The column as declared: its dtype, then each field that differs
        from its default -- `ColSpec(Int64, bounds=(0, 10))`. An Enum's
        categories, and weighted choices, are written as `choices`, each
        beside its weight: `ColSpec(Enum, choices={'A': 0.7, 'B': 0.3})`."""
        parts, shown_as_choices = self._choices_repr()
        for f in dataclasses.fields(self):
            if f.name == "dtype" or not f.repr or f.name in shown_as_choices:
                continue
            value = getattr(self, f.name)
            default = (
                f.default_factory()
                if f.default_factory is not dataclasses.MISSING
                else f.default
            )
            if f.name == "null_probability" and not self.nullable:
                continue  # a rate of nulls a column holds none of says nothing
            if isinstance(value, Pred):
                parts.append(f"{f.name}={value!r}")
                continue  # `==` on a predicate builds one
            if value == default:
                continue
            shown = (value.min, value.max) if isinstance(value, Bound) else value
            parts.append(f"{f.name}={shown!r}")
        return f"ColSpec({', '.join(parts)})"

    def _choices_repr(self) -> tuple[list[str], frozenset[str]]:
        """The `repr`'s opening parts -- the dtype, and the choices when
        they are written with it -- and the fields those parts show."""
        dtype, values, weights = self.dtype, self.choices, self.weights
        if isinstance(dtype, pl.Enum) and values is None:
            # Its categories are its choices: the spelling it accepts.
            values = tuple(dtype.categories.to_list())
            first = "Enum"
        elif values is not None and weights is not None:
            first = repr(dtype)
        else:
            return [repr(dtype)], frozenset()
        if weights is None:
            shown = f"choices={list(values)!r}"
        else:
            pairs = ", ".join(
                f"{v!r}: {w!r}" for v, w in zip(values, weights, strict=True)
            )
            shown = f"choices={{{pairs}}}"
        return [first, shown], frozenset({"choices", "weights"})

    def __eq__(self, other: object) -> bool:
        """Field by field, as a dataclass compares -- a `col()` expression
        structurally, since `==` on one builds a predicate."""
        if other.__class__ is not self.__class__:
            return NotImplemented
        for f in dataclasses.fields(self):
            mine, theirs = getattr(self, f.name), getattr(other, f.name)
            if isinstance(mine, Pred) or isinstance(theirs, Pred):
                if not (isinstance(mine, Pred) and mine.equals(theirs)):
                    return False
            elif mine != theirs:
                return False
        return True

    def replace(self, **changes: Any) -> ColSpec:
        """This column with `changes` made, checked as a new declaration is:
        `Orders.col("total").replace(bounds=None)`."""
        return dataclasses.replace(self, **changes)

    @property
    def value_dtype(self) -> pl.DataType:
        """The dtype each *value* has: the element's for a `List` or `Array`
        column, the column's own otherwise. Every field that describes a
        value is checked against this."""
        return value_dtype_of(self.dtype)


def claims_refused(spec: ColSpec, values: pl.Series) -> list[str]:
    """The claims about values, of a scalar column, that `values` break,
    each named by the finding code validation would give it.

    A default and a pinned column are held to them before either is used.
    The bounds, choices and lengths are asked through the masks validation
    builds its findings from (`polspec._colspec.masks`), so the two cannot
    disagree about them; a test holds the rest to validation's answers. A claim among columns
    -- a rule, a check, a validator reading another column -- is not
    asked: `values` is one column. `values` is named as the column is, so
    a validator naming it reads it.
    """
    frame = values.to_frame()
    column = frames.column(values.name)
    present = values.drop_nulls()
    broken: list[str] = []

    def fails(mask: pl.Expr) -> bool:
        return bool(frame.select(mask.fill_null(False).any()).item())

    if not spec.nullable and values.null_count():
        broken.append("nullability")
    if values.dtype.is_float() and not spec.nan_probability and fails(column.is_nan()):
        broken.append("nan")
    if spec.unique and present.is_duplicated().any():
        broken.append("unique")
    bounds = spec.bounds
    if (
        bounds is not None
        and not bounds.is_open_both
        and fails(out_of_bounds(column, bounds, values.dtype))
    ):
        broken.append("bounds")
    if spec.choices is not None and fails(
        ~in_values(column, spec.choices, spec.dtype, values.dtype)
    ):
        broken.append("choices")
    if spec.format is not None:
        fmt = _lookup_format(spec.format)
        extras = list(spec.extra_values or {})
        if fails(~(fmt.check(column) | column.is_in(extras))):
            broken.append("format")
    if spec.pattern is not None and fails(~column.str.contains(spec.pattern)):
        broken.append("pattern")
    size = measure_length(column, values.dtype)
    lengths = spec.string_length
    if (
        lengths is not None
        and size is not None
        and fails(out_of_bounds(size, lengths, pl.UInt32()))
    ):
        broken.append("string_length")
    for validator in spec.validators:
        if set(validator.expr.meta.root_names()) == {values.name} and fails(
            failure_mask(validator)
        ):
            broken.append("validator")
    return sorted(set(broken))


def field_of(spec: ColSpec, name: str) -> ColSpec:
    """What one field of a struct value is claimed to be: what `fields`
    says, or its dtype alone when `fields` says nothing about it."""
    declared = (spec.fields or {}).get(name)
    if declared is not None:
        return declared
    dtype = spec.value_dtype
    assert isinstance(dtype, pl.Struct)  # noqa: S101 - only a struct has fields
    field = field_dtypes(dtype)[name]
    # A Null field can only be declared nullable; say so for it.
    return ColSpec(field, nullable=field == pl.Null)


def element_of(spec: ColSpec) -> ColSpec:
    """What a `List`'s elements are claimed to be: this declaration with
    the list's own claims -- its length, whether a cell is null -- taken
    off, and its element null rate as the element's own. Generation
    draws an element from it and validation checks one against it, so
    the two read one answer."""
    return dataclasses.replace(
        spec,
        dtype=spec.value_dtype,
        list_length=None,
        nullable=spec.element_null_probability > 0,
        null_probability=spec.element_null_probability,
        element_null_probability=0.0,
    )


def entries_of(spec: ColSpec) -> ColSpec:
    """A `Map` column as the list it is: this declaration with the dtype
    `List(Struct({"key": K, "value": V}))`, which Polars casts to and from
    the map. Everything a list of structs has -- its length, its entries'
    `fields` -- is then read by the code that reads a list of structs."""
    entries = map_entries(spec.dtype)
    assert entries is not None  # noqa: S101 - only a map is a list of entries
    return dataclasses.replace(spec, dtype=entries)


def drawn_entries_of(spec: ColSpec) -> ColSpec:
    """`entries_of`, with the length generation draws.

    A map with no `list_length` of its own is as long as a list, unless
    its keys run out first: a map of `Boolean` keys holds at most two
    entries, without having to be told. Validation reads `entries_of`,
    where an undeclared length is no claim at all."""
    entries = entries_of(spec)
    if entries.list_length is not None:
        return entries
    lo, hi = DEFAULT_LIST_LEN
    keys = distinct_values(field_of(spec, "key"))
    length = Bound(lo, hi if keys is None else min(hi, keys))
    return dataclasses.replace(entries, list_length=length)
