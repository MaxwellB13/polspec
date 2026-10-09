"""The comparator registry: one comparison per field of a declaration.

Every field of `ColSpec` has an entry in `FIELD_COMPARATORS`, and every
field of `TableSpec` one in `TABLE_COMPARATORS` -- a parity test asserts
both. So when a field is added to a declaration, the comparison for it is
one entry here or one red test, the same way a new field is one `Field` in
`serialization.fields`. Several fields share one comparator (`bounds`,
`choices` and `format` are all the column's domain); the runner dedupes by
identity so it runs once.

Each comparator takes a `Pair` -- the declared column on one side and, on
the other, either a second declaration or what a frame's column was
observed to hold -- and returns the findings it can see. A comparator whose
field one side cannot see returns nothing: data has no `tags`; a second
declaration has no null *rate* to measure.

Severity follows one rule, and where the rule is about dtypes it is the
same function validation uses, so drift never has a second opinion about
what would fail.

Internal: not part of the public API.
"""

from __future__ import annotations

import datetime
import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from polspec.domain import Domain, describe_values
from polspec.drift import stats
from polspec.drift.data import Observed
from polspec.drift.report import DriftFinding, Severity
from polspec.dtypes import bound_endpoint_to_physical, field_dtypes
from polspec.errors import PolspecError
from polspec.formats import lookup as _lookup_format
from polspec.generation import generate
from polspec.pass_order import DISTRIBUTION, NAN_RATE, NULL_RATE, VALUES, WEIGHTS
from polspec.shape import ks_distance, physical
from polspec.validation.constraints import is_dtype_compatible

# The sample a declared distribution is drawn as, to hold values against:
# large enough that its own noise is a small part of the KS distance, and
# drawn from a fixed seed, so a report is the same every run.
REFERENCE_ROWS = 50_000
REFERENCE_SEED = 20_260_929

if TYPE_CHECKING:
    from polspec.bound import Bound
    from polspec.drift import DriftOptions
    from polspec.spec import ColSpec
    from polspec.tablespec import TableSpec


@dataclass(frozen=True, slots=True)
class Pair:
    """One column, on both sides of a comparison.

    `declared` is always a `ColSpec`. `other` is a `ColSpec` when two specs
    are being diffed and an `Observed` when a spec is being held against a
    frame; `mode` says which. `where` names a field inside the column
    (`point.lat`) when the pair is one of a struct's fields: a finding's key
    and message name it, while its `columns` stay the column's.
    `overridden` names the claims a pass writing the column can leave untrue
    of it (`Pass.overrides`) -- a rule's weights, a hierarchy's null rate --
    which are not compared.
    """

    column: str
    declared: ColSpec
    other: ColSpec | Observed
    options: DriftOptions
    where: str = ""
    overridden: frozenset[str] = frozenset()

    @property
    def label(self) -> str:
        """What the pair is about: the column, or the field within it."""
        return self.where or self.column

    @property
    def mode(self) -> Literal["diff", "drift"]:
        return "drift" if isinstance(self.other, Observed) else "diff"

    @property
    def new(self) -> ColSpec:
        """The other side as a declaration; only meaningful in `diff` mode."""
        assert not isinstance(self.other, Observed)  # noqa: S101 - mode guard
        return self.other

    @property
    def observed(self) -> Observed:
        """The other side as data; only meaningful in `drift` mode."""
        assert isinstance(self.other, Observed)  # noqa: S101 - mode guard
        return self.other

    def finding(
        self,
        code: Any,
        severity: Severity,
        message: str,
        *,
        suffix: str,
        **details: Any,
    ) -> DriftFinding:
        return DriftFinding(
            code=code,
            severity=severity,
            key=f"{self.label}__{suffix}",
            message=f"Column '{self.label}': {message}",
            columns=(self.column,),
            details=details,
        )


Comparator = Callable[[Pair], list[DriftFinding]]


def _ignore(pair: Pair) -> list[DriftFinding]:
    return []


# ---------------------------------------------------------------------------
# dtype and nullability
# ---------------------------------------------------------------------------


def _same_dtype(a: pl.DataType, b: pl.DataType) -> bool:
    if a in (pl.String, pl.Utf8):
        return b in (pl.String, pl.Utf8)
    return a == b


def _compare_dtype(pair: Pair) -> list[DriftFinding]:
    """A dtype change is breaking exactly when validation would refuse the
    values under the declaration they are now checked against."""
    if pair.mode == "diff":
        # Old data is what the new declaration would be checked against.
        actual, expected = pair.declared.dtype, pair.new.dtype
    else:
        actual, expected = pair.observed.dtype, pair.declared.dtype
    if _same_dtype(actual, expected):
        return []
    compatible = is_dtype_compatible(
        expected, actual, strict=pair.options.strict_dtypes
    )
    if compatible and pair.mode == "drift":
        # What a file hands back -- a String for an Enum from a CSV, an
        # Int64 for an Int32 -- validates as it is; data that passes
        # validation has not drifted on dtype. `strict_dtypes` reports it.
        return []
    if pair.mode == "diff":
        message = f"dtype changed from {actual} to {expected}" + (
            "" if compatible else "; values of the old type no longer validate"
        )
    else:
        message = f"holds {actual}, declared {expected}" + (
            "" if compatible else "; the column fails validation on dtype"
        )
    return [
        pair.finding(
            "dtype_changed",
            "compatible" if compatible else "breaking",
            message,
            suffix="dtype",
            old=str(actual),
            new=str(expected),
        )
    ]


def _compare_nullable(pair: Pair) -> list[DriftFinding]:
    if pair.mode == "diff":
        old, new = pair.declared.nullable, pair.new.nullable
        if old == new:
            return []
        if new:
            return [
                pair.finding(
                    "nullability_changed",
                    "compatible",
                    "now nullable; nulls that failed before are accepted",
                    suffix="nullable",
                    old=old,
                    new=new,
                )
            ]
        return [
            pair.finding(
                "nullability_changed",
                "breaking",
                "no longer nullable; any null that validated before now fails",
                suffix="nullable",
                old=old,
                new=new,
            )
        ]
    if pair.declared.nullable or not pair.observed.has_nulls:
        return []
    return [
        pair.finding(
            "nullability_changed",
            "breaking",
            f"non-nullable, but holds {pair.observed.null_count} null(s). "
            "Declare nullable=True, or fix the source",
            suffix="nullable",
            null_count=pair.observed.null_count,
        )
    ]


# ---------------------------------------------------------------------------
# The domain: bounds, choices, format -- and string_length beside them
# ---------------------------------------------------------------------------

Relation = Literal["equal", "widened", "narrowed", "changed", "unknown"]


def _relation(old: Domain, new: Domain) -> Relation:
    """How `new` relates to `old`, from `Domain.rejects` run both ways.

    `rejects` says nothing about domains it cannot compare, so two of them
    that reject nothing either way are equal-or-unknown; they are reported
    as equal, since nothing observable changed.
    """
    new_rejects_old = new.rejects(old) is not None
    old_rejects_new = old.rejects(new) is not None
    if not new_rejects_old and not old_rejects_new:
        return "equal"
    if new_rejects_old and old_rejects_new:
        return "changed"
    return "narrowed" if new_rejects_old else "widened"


_RELATION_FINDINGS: dict[Relation, tuple[str, Severity, str]] = {
    "widened": (
        "domain_widened",
        "compatible",
        "values that failed before are now accepted",
    ),
    "narrowed": (
        "domain_narrowed",
        "breaking",
        "values that validated before now fail",
    ),
    "changed": (
        "domain_changed",
        "breaking",
        "the two do not overlap cleanly; values that validated before may fail",
    ),
}


def _compare_domain(pair: Pair) -> list[DriftFinding]:
    """`bounds`, `choices` and `format` are one claim: what values the column holds."""
    if pair.mode == "diff":
        old, new = Domain.of(pair.declared), Domain.of(pair.new)
        relation = _relation(old, new)
        if relation not in _RELATION_FINDINGS:
            return []
        code, severity, consequence = _RELATION_FINDINGS[relation]
        return [
            pair.finding(
                code,
                severity,
                f"domain {relation} from {old} to {new}; {consequence}",
                suffix="domain",
                old=str(old),
                new=str(new),
            )
        ]
    return [*_observed_bounds(pair), *_observed_values(pair), *_observed_format(pair)]


def _exceeded(
    name: str, declared: Bound, found: Bound, dtype: pl.DataType
) -> dict[str, Any]:
    """The facts of an observed extent escaping a declared bound, or empty.

    A temporal bound may be declared as the physical integer the dtype
    stores (a `Duration` in microseconds) while the data reads back as a
    `timedelta`; both sides are compared in the physical form, the same way
    generation and validation read them, and reported as found.
    """

    def physical(value: Any) -> Any:
        return (
            bound_endpoint_to_physical(value, dtype) if dtype.is_temporal() else value
        )

    facts: dict[str, Any] = {}
    if declared.min is not None and physical(found.min) < physical(declared.min):
        facts["below_by"] = _distance(declared.min, found.min, physical)
    if declared.max is not None and physical(found.max) > physical(declared.max):
        facts["above_by"] = _distance(found.max, declared.max, physical)
    if facts:
        facts.update(field=name, min_found=found.min, max_found=found.max)
    return facts


def _distance(a: Any, b: Any, physical: Callable[[Any], Any]) -> Any:
    """`a - b` in the values' own terms (a `timedelta` for dates) where the
    types allow it, else in the dtype's physical units."""
    try:
        return a - b
    except TypeError:
        return physical(a) - physical(b)


def _human(value: Any) -> str:
    """A value as a reader would write it: `2026-02-05`, `400 days`, `20`."""
    if (
        isinstance(value, datetime.timedelta)
        and value.seconds == 0
        and value.microseconds == 0
    ):
        return f"{value.days} day{'s' if value.days != 1 else ''}"
    return str(value)


def _describe_excess(facts: Mapping[str, Any]) -> str:
    parts = []
    if "below_by" in facts:
        by = f" by {_human(facts['below_by'])}" if facts["below_by"] is not None else ""
        parts.append(f"min found {_human(facts['min_found'])}{by} below")
    if "above_by" in facts:
        by = f" by {_human(facts['above_by'])}" if facts["above_by"] is not None else ""
        parts.append(f"max found {_human(facts['max_found'])}{by} above")
    return ", ".join(parts)


def _observed_bounds(pair: Pair) -> list[DriftFinding]:
    declared, found = pair.declared.bounds, pair.observed.extent
    if declared is None or found is None:
        return []
    facts = _exceeded("bounds", declared, found, pair.observed.dtype)
    if not facts:
        return []
    return [
        pair.finding(
            "bounds_exceeded",
            "breaking",
            f"values escape bounds {declared}: {_describe_excess(facts)}. "
            "Widen the bounds, or fix the source",
            suffix="bounds",
            **facts,
        )
    ]


def _observed_values(pair: Pair) -> list[DriftFinding]:
    domain = Domain.of(pair.declared)
    if domain.values is None:
        return []
    findings = []
    count, samples = pair.observed.outside
    if count:
        findings.append(
            pair.finding(
                "new_values",
                "breaking",
                f"{count} row(s) hold values outside {domain}: {describe_values(samples)}. "
                "Add them to the declaration, or fix the source",
                suffix="values",
                count=count,
                values=list(samples),
            )
        )
    unseen = _improbably_unseen(pair, domain) if pair.options.unseen_values else ()
    if unseen:
        claimed = _claimed(pair.declared)
        declared = len(claimed) if claimed is not None else len(domain.values)
        findings.append(
            pair.finding(
                "cardinality_moved",
                "compatible",
                f"{len(unseen)} of {declared} declared value(s) never "
                f"appear: {describe_values(unseen)}",
                suffix="unseen",
                unseen=list(unseen),
                declared=declared,
                observed=declared - len(unseen),
            )
        )
    return findings


def _improbably_unseen(pair: Pair, domain: Domain) -> tuple[Any, ...]:
    """The declared values the data never holds and, at the share each is
    generated at, would not miss by chance: a value of weight 1% is absent
    from fifty rows six times in ten, and from a thousand almost never. A
    column a pass rewrites has no share to weigh a value at -- a rule may
    leave one out of every row -- so none is reported on it."""
    unseen = pair.observed.unseen
    if not unseen or VALUES in pair.overridden:
        return ()
    claimed = _claimed(pair.declared)
    if claimed is not None:
        unseen = tuple(value for value in unseen if value in claimed)
    values = list(domain.values or ())
    shares = _generation_shares(pair.declared, values)
    share_of = dict(zip(values, shares, strict=True))
    draws = pair.observed.present_count
    return tuple(
        value
        for value in unseen
        if stats.unseen_p(share_of.get(value, 0.0), draws) < pair.options.significance
    )


def _claimed(declared: ColSpec) -> frozenset[Any] | None:
    """The declared values whose absence is news, when not every value in
    the domain is one: a finite format's codes are what a value may be, not
    a list each of which should appear -- a format promises syntax, not
    existence -- so only its `extra_values` are claimed. None: all of them."""
    if declared.format is None or declared.choices is not None:
        return None
    if not _lookup_format(declared.format).is_finite:
        return None
    return frozenset(declared.extra_values or ())


def _generation_shares(declared: ColSpec, values: Sequence[Any]) -> list[float]:
    """How often generation draws each of `values`, the column's finite
    domain: its weights, a finite format's extras at their shares and its
    own values evenly, or every value evenly."""
    if not values:
        return []
    even = [1 / len(values)] * len(values)
    if declared.weights is not None and len(declared.weights) == len(values):
        total = sum(declared.weights)
        return [w / total for w in declared.weights]
    extras = declared.extra_values or {}
    if extras and declared.format is not None and len(values) > len(extras):
        rest = max(0.0, 1.0 - sum(extras.values()))
        own = len(values) - len(extras)
        return [rest / own] * own + list(extras.values())
    return even


def _observed_format(pair: Pair) -> list[DriftFinding]:
    count, samples = pair.observed.format_failures
    if not count:
        return []
    return [
        pair.finding(
            "format_violated",
            "breaking",
            f"{count} row(s) are not {_lookup_format(pair.declared.format or '')}: "
            f"{describe_values(samples)}. "
            "Change the format, or fix the source",
            suffix="format",
            format=pair.declared.format,
            count=count,
            samples=list(samples),
        )
    ]


def _length_relation(old: Bound | None, new: Bound | None) -> Relation:
    """How a closed `[min, max]` moved; `None` is unconstrained."""
    if old == new:
        return "equal"
    if old is None:
        return "narrowed"
    if new is None:
        return "widened"
    (old_min, old_max), (new_min, new_max) = old.closed(), new.closed()
    widened = new_min < old_min or new_max > old_max
    narrowed = new_min > old_min or new_max < old_max
    if widened and narrowed:
        return "changed"
    return "widened" if widened else "narrowed"


def _compare_length(field: str, observed_attr: str) -> Comparator:
    """A comparator for a closed length range -- `string_length` over each
    value's characters, `list_length` over each list's elements."""

    def compare(pair: Pair) -> list[DriftFinding]:
        if pair.mode == "diff":
            old = getattr(pair.declared, field)
            new = getattr(pair.new, field)
            relation = _length_relation(old, new)
            if relation not in _RELATION_FINDINGS:
                return []
            code, severity, consequence = _RELATION_FINDINGS[relation]
            return [
                pair.finding(
                    code,
                    severity,
                    f"{field} {relation} from {old or 'unconstrained'} to "
                    f"{new or 'unconstrained'}; {consequence}",
                    suffix=field,
                    field=field,
                    old=[old.min, old.max] if old else None,
                    new=[new.min, new.max] if new else None,
                )
            ]
        declared = getattr(pair.declared, field)
        found = getattr(pair.observed, observed_attr)
        if declared is None or found is None:
            return []
        facts = _exceeded(field, declared, found, pl.Int64())
        if not facts:
            return []
        return [
            pair.finding(
                "bounds_exceeded",
                "breaking",
                f"lengths escape {field} {declared}: {_describe_excess(facts)}. "
                "Widen the length, or fix the source",
                suffix=field,
                **facts,
            )
        ]

    return compare


# ---------------------------------------------------------------------------
# Rates, uniqueness, rules, validators, and the fields that only describe
# generation
# ---------------------------------------------------------------------------


def _rate_moved(
    pair: Pair,
    code: str,
    what: str,
    field: str,
    declared: float,
    hits: int,
    trials: int,
    suffix: str,
) -> list[DriftFinding]:
    """A rate that moved: further from `declared` than the tolerance, and
    further than sampling noise over `trials` would put it."""
    if not trials:
        return []
    observed = hits / trials
    tolerance = pair.options.null_rate_tolerance
    if abs(observed - declared) <= tolerance:
        return []
    p_value = stats.binomial_p(hits, trials, declared)
    if p_value >= pair.options.significance:
        return []
    return [
        pair.finding(
            code,
            "compatible",
            f"{observed:.1%} {what}, declared {field}={declared}; beyond the "
            f"{tolerance:.0%} tolerance over {trials:,} value(s) "
            f"(p={p_value:.2g})",
            suffix=suffix,
            declared=declared,
            observed=observed,
            tolerance=tolerance,
            count=trials,
            p_value=p_value,
        )
    ]


def _compare_null_rate(pair: Pair) -> list[DriftFinding]:
    if pair.mode == "diff":
        old, new = pair.declared.null_probability, pair.new.null_probability
        if old == new or not (pair.declared.nullable and pair.new.nullable):
            return []  # a rate on a non-nullable column means nothing
        return [_field_changed(pair, "null_probability", old, new)]
    if not pair.declared.nullable or NULL_RATE in pair.overridden:
        return []
    observed = pair.observed
    return _rate_moved(
        pair,
        "null_rate_moved",
        "null",
        "null_probability",
        pair.declared.null_probability,
        observed.null_count,
        observed.height,
        "null_rate",
    )


def _compare_nans(pair: Pair) -> list[DriftFinding]:
    """NaN in a float value. A share of 0 claims there are none, so
    allowing them widens what validates and forbidding them narrows it."""
    declared = pair.declared.nan_probability
    if pair.mode == "diff":
        new = pair.new.nan_probability
        if declared == new:
            return []
        if not declared or not new:
            allowed = bool(new)
            return [
                pair.finding(
                    "domain_widened" if allowed else "domain_narrowed",
                    "compatible" if allowed else "breaking",
                    "values may now be NaN; a NaN that failed before is accepted"
                    if allowed
                    else "values may no longer be NaN; any NaN that validated "
                    "before now fails",
                    suffix="nan",
                    old=declared,
                    new=new,
                )
            ]
        return [_field_changed(pair, "nan_probability", declared, new)]
    observed = pair.observed
    if not declared:
        if not observed.nan_count:
            return []
        return [
            pair.finding(
                "new_values",
                "breaking",
                f"holds {observed.nan_count} NaN value(s) and declares none. "
                "Declare nan_probability, or fix the source",
                suffix="nan",
                nan_count=observed.nan_count,
            )
        ]
    if NAN_RATE in pair.overridden:
        return []
    return _rate_moved(
        pair,
        "nan_rate_moved",
        "of values NaN",
        "nan_probability",
        declared,
        observed.nan_count,
        observed.value_count,
        "nan_rate",
    )


def _compare_element_nulls(pair: Pair) -> list[DriftFinding]:
    """Nulls inside a list. A rate of 0 claims there are none, so allowing
    them widens what validates and forbidding them narrows it, as
    `nullable` does for the cell."""
    declared = pair.declared.element_null_probability
    if pair.mode == "diff":
        new = pair.new.element_null_probability
        if declared == new:
            return []
        if not declared or not new:
            allowed = bool(new)
            return [
                pair.finding(
                    "nullability_changed",
                    "compatible" if allowed else "breaking",
                    "list elements may now be null; lists holding one that failed "
                    "before are accepted"
                    if allowed
                    else "list elements may no longer be null; any list holding "
                    "one that validated before now fails",
                    suffix="element_nulls",
                    old=declared,
                    new=new,
                )
            ]
        return [_field_changed(pair, "element_null_probability", declared, new)]
    observed = pair.observed
    if not declared:
        if not observed.element_null_count:
            return []
        return [
            pair.finding(
                "nullability_changed",
                "breaking",
                f"elements are never null, but {observed.element_null_count} "
                "are. Declare element_null_probability, or fix the source",
                suffix="element_nulls",
                null_count=observed.element_null_count,
            )
        ]
    return _rate_moved(
        pair,
        "null_rate_moved",
        "of elements null",
        "element_null_probability",
        declared,
        observed.element_null_count,
        observed.element_count,
        "element_null_rate",
    )


def _constraint(pair: Pair, kind: str, added: bool, what: str) -> DriftFinding:
    return pair.finding(
        "constraint_added" if added else "constraint_removed",
        "breaking" if added else "compatible",
        f"{what} {'added' if added else 'removed'}"
        + ("; rows that validated before may fail" if added else ""),
        suffix=f"{kind}",
        kind=kind,
    )


def _compare_unique(pair: Pair) -> list[DriftFinding]:
    if pair.mode == "drift" or pair.declared.unique == pair.new.unique:
        return []  # uniqueness in data is validation's job, not drift's
    return [_constraint(pair, "unique", added=pair.new.unique, what="unique=True")]


def _compare_named(
    pair: Pair, kind: str, old: Sequence[Any], new: Sequence[Any], name: Callable
) -> list[DriftFinding]:
    """Added and removed members of a constraint set, matched by `name`.

    A member that changed under the same name is reported as removed and
    added; nothing is guessed about whether it is "the same" constraint.
    """
    old_by, new_by = {name(o): o for o in old}, {name(n): n for n in new}
    findings = []
    for key, item in old_by.items():
        if new_by.get(key) != item:
            findings.append(_member(pair, kind, key, added=False))
    for key, item in new_by.items():
        if old_by.get(key) != item:
            findings.append(_member(pair, kind, key, added=True))
    return findings


def _member(pair: Pair, kind: str, key: str, *, added: bool) -> DriftFinding:
    finding = _constraint(pair, kind, added, f"{kind} {key!r}")
    return DriftFinding(
        code=finding.code,
        severity=finding.severity,
        key=f"{pair.label}__{kind}:{key}",
        message=finding.message,
        columns=finding.columns,
        details={**finding.details, "name": key},
    )


def _compare_rules(pair: Pair) -> list[DriftFinding]:
    if pair.mode == "drift":
        return []
    return _compare_named(
        pair, "rule", pair.declared.rules, pair.new.rules, lambda r: repr(r.when)
    )


def _compare_validators(pair: Pair) -> list[DriftFinding]:
    if pair.mode == "drift":
        return []
    return _compare_named(
        pair,
        "validator",
        pair.declared.validators,
        pair.new.validators,
        lambda c: c.name,
    )


def _field_changed(pair: Pair, name: str, old: Any, new: Any) -> DriftFinding:
    return pair.finding(
        "field_changed",
        "compatible",
        f"{name} changed from {old!r} to {new!r}",
        suffix=name,
        field=name,
        old=old,
        new=new,
    )


def _compare_struct_fields(pair: Pair) -> list[DriftFinding]:
    """A struct's fields, each compared as a column is.

    Every comparator runs again on the field pair -- the declared field
    against the other side's, a second declaration in `diff` and what the
    data's field holds in `drift` -- keyed by its path (`point.lat`). So a
    field's bounds narrowing is breaking for the reason a column's is, and
    one field's history reads the same whether it sits in a struct or
    beside one. A field `fields` does not describe is compared as its dtype
    alone; a field only one side's dtype has is the dtype comparator's
    finding.
    """
    declared = pair.declared.value_dtype
    if not isinstance(declared, pl.Struct):
        return []
    if pair.mode == "diff":
        new = pair.new.value_dtype
        if not isinstance(new, pl.Struct):
            return []
        others: dict[str, ColSpec | Observed] = {
            name: pair.new._field(name)
            for name in field_dtypes(new)
            if name in field_dtypes(declared)
        }
    else:
        others = dict(pair.observed.fields)
    findings: list[DriftFinding] = []
    for name, other in others.items():
        field_pair = Pair(
            pair.column,
            pair.declared._field(name),
            other,
            pair.options,
            where=f"{pair.label}.{name}",
            overridden=pair.overridden,
        )
        findings.extend(compare_column(field_pair))
    return findings


def _compare_weights(pair: Pair) -> list[DriftFinding]:
    """Declared weights against the frequencies the data holds: moved when
    the shares differ by at least `frequency_tolerance` in total variation,
    and by more than chance would move them over the rows seen."""
    if pair.mode == "diff":
        old, new = pair.declared.weights, pair.new.weights
        return [] if old == new else [_field_changed(pair, "weights", old, new)]
    counts = pair.observed.counts
    declared = pair.declared
    if (
        WEIGHTS in pair.overridden
        or counts is None
        or not sum(counts)
        or declared.weights is None
    ):
        return []
    if declared.value_dtype == pl.Boolean:
        values: list[Any] = [False, True]
    else:
        values = list(Domain.of(declared).values or ())
    if len(values) != len(counts):
        return []
    shares = _generation_shares(declared, values)
    distance = stats.total_variation(counts, shares)
    if distance < pair.options.frequency_tolerance:
        return []
    p_value = stats.chi_square_p(counts, shares)
    if p_value >= pair.options.significance:
        return []
    total = sum(counts)
    moved = sorted(
        zip(values, shares, counts, strict=True),
        key=lambda row: -abs(row[2] / total - row[1]),
    )[:3]
    described = ", ".join(
        f"{value!r} {count / total:.1%} (declared {share:.1%})"
        for value, share, count in moved
    )
    return [
        pair.finding(
            "frequencies_moved",
            "compatible",
            f"frequencies moved by {distance:.1%} over {total:,} value(s): "
            f"{described} (p={p_value:.2g})",
            suffix="frequencies",
            declared={_key(v): share for v, share in zip(values, shares, strict=True)},
            observed={_key(v): c / total for v, c in zip(values, counts, strict=True)},
            distance=distance,
            tolerance=pair.options.frequency_tolerance,
            count=total,
            p_value=p_value,
        )
    ]


def _key(value: Any) -> Any:
    """A value as a details key: JSON keys are strings."""
    return value if isinstance(value, str) else str(value)


def _compare_shape(pair: Pair) -> list[DriftFinding]:
    """A declared distribution against the values: moved when their
    Kolmogorov-Smirnov distance from a sample drawn from the declaration is
    at least `distribution_tolerance`, and more than chance would put
    between two samples of those sizes. Drawn, not computed: the reference
    is what `generate()` makes of the declaration -- clamping at its bounds
    and rounding an integer included -- as `shape.fit` compares candidates."""
    if pair.mode == "diff":
        return [
            _field_changed(
                pair, name, getattr(pair.declared, name), getattr(pair.new, name)
            )
            for name in ("distribution", "distribution_params")
            if getattr(pair.declared, name) != getattr(pair.new, name)
        ]
    declared, values = pair.declared, pair.observed.physical
    if (
        DISTRIBUTION in pair.overridden
        or declared.distribution is None
        or values is None
    ):
        return []
    if len(values) < 2:
        return []
    reference = _reference(
        declared.value_dtype,
        declared.bounds,
        declared.distribution,
        tuple(sorted((declared.distribution_params or {}).items())),
    )
    if reference is None:
        return []
    distance = ks_distance(values, reference)
    if distance < pair.options.distribution_tolerance:
        return []
    p_value = stats.ks_p(distance, len(values), len(reference))
    if p_value >= pair.options.significance:
        return []
    return [
        pair.finding(
            "distribution_moved",
            "compatible",
            f"values no longer follow the declared {declared.distribution} "
            f"distribution: a distance of {distance:.3f} from a sample drawn "
            f"from the declaration, over {len(values):,} value(s) "
            f"(p={p_value:.2g})",
            suffix="distribution",
            distribution=declared.distribution,
            distance=distance,
            tolerance=pair.options.distribution_tolerance,
            count=len(values),
            p_value=p_value,
        )
    ]


@functools.lru_cache(maxsize=64)
def _reference(
    dtype: pl.DataType,
    bounds: Bound | None,
    distribution: str,
    params: tuple[tuple[str, float], ...],
) -> pl.Series | None:
    """A sample of what the declaration generates, in physical units; None
    for a declaration the engine refuses."""
    from polspec.spec import ColSpec
    from polspec.tablespec import TableSpec

    column = ColSpec(
        dtype,
        bounds=bounds,
        distribution=distribution,
        distribution_params=dict(params) or None,
    )
    try:
        frame = generate(
            TableSpec("Reference", {"v": column}), REFERENCE_ROWS, seed=REFERENCE_SEED
        )
    except (PolspecError, ValueError):
        return None
    return physical(frame["v"]).cast(pl.Float64)


def _compare_field(name: str) -> Comparator:
    """A comparator for a field that shapes generation but not validation."""

    def compare(pair: Pair) -> list[DriftFinding]:
        if pair.mode == "drift":
            return []
        old, new = getattr(pair.declared, name), getattr(pair.new, name)
        if old == new:
            return []
        return [_field_changed(pair, name, old, new)]

    compare.__name__ = f"_compare_{name}"
    return compare


FIELD_COMPARATORS: dict[str, Comparator] = {
    "dtype": _compare_dtype,
    "col_name": _ignore,  # the column's key already is its name
    "nullable": _compare_nullable,
    "bounds": _compare_domain,
    "choices": _compare_domain,
    "format": _compare_domain,
    # The extras widen the format's domain: one removed narrows it, one
    # added widens it, as a choice would. A share is a claim about
    # generation alone, which drift does not judge.
    "extra_values": _compare_domain,
    "string_length": _compare_length("string_length", "length_extent"),
    "list_length": _compare_length("list_length", "list_length_extent"),
    "element_null_probability": _compare_element_nulls,
    # A pattern is not part of `Domain`: whether one regex contains another
    # is not a decision worth guessing, so a change is reported as a change.
    "pattern": _compare_field("pattern"),
    # Which name a column is seeded from cannot affect what validation accepts.
    "seed_name": _compare_field("seed_name"),
    "fields": _compare_struct_fields,
    "null_probability": _compare_null_rate,
    "nan_probability": _compare_nans,
    "unique": _compare_unique,
    "rules": _compare_rules,
    "validators": _compare_validators,
    "tags": _compare_field("tags"),
    "weights": _compare_weights,
    # One claim in two fields: the distribution and its parameters.
    "distribution": _compare_shape,
    "distribution_params": _compare_shape,
}


def compare_column(pair: Pair) -> list[DriftFinding]:
    """Every finding for one column, each comparator run once."""
    findings: list[DriftFinding] = []
    seen: set[int] = set()
    for comparator in FIELD_COMPARATORS.values():
        if id(comparator) in seen:
            continue
        seen.add(id(comparator))
        findings.extend(comparator(pair))
    return findings


# ---------------------------------------------------------------------------
# Table-level constraints
# ---------------------------------------------------------------------------

TableComparator = Callable[["TableSpec", "TableSpec"], list[DriftFinding]]


def _table_ignore(old: TableSpec, new: TableSpec) -> list[DriftFinding]:
    return []


def _table_member(
    kind: str, key: str, *, added: bool, columns: tuple[str, ...]
) -> DriftFinding:
    return DriftFinding(
        code="constraint_added" if added else "constraint_removed",
        severity="breaking" if added else "compatible",
        key=f"{kind}:{key}",
        message=f"{kind} {key!r} {'added' if added else 'removed'}"
        + ("; rows that validated before may fail" if added else ""),
        columns=columns,
        details={"kind": kind, "name": key},
    )


def _table_named(
    kind: str,
    old: Sequence[Any],
    new: Sequence[Any],
    name: Callable[[Any], str],
    columns: Callable[[Any], tuple[str, ...]],
) -> list[DriftFinding]:
    old_by, new_by = {name(o): o for o in old}, {name(n): n for n in new}
    findings = []
    for key, item in old_by.items():
        if new_by.get(key) != item:
            findings.append(
                _table_member(kind, key, added=False, columns=columns(item))
            )
    for key, item in new_by.items():
        if old_by.get(key) != item:
            findings.append(_table_member(kind, key, added=True, columns=columns(item)))
    return findings


def _compare_checks(old: TableSpec, new: TableSpec) -> list[DriftFinding]:
    return _table_named(
        "check",
        old.checks,
        new.checks,
        lambda c: c.name,
        lambda c: tuple(c.expr.meta.root_names()),
    )


def _compare_unique_together(old: TableSpec, new: TableSpec) -> list[DriftFinding]:
    return _table_named(
        "unique_together",
        [tuple(g) for g in old.unique_together],
        [tuple(g) for g in new.unique_together],
        lambda g: ", ".join(g),
        lambda g: tuple(g),
    )


def _compare_foreign_keys(old: TableSpec, new: TableSpec) -> list[DriftFinding]:
    def identity(fk: Any) -> tuple:
        return (fk.name, fk.columns, fk.references, fk.ref_columns)

    return _table_named(
        "foreign_key",
        [identity(fk) for fk in old.foreign_keys],
        [identity(fk) for fk in new.foreign_keys],
        lambda fk: fk[0],
        lambda fk: fk[1],
    )


def _compare_hierarchy(old: TableSpec, new: TableSpec) -> list[DriftFinding]:
    return _table_named(
        "hierarchy",
        [old.hierarchy] if old.hierarchy is not None else [],
        [new.hierarchy] if new.hierarchy is not None else [],
        lambda h: f"{h.child}->{h.parent}",
        lambda h: (h.child, h.parent),
    )


TABLE_COMPARATORS: dict[str, TableComparator] = {
    "name": _table_ignore,  # a spec's name is what the report is *about*
    "columns": _table_ignore,  # compared column by column by the runner
    "checks": _compare_checks,
    "unique_together": _compare_unique_together,
    "foreign_keys": _compare_foreign_keys,
    "hierarchy": _compare_hierarchy,
}


def compare_table(old: TableSpec, new: TableSpec) -> list[DriftFinding]:
    """Every table-level finding between two specs."""
    findings: list[DriftFinding] = []
    for comparator in TABLE_COMPARATORS.values():
        findings.extend(comparator(old, new))
    return findings
