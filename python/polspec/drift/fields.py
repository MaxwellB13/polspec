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
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import polars as pl

from polspec.constraints import Domain
from polspec.drift.data import Observed
from polspec.drift.report import DriftFinding, Severity
from polspec.dtypes import _bound_endpoint_to_physical
from polspec.formats import lookup as _lookup_format
from polspec.validation.constraints import _is_dtype_compatible

if TYPE_CHECKING:
    from polspec.bound import Bound
    from polspec.check import Check
    from polspec.drift import DriftOptions
    from polspec.spec import ColSpec
    from polspec.tablespec import TableSpec

MAX_LISTED = 8


@dataclass(frozen=True, slots=True)
class Pair:
    """One column, on both sides of a comparison.

    `declared` is always a `ColSpec`. `other` is a `ColSpec` when two specs
    are being diffed and an `Observed` when a spec is being held against a
    frame; `mode` says which.
    """

    column: str
    declared: ColSpec
    other: ColSpec | Observed
    options: DriftOptions

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
            key=f"{self.column}__{suffix}",
            message=f"Column '{self.column}': {message}",
            columns=(self.column,),
            details=details,
        )


Comparator = Callable[[Pair], list[DriftFinding]]


# `ColSpec` annotates what its constructor accepts; `__post_init__` narrows
# every field to one form. These read the narrowed form, so this module is
# type-checked against what a constructed ColSpec actually holds.


def _dtype(spec: ColSpec) -> pl.DataType:
    return cast("pl.DataType", spec.dtype)


def _bound(value: Any) -> Bound | None:
    return cast("Bound | None", value)


def _closed(bound: Bound) -> tuple[Any, Any]:
    """Both endpoints of a bound that has both, as `string_length` does."""
    return bound.min, bound.max


def _validators(spec: ColSpec) -> tuple[Check, ...]:
    return cast("tuple[Check, ...]", spec.validators)


def _listed(values: Sequence[Any]) -> str:
    shown = [repr(v) for v in values[:MAX_LISTED]]
    if len(values) > MAX_LISTED:
        shown.append(f"... {len(values) - MAX_LISTED} more")
    return f"[{', '.join(shown)}]"


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
        actual, expected = _dtype(pair.declared), _dtype(pair.new)
    else:
        actual, expected = pair.observed.dtype, _dtype(pair.declared)
    if _same_dtype(actual, expected):
        return []
    compatible = _is_dtype_compatible(
        expected, actual, strict=pair.options.strict_dtypes
    )
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
            _bound_endpoint_to_physical(value, dtype) if dtype.is_temporal() else value
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
    declared, found = _bound(pair.declared.bounds), pair.observed.extent
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
                f"{count} row(s) hold values outside {domain}: {_listed(samples)}. "
                "Add them to the declaration, or fix the source",
                suffix="values",
                count=count,
                values=list(samples),
            )
        )
    if pair.options.unseen_values and pair.observed.unseen:
        unseen = pair.observed.unseen
        findings.append(
            pair.finding(
                "cardinality_moved",
                "compatible",
                f"{len(unseen)} of {len(domain.values)} declared value(s) never "
                f"appear: {_listed(unseen)}",
                suffix="unseen",
                unseen=list(unseen),
                declared=len(domain.values),
                observed=len(domain.values) - len(unseen),
            )
        )
    return findings


def _observed_format(pair: Pair) -> list[DriftFinding]:
    count, samples = pair.observed.format_failures
    if not count:
        return []
    return [
        pair.finding(
            "format_violated",
            "breaking",
            f"{count} row(s) are not {_lookup_format(pair.declared.format or '')}: "
            f"{_listed(samples)}. "
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
    (old_min, old_max), (new_min, new_max) = _closed(old), _closed(new)
    widened = new_min < old_min or new_max > old_max
    narrowed = new_min > old_min or new_max < old_max
    if widened and narrowed:
        return "changed"
    return "widened" if widened else "narrowed"


def _compare_string_length(pair: Pair) -> list[DriftFinding]:
    if pair.mode == "diff":
        old = _bound(pair.declared.string_length)
        new = _bound(pair.new.string_length)
        relation = _length_relation(old, new)
        if relation not in _RELATION_FINDINGS:
            return []
        code, severity, consequence = _RELATION_FINDINGS[relation]
        return [
            pair.finding(
                code,
                severity,
                f"string_length {relation} from {old or 'unconstrained'} to "
                f"{new or 'unconstrained'}; {consequence}",
                suffix="string_length",
                field="string_length",
                old=[old.min, old.max] if old else None,
                new=[new.min, new.max] if new else None,
            )
        ]
    declared = _bound(pair.declared.string_length)
    found = pair.observed.length_extent
    if declared is None or found is None:
        return []
    facts = _exceeded("string_length", declared, found, pl.Int64())
    if not facts:
        return []
    return [
        pair.finding(
            "bounds_exceeded",
            "breaking",
            f"lengths escape string_length {declared}: {_describe_excess(facts)}. "
            "Widen the length, or fix the source",
            suffix="string_length",
            **facts,
        )
    ]


# ---------------------------------------------------------------------------
# Rates, uniqueness, rules, validators, and the fields that only describe
# generation
# ---------------------------------------------------------------------------


def _compare_null_rate(pair: Pair) -> list[DriftFinding]:
    if pair.mode == "diff":
        old, new = pair.declared.null_probability, pair.new.null_probability
        if old == new or not (pair.declared.nullable and pair.new.nullable):
            return []  # a rate on a non-nullable column means nothing
        return [_field_changed(pair, "null_probability", old, new)]
    if not pair.declared.nullable or pair.observed.null_rate is None:
        return []
    declared, observed = pair.declared.null_probability, pair.observed.null_rate
    if abs(observed - declared) <= pair.options.null_rate_tolerance:
        return []
    return [
        pair.finding(
            "null_rate_moved",
            "compatible",
            f"{observed:.1%} null, declared null_probability={declared}; "
            f"beyond the {pair.options.null_rate_tolerance:.0%} tolerance",
            suffix="null_rate",
            declared=declared,
            observed=observed,
            tolerance=pair.options.null_rate_tolerance,
        )
    ]


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
        key=f"{pair.column}__{kind}:{key}",
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
        _validators(pair.declared),
        _validators(pair.new),
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
    "string_length": _compare_string_length,
    # A pattern is not part of `Domain`: whether one regex contains another
    # is not a decision worth guessing, so a change is reported as a change.
    "pattern": _compare_field("pattern"),
    # Which name a column is seeded from cannot affect what validation accepts.
    "seed_name": _compare_field("seed_name"),
    "null_probability": _compare_null_rate,
    "unique": _compare_unique,
    "rules": _compare_rules,
    "validators": _compare_validators,
    "tags": _compare_field("tags"),
    "weights": _compare_field("weights"),
    "distribution": _compare_field("distribution"),
    "distribution_params": _compare_field("distribution_params"),
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
