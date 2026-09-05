"""The order the post-generation passes run in.

The engine fills every column independently; rules and foreign keys then
rewrite some of them. Which pass must run before which follows from what
each one reads and writes -- a rule keyed on a foreign-keyed column has to
see the parent's values, not the freely generated ones it replaced -- and
that ordering is what makes generated data satisfy the same declarations
validation checks it against.

The graph is decided by the spec alone, so a spec whose passes cannot be
ordered is rejected at declaration rather than silently generating data that
fails its own round-trip.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from polspec.errors import SpecError

if TYPE_CHECKING:
    from polspec.tablespec import TableSpec


@dataclass(frozen=True, slots=True)
class Pass:
    """One rewrite of an already-generated frame.

    `key` identifies it (`rules:total`, `fk:fk_customer_id__Customers`),
    `label` names it in an error, and `reads`/`writes` are the columns it
    consumes and replaces.

    A pass may list the same column in both. Reading what it writes is not a
    dependency on *itself* -- it reads that column's pre-pass values -- but it
    is a dependency on whatever else writes that column, which is exactly what
    a composite-key repair needs: it has to see the values the rules and
    foreign keys settled on.
    """

    key: str
    label: str
    writes: frozenset[str]
    reads: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "writes", frozenset(self.writes))
        object.__setattr__(self, "reads", frozenset(self.reads))


def passes_of(spec: TableSpec) -> list[Pass]:
    """Every post-generation pass `spec` implies, in declaration order.

    Column rules come first, then foreign keys, which is only the order they
    are declared in -- `order` decides the order they run in.
    """
    passes: list[Pass] = []
    for name, col in spec.columns.items():
        if not col.rules:
            continue
        reads = {n for rule in col.rules for n in rule.when.root_names()}
        passes.append(
            Pass(
                key=f"rules:{name}",
                label=f"the rules on {name!r}",
                writes=frozenset({name}),
                reads=frozenset(reads),
            )
        )
    for fk in spec.foreign_keys:
        # A key against another spec draws from that spec's frame, so it
        # reads nothing here; a self-referencing one reads this frame.
        reads = set(fk.ref_columns) if fk.references == "self" else set()
        passes.append(
            Pass(
                key=f"fk:{fk.name}",
                label=f"the foreign key {fk.name!r}",
                writes=frozenset(fk.columns),
                reads=frozenset(reads),
            )
        )
    for index, group in enumerate(spec.unique_together):
        members = tuple(group)
        if any(spec.columns[m].unique for m in members):
            continue  # one distinct member already makes the combination distinct
        passes.append(
            Pass(
                key=f"unique_together:{index}",
                label=f"the composite key {list(members)}",
                writes=rewritable_members(spec, members),
                reads=frozenset(members),
            )
        )
    return passes


def rewritable_members(spec: TableSpec, members: Sequence[str]) -> frozenset[str]:
    """The columns of a composite key a repair may resample.

    A foreign-keyed column is off limits: replacing its value would break the
    referential integrity the key just established. What is left is what the
    repair has to work with, and an empty set means it has nothing.
    """
    keyed = {column for fk in spec.foreign_keys for column in fk.columns}
    return frozenset(m for m in members if m not in keyed)


def order(passes: list[Pass], spec_name: str) -> list[Pass]:
    """`passes` sorted so every pass runs after the ones it reads from.

    Ties keep declaration order, so the sequence is stable. A cycle -- two
    passes each waiting on what the other writes -- raises `SpecError`,
    because no order satisfies both and generated data would fail its own
    validation whichever way round they ran.
    """
    if len(passes) < 2:
        return list(passes)

    writer_of: dict[str, list[int]] = {}
    for index, p in enumerate(passes):
        for column in p.writes:
            writer_of.setdefault(column, []).append(index)

    # dependencies[i] is every pass i must wait for.
    dependencies = [
        {w for column in p.reads for w in writer_of.get(column, ()) if w != index}
        for index, p in enumerate(passes)
    ]

    ordered: list[Pass] = []
    done: set[int] = set()
    while len(ordered) < len(passes):
        ready = [
            index
            for index in range(len(passes))
            if index not in done and dependencies[index] <= done
        ]
        if not ready:
            stuck = sorted(
                passes[index].label for index in range(len(passes)) if index not in done
            )
            raise SpecError(
                f"Cannot order the generation passes on {spec_name}: "
                f"{', '.join(stuck)} each depend on a column another of them "
                "rewrites, so no order lets them all see their inputs. Break "
                "the cycle by keying one of them on a column that nothing else "
                "rewrites."
            )
        ordered.append(passes[ready[0]])
        done.add(ready[0])
    return ordered


def ordered_passes(spec: TableSpec) -> list[Pass]:
    """`spec`'s post-generation passes, in the order they must run."""
    return order(passes_of(spec), spec.name)
