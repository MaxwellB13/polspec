"""The generation pipeline: a frame from the engine, then the passes.

`generate` is the whole of it: the Rust engine fills every column
independently, then the passes in `polspec.generation.passes` rewrite the
finished frame, in the order `polspec.pass_order` derives from what each
reads and writes. `generate_batches` runs the same pipeline window by
window; `scan` and the sinks are built on it.

Internal: not part of the public API.
"""

from __future__ import annotations

import dataclasses
import difflib
import random
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl

from polspec import frames
from polspec._ffi import chunk_size
from polspec.constants import DEFAULT_WIDE_INT_BOUND, LARGE_FRAME_BYTES
from polspec.dtypes import map_parts, unholdable
from polspec.errors import GenerationError, SpecError
from polspec.foreign_key import ForeignKey
from polspec.frames import Method, References, to_eager
from polspec.generation.engine import generate_cartesian, generate_random, readable_at
from polspec.generation.keeping import (
    KEY_ROOM,
    kept_or_warned,
    resolve_checks,
)
from polspec.generation.passes.derived import apply_derived
from polspec.generation.passes.foreign_keys import (
    apply_foreign_key,
    apply_windowed_self_key,
    parent_keys_of,
    unique_parent_shortfall,
)
from polspec.generation.passes.hierarchy import (
    apply_hierarchy,
    apply_windowed_hierarchy,
)
from polspec.generation.passes.rules import apply_column_rules
from polspec.generation.passes.unique_together import apply_unique_together
from polspec.generation.seeds import pass_seed
from polspec.pass_order import ordered_passes, passes_of, rewritable_members
from polspec.sizing import estimated_size
from polspec.spec import ColSpec, claims_refused
from polspec.tablespec import (
    SpecLike,
    TableSpec,
    as_table_spec,
    require_columns,
    resolve_references,
)

__all__ = [
    "check_counts",
    "check_method",
    "describe_bytes",
    "generate",
    "generate_batches",
    "requires_whole_frame",
]


def check_faults(spec: TableSpec, cycles: int, self_references: int) -> None:
    """Faults are hierarchy-shaped, so a spec without one cannot take them."""
    if cycles < 0 or self_references < 0:
        raise ValueError("cycles and self_references must be >= 0")
    if (cycles or self_references) and spec.hierarchy is None:
        raise SpecError(
            f"{spec.name} declares no __hierarchy__, so there are no links to "
            "damage. cycles= and self_references= describe a parent/child edge "
            "list; declare a Hierarchy, or drop the arguments."
        )


def requires_whole_frame(spec: TableSpec, verb: str) -> None:
    """Refuses the streaming verbs for a spec whose shape spans the frame.

    A hierarchy laid out whole (`layout="whole"`, 2.0's builder) is built
    over every row at once, so batches made apart would be a pile of
    unrelated fragments rather than a shallow tree, which is a worse answer
    than saying no. The windowed layout, the default since 2.1, draws each
    row's place from its position and streams. An order needs every row
    before the first is written.
    """
    if spec.hierarchy is not None and spec.hierarchy.layout == "whole":
        raise SpecError(
            f"{spec.name} declares a __hierarchy__ with layout='whole', which "
            f"{verb} cannot produce: the forest is built over the whole frame, "
            "and each batch is generated on its own. Declare the default, "
            "layout='windowed', to stream it, or use generate() and write the "
            "frame out yourself."
        )
    if spec.order_by:
        # Rows in order across a stream would need every row before the
        # first could be written -- and an order within each key, every row
        # of that key, which batches scatter.
        raise SpecError(
            f"{spec.name} declares order_by {list(spec.order_by)}, which {verb} "
            "cannot produce: the rows' order is a property of the whole frame, "
            "and each batch is generated on its own. Use generate() for an "
            "ordered frame, and write it out yourself."
        )


# The integer dtypes whose undeclared range is the default, not the dtype's.
_WIDE_INTS = (pl.Int64, pl.UInt64, pl.Int128, pl.UInt128)


def room_for_open_keys(spec: TableSpec, n: int) -> TableSpec:
    """`spec` with each unique 64- or 128-bit integer column that declares no
    bounds given room for `n` distinct values.

    With no bounds, such a column draws from the default range,
    +-1,000,000, which holds 2,000,001 values -- a refusal for a key column
    of a larger frame, in a dtype that holds 9e18. The range widens upward
    from the default's lower end until it holds `n`. Only where the default
    is too small, so a frame that generates today generates the same; a
    column that declares bounds keeps them, as they are a claim.
    """
    widened: dict[str, ColSpec] = {}
    for name, column in spec.columns.items():
        if (
            not column.unique
            or column.bounds is not None
            or column.choices is not None
            or column.format is not None
            or column.dtype not in _WIDE_INTS
        ):
            continue
        low = 0 if column.dtype.is_unsigned_integer() else -DEFAULT_WIDE_INT_BOUND
        if DEFAULT_WIDE_INT_BOUND - low + 1 >= n:
            continue
        widened[name] = dataclasses.replace(column, bounds=(low, low + n - 1))
    return spec.with_columns(widened) if widened else spec


def room_keeping_checks(spec: TableSpec, n: int) -> TableSpec:
    """`room_for_open_keys` for a frame whose checks are kept, which may read
    more than `n` rows. Where the default range holds `n`, it is left as
    `checks=False` leaves it, so a frame whose rows all pass is that one;
    where it does not, and would hold exactly `n`, a key is given room for
    the rows a check rejects."""
    widened = room_for_open_keys(spec, n)
    return widened if widened is spec else room_for_open_keys(spec, n * KEY_ROOM)


def check_counts(n: int, batch_size: int | None = None) -> None:
    if n < 0:
        raise ValueError("n must be >= 0")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be > 0")


def check_method(method: str) -> None:
    """Refuses a `method=` that is neither of the two there are."""
    if method not in ("random", "cartesian"):
        raise ValueError(f"Unknown method {method!r}; expected 'random' or 'cartesian'")


def _warn_unused_references(spec: TableSpec, parents: dict[str, Any]) -> None:
    """Warns when a supplied parent went unused *and* a key went unfilled.

    A foreign key whose target has no data is left exactly as freely
    generated, which is documented and deliberate -- `references` is optional.
    What is not deliberate is supplying a parent under a name nothing asked
    for: the caller believes that key was satisfied, generation quietly did
    not satisfy it, and `validate()` then reports the key as unresolved. A
    misspelled spec name is the usual cause.

    Both halves have to hold before this says anything. A `Registry` hands
    every spec the whole set of frames generated so far, most of which any one
    spec has no key for, so an unused parent on its own is ordinary.
    """
    if not parents:
        return
    targets = {fk.references for fk in spec.foreign_keys if fk.references != "self"}
    unfilled = sorted(targets - parents.keys())
    unused = sorted(parents.keys() - targets)
    if not unfilled or not unused:
        return
    hints = []
    for name in unfilled:
        close = difflib.get_close_matches(name, unused, n=1)
        hints.append(f"{name!r}{f' (supplied {close[0]!r}?)' if close else ''}")
    warnings.warn(
        f"{spec.name}: references={{...}} supplied {unused} that no foreign key "
        f"points at, while {', '.join(hints)} went unfilled. Those columns are "
        "generated freely, so validate() will report the key as unresolved. "
        f"{spec.name} references: {sorted(targets)}.",
        stacklevel=3,
    )


def generate(
    spec: SpecLike,
    n: int,
    *,
    method: Method = "random",
    seed: int | None = None,
    references: References = None,
    cycles: int = 0,
    self_references: int = 0,
    max_bytes: int | None = None,
    fixed: Mapping[str, Any] | None = None,
    checks: bool | None = None,
) -> pl.DataFrame:
    """Generates a DataFrame matching `spec`.

    method="random" (default): `n` rows, each column drawn independently.

    method="cartesian": guarantees a minimum level of coverage. Builds the
    cartesian product of every Enum/Boolean column's full set of values,
    crossed with the negative/zero/positive/null partitions of every bounded
    numeric column, so every enum combination appears alongside every numeric
    sign/null case. `n` is then a *minimum*: if that coverage set has fewer
    than `n` rows it is padded with random rows; if it has more, all of it is
    kept.

    `ColSpec.rules` and any `ForeignKey` the spec declares are then applied as
    vectorised passes over the generated frame, regardless of method. Each
    pass sees the frame the passes before it produced, and they run in the
    order their reads and writes imply -- a rule keyed on a foreign-keyed
    column reads the parent's values, not the freely generated ones they
    replaced -- so the result satisfies the same declarations `validate`
    checks it against.

    A foreign key is only made referentially consistent where data for its
    target is available: self-referencing keys always are, sampled from this
    same frame; a key referencing another spec only is if `references`
    carries an entry for it, keyed by the spec, its class, or its name --
    otherwise that column is left exactly as freely generated. Composite keys
    are sampled as one joint pick per row; a single-column key whose ColSpec
    is `unique=True` samples without replacement when the parent has enough
    distinct rows to cover `n`.

    A `unique=True` column is drawn without replacement by the engine itself,
    and a `__unique_together__` group afterwards: as one permutation of its
    members' combinations when each has a finite or bounded space, or by
    resampling the rows that repeat a combination. Either refuses, naming the
    column or the group, when the domain is too small to cover `n`.

    A spec declaring a `Hierarchy` has its two link columns rewritten as a
    forest of the declared depth. `cycles` and `self_references` then damage it
    on purpose -- closing that many chains into loops, and pointing that many
    rows at themselves -- which is how a graph walk gets something to fail
    against. Both default to zero, and `validate()` reports whatever they
    injected.

    Before anything is allocated the frame's size is estimated from the
    declaration. Past four gibibytes that is a warning naming the estimate;
    `max_bytes=` makes it a refusal instead, and `max_bytes=0` silences both.

    A spec declaring `order_by` has its rows sorted by those columns last,
    after every pass: each value keeps its row, so every claim the rows make
    about their own columns still holds. Nulls come last.

    `fixed` pins columns to values given: `{"status": ["open", "closed"],
    "region": "EU"}`, a scalar for every row or a sequence or Series of
    exactly `n`, cast strictly to the column's dtype. Every other column is
    what it would have been, since each is seeded by its own name; a rule
    reading a pinned column follows the pin. A pin must be values its
    column accepts, and a column a pass writes -- a rule's, a foreign
    key's, a hierarchy's, a composite key's -- cannot be pinned. With
    `order_by`, pinned rows move with the sort like any other.
    `method="cartesian"`, whose row count is not `n`, takes no pins.

    `checks` keeps the spec's `__checks__` and column validators: the frame
    is drawn as always, the rows every check passes are kept, and when fewer
    than `n` pass, more of the same frame is read -- only the rows past
    those already read -- and its first `n` passing rows taken. A spec whose
    rows all pass is the frame `checks=False` gives. Rates a check reads --
    a null rate, a weight -- become rates among the rows that pass.

    Some checks cannot be kept: beside a hierarchy or a key into the spec's
    own rows, under `method="cartesian"`, with a pin of all `n` values, and
    a check that rejects nearly every row. The default (`checks=None`) then
    generates the frame `checks=False` gives and warns, naming why;
    `checks=True` refuses instead -- a `SpecError`, or a `GenerationError`
    naming the check. `checks=False` leaves them to validation. *Kept by
    default since 2.0; added in 1.7.*

    The whole frame is built before this returns. `scan()` is the lazy
    verb: it generates as the plan is collected, so only the columns and
    rows a plan asks for are made.

    Examples
    --------
    >>> spec = TableSpec("Orders", {
    ...     "id": ColSpec(pl.Int64, unique=True),
    ...     "total": ColSpec(pl.Float64, bounds=(0, 100)),
    ... })
    >>> frame = generate(spec, 1_000, seed=1)
    >>> frame.shape
    (1000, 2)
    >>> frame["id"].is_unique().all()
    True
    """
    spec = as_table_spec(spec)
    require_columns(spec)
    check_counts(n)
    check_faults(spec, cycles, self_references)
    _check_size(spec, n, max_bytes)
    check_method(method)
    checked = checked_pins(spec, fixed, n, method) if fixed else None
    keeping = resolve_checks(
        spec,
        checks,
        method,
        sequence_pins=checked is not None and checked.has_sequences,
    )
    frame_seed = _frame_seed(seed)

    def reader(planned: TableSpec) -> Callable[[int, int], pl.DataFrame]:
        def read(offset: int, rows: int) -> pl.DataFrame:
            # Rows `[offset, offset + rows)` of the one frame the seed
            # describes: a longer read is the same rows and more.
            return _read(planned, offset, rows)

        return read

    def _read(planned: TableSpec, offset: int, rows: int) -> pl.DataFrame:
        return _window(
            planned,
            rows,
            frame_seed,
            method=method,
            references=references,
            row_offset=offset,
            cycles=cycles,
            self_references=self_references,
            pins=checked.window(offset, rows) if checked is not None else {},
            frame_rows=n,
        )

    if keeping:
        planned = room_keeping_checks(spec, n)
        kept = kept_or_warned(planned, n, 0, reader(planned), strict=checks is True)
        if kept is not None:
            return ordered(kept[0], planned)
    # Not kept -- by choice, or because they cannot be: the frame
    # `checks=False` gives.
    planned = room_for_open_keys(spec, n)
    return ordered(_read(planned, 0, n), planned)


@dataclass(frozen=True)
class Pins:
    """`fixed=` checked: a scalar pin as its one value, a sequence as all `n`
    of them, each the column's dtype. `window` hands any stretch of rows its
    pins, so a batch costs a scalar pin nothing however large `n` is, and a
    sequence pin's batches are the rows `generate` would give it."""

    values: Mapping[str, pl.Series]
    scalars: frozenset[str]

    def window(self, offset: int, length: int) -> dict[str, pl.Series]:
        return {
            name: (
                values.new_from_index(0, length)
                if name in self.scalars
                else values.slice(offset, length)
            )
            for name, values in self.values.items()
        }

    @property
    def has_sequences(self) -> bool:
        """Whether a pin gives a column all `n` values, not one."""
        return bool(set(self.values) - self.scalars)

    def restricted(self, columns: Iterable[str]) -> Pins:
        """The pins of `columns` only: the rest are not drawn."""
        keep = set(columns)
        return Pins(
            {k: v for k, v in self.values.items() if k in keep},
            self.scalars & keep,
        )


def checked_pins(
    spec: TableSpec, fixed: Mapping[str, Any], n: int, method: Method
) -> Pins:
    """`fixed` as one Series per column, each the column's dtype -- one value
    for a scalar, `n` for a sequence -- or a `GenerationError` naming the
    column that cannot be pinned.

    A pin is held to the promise `generate` makes, that its output
    validates: values the column refuses are refused here, and a column a
    pass rewrites would either overwrite the pin or break the claim the
    pass exists to keep.
    """
    if not isinstance(fixed, Mapping):
        raise GenerationError(
            f"fixed= must map column names to values, got {type(fixed).__name__}"
        )
    if method == "cartesian":
        raise GenerationError(
            "fixed= does not combine with method='cartesian': its row count is "
            "the coverage it needs, not n, so a pin's length cannot be known."
        )
    writers = {column: p.label for p in passes_of(spec) for column in sorted(p.writes)}
    pins: dict[str, pl.Series] = {}
    scalars: set[str] = set()
    for name, given in fixed.items():
        if name not in spec.columns:
            close = difflib.get_close_matches(str(name), list(spec.columns), n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise GenerationError(
                f"fixed= names {name!r}, which {spec.name} does not declare.{hint}"
            )
        if name in writers:
            raise GenerationError(
                f"fixed= cannot pin {name!r}: {writers[name]} writes it after "
                "the columns are drawn, and would overwrite the pin or break "
                "for it. Pin a column it reads instead."
            )
        declared = spec.columns[name]
        dtype = declared.dtype
        if (
            isinstance(dtype, (pl.List, pl.Array, pl.Struct))
            or map_parts(dtype) is not None
        ):
            raise GenerationError(
                f"fixed= cannot pin {name!r}: a {dtype!r} column cannot be "
                "pinned yet. Pin a scalar column."
            )
        values, scalar = _pin_values(name, given, n, dtype)
        broken = claims_refused(declared, values)
        if broken:
            raise GenerationError(
                f"fixed= pins {name!r} to values it does not accept: they break "
                f"its {', '.join(broken)}. generate() makes a frame its spec "
                "accepts, so a pin has to be one too."
            )
        pins[name] = values
        if scalar:
            scalars.add(name)
    return Pins(pins, frozenset(scalars))


def _pin_values(
    name: str, given: Any, n: int, dtype: pl.DataType
) -> tuple[pl.Series, bool]:
    """One pin as values of `dtype` -- its one value, for a scalar, and
    whether it is one -- cast strictly: a value the dtype holds only as
    another -- `1.5` on an `Int64` -- is refused."""
    scalar = isinstance(given, (str, bytes)) or not isinstance(
        given, (Sequence, pl.Series)
    )
    values = [given] if scalar else list(given)
    if not scalar and len(values) != n:
        raise GenerationError(
            f"fixed= gives {name!r} {len(values)} value(s) for {n} row(s): give "
            "one value for every row, or a sequence of exactly n."
        )
    try:
        series = pl.Series(name, values, dtype=dtype, strict=True)
    except (TypeError, ValueError, pl.exceptions.PolarsError) as exc:
        raise GenerationError(
            f"fixed= gives {name!r} values {dtype!r} cannot hold: "
            f"{str(exc).splitlines()[0]}"
        ) from exc
    offending = unholdable([v for v in values if v is not None], dtype)
    if offending:
        raise GenerationError(
            f"fixed= gives {name!r} {offending[:5]}, which {dtype!r} would hold "
            "only as different values."
        )
    return series, scalar


def ordered(frame: pl.DataFrame, spec: TableSpec) -> pl.DataFrame:
    """`frame`'s rows sorted as `spec.order_by` declares: whole rows, so
    every value stays beside the others in its row. Stable, so rows that
    tie keep the order they were generated in."""
    if not spec.order_by:
        return frame
    return frame.sort(
        frames.columns(spec.order_by),
        descending=list(spec.descending),
        nulls_last=True,
        maintain_order=True,
    )


def describe_bytes(size: int) -> str:
    """A byte count as the unit a reader thinks in."""
    if size < 1024:
        return f"{size:,} bytes"
    scaled = float(size)
    for unit in ("KiB", "MiB", "GiB"):
        scaled /= 1024
        if scaled < 1024 or unit == "GiB":
            return f"{scaled:,.1f} {unit}"
    raise AssertionError  # pragma: no cover - the loop returns at GiB


def _check_size(spec: TableSpec, n: int, max_bytes: int | None) -> None:
    """Says how large the frame will be before it is allocated.

    The first sign that a spec is too big for the machine is usually the
    machine swapping, and the estimate costs nothing -- it is read off the
    declaration. A warning rather than a refusal by default, because a ten
    gigabyte frame on a machine with sixty-four is a reasonable thing to
    ask for; `max_bytes` is for a caller who wants it refused, and
    `max_bytes=0` for one who wants neither.
    """
    if max_bytes == 0:
        return
    estimate = estimated_size(spec, n)
    limit = LARGE_FRAME_BYTES if max_bytes is None else max_bytes
    if estimate <= limit:
        return
    advice = (
        "scan() generates it as it is collected, and generate_batches() "
        "in batches; either holds a fraction of that at once."
    )
    if max_bytes is not None:
        raise GenerationError(
            f"{spec.name} at {n:,} rows is an estimated "
            f"{describe_bytes(estimate)}, over the max_bytes of "
            f"{describe_bytes(max_bytes)}. {advice}"
        )
    warnings.warn(
        f"{spec.name} at {n:,} rows is an estimated {describe_bytes(estimate)}, "
        f"which generate() allocates before it returns. {advice} Pass "
        "max_bytes=0 to silence this.",
        stacklevel=3,
    )


def _frame_seed(seed: int | None) -> int:
    """The frame seed: the first draw of the caller's seed, as it always was,
    so every column's values are what they were. Everything else -- each
    column, each pass, each batch -- derives its own seed from this one and
    a stable key."""
    return random.Random(seed).randrange(2**63)


def _window(
    spec: TableSpec,
    n: int,
    frame_seed: int,
    *,
    method: Method,
    references: References,
    row_offset: int = 0,
    cycles: int = 0,
    self_references: int = 0,
    pins: Mapping[str, pl.Series] | None = None,
    parent_keys: Mapping[str, pl.DataFrame] | None = None,
    drawn: pl.DataFrame | None = None,
    frame_rows: int | None = None,
) -> pl.DataFrame:
    """Rows `[row_offset, row_offset + n)` of the frame `frame_seed` describes,
    with the passes run over them. `frame_rows` is how many rows the whole
    frame holds, which a windowed hierarchy lays its forest out over; `n`,
    when not given. `drawn` is those rows' columns as the
    engine fills them, when a caller drew them already (`_Blocks`).

    `pins` replace their columns after the draw and before the passes, so
    a pass reading a pinned column reads the pin. `parent_keys` are each
    foreign key's parent keys, by the key's name, read once by a caller
    drawing several windows from the same parents.

    The columns are a window onto one frame: the engine numbers its chunks
    from the offset, so the same rows come back whatever `n` the call asks
    for. The passes are drawn per window, keyed by the offset, so two
    windows never repeat each other's draws -- except a unique foreign key,
    which is a permutation of its parent's keys keyed by the frame, so a
    window takes the keys the whole frame takes there.
    """
    columns = dict(spec.columns)
    if drawn is not None:
        df = drawn
    elif method == "cartesian":
        df = generate_cartesian(columns, n, frame_seed)
    else:
        df = generate_random(columns, n, frame_seed, row_offset)
    if pins:
        df = df.with_columns(pins.values())
    pass_seed_ = (
        frame_seed if row_offset == 0 else pass_seed(frame_seed, f"window:{row_offset}")
    )
    return _run_passes(
        spec,
        columns,
        df,
        references,
        pass_seed_,
        row_offset=row_offset,
        whole_seed=frame_seed,
        cycles=cycles,
        self_references=self_references,
        parent_keys=parent_keys,
        cartesian=method == "cartesian",
        frame_rows=frame_rows,
        pinned=frozenset(pins or ()),
    )


def windowed_self_key(
    spec: TableSpec,
    fk: ForeignKey,
    *,
    cartesian: bool = False,
    pinned: frozenset[str] = frozenset(),
) -> bool:
    """Whether a key into `spec`'s own rows is drawn from the whole frame
    (`apply_windowed_self_key`): a one-column key whose referenced column is
    one the engine draws and nothing changes after -- `readable_at` reads
    it, no pass rewrites it, no pin replaces it, and the frame is not
    cartesian. Any other self key draws from its window's own rows."""
    if cartesian or len(fk.columns) != 1:
        return False
    ref = fk.ref_columns[0]
    if ref in pinned or not readable_at(spec.columns[ref]):
        return False
    return not any(ref in p.writes for p in passes_of(spec))


def _run_passes(
    spec: TableSpec,
    columns: dict[str, ColSpec],
    df: pl.DataFrame,
    references: References,
    frame_seed: int,
    *,
    row_offset: int = 0,
    whole_seed: int | None = None,
    cycles: int = 0,
    self_references: int = 0,
    parent_keys: Mapping[str, pl.DataFrame] | None = None,
    cartesian: bool = False,
    frame_rows: int | None = None,
    pinned: frozenset[str] = frozenset(),
) -> pl.DataFrame:
    """Applies every rule and foreign-key pass, in dependency order.

    Each pass is seeded from the frame seed and a key naming what it is for
    -- a rules column's seed name, a foreign key's name, a composite key's
    members -- so neither the order the passes run in nor the columns
    declared around them change the values any one of them samples. A
    column inserted ahead of a rules column leaves it alone; a rules column
    renamed with `seed_name` keeps its rule's draw as well as its own.

    `frame_seed` seeds this window's passes; `whole_seed` is the seed of the
    frame the window belongs to and `row_offset` where the window starts,
    which a unique foreign key reads so that it is a window onto the whole
    frame's draw.
    """
    runners: dict[str, Callable[[pl.DataFrame], pl.DataFrame]] = {}

    # Rules and a foreign key into another table are windows: seeded by the
    # whole frame and drawn at the window's rows, so a batch holds what the
    # frame holds there.
    frame_key = whole_seed if whole_seed is not None else frame_seed
    for name, col in columns.items():
        if not col.rules:
            continue
        seed = pass_seed(frame_key, f"rules:{col.seed_name or name}")
        runners[f"rules:{name}"] = lambda frame, name=name, col=col, seed=seed: (
            apply_column_rules(frame, name, col, seed, row_offset=row_offset)
        )

    for name, col in columns.items():
        if col.derive is not None:
            # Computed, not drawn: no seed.
            runners[f"derive:{name}"] = lambda frame, name=name, col=col: apply_derived(
                frame, name, col
            )

    if (hierarchy := spec.hierarchy) is not None and hierarchy.layout == "whole":
        seed = pass_seed(frame_seed, "hierarchy")
        runners["hierarchy"] = lambda frame, seed=seed: apply_hierarchy(
            frame,
            columns,
            hierarchy,
            seed,
            cycles=cycles,
            self_references=self_references,
        )
    elif hierarchy is not None:
        # Keyed by the whole frame, and laid out over all of its rows, so a
        # window holds the forest's rows the whole frame holds there.
        seed = pass_seed(frame_key, "hierarchy")
        runners["hierarchy"] = lambda frame, seed=seed: apply_windowed_hierarchy(
            frame,
            columns,
            hierarchy,
            seed,
            frame_rows=frame_rows if frame_rows is not None else frame.height,
            row_offset=row_offset,
            cycles=cycles,
            self_references=self_references,
        )

    if spec.foreign_keys:
        parents = resolve_references(references, to_eager)
        _warn_unused_references(spec, parents)
        for fk in spec.foreign_keys:
            seed = pass_seed(frame_seed, f"fk:{fk.name}")
            if fk.references == "self" and windowed_self_key(
                spec, fk, cartesian=cartesian, pinned=pinned
            ):
                # Drawn from the whole frame, keyed by it, so a window's
                # children point where the whole frame's do.
                key_seed = pass_seed(frame_key, f"fk:{fk.name}")
                runners[f"fk:{fk.name}"] = lambda frame, fk=fk, k=key_seed: (
                    apply_windowed_self_key(
                        frame,
                        columns,
                        fk,
                        frame_seed=frame_key,
                        key_seed=k,
                        frame_rows=frame_rows
                        if frame_rows is not None
                        else frame.height,
                        row_offset=row_offset,
                    )
                )
            elif fk.references == "self":
                # The parent is this frame as it stands when the pass runs.
                runners[f"fk:{fk.name}"] = lambda frame, fk=fk, seed=seed: (
                    apply_foreign_key(frame, columns, fk, frame, seed)
                )
            elif (parent := parents.get(fk.references)) is not None:
                key_seed = pass_seed(
                    whole_seed if whole_seed is not None else frame_seed,
                    f"fk:{fk.name}",
                )
                keys = None if parent_keys is None else parent_keys.get(fk.name)
                runners[f"fk:{fk.name}"] = (
                    lambda frame, fk=fk, parent=parent, seed=seed, k=key_seed, keys=keys: (
                        apply_foreign_key(
                            frame,
                            columns,
                            fk,
                            parent,
                            seed,
                            row_offset=row_offset,
                            key_seed=k,
                            parent_keys=keys,
                        )
                    )
                )

    for index, group in enumerate(spec.unique_together):
        members, writable = tuple(group), rewritable_members(spec, group)
        name = f"unique_together:{','.join(sorted(members))}"
        seed = pass_seed(frame_seed, name)
        # The permutation is keyed by the whole frame, so a window takes the
        # combinations the whole frame takes there.
        joint_seed = pass_seed(frame_key, name)
        key = f"unique_together:{index}"
        runners[key] = lambda frame, m=members, w=writable, seed=seed, j=joint_seed: (
            apply_unique_together(
                frame,
                columns,
                m,
                w,
                seed,
                key_seed=j,
                row_offset=row_offset,
                jointly=not cartesian,
            )
        )

    for step in ordered_passes(spec):
        run = runners.get(step.key)
        if run is not None:
            df = run(df)
    return df


def generate_batches(
    spec: SpecLike,
    n: int,
    *,
    batch_size: int = 100_000,
    method: Method = "random",
    seed: int | None = None,
    references: References = None,
    fixed: Mapping[str, Any] | None = None,
    checks: bool | None = None,
    cycles: int = 0,
    self_references: int = 0,
) -> Iterator[pl.DataFrame]:
    """Yields chunks of generated rows without holding all `n` in memory.

    Each batch is a window onto the one frame `seed` describes: a column no
    pass rewrites holds, batch by batch, exactly the rows `generate(n, seed=seed)`
    would, whatever `batch_size` is -- a nested column, and one rules or a
    foreign key into another table rewrite, and a composite key whose
    members have a finite or bounded space, included. What is drawn per
    batch instead -- deterministic, but not row for row the whole frame's --
    is any other composite key and a key into the spec's own rows.

    A `unique=True` column is a window too: its values are a permutation of
    its value space -- or, filled by a foreign key, of its parent's keys --
    unique across the whole `n` however it is batched; a parent too small
    for all `n` is refused before the first batch. A `__unique_together__`
    group it repairs rather than permutes, and a unique key into the spec's
    own rows, are drawn afresh per batch, so they are distinct within a
    batch and collide across batches only by chance.

    `fixed` pins columns as `generate()` takes it: a scalar for every row, or
    a sequence of all `n` values, which each batch takes its stretch of.
    *Added in 1.4.*

    `checks` keeps the spec's checks, as `generate()` keeps them, by
    default: each batch is the first rows that pass from where the last
    batch stopped reading, so a `unique=True` column stays unique across
    `n`, and the batches are `generate()`'s rows. Batches whose rows all
    pass are the batches `checks=False` gives. *Kept by default since 2.0;
    added in 1.7.*

    `cycles` and `self_references` damage a hierarchy as `generate()` does,
    and every batch holds the damage the whole frame holds there: the edits
    are drawn from the seed alone. *Added in 2.1.*

    Examples
    --------
    >>> spec = TableSpec("Orders", {"id": ColSpec(pl.Int64, unique=True)})
    >>> [batch.height for batch in generate_batches(spec, 25_000, batch_size=10_000, seed=1)]
    [10000, 10000, 5000]
    """
    spec = as_table_spec(spec)
    require_columns(spec)
    requires_whole_frame(spec, "generate_batches")
    check_counts(n, batch_size)
    check_faults(spec, cycles, self_references)
    check_method(method)
    pins = checked_pins(spec, fixed, n, method) if fixed else None
    keeping = resolve_checks(
        spec, checks, method, sequence_pins=pins is not None and pins.has_sequences
    )
    return batch_stream(
        spec,
        n,
        batch_size=batch_size,
        method=method,
        seed=seed,
        references=references,
        pins=pins,
        checks=keeping,
        strict=checks is True,
        cycles=cycles,
        self_references=self_references,
    )


def batch_stream(
    spec: TableSpec,
    n: int,
    *,
    batch_size: int,
    method: Method,
    seed: int | None,
    references: References,
    pins: Pins | None,
    checks: bool = False,
    strict: bool = False,
    cycles: int = 0,
    self_references: int = 0,
    frame_rows: int | None = None,
) -> Iterator[pl.DataFrame]:
    """`generate_batches` once its arguments are checked, as a generator --
    so the checks above run when it is called, not when it is first read.
    `checks` is whether to keep the spec's checks in each batch, as
    `resolve_checks` decided it beforehand; `strict` (`checks=True`) raises
    if they reject nearly every row, where the default stops keeping them
    from that batch on and says so. `frame_rows` is how many rows the whole
    frame holds when fewer are streamed -- a scan's `head` -- so a windowed
    hierarchy and self key lay themselves out over the frame, not the head."""
    if n == 0:
        return
    # The whole frame's `n`, so every batch draws from one key range.
    keeping = checks
    spec = room_keeping_checks(spec, n) if keeping else room_for_open_keys(spec, n)

    # A parent frame is the same frame for every batch, so collect it once.
    # Left as given, a `LazyFrame` reference would be collected inside each
    # batch's foreign-key pass -- once per batch rather than once per call,
    # which is the whole parent re-read however many batches there are.
    resolved = resolve_references(references, to_eager)
    parent_keys: dict[str, pl.DataFrame] = {}
    if spec.foreign_keys:
        # Once per call, not once per batch: the parents are the same every
        # time round, so the batches below are handed only the keys a foreign
        # key actually points at and find nothing left to report.
        _warn_unused_references(spec, resolved)
        targets = {fk.references for fk in spec.foreign_keys if fk.references != "self"}
        resolved = {k: v for k, v in resolved.items() if k in targets}
        # Each key's parent keys are read once, here, rather than once per
        # batch: they are the same every time round, and reading them hashes
        # every parent row. A unique foreign key draws the whole frame's keys
        # window by window, so a parent too small for all `n` rows is refused
        # now, before the first batch -- not when a later one runs past it,
        # halfway through a sink.
        for fk in spec.foreign_keys:
            parent = resolved.get(fk.references)
            if parent is not None:
                parent_keys[fk.name] = parent_keys_of(fk, parent)
                shortfall = unique_parent_shortfall(
                    fk, spec.columns, parent_keys[fk.name].height, n
                )
                if shortfall is not None:
                    raise GenerationError(shortfall)
    references = resolved or None

    frame_seed = _frame_seed(seed)
    produced = 0

    if method == "cartesian":
        # The coverage set is a property of the spec, not of a window: it is
        # built once, as `generate` builds it, and sliced to the batch size
        # (it is a cross-product, not a row count cap). The padding after it
        # is windowed like a random batch stream.
        first = _window(
            spec,
            min(n, batch_size),
            frame_seed,
            method="cartesian",
            references=references,
            parent_keys=parent_keys,
        )
        for offset in range(0, first.height, batch_size):
            yield first.slice(offset, batch_size)
        produced = first.height

    blocks = _Blocks(dict(spec.columns), frame_seed, n, batch_size)

    def read(offset: int, rows: int) -> pl.DataFrame:
        return _window(
            spec,
            rows,
            frame_seed,
            method="random",
            references=references,
            row_offset=offset,
            parent_keys=parent_keys,
            pins=pins.window(offset, rows) if pins is not None else None,
            # Blocks hold the first `n` rows; kept rows can read past them.
            drawn=blocks.window(offset, rows) if offset + rows <= n else None,
            frame_rows=frame_rows if frame_rows is not None else n,
            cycles=cycles,
            self_references=self_references,
        )

    # Where the frame has been read to: `produced` while every row passes.
    cursor = produced
    while produced < n:
        current = min(n - produced, batch_size)
        kept = (
            kept_or_warned(spec, current, cursor, read, strict=strict)
            if keeping
            else None
        )
        if kept is not None:
            batch, used = kept
        else:
            keeping = False
            batch, used = read(cursor, current), current
        yield batch
        produced += current
        cursor += used


# The most a block of batch rows holds, in bytes, by `estimated_size`: enough
# whole chunks to fill every core for a narrow spec, few enough to bound the
# memory of a wide one.
BLOCK_BYTES = 64 * 1024**2


class _Blocks:
    """A stream's engine columns, drawn a block of whole chunks at a time.

    The engine seeds each chunk of a column by its index, so a window that
    starts inside a chunk fills the chunk from its start and drops the head,
    and a window of one chunk fills it on one thread. A block starts on a
    chunk and spans many, filled in parallel, and each window is a slice of
    it: the rows the window would have drawn itself, without the waste.
    Every column is a window onto the frame -- a list's elements too, since
    2.0 -- so every column is blocked.
    """

    def __init__(
        self, columns: dict[str, ColSpec], seed: int, n: int, batch_size: int
    ) -> None:
        self._columns, self._seed, self._n = columns, seed, n
        chunk = self._chunk = chunk_size()
        rows = chunk
        if columns:
            per_chunk = estimated_size(TableSpec("block", columns), chunk)
            rows = max(chunk, BLOCK_BYTES // max(per_chunk, 1) * chunk)
        # Never fewer rows than a window, so a window is one slice of one block.
        self._rows = max(rows, -(-batch_size // chunk) * chunk)
        self._block: pl.DataFrame | None = None
        self._start = 0

    def window(self, offset: int, n: int) -> pl.DataFrame:
        """Rows `[offset, offset + n)`'s engine columns, in declaration order."""
        end = offset + n
        block = self._block
        if block is None or offset < self._start or end > self._start + block.height:
            start = offset - offset % self._chunk
            stop = min(
                self._n, max(start + self._rows, -(-end // self._chunk) * self._chunk)
            )
            block = self._block = generate_random(
                self._columns, stop - start, self._seed, start
            )
            self._start = start
        return block.slice(offset - self._start, n)
