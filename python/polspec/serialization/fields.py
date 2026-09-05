"""The field registry: one description of each declaration's fields.

YAML writing, YAML reading, Python-source emission and the "does this file
need `import datetime`" question are all derived from the tables below, so a
new field on `ColSpec` (or `ColRule`, `Check`, `ForeignKey`, `TableSpec`) is
one entry here. A parity test asserts every dataclass field has one.

Each `Field` knows its key, how to turn a value into YAML-safe data and back,
how to render it as Python source, and when to leave it out of a file.
"""

from __future__ import annotations

import difflib
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from polspec.bound import Bound
from polspec.check import Check
from polspec.constants import _DEFAULT_NULL_PROBABILITY
from polspec.errors import SerializationError
from polspec.expr import from_data as pred_from_data
from polspec.foreign_key import ForeignKey, _default_fk_name
from polspec.rules import ColRule
from polspec.serialization.dtypes import dtype_from_data, dtype_to_data, dtype_to_source
from polspec.spec import ColSpec
from polspec.tablespec import TableSpec

if TYPE_CHECKING:
    from polspec.catspec import CatSpec


@dataclass(frozen=True, slots=True)
class Ctx:
    """What a reader needs beyond the data: the registry, and how strict to be."""

    categories: CatSpec | None = None
    strict: bool = True


def _identity(value: Any) -> Any:
    return value


def _read_identity(value: Any, ctx: Ctx, path: str) -> Any:
    return value


def _never(value: Any, obj: Any) -> bool:
    return False


def _if_none(value: Any, obj: Any) -> bool:
    return value is None


def _if_falsy(value: Any, obj: Any) -> bool:
    return not value


def _if_false(value: Any, obj: Any) -> bool:
    return value is False


@dataclass(frozen=True, slots=True)
class Field:
    name: str
    attr: str | None = None
    positional: bool = False
    omit_if: Callable[[Any, Any], bool] = _if_none
    to_data: Callable[[Any], Any] = _identity
    from_data: Callable[[Any, Ctx, str], Any] = _read_identity
    to_source: Callable[[Any], str] = repr
    since: int = 1

    @property
    def attribute(self) -> str:
        return self.attr or self.name


# ---------------------------------------------------------------------------
# The generic operations
# ---------------------------------------------------------------------------


def encode(obj: Any, fields: Sequence[Field]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields:
        value = getattr(obj, f.attribute)
        if f.omit_if(value, obj):
            continue
        out[f.name] = f.to_data(value)
    return out


def check_unknown_keys(
    data: Mapping[str, Any], known: Sequence[str], ctx: Ctx, path: str
) -> None:
    """Rejects (or warns about) keys the reader does not know.

    Silently ignoring a key is the worst outcome: a renamed or misspelt
    option is then read as its default without a word.
    """
    unknown = [k for k in data if k not in known]
    if not unknown:
        return
    parts = []
    for key in unknown:
        where = f"{path}.{key}" if path else key
        close = difflib.get_close_matches(str(key), [str(k) for k in known], n=1)
        parts.append(f"{where!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    message = f"Unknown key(s) in spec file: {', '.join(parts)}"
    if ctx.strict:
        raise SerializationError(
            message + ". Pass strict=False to ignore unknown keys."
        )
    warnings.warn(message + "; ignored.", stacklevel=4)


def decode(
    cls: type,
    data: Mapping[str, Any],
    fields: Sequence[Field],
    ctx: Ctx,
    path: str,
    *,
    extra_known: Sequence[str] = (),
) -> Any:
    if not isinstance(data, Mapping):
        raise SerializationError(
            f"{path or 'the spec'} must be a mapping, got {type(data).__name__}"
        )
    check_unknown_keys(data, [f.name for f in fields] + list(extra_known), ctx, path)
    kwargs: dict[str, Any] = {}
    for f in fields:
        if f.name in data:
            # The key is also the constructor keyword; `attr` is only where
            # the value lives once constructed (Check reads `pred`, takes `expr`).
            kwargs[f.name] = f.from_data(
                data[f.name], ctx, f"{path}.{f.name}" if path else f.name
            )
    return cls(**kwargs)


def to_source(obj: Any, fields: Sequence[Field], ctor: str) -> str:
    args: list[str] = []
    for f in fields:
        value = getattr(obj, f.attribute)
        if f.omit_if(value, obj):
            continue
        rendered = f.to_source(value)
        args.append(rendered if f.positional else f"{f.name}={rendered}")
    return f"{ctor}({', '.join(args)})"


# ---------------------------------------------------------------------------
# Value codecs shared by several fields
# ---------------------------------------------------------------------------


def _bound_to_data(value: Bound) -> list[Any]:
    return [value.min, value.max]


def _bound_from_data(value: Any, ctx: Ctx, path: str) -> tuple[Any, Any]:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise SerializationError(f"{path} must be a two-element list, got {value!r}")
    return (value[0], value[1])


def _bound_to_source(value: Bound) -> str:
    return f"({value.min!r}, {value.max!r})"


def _tags_to_data(value: tuple[str, ...]) -> str | list[str]:
    return value[0] if len(value) == 1 else list(value)


def _tags_to_source(value: tuple[str, ...]) -> str:
    return repr(value[0]) if len(value) == 1 else repr(list(value))


def _when_from_data(value: Any, ctx: Ctx, path: str) -> Any:
    if isinstance(value, dict) and "column" in value:
        # A version 1 condition in a file claiming to be version 2: the
        # migration converts these, so reaching here means the version key
        # is wrong rather than the condition.
        raise SerializationError(
            f"{path}: {value!r} is a version 1 rule condition, but the file "
            "declares a later version. Remove the `version:` key to have it "
            "migrated, or write the condition with col()."
        )
    return pred_from_data(value)


def _pred_to_data(check: Check) -> Any:
    if check.pred is None:
        raise SerializationError(f"Check {check.name!r} wraps a raw polars.Expr")
    return check.pred.to_data()


# ---------------------------------------------------------------------------
# Check / validators
# ---------------------------------------------------------------------------


def _default_check_name(value: Any, check: Check) -> bool:
    return check.pred is not None and value == repr(check.pred)


CHECK_FIELDS: tuple[Field, ...] = (
    Field(
        "expr",
        attr="pred",
        positional=True,
        omit_if=_never,
        to_data=lambda pred: pred.to_data(),
        from_data=lambda v, ctx, path: pred_from_data(v),
        to_source=lambda pred: pred.to_source(),
    ),
    Field("name", omit_if=_default_check_name),
    Field("description"),
    Field("ignore_nulls", omit_if=lambda v, obj: v is True),
)


def check_to_data(check: Check) -> dict[str, Any]:
    if check.pred is None:
        raise SerializationError(f"Check {check.name!r} wraps a raw polars.Expr")
    return encode(check, CHECK_FIELDS)


def check_from_data(value: Any, ctx: Ctx, path: str) -> Check:
    if not isinstance(value, Mapping) or "expr" not in value:
        raise SerializationError(
            f"{path}: a check or validator needs an 'expr' key, got {value!r}"
        )
    return decode(Check, value, CHECK_FIELDS, ctx, path)


def check_to_source(check: Check) -> str:
    if check.pred is None:
        raise SerializationError(f"Check {check.name!r} wraps a raw polars.Expr")
    return to_source(check, CHECK_FIELDS, "Check")


def _persistable(validators: tuple[Check, ...]) -> list[Check]:
    return [v for v in validators if v.pred is not None]


# ---------------------------------------------------------------------------
# ColRule
# ---------------------------------------------------------------------------

COLRULE_FIELDS: tuple[Field, ...] = (
    Field(
        "when",
        omit_if=_never,
        to_data=lambda pred: pred.to_data(),
        from_data=_when_from_data,
        to_source=lambda pred: pred.to_source(),
    ),
    Field("choices", omit_if=_never, to_data=list, to_source=lambda v: repr(list(v))),
    Field("weights", to_data=list, to_source=lambda v: repr(list(v))),
)


def _rule_from_data(value: Any, ctx: Ctx, path: str) -> ColRule:
    return decode(ColRule, value, COLRULE_FIELDS, ctx, path)


# ---------------------------------------------------------------------------
# ColSpec
# ---------------------------------------------------------------------------

COLSPEC_FIELDS: tuple[Field, ...] = (
    Field(
        "dtype",
        positional=True,
        omit_if=_never,
        to_data=dtype_to_data,
        from_data=lambda v, ctx, path: dtype_from_data(v, ctx.categories),
        to_source=dtype_to_source,
    ),
    # The column's key already is its name; col_name is a class-body affordance.
    Field("col_name", omit_if=lambda v, obj: True),
    Field("nullable", omit_if=_if_false),
    Field(
        "bounds",
        to_data=_bound_to_data,
        from_data=_bound_from_data,
        to_source=_bound_to_source,
    ),
    Field("tags", omit_if=_if_falsy, to_data=_tags_to_data, to_source=_tags_to_source),
    Field("unique", omit_if=_if_false),
    Field(
        "null_probability",
        omit_if=lambda v, obj: v == _DEFAULT_NULL_PROBABILITY,
    ),
    Field(
        "string_length",
        to_data=_bound_to_data,
        from_data=_bound_from_data,
        to_source=_bound_to_source,
    ),
    Field("distribution"),
    Field(
        "distribution_params",
        omit_if=_if_falsy,
        to_data=dict,
        to_source=lambda v: repr(dict(v)),
    ),
    Field("choices", to_data=list, to_source=lambda v: repr(list(v))),
    Field("weights", to_data=list, to_source=lambda v: repr(list(v))),
    Field(
        "rules",
        omit_if=_if_falsy,
        to_data=lambda rules: [encode(r, COLRULE_FIELDS) for r in rules],
        from_data=lambda v, ctx, path: tuple(
            _rule_from_data(r, ctx, f"{path}[{i}]") for i, r in enumerate(v)
        ),
        to_source=lambda rules: (
            "["
            + ", ".join(to_source(r, COLRULE_FIELDS, "ColRule") for r in rules)
            + "]"
        ),
    ),
    Field(
        "validators",
        omit_if=lambda v, obj: not _persistable(v),
        to_data=lambda v: [check_to_data(c) for c in _persistable(v)],
        from_data=lambda v, ctx, path: tuple(
            check_from_data(c, ctx, f"{path}[{i}]") for i, c in enumerate(v)
        ),
        to_source=lambda v: (
            "[" + ", ".join(check_to_source(c) for c in _persistable(v)) + "]"
        ),
        since=2,
    ),
)


def colspec_to_data(spec: ColSpec) -> dict[str, Any]:
    return encode(spec, COLSPEC_FIELDS)


def colspec_from_data(value: Any, ctx: Ctx, path: str) -> ColSpec:
    if not isinstance(value, Mapping) or "dtype" not in value:
        raise SerializationError(f"{path}: a column needs a 'dtype' key, got {value!r}")
    return decode(ColSpec, value, COLSPEC_FIELDS, ctx, path)


def colspec_to_source(spec: ColSpec) -> str:
    return to_source(spec, COLSPEC_FIELDS, "ColSpec")


# ---------------------------------------------------------------------------
# ForeignKey
# ---------------------------------------------------------------------------


def _same_as_columns(value: Any, fk: ForeignKey) -> bool:
    return value == fk.columns


def _default_fk(value: Any, fk: ForeignKey) -> bool:
    return value == _default_fk_name(fk.columns, fk.references)


FK_FIELDS: tuple[Field, ...] = (
    Field("columns", omit_if=_never, to_data=list, to_source=lambda v: repr(list(v))),
    Field("references", omit_if=_never),
    Field(
        "ref_columns",
        omit_if=_same_as_columns,
        to_data=list,
        to_source=lambda v: repr(list(v)),
    ),
    Field("name", omit_if=_default_fk),
    # Bound at declaration from a spec object; a file only ever holds the name.
    Field("target", omit_if=lambda v, obj: True),
)


def fk_from_data(value: Any, ctx: Ctx, path: str) -> ForeignKey:
    if not isinstance(value, Mapping) or "columns" not in value:
        raise SerializationError(
            f"{path}: a foreign key needs a 'columns' key, got {value!r}"
        )
    return decode(ForeignKey, value, FK_FIELDS, ctx, path)


def fk_to_source(fk: ForeignKey) -> str:
    return to_source(fk, FK_FIELDS, "ForeignKey")


# ---------------------------------------------------------------------------
# TableSpec
# ---------------------------------------------------------------------------


def _columns_from_data(value: Any, ctx: Ctx, path: str) -> dict[str, ColSpec]:
    if not isinstance(value, Mapping) or not value:
        raise SerializationError("the spec declares no columns")
    return {
        str(name): colspec_from_data(col, ctx, f"{path}.{name}")
        for name, col in value.items()
    }


TABLESPEC_FIELDS: tuple[Field, ...] = (
    Field("name", omit_if=_never),
    Field(
        "columns",
        omit_if=_never,
        to_data=lambda cols: {name: colspec_to_data(cs) for name, cs in cols.items()},
        from_data=_columns_from_data,
    ),
    Field(
        "unique_together",
        omit_if=_if_falsy,
        to_data=lambda groups: [list(g) for g in groups],
    ),
    Field(
        "foreign_keys",
        omit_if=_if_falsy,
        to_data=lambda fks: [encode(fk, FK_FIELDS) for fk in fks],
        from_data=lambda v, ctx, path: [
            fk_from_data(fk, ctx, f"{path}[{i}]") for i, fk in enumerate(v)
        ],
    ),
    Field(
        "checks",
        omit_if=lambda v, obj: not _persistable(v),
        to_data=lambda v: [check_to_data(c) for c in _persistable(v)],
        from_data=lambda v, ctx, path: [
            check_from_data(c, ctx, f"{path}[{i}]") for i, c in enumerate(v)
        ],
        since=2,
    ),
)

# Keys a spec file may carry that are not TableSpec fields.
FILE_KEYS: tuple[str, ...] = ("version", "categories")


def tablespec_to_data(spec: TableSpec) -> dict[str, Any]:
    return encode(spec, TABLESPEC_FIELDS)


def tablespec_from_data(data: Mapping[str, Any], ctx: Ctx) -> TableSpec:
    if not isinstance(data, Mapping):
        raise SerializationError(
            f"A spec file must hold a mapping, got {type(data).__name__}"
        )
    body = {k: v for k, v in data.items() if k not in FILE_KEYS}
    if "columns" not in body or not body["columns"]:
        raise SerializationError("the spec declares no columns")
    body.setdefault("name", "LoadedFrameSpec")
    return decode(TableSpec, body, TABLESPEC_FIELDS, ctx, "")


# ---------------------------------------------------------------------------
# `import datetime` in generated Python
# ---------------------------------------------------------------------------


def needs_datetime_import(value: Any) -> bool:
    """Whether any literal in an encoded value is a date, time or timedelta."""
    import datetime

    if isinstance(value, (datetime.date, datetime.time, datetime.timedelta)):
        return True
    if isinstance(value, Mapping):
        return any(needs_datetime_import(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(needs_datetime_import(v) for v in value)
    return False
