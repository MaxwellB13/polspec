"""`seed_name=`: rename a column without changing the data it generates.

Each column is seeded from the frame seed and the column *name*, so a rename
changes its values. A column declared with `seed_name="old"` is seeded as
`"old"` and keeps producing what it did -- the generation-side twin of
`diff(renames=)`: a rename is declared, not guessed. What it does not do is
pinned in `test_roundtrip.py`.
"""

from __future__ import annotations

import polars as pl
import pytest
from polspec import (
    CatSpec,
    ColRule,
    ColSpec,
    FrameSpec,
    SpecError,
    TableSpec,
    col,
)
from polspec.drift import diff

ROWS, SEED = 200, 42


class Before(FrameSpec):
    keep = ColSpec(pl.String, nullable=True)
    old = ColSpec(pl.String, nullable=True)


class After(FrameSpec):
    keep = ColSpec(pl.String, nullable=True)
    new = ColSpec(pl.String, nullable=True, seed_name="old")


# ---------------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------------


def test_a_renamed_column_keeps_its_data():
    before = Before.generate(ROWS, seed=SEED)
    after = After.generate(ROWS, seed=SEED)
    assert after["keep"].equals(before["keep"])
    assert after["new"].equals(before["old"])
    assert after["new"].name == "new"


def test_the_frame_seed_still_counts():
    assert not After.generate(ROWS, seed=SEED + 1)["new"].equals(
        Before.generate(ROWS, seed=SEED)["old"]
    )


def test_through_cartesian_batches_and_a_rule():
    class OldRules(FrameSpec):
        flag = ColSpec(pl.Boolean)
        old = ColSpec(pl.Int64, bounds=(0, 100))
        ruled_old = ColSpec(
            pl.Int64,
            bounds=(0, 100),
            rules=[ColRule(when=col("flag"), choices=tuple(range(50, 100)))],
        )

    class NewRules(FrameSpec):
        flag = ColSpec(pl.Boolean)
        new = ColSpec(pl.Int64, bounds=(0, 100), seed_name="old")
        ruled_new = ColSpec(
            pl.Int64,
            bounds=(0, 100),
            seed_name="ruled_old",
            rules=[ColRule(when=col("flag"), choices=tuple(range(50, 100)))],
        )

    old = OldRules.generate(ROWS, seed=SEED)
    new = NewRules.generate(ROWS, seed=SEED)
    assert new["new"].equals(old["old"])
    # Both the column's own seed and its rule pass survive the rename: the
    # rule pass is seeded by declaration position, which did not move.
    assert new["ruled_new"].equals(old["ruled_old"])

    old = OldRules.generate(50, seed=SEED, method="cartesian")
    new = NewRules.generate(50, seed=SEED, method="cartesian")
    assert new["new"].equals(old["old"])

    old = pl.concat(OldRules.generate_batches(300, batch_size=100, seed=SEED))
    new = pl.concat(NewRules.generate_batches(300, batch_size=100, seed=SEED))
    assert new["new"].equals(old["old"])


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


def test_a_seed_name_must_be_a_non_empty_string():
    with pytest.raises(SpecError, match="non-empty string, got ''"):
        ColSpec(pl.Int64, seed_name="")
    with pytest.raises(SpecError, match="non-empty string, got 3"):
        ColSpec(pl.Int64, seed_name=3)  # type: ignore[arg-type]


def test_two_columns_cannot_share_a_seed_name():
    with pytest.raises(
        SpecError, match="Columns 'a' and 'b' would both be seeded as 'x'"
    ):
        TableSpec(
            "T",
            {
                "a": ColSpec(pl.Int64, seed_name="x"),
                "b": ColSpec(pl.Int64, seed_name="x"),
            },
        )


def test_a_seed_name_cannot_be_another_columns_name():
    with pytest.raises(
        SpecError, match="Columns 'a' and 'b' would both be seeded as 'a'"
    ):
        TableSpec("T", {"a": ColSpec(pl.Int64), "b": ColSpec(pl.Int64, seed_name="a")})


def test_a_columns_own_name_is_a_harmless_seed_name():
    spec = TableSpec("T", {"a": ColSpec(pl.Int64, seed_name="a")})
    assert spec.columns["a"].seed_name == "a"


def test_rename_leaves_seed_names_alone_and_refuses_a_clash():
    """A rename that must keep its data says so in the declaration; `rename`
    is structural and does not guess. Renaming *into* a seed name is the
    same clash as declaring it."""
    renamed = After.spec.rename({"new": "newer"})
    assert renamed.columns["newer"].seed_name == "old"
    assert renamed.columns["keep"].seed_name is None
    with pytest.raises(SpecError, match="would both be seeded as 'old'"):
        After.spec.rename({"keep": "old"})


# ---------------------------------------------------------------------------
# Everything around it
# ---------------------------------------------------------------------------


def test_a_seed_name_survives_a_file_round_trip(tmp_path):
    path = tmp_path / "s.yaml"
    After.to_yaml(path)
    assert "seed_name: old" in path.read_text(encoding="utf-8")
    assert FrameSpec.from_yaml(path).spec == After.spec
    py = tmp_path / "s.py"
    After.to_python(py)
    assert "seed_name='old'" in py.read_text(encoding="utf-8")


def test_a_changed_seed_name_is_a_compatible_change():
    (finding,) = diff(Before, After.spec.rename({"new": "old"})).findings
    assert finding.code == "field_changed" and not finding.breaking
    assert finding.details["field"] == "seed_name"


def test_retyping_to_a_registry_category_keeps_the_seed_name():
    cats = CatSpec(enums={"c": ["A", "B"]})
    spec = TableSpec("T", {"c": ColSpec(pl.String, seed_name="old")})
    assert spec.with_catspec(cats).columns["c"].seed_name == "old"
