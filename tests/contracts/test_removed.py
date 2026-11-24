"""What 2.0 renamed or removed fails at once, naming what to write instead.

2.0 changes these without a 1.x deprecation (see *Moving to 2.0*), so the
old spelling must not fail with Python's bare "unexpected keyword" or "has
no attribute": each case below calls it as 1.x code would and holds the
error to naming the replacement.
"""

from __future__ import annotations

import re
import types

import polars as pl
import polspec
import pytest
from polspec import CatSpec, ColRule, ColSpec, FrameSpec, Registry, assert_valid, col


class Spec(FrameSpec):
    status = ColSpec(pl.String, choices=["a", "b"], tags="pii")


CASES = {
    "assert_valid(df, spec)": (
        lambda: assert_valid(pl.DataFrame({"status": ["a"]}), Spec),
        TypeError,
        r"assert_valid\(spec, df\)",
    ),
    "FrameSpec.col": (lambda: Spec.col("status"), AttributeError, "FrameSpec.column"),
    "FrameSpec.tag": (lambda: Spec.tag("pii"), AttributeError, "FrameSpec.tagged"),
    "TableSpec.tag": (lambda: Spec.spec.tag("pii"), AttributeError, "TableSpec.tagged"),
    "CatSpec.from_framespec": (
        lambda: CatSpec.from_framespec(Spec),
        AttributeError,
        "CatSpec.from_spec",
    ),
    "CatSpec.infer(max_enum_cardinality=)": (
        lambda: CatSpec.infer(Spec, max_enum_cardinality=3),
        TypeError,
        "max_unique_enum=",
    ),
    "from_dict(categories=)": (
        lambda: polspec.from_dict({"name": "T", "columns": {}}, categories=CatSpec()),
        TypeError,
        "catspec=",
    ),
    "from_yaml(categories=)": (
        lambda: polspec.from_yaml("unread.yaml", categories=CatSpec()),
        TypeError,
        "catspec=",
    ),
    "FrameSpec.from_yaml(categories=)": (
        lambda: FrameSpec.from_yaml("unread.yaml", categories=CatSpec()),
        TypeError,
        "catspec=",
    ),
    "Registry(categories=)": (
        lambda: Registry(Spec, categories=CatSpec()),
        TypeError,
        "catspec=",
    ),
    "Registry.discover(categories=)": (
        lambda: Registry.discover("unread", categories=CatSpec()),
        TypeError,
        "catspec=",
    ),
    "Registry.from_module(categories=)": (
        lambda: Registry.from_module(types.ModuleType("m"), categories=CatSpec()),
        TypeError,
        "catspec=",
    ),
    "ColSpec(weights=)": (
        lambda: ColSpec(pl.Boolean, weights=[0.9, 0.1]),
        TypeError,
        re.escape("choices={value: weight}"),
    ),
    "ColRule(weights=)": (
        lambda: ColRule(when=col("status") == "a", choices=["x"], weights=[1.0]),
        TypeError,
        re.escape("choices={value: weight}"),
    ),
    "ColSpec.replace(weights=)": (
        lambda: Spec.column("status").replace(weights=[1.0, 2.0]),
        TypeError,
        re.escape("choices={value: weight}"),
    ),
    "Registry.categories": (
        lambda: Registry(Spec).categories,
        AttributeError,
        r"Registry.catspec\(\)",
    ),
}


@pytest.mark.parametrize("old", sorted(CASES))
def test_the_old_spelling_names_the_new(old):
    call, error, replacement = CASES[old]
    with pytest.raises(error, match=replacement) as raised:
        call()
    assert "since 2.0" in str(raised.value)


def test_the_new_spellings_do_what_the_old_did():
    assert assert_valid(Spec, pl.DataFrame({"status": ["a"]})).height == 1
    assert Spec.column("status") is Spec.spec["status"]
    assert Spec.tagged("pii") == Spec.spec.tagged("pii") == ["status"]
    assert CatSpec.from_spec(Spec).choices == {"status": ["a", "b"]}
    assert CatSpec.infer(Spec, max_unique_enum=3) is not None
    assert ColSpec(pl.Boolean, choices={True: 0.1, False: 0.9}).weights == (0.9, 0.1)
    assert (
        ColSpec(pl.Enum, choices={"a": 1, "b": 3})
        == polspec.from_dict(
            {
                "version": 4,
                "name": "T",
                "columns": {"c": {"dtype": {"Enum": ["a", "b"]}, "weights": [1, 3]}},
            }
        )["c"]
    )
    declared = CatSpec(enums={"S": ["a"]})
    assert Registry(Spec, catspec=declared).catspec() is declared


def test_a_writer_returns_its_text_when_it_writes_a_file(tmp_path):
    """1.x returned None given a path; 2.0 returns the text either way."""
    assert Spec.to_yaml(tmp_path / "s.yaml") == Spec.to_yaml()
    assert Spec.to_python(tmp_path / "s.py") == Spec.to_python()
