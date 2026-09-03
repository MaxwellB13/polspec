"""The `FrameSpec` class body: inheritance, tags, `__checks__`, `__unique_together__`,
`ColSpec.validators`, and the public import surface.

Declaration-time behaviour that does not fit `test_declaration.py`'s narrower
theme of contracts polspec used to break silently.
"""

import polars as pl
import pytest
from polspec import (
    Bound,
    Check,
    ColRule,
    ColSpec,
    FrameSpec,
    ValidationError,
)
from polspec.serialization import _colspec_to_yaml

# ---------------------------------------------------------------------------
# inheritance and the public surface
# ---------------------------------------------------------------------------


def test_subclass_attribute_overriding():
    class BaseSpec(FrameSpec):
        a = ColSpec(dtype=pl.Int32)
        b = ColSpec(dtype=pl.String)

    class DerivedSpec(BaseSpec):
        b = None  # removes column b
        c = ColSpec(dtype=pl.Float64)

    assert "b" not in DerivedSpec.spec.columns
    assert set(DerivedSpec.spec.columns.keys()) == {"a", "c"}
    df = DerivedSpec.generate(100, seed=42)
    assert set(df.columns) == {"a", "c"}


def test_modular_subpackage_imports():
    from polspec import (
        ColSpec,
        FrameSpec,
        profile_dataframe,
    )
    from polspec.bound import Bound as BoundDirect
    from polspec.engine import _generate_cartesian, _generate_random
    from polspec.framespec import (
        FrameSpec as FrameSpecDirect,
    )
    from polspec.profiler import profile_dataframe as profile_dataframe_direct
    from polspec.rules import ColRule as ColRuleDirect
    from polspec.serialization import _colspec_from_yaml
    from polspec.spec import ColSpec as ColSpecDirect
    from polspec.validation import (
        ValidationError as ValidationErrorDirect,
    )
    from polspec.validation import (
        _validate_dataframe,
    )

    assert Bound is BoundDirect
    assert ColRule is ColRuleDirect
    assert ColSpec is ColSpecDirect
    assert FrameSpec is FrameSpecDirect
    assert ValidationError is ValidationErrorDirect
    assert profile_dataframe is profile_dataframe_direct
    assert callable(_colspec_to_yaml)
    assert callable(_colspec_from_yaml)
    assert callable(_generate_random)
    assert callable(_generate_cartesian)
    assert callable(_validate_dataframe)


# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------


def test_colspec_and_framespec_tags(tmp_path):
    # 1. ColSpec initialization with tags
    c1 = ColSpec(pl.Int64, tags="aggregate")
    assert c1.tags == ("aggregate",)

    c2 = ColSpec(pl.Float64, tags=["aggregate", "index"])
    assert c2.tags == ("aggregate", "index")

    c3 = ColSpec(pl.String, tags=("feature", "metadata"))
    assert c3.tags == ("feature", "metadata")

    c4 = ColSpec(pl.Boolean, tags="")
    assert c4.tags == ()

    c5 = ColSpec(pl.Date, tags=None)
    assert c5.tags == ()

    c6 = ColSpec(pl.Time, tags=["dup", "dup", "unique"])
    assert c6.tags == ("dup", "unique")

    with pytest.raises(TypeError, match=r"ColSpec\.tags must be a string or sequence"):
        ColSpec(pl.Int64, tags=123)  # type: ignore[arg-type]

    # 2. FrameSpec.tag query
    class TaggedSpec(FrameSpec):
        id_col = ColSpec(pl.Int64, tags="index")
        agg_val = ColSpec(pl.Float64, tags=["aggregate", "metric"])
        total_sum = ColSpec(pl.Float64, tags=["aggregate", "index"])
        feature_1 = ColSpec(pl.String, tags="feature")
        untagged = ColSpec(pl.Boolean)

    # Single tag
    assert TaggedSpec.tag("aggregate") == ["agg_val", "total_sum"]
    assert TaggedSpec.tag("index") == ["id_col", "total_sum"]
    assert TaggedSpec.tag("feature") == ["feature_1"]
    assert TaggedSpec.tag("non_existent") == []

    # Sequence of tags (match="any" by default)
    assert TaggedSpec.tag(["aggregate", "feature"]) == [
        "agg_val",
        "total_sum",
        "feature_1",
    ]
    assert TaggedSpec.tag(["index", "feature"]) == ["id_col", "total_sum", "feature_1"]

    # Multiple positional arguments (match="any" by default)
    assert TaggedSpec.tag("aggregate", "feature") == [
        "agg_val",
        "total_sum",
        "feature_1",
    ]
    assert TaggedSpec.tag("index", "feature") == ["id_col", "total_sum", "feature_1"]

    # Multiple tags with match="all"
    assert TaggedSpec.tag(["aggregate", "index"], match="all") == ["total_sum"]
    assert TaggedSpec.tag("aggregate", "index", match="all") == ["total_sum"]
    assert TaggedSpec.tag(["aggregate", "feature"], match="all") == []
    assert TaggedSpec.tag("aggregate", "feature", match="all") == []

    # Empty tags returns empty list
    assert TaggedSpec.tag() == []

    # Invalid tag type
    with pytest.raises(TypeError, match="Tag must be a string or sequence"):
        TaggedSpec.tag(123)  # type: ignore[arg-type]

    # Invalid match mode
    with pytest.raises(ValueError, match="Invalid match mode"):
        TaggedSpec.tag(["aggregate"], match="invalid")  # type: ignore[arg-type]

    # FrameSpec.tags should not exist
    assert not hasattr(TaggedSpec, "tags")

    # 3. YAML serialization roundtrip preserving tags
    yaml_file = tmp_path / "tagged_spec.yaml"
    TaggedSpec.to_yaml(yaml_file)
    LoadedTagged = FrameSpec.from_yaml(yaml_file)
    assert LoadedTagged.tag("aggregate") == ["agg_val", "total_sum"]
    assert LoadedTagged.tag(["index", "feature"]) == [
        "id_col",
        "total_sum",
        "feature_1",
    ]
    assert LoadedTagged.spec.columns["id_col"].tags == ("index",)
    assert LoadedTagged.spec.columns["agg_val"].tags == ("aggregate", "metric")


# ---------------------------------------------------------------------------
# __checks__
# ---------------------------------------------------------------------------


def test_check_construction_rejects_non_expr():
    with pytest.raises(TypeError, match="Check expr must be a polars Expr"):
        Check("col(a) > col(b)")  # type: ignore[arg-type]


def test_checks_declaration_rejects_non_check_items():
    with pytest.raises(TypeError, match="Items in __checks__ must be Check instances"):

        class BadSpec(FrameSpec):
            x = ColSpec(pl.Int64)
            __checks__ = ["not_a_check"]  # type: ignore[list-item]


def test_check_anonymous_naming_defaults_to_expr_repr():
    class AnonSpec(FrameSpec):
        id_col = ColSpec(pl.Int64, unique=True, nullable=True)
        val = ColSpec(pl.Int64)

        __checks__ = [Check(pl.col("val") > 0)]

    check_obj = AnonSpec.checks()[0]
    assert "val" in check_obj.name


def test_checks_declaration_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate Check name 'positive'"):

        class DupCheckSpec(FrameSpec):
            a = ColSpec(pl.Int64)
            b = ColSpec(pl.Int64)
            __checks__ = [
                Check(pl.col("a") > 0, name="positive"),
                Check(pl.col("b") > 0, name="positive"),
            ]


# ---------------------------------------------------------------------------
# __unique_together__
# ---------------------------------------------------------------------------


def test_unique_together_declaration_variations_and_validation():
    # Single tuple of column names
    class SingleTupleSpec(FrameSpec):
        tenant_id = ColSpec(pl.Int32)
        account_id = ColSpec(pl.Int32)
        __unique_together__ = ("tenant_id", "account_id")

    assert SingleTupleSpec.unique_together() == (("tenant_id", "account_id"),)

    # Reference to non-existent column raises ValueError
    with pytest.raises(ValueError, match="references unknown column 'missing_col'"):

        class BadUniqueSpec(FrameSpec):
            a = ColSpec(pl.Int64)
            __unique_together__ = [("a", "missing_col")]


# ---------------------------------------------------------------------------
# ColSpec.validators
# ---------------------------------------------------------------------------


def test_colspec_validators_accepts_bare_expr_and_wraps_in_check():
    spec = ColSpec(pl.Int64, validators=[pl.col("x") % 2 == 0])
    assert len(spec.validators) == 1
    assert isinstance(spec.validators[0], Check)


def test_colspec_validators_accepts_single_expr_without_a_list():
    spec = ColSpec(pl.Int64, validators=pl.col("x") > 0)
    assert len(spec.validators) == 1


def test_colspec_validators_accepts_single_check_without_a_list():
    chk = Check(pl.col("x") > 0, name="positive")
    spec = ColSpec(pl.Int64, validators=chk)
    assert spec.validators == (chk,)


def test_colspec_validators_accepts_check_with_name_and_description():
    spec = ColSpec(
        pl.Float64,
        nullable=True,
        validators=[
            Check(
                pl.col("score") >= 0,
                name="score_non_negative",
                description="Scores can't be negative",
            )
        ],
    )
    assert spec.validators[0].name == "score_non_negative"
    assert spec.validators[0].description == "Scores can't be negative"


def test_colspec_validators_default_empty():
    assert ColSpec(pl.Int64).validators == ()


def test_colspec_validators_rejects_invalid_item_type():
    with pytest.raises(TypeError, match="must be a polars Expr, a predicate"):
        ColSpec(pl.Int64, validators=["not an expr"])  # type: ignore[list-item]


def test_colspec_validators_rejects_duplicate_names():
    with pytest.raises(ValueError, match="Duplicate validator name 'dup'"):
        ColSpec(
            pl.Int64,
            validators=[
                Check(pl.col("x") > 0, name="dup"),
                Check(pl.col("x") < 100, name="dup"),
            ],
        )


def test_colspec_validators_allows_reusing_identical_check():
    # Repeating the exact same Check (same name/expr) is allowed -- it's a
    # duplicate-name collision only when the *definitions* differ.
    chk = Check(pl.col("x") > 0, name="positive")
    spec = ColSpec(pl.Int64, validators=[chk, chk])
    assert spec.validators == (chk, chk)


def test_framespec_rejects_validator_referencing_another_column():
    with pytest.raises(ValueError, match="references other column"):

        class BadSpec(FrameSpec):
            a = ColSpec(pl.Int64, validators=[pl.col("b") > 0])
            b = ColSpec(pl.Int64)


def test_framespec_allows_validator_referencing_only_its_own_column():
    class GoodSpec(FrameSpec):
        a = ColSpec(pl.Int64, validators=[pl.col("a") > 0])

    assert len(GoodSpec.spec.columns["a"].validators) == 1
