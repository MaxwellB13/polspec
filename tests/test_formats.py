"""`format=`: a String column that says what its values look like.

The round trip itself -- every format generates data its own validator
accepts -- lives in `test_roundtrip.py` beside the other column cases. This
module covers everything around it: what a declaration refuses and how it
says so, what the `format` finding reports, how a format takes part in a
foreign key's domain check, and the paths through the engine.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest
from polspec import (
    CatSpec,
    ColSpec,
    ForeignKey,
    FrameSpec,
    GenerationError,
    Hierarchy,
    SpecError,
    TableSpec,
    generate_batches,
)
from polspec.constraints import Domain
from polspec.engine import _plan_column
from polspec.formats import FORMATS, Format, lookup, names


def _spec_for(column: ColSpec) -> type[FrameSpec]:
    return type("Formatted", (FrameSpec,), {"__columns__": {"c": column}})


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_the_documented_set_is_the_shipped_set():
    assert names() == [
        "uuid4",
        "email",
        "ipv4",
        "ipv6",
        "mac",
        "hostname",
        "iso_country",
        "iso_currency",
    ]


def test_iso_lists_are_well_formed():
    countries = FORMATS["iso_country"].values
    currencies = FORMATS["iso_currency"].values
    assert len(countries) == 249 and len(set(countries)) == 249
    assert all(len(c) == 2 and c.isupper() for c in countries)
    assert len(set(currencies)) == len(currencies)
    assert all(len(c) == 3 and c.isupper() for c in currencies)
    assert {"US", "GB", "DE", "JP"} <= set(countries)
    assert {"USD", "EUR", "GBP", "JPY"} <= set(currencies)


def test_a_format_is_a_template_or_a_list_never_both():
    with pytest.raises(ValueError, match="needs a template or values"):
        Format("x", "nothing")
    with pytest.raises(ValueError, match="no pattern"):
        Format("x", "shape", template=(("lit", ["a"], 0, 0),))


@pytest.mark.parametrize(
    "name, good, bad",
    [
        (
            "uuid4",
            [
                "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
                "3F2504E0-4F89-41D3-9A0C-0305E82C3301",
            ],
            [
                "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
                "3f2504e04f8941d39a0c0305e82c3301",
            ],
        ),
        (
            "email",
            ["a@b.co", "first.last+tag@sub.example.org"],
            ["a@b", "@b.co", "a b@c.d"],
        ),
        (
            "ipv4",
            ["0.0.0.0", "255.255.255.255", "10.0.0.1"],  # noqa: S104
            ["256.0.0.1", "1.2.3", "01.2.3.4"],
        ),
        (
            "ipv6",
            ["2001:db8:0:0:0:0:0:1", "ABCD:0:0:0:0:0:0:0"],
            ["2001:db8::1", "1:2:3:4:5:6:7"],
        ),
        (
            "mac",
            ["00:1a:2b:3c:4d:5e", "FF:FF:FF:FF:FF:FF"],
            ["00-1a-2b-3c-4d-5e", "00:1a:2b:3c:4d"],
        ),
        (
            "hostname",
            ["localhost", "a-b.example.com", "X.Y"],
            ["-a.com", "a..b", "a_b.com", "a" * 254],
        ),
        ("iso_country", ["US", "ZW"], ["us", "USA", "XX"]),
        ("iso_currency", ["USD", "ZWG"], ["usd", "US", "XTS"]),
    ],
)
def test_each_validator_accepts_its_shape_and_nothing_else(name, good, bad):
    """The check side of every format, against hand-written values.

    Generation only ever produces a subset of what a format accepts -- a
    lowercase UUID, an uncompressed IPv6 -- so the round trip cannot see
    whether the validator is too *narrow* for real data. This can.
    """
    fmt = lookup(name)
    result = pl.DataFrame({"v": good + bad}).select(fmt.check(pl.col("v")))["v"]
    assert result.to_list() == [True] * len(good) + [False] * len(bad)


def test_null_is_neither_valid_nor_invalid():
    fmt = lookup("email")
    result = pl.DataFrame({"v": [None, "a@b.c"]}).select(fmt.check(pl.col("v")))["v"]
    assert result.to_list() == [None, True]


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


def test_an_unknown_format_names_the_nearest_one():
    with pytest.raises(SpecError) as exc:
        ColSpec(pl.String, format="uuid")
    message = str(exc.value)
    assert "'uuid' is not a known format" in message
    assert "Did you mean 'uuid4'?" in message
    assert "Known formats: uuid4, email, ipv4" in message


def test_an_unknown_format_with_no_near_miss_still_lists_them():
    with pytest.raises(SpecError, match="Known formats:") as exc:
        ColSpec(pl.String, format="zzzz")
    assert "Did you mean" not in str(exc.value)


@pytest.mark.parametrize("dtype", [pl.Int64, pl.Binary, pl.Categorical, pl.Enum(["a"])])
def test_a_format_needs_a_string_column(dtype):
    with pytest.raises(SpecError, match=r"only supported for pl\.String"):
        ColSpec(dtype, format="uuid4")


def test_a_format_and_choices_are_two_definitions_of_one_domain():
    with pytest.raises(SpecError, match="both format='email' and choices"):
        ColSpec(pl.String, format="email", choices=["a@b.c"])


def test_a_format_owns_the_length():
    with pytest.raises(SpecError, match="both format='uuid4' and string_length"):
        ColSpec(pl.String, format="uuid4", string_length=(36, 36))


def test_a_format_allows_what_it_does_not_contradict():
    spec = ColSpec(
        pl.String,
        format="email",
        nullable=True,
        null_probability=0.5,
        unique=True,
        tags="pii",
        validators=[pl.col("c").str.contains("@")],
    )
    assert spec.format == "email"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_the_format_finding_names_the_format_and_samples():
    spec_cls = _spec_for(ColSpec(pl.String, format="ipv4", nullable=True))
    df = pl.DataFrame({"c": ["10.0.0.1", "300.1.1.1", None, "abc", "300.1.1.1"]})
    report = spec_cls.inspect(df)
    (finding,) = report.findings
    assert finding.code == "format"
    assert finding.count == 3
    assert finding.columns == ("c",)
    assert finding.samples == ("300.1.1.1", "abc")
    assert dict(finding.details) == {"format": "ipv4"}
    assert finding.message == (
        "Column 'c': found 3 value(s) that are not ipv4 (four dotted decimal "
        "octets, each 0-255). Invalid samples: ['300.1.1.1', 'abc']"
    )
    rows = report.rows(finding).collect()
    assert rows["c"].to_list() == ["300.1.1.1", "abc", "300.1.1.1"]


def test_a_finite_format_reports_as_format_not_choices():
    spec_cls = _spec_for(ColSpec(pl.String, format="iso_currency"))
    report = spec_cls.inspect(pl.DataFrame({"c": ["USD", "usd", "XTS"]}))
    (finding,) = report.findings
    assert finding.code == "format"
    assert "not iso_currency (an ISO 4217 alpha-3 currency code)" in finding.message
    assert finding.samples == ("usd", "XTS")


def test_a_format_is_not_checked_on_a_column_of_the_wrong_dtype():
    """Only the dtype finding is worth reporting: the format check would be
    noise on top of it, or an expression Polars refuses to run on integers.
    """
    spec_cls = _spec_for(ColSpec(pl.String, format="uuid4"))
    report = spec_cls.inspect(pl.DataFrame({"c": [1, 2, 3]}))
    assert [f.code for f in report.findings] == ["dtype"]


# ---------------------------------------------------------------------------
# Domain: a format in a foreign key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "child, parent, fits",
    [
        # a formatted child needs a parent of the same format
        (ColSpec(pl.String, format="uuid4"), ColSpec(pl.String, format="uuid4"), True),
        (ColSpec(pl.String, format="uuid4"), ColSpec(pl.String, format="mac"), False),
        (ColSpec(pl.String, format="uuid4"), ColSpec(pl.String), False),
        # a plain child accepts a formatted parent
        (ColSpec(pl.String), ColSpec(pl.String, format="uuid4"), True),
        # a finite parent is checked value by value against the format
        (
            ColSpec(pl.String, format="ipv4"),
            ColSpec(pl.String, choices=["10.0.0.1", "10.0.0.2"]),
            True,
        ),
        (
            ColSpec(pl.String, format="ipv4"),
            ColSpec(pl.String, choices=["10.0.0.1", "nope"]),
            False,
        ),
        # a finite format is a finite domain like any other
        (
            ColSpec(pl.String, format="iso_country"),
            ColSpec(pl.String, choices=["US"]),
            True,
        ),
        (
            ColSpec(pl.String, format="iso_country"),
            ColSpec(pl.String, choices=["XX"]),
            False,
        ),
        (
            ColSpec(pl.String, choices=["US", "GB"]),
            ColSpec(pl.String, format="iso_country"),
            False,
        ),
        (
            ColSpec(pl.String, format="iso_country"),
            ColSpec(pl.String, format="iso_country"),
            True,
        ),
        (
            ColSpec(pl.Enum(["US", "GB"])),
            ColSpec(pl.String, format="iso_country"),
            False,
        ),
    ],
)
def test_domain_knows_whether_a_formatted_parent_fits(child, parent, fits):
    assert (Domain.of(child).rejects(Domain.of(parent)) is None) is fits


def test_domain_describes_a_format_by_name():
    assert str(Domain.of(ColSpec(pl.String, format="uuid4"))) == "format 'uuid4'"
    finite = Domain.of(ColSpec(pl.String, format="iso_country"))
    assert finite.format is None and len(finite.values) == 249


def test_a_foreign_key_into_a_formatted_parent_is_checked_at_declaration():
    class Users(FrameSpec):
        user_id = ColSpec(pl.String, format="uuid4", unique=True)

    with pytest.raises(SpecError, match="format 'uuid4' is not format 'mac'"):

        class Wrong(FrameSpec):
            user_id = ColSpec(pl.String, format="mac")
            __foreign_keys__ = [ForeignKey("user_id", references=Users)]

    class Right(FrameSpec):
        user_id = ColSpec(pl.String, format="uuid4")
        __foreign_keys__ = [ForeignKey("user_id", references=Users)]

    users = Users.generate(50, seed=1)
    df = Right.generate(200, seed=2, references={Users: users})
    assert set(df["user_id"].to_list()) <= set(users["user_id"].to_list())
    Right.validate(df, references={Users: users})


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def test_a_template_format_reaches_the_engine_as_a_template():
    plan, domain = _plan_column("c", ColSpec(pl.String, format="email"))
    assert plan.kind == "template" and domain is None


def test_a_finite_format_reaches_the_engine_as_an_index():
    plan, domain = _plan_column("c", ColSpec(pl.String, format="iso_currency"))
    assert plan.kind == "index"
    assert domain.dtype == pl.String and domain.len() == plan.n_categories


def test_a_unique_finite_format_refuses_when_the_list_runs_out():
    spec_cls = _spec_for(ColSpec(pl.String, format="iso_country", unique=True))
    with pytest.raises(GenerationError, match="only 249 distinct value"):
        spec_cls.generate(250, seed=1)
    df = spec_cls.generate(249, seed=1)
    assert df["c"].n_unique() == 249


def test_a_unique_template_format_draws_without_replacement():
    spec_cls = _spec_for(ColSpec(pl.String, format="mac", unique=True))
    df = spec_cls.generate(20_000, seed=1)
    assert df["c"].n_unique() == 20_000
    spec_cls.validate(df)


def test_nulls_follow_the_declared_rate():
    spec_cls = _spec_for(
        ColSpec(pl.String, format="ipv6", nullable=True, null_probability=0.4)
    )
    df = spec_cls.generate(5_000, seed=1)
    assert 0.35 < df["c"].null_count() / 5_000 < 0.45
    spec_cls.validate(df)


def test_the_same_seed_gives_the_same_values():
    spec_cls = _spec_for(ColSpec(pl.String, format="uuid4"))
    assert spec_cls.generate(100, seed=9).equals(spec_cls.generate(100, seed=9))
    assert not spec_cls.generate(100, seed=9).equals(spec_cls.generate(100, seed=10))


def test_a_format_column_is_filled_under_cartesian_and_in_batches():
    class Mixed(FrameSpec):
        flag = ColSpec(pl.Boolean)
        host = ColSpec(pl.String, format="hostname")

    Mixed.validate(Mixed.generate(10, method="cartesian", seed=1))
    for batch in generate_batches(Mixed.spec, 250, batch_size=100, seed=1):
        Mixed.validate(batch)


def test_a_hierarchy_draws_its_references_in_the_column_format():
    class Node(FrameSpec):
        ref = ColSpec(pl.String, format="uuid4")
        parent = ColSpec(pl.String, format="uuid4")
        __hierarchy__ = Hierarchy(child="ref", parent="parent", max_depth=3)

    df = Node.generate(40, seed=1)
    Node.validate(df)


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


def test_retyping_to_a_registry_category_drops_the_format_with_a_warning():
    cats = CatSpec(enums={"country": ["US", "GB"]})
    spec = TableSpec("T", {"country": ColSpec(pl.String, format="iso_country")})
    with pytest.warns(UserWarning, match="dropping format='iso_country'"):
        retyped = spec.with_catspec(cats)
    assert retyped.columns["country"].format is None
    assert isinstance(retyped.columns["country"].dtype, pl.Enum)


def test_renaming_keeps_the_format():
    spec = TableSpec("T", {"a": ColSpec(pl.String, format="email")})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        renamed = spec.rename({"a": "b"})
    assert renamed.columns["b"].format == "email"
