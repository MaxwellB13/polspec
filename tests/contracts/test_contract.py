"""What 1.0 promises, each promise held to a golden copy.

The Stability page (`docs/explanation/stability.md`) states the promises;
each test here is named for one. A golden copy lives in `golden/` beside
this file, and is rewritten only on purpose:

    POLSPEC_WRITE_GOLDEN=1 uv run pytest tests/contracts/test_contract.py

A rewrite is a change to the contract, so its diff is read as one: an
added name or key is a deliberate addition, and a removed one must have
been through a deprecation. The spec files of older format versions are
written by hand and never rewritten -- they are what an old polspec wrote.
"""

from __future__ import annotations

import json
import os
import re
import typing
from pathlib import Path

import polars as pl
import polspec
import pytest
from polspec import (
    FORMAT_VERSION,
    Check,
    ColRule,
    ColSpec,
    DriftFinding,
    Finding,
    ForeignKey,
    Hierarchy,
    Registry,
    TableSpec,
    col,
    from_yaml,
    generate,
    to_yaml,
)
from polspec.cli import main

GOLDEN = Path(__file__).with_name("golden")
WRITE = os.environ.get("POLSPEC_WRITE_GOLDEN") == "1"
DOCS = Path(__file__).resolve().parents[2] / "docs"
PUBLIC = ("polspec",)


def _golden(name: str, actual: str) -> None:
    """`actual` against the golden copy `name` -- or, writing, its new copy."""
    path = GOLDEN / name
    if WRITE:
        path.write_text(actual, encoding="utf-8", newline="\n")
        return
    assert path.exists(), f"no golden {name}: run with POLSPEC_WRITE_GOLDEN=1"
    assert actual == path.read_text(encoding="utf-8"), (
        f"{name} moved: if on purpose, rewrite it (see this module's docstring)"
    )


# ---------------------------------------------------------------------------
# The public API
# ---------------------------------------------------------------------------


def test_the_public_api_is_the_golden_list():
    """An addition is a deliberate edit of `golden/public_api.json`; a
    removal waits for a deprecation, and then the next major release."""
    import importlib

    names = {m: sorted(importlib.import_module(m).__all__) for m in PUBLIC}
    _golden("public_api.json", json.dumps(names, indent=1) + "\n")


# ---------------------------------------------------------------------------
# Spec files: every version reads, today's is written byte for byte
# ---------------------------------------------------------------------------

# One spec, in every format version. Each old file is what that version
# wrote for it: version 1 spelt tags `category`, conditions as one-column
# dicts, and distribution parameters by alias.
EVERY_VERSION = TableSpec(
    "Orders",
    {
        "region": ColSpec(pl.Enum(["UK", "US"]), tags="geo"),
        "carrier": ColSpec(
            pl.Enum(["RM", "UPS"]),
            rules=[ColRule(when=col("region") == "UK", choices=["RM"])],
        ),
        "amount": ColSpec(
            pl.Float64,
            bounds=(0.0, 100.0),
            distribution="normal",
            distribution_params={"mean": 50.0, "std": 10.0},
        ),
        "note": ColSpec(pl.String, nullable=True, null_probability=0.25),
    },
)


@pytest.mark.parametrize("version", range(1, FORMAT_VERSION + 1))
def test_a_spec_file_of_every_version_reads_as_the_spec_it_declared(version):
    spec = from_yaml(GOLDEN / f"spec_v{version}.yaml")
    assert spec == EVERY_VERSION
    for name, column in EVERY_VERSION.columns.items():
        for got, want in zip(spec[name].rules, column.rules, strict=True):
            assert got.when.equals(want.when), (version, name)


def test_todays_format_is_written_byte_for_byte():
    _golden(f"spec_v{FORMAT_VERSION}.yaml", to_yaml(EVERY_VERSION))


# Everything a spec file can say, at once.
RICH = TableSpec(
    "Shipments",
    {
        "id": ColSpec(pl.Int64, bounds=(1, 1_000_000), unique=True),
        "ref": ColSpec(pl.String, format="uuid4", unique=True),
        "country": ColSpec(
            pl.String, format="iso_country", extra_values={"UK (ISO)": 0.02}
        ),
        "status": ColSpec(
            pl.Enum(["NEW", "SENT", "LOST"]), weights=[0.5, 0.45, 0.05], tags="state"
        ),
        "weight_kg": ColSpec(
            pl.Decimal(8, 2), bounds=(0, 999), nullable=True, null_probability=0.1
        ),
        "ratio": ColSpec(pl.Float32, bounds=(0.0, 1.0), nan_probability=0.01),
        "sent_at": ColSpec(pl.Datetime("us", "UTC"), nullable=True),
        "labels": ColSpec(pl.List(pl.String), list_length=(0, 3), string_length=(1, 8)),
        "dims": ColSpec(
            pl.Struct({"w": pl.Int16, "h": pl.Int16}),
            fields={"w": ColSpec(pl.Int16, bounds=(1, 300))},
        ),
        "code": ColSpec(pl.String, pattern=r"^[A-Z]{3}-\d{4}$", string_length=(8, 8)),
        "parent_id": ColSpec(pl.Int64, nullable=True, null_probability=0.3),
    },
    foreign_keys=[ForeignKey("parent_id", references="self", ref_columns="id")],
    checks=[Check(col("ratio") <= 1.0, name="ratio_at_most_one")],
    unique_together=[("country", "code")],
    order_by=["id"],
)

TREE = TableSpec(
    "Categories",
    {"node": ColSpec(pl.String), "parent": ColSpec(pl.String)},
    hierarchy=Hierarchy(child="node", parent="parent", max_depth=4),
)


def test_everything_a_spec_file_says_is_written_byte_for_byte():
    _golden("spec_rich.yaml", to_yaml(RICH))
    assert from_yaml(GOLDEN / "spec_rich.yaml") == RICH


def test_a_registry_file_is_written_byte_for_byte():
    registry = Registry(RICH, TREE, categories=polspec.CatSpec(enums={"STATE": ["A"]}))
    _golden("registry.yaml", registry.to_yaml())


def test_schema_infer_writes_its_golden_draft(tmp_path, capsys):
    """What `polspec schema infer` writes, rounding and all, for one frame."""
    n = 1_000
    frame = pl.DataFrame(
        {
            "order_id": range(1, n + 1),
            "status": [["NEW", "PAID", "PAID", "SHIPPED"][i % 4] for i in range(n)],
            "total": [round(3.17 + (i * 37 % 997) * 9.87, 2) for i in range(n)],
            "email": [f"user{i % 300}@example.com" for i in range(n)],
            "note": [None if i % 9 else f"note {i}" for i in range(n)],
        }
    )
    frame.write_parquet(tmp_path / "orders.parquet")
    out = tmp_path / "orders.yaml"
    flags = ["--weights", "--keys", "--order"]
    assert (
        main(
            [
                "schema",
                "infer",
                str(tmp_path / "orders.parquet"),
                "-o",
                str(out),
                *flags,
            ]
        )
        == 0
    )
    _golden("schema_infer.yaml", out.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The command line: --json shapes and exit codes
# ---------------------------------------------------------------------------


def _type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "array" if isinstance(value, list) else "object"


# What varies with the finding, not the document: held to its type only.
OPEN = {"details", "samples"}


def _shape(value: object) -> object:
    """A document's keys and the types of their values, not the values."""
    if isinstance(value, dict):
        return {k: _type(v) if k in OPEN else _shape(v) for k, v in value.items()}
    if isinstance(value, list):
        shapes = {json.dumps(_shape(v), sort_keys=True) for v in value}
        return [json.loads(s) for s in sorted(shapes)]
    return _type(value)


@pytest.fixture
def project(tmp_path):
    """Two specs, data that passes one and fails the other, and a changed spec."""
    passing = TableSpec("Customers", {"id": ColSpec(pl.Int64, bounds=(1, 100))})
    failing = TableSpec(
        "Orders",
        {
            "total": ColSpec(pl.Float64, bounds=(0.0, 10.0)),
            "status": ColSpec(pl.Enum(["NEW", "PAID"]), weights=[0.5, 0.5]),
        },
    )
    specs, data = tmp_path / "specs", tmp_path / "data"
    specs.mkdir()
    data.mkdir()
    to_yaml(passing, specs / "Customers.yaml")
    to_yaml(failing, specs / "Orders.yaml")
    generate(passing, 50, seed=1).write_csv(data / "Customers.csv")
    pl.DataFrame({"total": [1.0, -5.0, 12.0] * 30, "status": ["NEW"] * 90}).write_csv(
        data / "Orders.csv"
    )
    changed = failing.with_columns(total=ColSpec(pl.Float64, bounds=(0.0, 5.0)))
    to_yaml(changed, tmp_path / "Orders_changed.yaml")
    return tmp_path


def _documents(root: Path) -> dict[str, list[str]]:
    specs, data = root / "specs", root / "data"
    return {
        "validate": ["validate", f"{specs}/Orders.yaml", f"{data}/Orders.csv"],
        "validate --all": ["validate", "--all", str(specs), str(data)],
        "diff": ["diff", f"{specs}/Orders.yaml", f"{root}/Orders_changed.yaml"],
        "drift": ["drift", f"{specs}/Orders.yaml", f"{data}/Orders.csv"],
        "drift --all": ["drift", "--all", str(specs), str(data)],
    }


def test_every_json_document_has_its_golden_shape(project, capsys):
    shapes = {}
    for name, argv in _documents(project).items():
        capsys.readouterr()
        main([*argv, "--json"])
        document = json.loads(capsys.readouterr().out)
        assert document["schema_version"] == 1, name
        shapes[name] = _shape(document)
    _golden("cli_json.json", json.dumps(shapes, indent=1, sort_keys=True) + "\n")


def test_every_exit_code_means_what_the_stability_page_says(project, capsys):
    specs, data = project / "specs", project / "data"
    assert main(["validate", f"{specs}/Customers.yaml", f"{data}/Customers.csv"]) == 0
    assert main(["validate", f"{specs}/Orders.yaml", f"{data}/Orders.csv"]) == 1
    assert main(["validate", f"{specs}/Orders.yaml", f"{data}/nope.csv"]) == 1
    assert "error: " in capsys.readouterr().err
    for wrong in (
        ["nope"],
        ["validate", "--nope"],
        ["diff", "a", "b", "--fail-on", "x"],
    ):
        with pytest.raises(SystemExit) as exited:
            main(wrong)
        assert exited.value.code == 2, wrong


# ---------------------------------------------------------------------------
# Finding and drift codes
# ---------------------------------------------------------------------------


def _documented(heading: str) -> set[str]:
    """The codes in the first column of the table under `heading`."""
    page = (DOCS / "reference" / "errors.md").read_text(encoding="utf-8")
    section = page.split(f"## {heading}\n", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"^\| `([a-z_]+)` \|", section, re.M))


@pytest.mark.parametrize(
    ("heading", "finding"),
    [("Finding codes", Finding), ("Drift codes", DriftFinding)],
)
def test_the_codes_and_their_documentation_are_the_same_list(heading, finding):
    codes = set(typing.get_args(typing.get_type_hints(finding)["code"]))
    assert codes == _documented(heading)
