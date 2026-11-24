"""Spec files of another format version, read by today's polspec.

polspec 2.0 reads format version 4 alone. An older file -- 1, 2 or 3, from
polspec 0.13 and before -- is refused, naming its version and saying that
polspec 1.x reads it. `fixtures/` holds files old polspecs wrote, one of
every kind, so the refusal is pinned against real files. A fixture is never
regenerated; it is what an old polspec wrote.

*Since 2.0: 1.x migrated versions 1 to 3 as it read them.*
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polspec import (
    FORMAT_VERSION,
    CatSpec,
    ColSpec,
    Registry,
    SerializationError,
    TableSpec,
    from_dict,
    from_yaml,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("read", "fixture", "version"),
    [
        (from_yaml, "spec_v2.yaml", 2),
        (from_yaml, "spec_v3_temporal.yaml", 3),
        (CatSpec.from_yaml, "catspec_v2.yaml", 2),
        (Registry.from_yaml, "registry_v2.yaml", 2),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_an_old_file_is_refused_naming_its_version_and_1x(read, fixture, version):
    with pytest.raises(SerializationError, match=f"format version {version}") as raised:
        read(FIXTURES / fixture)
    message = str(raised.value)
    assert "polspec 1.x reads it" in message
    assert fixture in message  # where, for a person with many files


_OLD = {
    "spec": (from_dict, {"name": "T", "columns": {"a": {"dtype": "Int64"}}}),
    "catspec": (CatSpec.from_dict, {"enums": {"S": ["a"]}}),
    "registry": (Registry.from_dict, {"specs": {"T": {"columns": {"a": "Int64"}}}}),
}


@pytest.mark.parametrize("version", range(1, FORMAT_VERSION))
@pytest.mark.parametrize("kind", sorted(_OLD))
def test_every_kind_of_file_is_refused_at_every_old_version(kind, version):
    read, data = _OLD[kind]
    with pytest.raises(SerializationError, match=f"format version {version}"):
        read({"version": version, **data})


@pytest.mark.parametrize("kind", sorted(_OLD))
def test_every_kind_reads_at_the_current_version(kind):
    read, data = _OLD[kind]
    assert read({"version": FORMAT_VERSION, **data}) is not None


@pytest.mark.parametrize(
    ("read", "fixture"),
    [(from_yaml, "spec_v1.yaml"), (CatSpec.from_yaml, "catspec_v1.yaml")],
)
def test_a_format_1_file_with_no_version_key_fails_loudly_and_says_why(read, fixture):
    """Format 1 had no `version:` key, so such a file reads as today's
    format; what format 1 spelt differently is an unknown key, and the
    error says where the file can still be read."""
    with pytest.raises(SerializationError, match="Unknown key") as raised:
        read(FIXTURES / fixture)
    assert "may be format 1" in str(raised.value)
    assert "polspec 1.x" in str(raised.value)


def test_a_mapping_with_no_version_reads_as_the_current_format():
    spec = from_dict({"name": "T", "columns": {"a": "Int64"}})
    assert spec == TableSpec("T", {"a": ColSpec(pl.Int64)})


def test_a_versionless_mapping_with_an_unknown_key_names_format_1():
    with pytest.raises(SerializationError, match="may be format 1"):
        from_dict({"name": "T", "columns": {"a": {"dtype": "Int64", "category": "x"}}})


def test_a_versioned_file_with_an_unknown_key_does_not_mention_format_1():
    with pytest.raises(SerializationError, match="Unknown key") as raised:
        from_dict(
            {
                "version": FORMAT_VERSION,
                "name": "T",
                "columns": {"a": {"dtype": "Int64", "category": "x"}},
            }
        )
    assert "format 1" not in str(raised.value)


@pytest.mark.parametrize("version", [0, -1, "4", 4.0, True])
def test_a_version_that_is_not_a_positive_integer_is_refused(version):
    with pytest.raises(SerializationError, match="positive integer"):
        from_dict({"version": version, "name": "T", "columns": {}})


def test_a_file_that_is_not_a_mapping_is_refused():
    with pytest.raises(SerializationError, match="expected a mapping"):
        from_dict(["not", "a", "spec"])  # type: ignore[arg-type]
