"""The spec-file reference against the reader's own vocabulary.

Every other reference test asks the public interface. This one cannot: the
dtype names and predicate operations a file may use have no public list,
and a new one added to the reader without a row on the page is exactly what
this exists to catch. So it reads the reader's own tables, and is the one
docs test the public-API policy lists as an exception.
"""

import re
from pathlib import Path

from polspec import FORMAT_VERSION

REFERENCE = Path(__file__).resolve().parents[2] / "docs" / "reference"
TABLE_KEY = re.compile(r"^\| `([^`]+)` \|(?: `([^`]+)` \|)?", re.M)


def _section(page: Path, heading: str) -> str:
    """The text under `heading`, up to the next heading of any level."""
    text = page.read_text(encoding="utf-8")
    start = text.index(f"\n{heading}\n") + len(heading) + 2
    following = re.search(r"^#{1,6} ", text[start:], re.M)
    return text[start : start + following.start()] if following else text[start:]


def _first_column(page: Path, heading: str) -> list[str]:
    return [m.group(1) for m in TABLE_KEY.finditer(_section(page, heading))]


def test_the_spec_file_reference_names_every_dtype_operation_and_version():
    from polspec.expr import KNOWN_OPS
    from polspec.serialization.dtypes import _BUILDERS, DTYPE_NAMES

    page = REFERENCE / "spec-files.md"
    dtypes = _section(page, "## Dtypes")
    for name in set(DTYPE_NAMES.values()) | set(_BUILDERS):
        assert f"`{name}`" in dtypes, name
    operations = {
        op.strip("` ")
        for cell in _first_column(page, "## Predicates")
        for op in cell.split(",")
    }
    written_out = {
        op.strip("` ")
        for row in re.findall(
            r"^\| (`[^|]+`) \|", _section(page, "## Predicates"), re.M
        )
        for op in row.split(",")
    }
    assert operations | written_out == set(KNOWN_OPS)
    assert f"`version: {FORMAT_VERSION}`" in page.read_text(encoding="utf-8")
    assert _first_column(page, "## Versions") == [] and re.findall(
        r"^\| (\d+) \|", _section(page, "## Versions"), re.M
    ) == [str(v) for v in range(1, FORMAT_VERSION + 1)]
