"""The documentation cannot quietly fall behind the code.

The API reference is rendered from docstrings, so it cannot describe a
signature that does not exist -- but it can silently *omit* one. These tests
pin the other direction: everything polspec exports is documented, every page
the nav names exists, every relative link between pages resolves, every yaml
example carries the format version `to_yaml()` writes today, and the files a
language model reads are the ones the current docs produce.
"""

import importlib.util
import inspect
import re
import tomllib
from pathlib import Path

import polspec
from polspec.serialization import FORMAT_VERSION

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
API = DOCS / "reference" / "api"

DOC_PAGES = sorted(DOCS.rglob("*.md"))
DIRECTIVE = re.compile(r"^::: +polspec\.(\w+)\s*$", re.M)
RELATIVE_LINK = re.compile(r"\]\((?!https?:|/|#)([^)#]+\.md)(#[^)]*)?\)")
# A link to a heading, on this page (`](#anchor)`) or another
# (`](page.md#anchor)`). The second group of RELATIVE_LINK catches the
# cross-page form; this one catches both.
ANCHOR_LINK = re.compile(r"\]\((?!https?:)([^)#]*\.md)?#([^)]+)\)")
HEADING = re.compile(r"^#{1,6} +(.+?)\s*$", re.M)
# A `version:` line in a yaml fence, or `version: N` quoted in prose.
YAML_VERSION = re.compile(r"(?:^|`)version: (\d+)(?:$|`)", re.M)


def _nav_pages(nav: object, into: list[str]) -> list[str]:
    """Every page path the nav names, however deeply it is nested."""
    if isinstance(nav, str):
        into.append(nav)
    elif isinstance(nav, dict):
        for value in nav.values():
            _nav_pages(value, into)
    elif isinstance(nav, list):
        for item in nav:
            _nav_pages(item, into)
    return into


def test_every_exported_name_is_in_the_api_reference():
    """The verification the v0.2 plan asks for: the API reference renders
    every public name in `polspec.__all__`.
    """
    documented = {
        name
        for page in API.glob("*.md")
        for name in DIRECTIVE.findall(page.read_text(encoding="utf-8"))
    }
    missing = sorted(set(polspec.__all__) - documented)
    assert not missing, f"exported but not in docs/reference/api: {missing}"

    # And nothing documented has since stopped being exported.
    stale = sorted(documented - set(polspec.__all__))
    assert not stale, f"documented but no longer exported: {stale}"


def test_the_api_index_lists_every_name():
    index = (API / "index.md").read_text(encoding="utf-8")
    missing = [n for n in polspec.__all__ if f"[`{n}`]" not in index]
    assert not missing, f"absent from the API reference index table: {missing}"


def test_every_page_is_reachable_from_the_nav():
    config = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))
    listed = set(_nav_pages(config["project"]["nav"], []))
    on_disk = {str(p.relative_to(DOCS)).replace("\\", "/") for p in DOC_PAGES}
    assert on_disk - listed == set(), f"page not in the nav: {sorted(on_disk - listed)}"
    assert listed - on_disk == set(), (
        f"nav names a missing page: {sorted(listed - on_disk)}"
    )


def test_every_relative_link_resolves():
    broken = []
    for page in DOC_PAGES:
        for target, _anchor in RELATIVE_LINK.findall(page.read_text(encoding="utf-8")):
            if not (page.parent / target).resolve().exists():
                broken.append(f"{page.relative_to(DOCS)} -> {target}")
    assert not broken, f"broken links: {broken}"


def _slug(heading: str) -> str:
    """The anchor the site generator gives a heading.

    Lowercased, with everything but words, spaces and hyphens dropped, and
    runs of whitespace collapsed to one hyphen -- so `## Lazy output --
    `scan()`` is `#lazy-output-scan`, the dash and the backticks leaving
    nothing behind. Mirrors what `zensical build --strict` checks, so a
    broken anchor fails here rather than in CI.
    """
    text = re.sub(r"[^\w\s-]", "", heading.replace("—", " ").replace("--", " "))
    return re.sub(r"[\s_]+", "-", text.strip().lower())


def _anchors_of(page: Path) -> set[str]:
    """Every heading on `page`, as the anchor a link would use."""
    return {_slug(h) for h in HEADING.findall(page.read_text(encoding="utf-8"))}


def test_every_anchor_link_points_at_a_heading():
    """A link to a heading that does not exist builds, renders, and goes
    nowhere -- and `zensical build --strict` fails on it, which is a slow
    way to find out.
    """
    known: dict[Path, set[str]] = {}
    broken = []
    for page in DOC_PAGES:
        if page.name.startswith("llms"):
            continue  # generated from the pages below
        for target, anchor in ANCHOR_LINK.findall(page.read_text(encoding="utf-8")):
            destination = (page.parent / target).resolve() if target else page
            if not destination.exists():
                continue  # the link test above reports this one
            anchors = known.setdefault(destination, _anchors_of(destination))
            if anchor not in anchors:
                broken.append(f"{page.relative_to(DOCS)} -> {target}#{anchor}")
    assert not broken, f"links to headings that do not exist: {broken}"


def test_every_yaml_example_carries_the_current_format_version():
    """A page that shows what `to_yaml()` writes shows today's version. The
    prose that mentions the number is held to the same standard.
    """
    stale = []
    for page in DOC_PAGES:
        if page.name.startswith("llms"):
            continue  # generated from the pages below
        for version in YAML_VERSION.findall(page.read_text(encoding="utf-8")):
            if int(version) != FORMAT_VERSION:
                stale.append(f"{page.relative_to(DOCS)}: version: {version}")
    assert not stale, f"yaml examples not at FORMAT_VERSION={FORMAT_VERSION}: {stale}"


def _generator():
    """The `scripts/generate_llms_txt.py` module, which is not importable."""
    path = ROOT / "scripts" / "generate_llms_txt.py"
    spec = importlib.util.spec_from_file_location("generate_llms_txt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_public_name_carries_a_docstring():
    """The API reference and `llms-full.txt` are both built from docstrings,
    so an undocumented public method is a blank entry in each.
    """
    undocumented = []
    for name in polspec.__all__:
        obj = getattr(polspec, name)
        if not inspect.getdoc(obj):
            undocumented.append(name)
        if not inspect.isclass(obj):
            continue
        for member_name, member in sorted(vars(obj).items()):
            if member_name.startswith("_"):
                continue
            func = getattr(member, "__func__", member)
            if callable(func) and not inspect.getdoc(func):
                undocumented.append(f"{name}.{member_name}")
    assert not undocumented, f"public but undocumented: {undocumented}"


def test_the_llms_files_are_built_not_committed():
    """The docs workflow builds them before the site; a committed copy would
    be a stale one, and every edit used to regenerate 400 KB of it."""
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert {"docs/llms.txt", "docs/llms-full.txt"} <= set(ignored)
    workflow = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    generate = workflow.index("scripts/generate_llms_txt.py\n")
    assert generate < workflow.index("zensical build")


def test_llms_full_txt_carries_every_page():
    _index, full = _generator().build()
    config = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))
    for _section, title, _relative in _generator().walk_nav(config["project"]["nav"]):
        assert title in full, f"{title} is missing from llms-full.txt"
    assert len(full) > 100_000  # the pages' text, not only their titles


def test_polspec_imports_from_a_checkout_that_was_never_installed():
    """How the docs workflow imports it, to read the docstrings: without
    package metadata there is no version to report, and no error."""
    import subprocess
    import sys

    probe = (
        "import importlib.metadata as m\n"
        "def missing(name): raise m.PackageNotFoundError(name)\n"
        "m.version = missing\n"
        "import polspec\n"
        "print(polspec.__version__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0+unknown"


def test_llms_txt_follows_the_convention():
    text, _full = _generator().build()
    lines = text.splitlines()
    assert lines[0].startswith("# "), "llms.txt opens with the project name as H1"
    assert any(line.startswith("> ") for line in lines[:5]), (
        "llms.txt carries a blockquote summary"
    )
    # Every page in the nav is reachable from the index.
    config = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))
    site = config["project"]["site_url"].rstrip("/") + "/"
    generator = _generator()
    for _section, title, relative in generator.walk_nav(config["project"]["nav"]):
        url = generator.url_for(site, relative)
        assert f"[{title}]({url})" in text, f"{title} is missing from llms.txt"


def test_every_docstring_parses_as_numpy():
    """What the API reference renders is what griffe parses, and a line of
    prose left under *Parameters* renders as a parameter named `A` or
    `that`. The strict docs build only logs that, so it is pinned here:
    every docstring in the package parses without a warning."""
    import logging

    import griffe

    warnings: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            warnings.append(record.getMessage())

    logger = logging.getLogger("griffe")
    handler = Collect(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        package = griffe.load(
            "polspec", search_paths=[str(ROOT / "python")], docstring_parser="numpy"
        )
        seen: set[str] = set()
        stack = [package]
        while stack:
            obj = stack.pop()
            if obj.path in seen:
                continue
            seen.add(obj.path)
            if obj.docstring is not None:
                obj.docstring.parsed  # noqa: B018 - parsing is what warns
            stack.extend(m for m in obj.members.values() if not m.is_alias)
    finally:
        logger.removeHandler(handler)
    assert not warnings, "\n".join(sorted(set(warnings)))


def test_the_package_says_which_version_it_is():
    """`polspec.__version__` is the installed version, which is the one
    `pyproject.toml` sets -- a stale build fails here, not in a bug report."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert polspec.__version__ == pyproject["project"]["version"]


def test_every_module_is_on_the_architecture_page():
    """The module table is the map of the package; a module added without a
    row is how it fell six modules behind before."""
    page = (DOCS / "explanation" / "architecture.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"^\| `([A-Za-z_]+)` \|", page, re.M))
    package = ROOT / "python" / "polspec"
    modules = {
        p.stem
        for p in package.iterdir()
        if (p.suffix == ".py" and p.stem not in ("__init__", "__main__"))
        or (p.is_dir() and (p / "__init__.py").exists())
    }
    assert modules - listed == set(), "add a row to architecture.md"
    assert listed - modules == set(), "architecture.md names a module that is gone"


def test_every_test_file_lives_in_a_folder_and_says_what_it_covers():
    """`tests/` is grouped by what a file covers, and each file's docstring is
    the map -- so a file at the top level, or one without a docstring, is how
    the layout starts to fall apart again."""
    import ast

    tests = ROOT / "tests"
    assert sorted(p.name for p in tests.glob("test_*.py")) == []
    undocumented = [
        path.relative_to(tests).as_posix()
        for path in sorted(tests.rglob("test_*.py"))
        if not ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert undocumented == []


# ---------------------------------------------------------------------------
# The reference pages that describe the code rather than render it
# ---------------------------------------------------------------------------

REFERENCE = DOCS / "reference"
TABLE_KEY = re.compile(r"^\| `([^`]+)` \|(?: `([^`]+)` \|)?", re.M)


def _section(page: Path, heading: str) -> str:
    """The text under `heading`, up to the next heading of any level."""
    text = page.read_text(encoding="utf-8")
    start = text.index(f"\n{heading}\n") + len(heading) + 2
    following = re.search(r"^#{1,6} ", text[start:], re.M)
    return text[start : start + following.start()] if following else text[start:]


def _first_column(page: Path, heading: str) -> list[str]:
    return [m.group(1) for m in TABLE_KEY.finditer(_section(page, heading))]


def test_the_colspec_field_reference_has_a_row_per_field():
    """One row per `ColSpec` field, in declaration order, each with the key
    a spec file writes it under."""
    import dataclasses

    from polspec import ColSpec
    from polspec.serialization.fields import COLSPEC_FIELDS

    rows = TABLE_KEY.findall(
        (REFERENCE / "colspec-fields.md").read_text(encoding="utf-8")
    )
    assert [field for field, _ in rows] == [f.name for f in dataclasses.fields(ColSpec)]
    file_key = {f.attribute: f.name for f in COLSPEC_FIELDS}
    assert dict(rows) == file_key


def test_the_spec_file_reference_names_every_key_the_reader_takes():
    """Each table on the page is the reader's own list of keys for that
    kind of mapping -- no more, no fewer."""
    from polspec.serialization import __dict__ as serialization
    from polspec.serialization import fields
    from polspec.serialization.fields import (
        CHECK_FIELDS,
        COLRULE_FIELDS,
        COLSPEC_FIELDS,
        FK_FIELDS,
        TABLESPEC_FIELDS,
    )

    page = REFERENCE / "spec-files.md"
    written = {f.name for f in FK_FIELDS} - {"target"}  # never written to a file
    expected = {
        "## A spec file": {f.name for f in TABLESPEC_FIELDS} | set(fields.FILE_KEYS),
        "### A column": {f.name for f in COLSPEC_FIELDS},
        "### A rule": {f.name for f in COLRULE_FIELDS},
        "### A check or validator": {f.name for f in CHECK_FIELDS},
        "### A foreign key": written,
        "### A hierarchy": {f.name for f in fields.HIERARCHY_FIELDS},
        "## A category registry": set(serialization["_CATSPEC_KEYS"]),
        "## A registry of specs": set(serialization["_REGISTRY_KEYS"]),
    }
    for heading, keys in expected.items():
        assert set(_first_column(page, heading)) == keys, heading


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


def test_the_cli_reference_is_current():
    """`docs/reference/cli.md` is generated from the parser, so a flag added
    or reworded without regenerating it is a failure here."""
    path = ROOT / "scripts" / "generate_cli_reference.py"
    spec = importlib.util.spec_from_file_location("generate_cli_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (REFERENCE / "cli.md").read_text(encoding="utf-8") == module.build(), (
        "docs/reference/cli.md is stale: run `uv run python scripts/generate_llms_txt.py`"
    )
