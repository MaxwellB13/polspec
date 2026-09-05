"""The documentation cannot quietly fall behind the code.

The API reference is rendered from docstrings, so it cannot describe a
signature that does not exist -- but it can silently *omit* one. These tests
pin the other direction: everything polspec exports is documented, every page
the nav names exists, every relative link between pages resolves, and the
files a language model reads are the ones the current docs produce.
"""

import importlib.util
import inspect
import re
import tomllib
from pathlib import Path

import polspec

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
API = DOCS / "reference" / "api"

DOC_PAGES = sorted(DOCS.rglob("*.md"))
DIRECTIVE = re.compile(r"^::: +polspec\.(\w+)\s*$", re.M)
RELATIVE_LINK = re.compile(r"\]\((?!https?:|/|#)([^)#]+\.md)(#[^)]*)?\)")


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


def test_the_llms_files_are_current():
    """`llms.txt` and `llms-full.txt` are generated and committed, so a page
    added or reworded without regenerating them is a failure here.
    """
    index, full = _generator().build()
    assert (DOCS / "llms.txt").read_text(encoding="utf-8") == index, (
        "docs/llms.txt is stale: run `uv run python scripts/generate_llms_txt.py`"
    )
    assert (DOCS / "llms-full.txt").read_text(encoding="utf-8") == full, (
        "docs/llms-full.txt is stale: run `uv run python scripts/generate_llms_txt.py`"
    )


def test_llms_txt_follows_the_convention():
    text = (DOCS / "llms.txt").read_text(encoding="utf-8")
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
