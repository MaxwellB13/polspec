"""Generates `docs/llms.txt` and `docs/llms-full.txt`.

`llms.txt` is the convention from <https://llmstxt.org>: a markdown index at
the site root that tells a language model what a project is and where its
documentation lives. `llms-full.txt` is the companion -- every page's text in
one file, so a model can take the whole thing in one fetch instead of
crawling.

Both are derived from the nav in `zensical.toml` and the pages themselves, so
neither can describe a page that no longer exists or miss one that does.

The API reference needs special handling. Those pages are `::: polspec.X`
directives that mkdocstrings fills in at build time, so their *source* is
empty -- and the signatures are exactly what a model reading this needs. So
the directives are expanded here from the live objects.

Run with `uv run python scripts/generate_llms_txt.py`;
`tests/test_docs.py` fails if the committed files are out of date.
"""

from __future__ import annotations

import importlib
import inspect
import posixpath
import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INDEX = DOCS / "llms.txt"
FULL = DOCS / "llms-full.txt"

DIRECTIVE = re.compile(r"^::: +([\w.]+)\s*$", re.M)
SNIPPET = re.compile(r'^--8<-- +"(.+)"\s*$', re.M)
MD_LINK = re.compile(r"\]\((?!https?:|/|#)([^)#]+)\.md(#[^)]*)?\)")


# ---------------------------------------------------------------------------
# The nav, flattened
# ---------------------------------------------------------------------------


def walk_nav(node: Any, section: str = "") -> list[tuple[str, str, str]]:
    """Every page the nav names, as (section, title, path)."""
    out: list[tuple[str, str, str]] = []
    if isinstance(node, list):
        for item in node:
            out.extend(walk_nav(item, section))
    elif isinstance(node, dict):
        for title, value in node.items():
            if isinstance(value, str):
                out.append((section, title, value))
            else:
                # A nested group keeps the outermost section's name, so
                # "Reference / API / Columns" files under "Reference".
                out.extend(walk_nav(value, section or title))
    return out


# ---------------------------------------------------------------------------
# Page text
# ---------------------------------------------------------------------------


def render_object(path: str) -> str:
    """One documented object as signature and docstring."""
    module_name, _, attr = path.rpartition(".")
    obj = getattr(importlib.import_module(module_name), attr)

    lines: list[str] = []
    try:
        lines.append(f"### {attr}{inspect.signature(obj)}")
    except (TypeError, ValueError):
        lines.append(f"### {attr}")
    doc = inspect.getdoc(obj)
    if doc:
        lines.append("")
        lines.append(doc)

    if inspect.isclass(obj):
        for name, member in sorted(vars(obj).items()):
            if name.startswith("_"):
                continue
            func = getattr(member, "__func__", member)
            if not callable(func):
                continue
            member_doc = inspect.getdoc(func)
            if not member_doc:
                continue
            try:
                signature = str(inspect.signature(func))
            except (TypeError, ValueError):
                signature = "(...)"
            summary = member_doc.strip().split("\n\n")[0].replace("\n", " ")
            lines.append("")
            lines.append(f"- `{attr}.{name}{signature}` -- {summary}")
    return "\n".join(lines)


def page_text(relative: str, site_url: str) -> str:
    """A page's markdown, with directives, snippets and links resolved.

    A relative link is meaningless once the page is lifted out of the site,
    so each one becomes the URL it would have resolved to.
    """
    text = (DOCS / relative).read_text(encoding="utf-8")
    text = DIRECTIVE.sub(lambda m: render_object(m.group(1)), text)
    text = SNIPPET.sub(lambda m: (ROOT / m.group(1)).read_text(encoding="utf-8"), text)

    here = posixpath.dirname(relative)

    def absolute(match: re.Match) -> str:
        target = posixpath.normpath(posixpath.join(here, match.group(1) + ".md"))
        return f"]({url_for(site_url, target)}{match.group(2) or ''})"

    return MD_LINK.sub(absolute, text).strip()


ANY_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def summarise(text: str) -> str:
    """The first sentence of prose under a page's title, as plain text.

    A description sits inside a link in `llms.txt`, so it cannot carry one of
    its own: link syntax is reduced to the words it wrapped. A sentence
    introducing a code block ends in a colon, which reads as truncation once
    the block is gone.
    """
    body = re.sub(r"^#\s+.*$", "", text, count=1, flags=re.M).strip()
    for block in body.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(("#", "```", "|", ">", "!!!", "-", "*")):
            continue
        block = ANY_LINK.sub(r"\1", block)
        sentence = re.split(r"(?<=[.!?])\s", block.replace("\n", " "))[0]
        return re.sub(r"\s+", " ", sentence).strip().rstrip(":")
    return ""


def url_for(site_url: str, relative: str) -> str:
    """The published URL of a page, as zensical lays the site out."""
    stem = relative[: -len(".md")]
    if stem == "index":
        return site_url
    if stem.endswith("/index"):
        stem = stem[: -len("/index")]
    return f"{site_url}{stem}/"


# ---------------------------------------------------------------------------
# The two files
# ---------------------------------------------------------------------------


def build() -> tuple[str, str]:
    config = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))
    project = config["project"]
    site_url = project["site_url"].rstrip("/") + "/"
    pages = walk_nav(project["nav"])

    index: list[str] = [
        f"# {project['site_name']}",
        "",
        f"> {project['site_description']}",
        "",
        "polspec is a Python library with a Rust extension. A schema is declared",
        "once -- as a `FrameSpec` class body or a YAML file -- and the same",
        "declaration both generates data that satisfies it and validates data",
        "against it. Full text of every page below is also available in one file:",
        f"{site_url}llms-full.txt",
        "",
    ]

    full: list[str] = [
        f"# {project['site_name']} documentation",
        "",
        f"> {project['site_description']}",
        "",
        f"Every page of {site_url}, in the order the documentation presents them.",
        "",
    ]

    current = None
    optional: list[str] = []
    for section, title, relative in pages:
        text = page_text(relative, site_url)
        url = url_for(site_url, relative)
        summary = summarise(text)
        entry = f"- [{title}]({url})" + (f": {summary}" if summary else "")

        # The changelog is reference material a model rarely needs in order to
        # use the library, which is exactly what `## Optional` is for.
        if relative == "changelog.md":
            optional.append(entry)
        else:
            heading = section or "Home"
            if heading != current:
                index.extend(["" if current else None, f"## {heading}", ""])
                index = [line for line in index if line is not None]
                current = heading
            index.append(entry)

        full.extend(["---", "", f"# {title}", f"Source: {url}", "", text, ""])

    if optional:
        index.extend(["", "## Optional", "", *optional])

    return "\n".join(index).rstrip() + "\n", "\n".join(full).rstrip() + "\n"


def main() -> None:
    index, full = build()
    INDEX.write_text(index, encoding="utf-8")
    FULL.write_text(full, encoding="utf-8")
    print(f"{INDEX.relative_to(ROOT)}: {len(index.splitlines())} lines")
    print(f"{FULL.relative_to(ROOT)}: {len(full.splitlines())} lines")


if __name__ == "__main__":
    main()
