"""Generates `docs/reference/cli.md` from the `polspec` command's own parser.

Every command, every argument, its help and its default are read off the
argparse parser the command runs, so the reference cannot list a flag that
is gone or miss one that was added. `generate_llms_txt.py` runs this first,
and `tests/docs/test_docs.py` fails if the committed page is out of date.

Run with `uv run python scripts/generate_cli_reference.py`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "reference" / "cli.md"

# argparse wraps usage to the terminal's width; a fixed one keeps the page
# the same on every machine.
os.environ["COLUMNS"] = "88"

HEADER = """# Command line

Every `polspec` command and argument, generated from the command's own
parser -- `polspec <command> --help` prints the same. [Command line](../how-to/cli.md)
shows the workflows, and what the exit codes mean.
"""


def _subcommands(
    parser: argparse.ArgumentParser,
) -> list[tuple[str, str, argparse.ArgumentParser]]:
    """The parser's subcommands, as (name, help, parser), in declared order."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            helps = {a.dest: a.help or "" for a in action._choices_actions}
            return [
                (name, helps.get(name, ""), sub) for name, sub in action.choices.items()
            ]
    return []


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _sentence(text: str) -> str:
    """`text` as a sentence: one line, ending in a full stop."""
    text = _cell(text)
    return text if not text or text.endswith((".", "!", "?")) else text + "."


def _argument_rows(parser: argparse.ArgumentParser) -> list[str]:
    rows = []
    for action in parser._actions:
        if isinstance(action, (argparse._HelpAction, argparse._SubParsersAction)):
            continue
        if action.option_strings:
            name = ", ".join(f"`{flag}`" for flag in action.option_strings)
            if action.nargs != 0:
                metavar = action.metavar or action.dest.upper()
                name += f" `{metavar}`"
        else:
            name = f"`{action.metavar or action.dest}`"
        help_text = action.help or ""
        if action.choices is not None and not isinstance(
            action, argparse._SubParsersAction
        ):
            choices = ", ".join(f"`{c}`" for c in action.choices)
            help_text += f" One of {choices}."
        if action.required and action.option_strings:
            help_text = f"**Required.** {help_text}"
        rows.append(f"| {name} | {_cell(help_text)} |")
    return rows


def _section(path: str, help_text: str, parser: argparse.ArgumentParser) -> list[str]:
    usage = parser.format_usage().removeprefix("usage: ").strip()
    lines = [f"## `{path}`", ""]
    if help_text:
        lines.extend([_sentence(help_text), ""])
    lines.extend(["```", usage, "```", ""])
    rows = _argument_rows(parser)
    if rows:
        lines.extend(["| Argument | Meaning |", "|:--|:--|", *rows, ""])
    return lines


def build() -> str:
    """The page, as it should be committed."""
    from polspec.cli import _build_parser

    parser = _build_parser()
    lines = [HEADER]
    lines.extend(_section("polspec", parser.description or "", parser)[2:])

    def walk(prefix: str, node: argparse.ArgumentParser) -> None:
        for name, help_text, sub in _subcommands(node):
            path = f"{prefix} {name}"
            if _subcommands(sub):
                lines.extend([f"## `{path}`", "", _sentence(help_text), ""])
                walk(path, sub)
            else:
                lines.extend(_section(path, help_text, sub))

    walk("polspec", parser)
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    PAGE.write_text(build(), encoding="utf-8")
    print(
        f"{PAGE.relative_to(ROOT)}: {len(PAGE.read_text(encoding='utf-8').splitlines())} lines"
    )


if __name__ == "__main__":
    main()
