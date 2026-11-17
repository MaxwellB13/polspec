"""The command line as a whole. Each command has a file of its own beside
this one; the CLI is thin argument-parsing over functions with their own
tests, so those cover what it adds: reading and writing files, templating,
and exit status.
"""

import pytest
from helpers import run_cli

# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli("--version")
    assert exc.value.code == 0
    assert "polspec" in capsys.readouterr().out


def test_no_command_is_an_error():
    with pytest.raises(SystemExit) as exc:
        run_cli()
    assert exc.value.code != 0
