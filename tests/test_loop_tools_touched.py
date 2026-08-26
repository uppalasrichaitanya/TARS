"""run_command now reports file paths in addition to the command."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tars.loop.tools import Toolbox
from tars.loop.types import ToolCall


def test_run_command_touched_includes_git_file(tmp_path: Path) -> None:
    # Init a git repo so git status enrichment fires
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("hi", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init", "-q"], cwd=tmp_path, check=True)

    toolbox = Toolbox(root=tmp_path, confirm=lambda _d: True)
    result = toolbox.dispatch(
        ToolCall("1", "run_command", {"argv": ["sh", "-c", "echo hello > newfile.txt"]})
    )
    assert result.ok
    # Command display first, then file path
    assert "sh -c" in result.touched[0]
    assert "newfile.txt" in result.touched


def test_run_command_touched_without_git_is_just_command(tmp_path: Path) -> None:
    # /tmp subdir without git — git status fails, only command is reported
    sub = tmp_path / "nogit"
    sub.mkdir()
    toolbox = Toolbox(root=sub, confirm=lambda _d: True)
    result = toolbox.dispatch(ToolCall("1", "run_command", {"argv": ["echo", "hi"]}))
    assert result.ok
    assert result.touched == ("echo hi",)
