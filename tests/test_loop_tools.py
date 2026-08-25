"""The model-facing tools are thin, safe adapters over ACT."""

from __future__ import annotations

from pathlib import Path

from tars.loop.tools import Toolbox
from tars.loop.types import ToolCall


def test_read_file_returns_contents(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello\n", encoding="utf-8")
    result = Toolbox(tmp_path).dispatch(
        ToolCall("1", "read_file", {"path": "note.txt"})
    )
    assert result.ok
    assert result.output == "hello\n"


def test_read_file_refuses_escape(tmp_path: Path) -> None:
    result = Toolbox(tmp_path).dispatch(ToolCall("1", "read_file", {"path": "../x"}))
    assert not result.ok
    assert "outside" in result.text


def test_list_dir_is_sorted_and_relative(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("", encoding="utf-8")
    (tmp_path / "a.txt").write_text("", encoding="utf-8")
    result = Toolbox(tmp_path).dispatch(ToolCall("1", "list_dir", {"path": "."}))
    assert result.ok
    assert result.output.splitlines() == ["a.txt", "b.txt"]


def test_apply_edit_previews_confirms_and_applies(tmp_path: Path) -> None:
    path = tmp_path / "code.py"
    path.write_text("x = 1\n", encoding="utf-8")
    seen: list[str] = []

    def confirm(diff: str) -> bool:
        seen.append(diff)
        return True

    tools = Toolbox(tmp_path, confirm=confirm)
    result = tools.dispatch(
        ToolCall(
            "1",
            "apply_edit",
            {"path": "code.py", "hunks": [{"old": "x = 1", "new": "x = 2"}]},
        )
    )
    assert result.ok
    assert seen and "-x = 1" in seen[0]
    assert path.read_text(encoding="utf-8") == "x = 2\n"


def test_apply_edit_reports_decline_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "code.py"
    path.write_text("x = 1\n", encoding="utf-8")
    result = Toolbox(tmp_path, confirm=lambda _diff: False).dispatch(
        ToolCall(
            "1",
            "apply_edit",
            {"path": "code.py", "hunks": [{"old": "x = 1", "new": "x = 2"}]},
        )
    )
    assert not result.ok
    assert "declined" in result.text
    assert path.read_text(encoding="utf-8") == "x = 1\n"


def test_run_command_uses_argv_and_root(tmp_path: Path) -> None:
    result = Toolbox(tmp_path, confirm=lambda _preview: True).dispatch(
        ToolCall("1", "run_command", {"argv": ["pwd"]})
    )
    assert result.ok
    assert Path(result.output.strip()) == tmp_path.resolve()


def test_run_command_shows_argv_and_decline_prevents_execution(tmp_path: Path) -> None:
    seen: list[str] = []

    def confirm(preview: str) -> bool:
        seen.append(preview)
        return False

    result = Toolbox(tmp_path, confirm=confirm).dispatch(
        ToolCall("1", "run_command", {"argv": ["touch", "ran.txt"]})
    )
    assert not result.ok
    assert "declined" in result.text
    assert seen == ["touch ran.txt"]
    assert not (tmp_path / "ran.txt").exists()


def test_unknown_tool_is_a_tool_error(tmp_path: Path) -> None:
    result = Toolbox(tmp_path).dispatch(ToolCall("1", "nope"))
    assert not result.ok
    assert "unknown tool" in result.text
