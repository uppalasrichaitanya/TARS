"""Chat, Plan, and Agent remain distinct interaction contracts."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from tars.loop.session import Session
from tars.providers import LLMResponse, ToolCall, Usage


def _response(text: str, calls=()) -> LLMResponse:
    return LLMResponse(text, tuple(calls), "fake", "fake", "end_turn", Usage())


def test_chat_persists_plain_conversation_without_tools(tmp_path: Path) -> None:
    from tars.loop.modes import ModeService

    seen = []

    def provider(messages, tools, model):
        seen.append((messages, tools, model))
        return _response("hello back")

    session = Session(root=tmp_path, model="fake")
    out = StringIO()
    assert ModeService(provider).chat(session, "hello", output=out) == "hello back"
    assert seen[0][1] == ()
    assert [message.role for message in session.messages] == ["user", "assistant"]
    assert out.getvalue() == "hello back\n"


def test_plan_uses_only_read_only_tools_and_saves_plan(tmp_path: Path) -> None:
    from tars.loop.modes import ModeService

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    responses = iter(
        [
            _response("", [ToolCall("r", "read_file", {"path": "a.py"})]),
            _response("1. Change x. 2. Run tests."),
        ]
    )
    offered = []

    def provider(messages, tools, model):
        offered.extend(tool.name for tool in tools)
        return next(responses)

    session = Session(root=tmp_path, model="fake")
    plan = ModeService(provider).plan(session, "change x")
    assert plan.startswith("1. Change")
    assert session.pending_plan is not None
    assert session.pending_plan.task == "change x"
    assert set(offered) <= {
        "read_file",
        "list_dir",
        "git_status",
        "git_log",
        "verification_surface",
    }


def test_plan_read_tools_refuse_paths_outside_the_repository(tmp_path: Path) -> None:
    from tars.loop.modes import read_only_tool

    outside = tmp_path.parent / "private.txt"
    outside.write_text("secret", encoding="utf-8")
    result = read_only_tool(tmp_path, "read_file", {"path": "../private.txt"})
    assert result == "path outside repository: ../private.txt"
    assert "secret" not in result
