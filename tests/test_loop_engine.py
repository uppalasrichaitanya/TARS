"""Part (a): phase composition, with no live model calls."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest

from tars.loop.session import Session
from tars.loop.types import RunPromptSpec, ToolResult
from tars.loop.types import ToolCall as LoopToolCall
from tars.memory import MemoryEntry
from tars.providers import LLMResponse, Message, ToolCall, Usage
from tars.router import Classification
from tars.verification import GateReport


def _response(text: str = "plan") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=(),
        model="fake",
        provider="fake",
        stop_reason="end_turn",
        usage=Usage(),
    )


def _tool_response(call: ToolCall, text: str = "") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=(call,),
        model="fake",
        provider="fake",
        stop_reason="tool_use",
        usage=Usage(),
    )


def test_engine_wires_all_phases_and_reports_uncertainties(
    tmp_path: Path,
) -> None:
    from tars.loop.engine import Engine

    events: list[str] = []
    output = StringIO()
    classification = Classification(
        tier=1,
        blast="medium",
        reversible=True,
        signals=("fake",),
        why="fake classification",
    )
    provider_calls = 0

    def toolbox(root: Path, **_kwargs):
        events.append("toolbox")
        assert root == tmp_path

        class NoopToolbox:
            def dispatch(self, _call: ToolCall) -> ToolResult:
                raise AssertionError("no tool call expected")

        return NoopToolbox()

    def recall(*, root: Path, **_kwargs):
        events.append("orient")
        return (
            MemoryEntry(
                id="decision-1",
                text="Use stdlib only.",
                kind="decision",
                domains=("loop",),
                why="Dependencies change the gate.",
                confidence=0.95,
            ),
        )

    def classify(task: str):
        events.append("classify")
        assert task == "fix it"
        return classification

    def generate(messages, tools, model):
        nonlocal provider_calls
        provider_calls += 1
        assert [tool.name for tool in tools] == [
            "read_file",
            "list_dir",
            "apply_edit",
            "run_command",
        ]
        assert model == "fake"
        if provider_calls == 1:
            events.append("plan")
            assert messages[-1] == Message.user("fix it")
            assert messages[0].role == "system"
            assert "Use stdlib only." in (messages[0].content or "")
            return _response("We decided to keep the change small.")
        events.append("act")
        assert "tier 1" in output.getvalue()
        assert messages[-1].role == "user"
        return _response("No tool calls were needed.")

    def verify(root: Path, **_kwargs):
        events.append("verify")
        return GateReport(root=str(root), results=())

    def propose(source, **_kwargs):
        events.append("reflect")
        assert source[0]["role"] == "user"
        from tars.memory import ReflectionReport

        return ReflectionReport(session="repl", notes=("uncertainty verbatim",))

    engine = Engine(
        provider=generate,
        recall=recall,
        classify=classify,
        verify=verify,
        propose=propose,
        toolbox=toolbox,
    )
    report = engine.run(
        RunPromptSpec("fix it"), Session(root=tmp_path, model="fake"), output=output
    )

    assert events == [
        "orient",
        "classify",
        "plan",
        "toolbox",
        "act",
        "verify",
        "reflect",
    ]
    assert output.getvalue().index("tier 1") < output.getvalue().index("ACT")
    assert report.task == "fix it"
    assert report.verdict == (
        "tier 1 · medium blast · reversible · PLAN model=fake · ACT model=fake"
    )
    assert report.verification_passed is False
    assert report.uncertainties == ("uncertainty verbatim",)
    assert report.turns == 2
    assert report.committed == 0
    assert report.queued == 0


def test_engine_records_model_plan_in_session(tmp_path: Path) -> None:
    from tars.loop.engine import Engine

    session = Session(root=tmp_path, model="fake")
    engine = Engine(provider=lambda *_args, **_kwargs: _response("plan text"))
    engine.run(RunPromptSpec("task"), session, output=StringIO())
    assert [message.content for message in session.messages] == [
        "task",
        "plan text",
        "Act on the plan above. Use the available tools when needed.",
        "plan text",
    ]


def test_engine_round_trips_a_tool_call_until_model_finishes(tmp_path: Path) -> None:
    from tars.loop.engine import Engine

    calls: list[tuple[tuple[Message, ...], tuple[str, ...]]] = []
    tool_call = ToolCall("call-1", "read_file", {"path": "note.txt"})
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")

    def provider(messages, tools, model):
        calls.append((tuple(messages), tuple(tool.name for tool in tools or ())))
        if len(calls) == 1:
            return _response("plan")
        if len(calls) == 2:
            return _tool_response(tool_call)
        assert messages[-1].role == "tool"
        assert messages[-1].tool_call_id == "call-1"
        assert messages[-1].content == "hello"
        return _response("done")

    session = Session(root=tmp_path, model="fake")
    Engine(provider=provider).run(
        RunPromptSpec("inspect it", max_turns=4), session, output=StringIO()
    )
    assert len(calls) == 3
    assert calls[0][1] == (
        "read_file",
        "list_dir",
        "apply_edit",
        "run_command",
    )
    assert calls[1][1] == calls[0][1]


def test_declined_edit_is_returned_to_model_and_reported(tmp_path: Path) -> None:
    from tars.loop.engine import Engine

    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    edit_call = ToolCall(
        "edit-1",
        "apply_edit",
        {"path": "code.py", "hunks": [{"old": "x = 1", "new": "x = 2"}]},
    )
    seen: list[Message] = []

    def provider(messages, tools, model):
        seen.extend(messages)
        if len([m for m in seen if m.role == "assistant"]) == 1:
            return _tool_response(edit_call)
        return _response("done")

    session = Session(root=tmp_path, model="fake")
    report = Engine(provider=provider).run(
        RunPromptSpec("edit it", max_turns=3), session, output=StringIO()
    )
    assert any(
        message.role == "tool" and "declined" in (message.content or "")
        for message in seen
    )
    assert report.declined == ("apply_edit: edit declined",)
    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "x = 1\n"


def test_auto_confirm_approves_edit_and_max_turns_is_reported(tmp_path: Path) -> None:
    from tars.loop.engine import Engine

    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    edit_call = ToolCall(
        "edit-1",
        "apply_edit",
        {"path": "code.py", "hunks": [{"old": "x = 1", "new": "x = 2"}]},
    )
    provider_calls = 0

    def provider(messages, tools, model):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            return _response("plan")
        return _tool_response(edit_call)

    session = Session(root=tmp_path, model="fake")
    report = Engine(provider=provider).run(
        RunPromptSpec("edit it", auto_confirm=True, max_turns=2),
        session,
        output=StringIO(),
    )
    assert provider_calls == 2
    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "x = 2\n"
    assert report.touched == ("code.py",)
    assert "max_turns reached while tool calls remained." in report.uncertainties


def test_dispatch_retries_timeout_once_and_non_timeout_never(tmp_path: Path) -> None:
    from tars.act import RETRYABLE_REASONS
    from tars.loop import engine as engine_module

    assert engine_module.RETRYABLE_REASONS is RETRYABLE_REASONS
    _dispatch = engine_module._dispatch

    call = ToolCall("c", "run_command", {"argv": ["test"]})

    class FakeToolbox:
        def __init__(self, results: list[ToolResult]) -> None:
            self.results = results
            self.calls = 0

        def dispatch(self, call: LoopToolCall) -> ToolResult:
            self.calls += 1
            return self.results[min(self.calls - 1, len(self.results) - 1)]

    timeout_box = FakeToolbox(
        [
            ToolResult.error_result("run_command", "timeout: overloaded"),
            ToolResult.ok_result("run_command", "done"),
        ]
    )
    assert _dispatch(timeout_box, call).ok
    assert timeout_box.calls == 2

    failed_box = FakeToolbox(
        [ToolResult.error_result("run_command", "failed: exit code 1")]
    )
    assert not _dispatch(failed_box, call).ok
    assert failed_box.calls == 1


def test_successful_run_command_is_reported_as_touched(tmp_path: Path) -> None:
    from tars.loop.engine import Engine

    call = ToolCall("c", "run_command", {"argv": ["touch", "created.txt"]})

    class FakeToolbox:
        def dispatch(self, _call: LoopToolCall) -> ToolResult:
            return ToolResult.ok_result(
                "run_command", "", touched=("touch created.txt",)
            )

    responses = iter((_response("plan"), _tool_response(call), _response("done")))
    session = Session(root=tmp_path, model="fake")
    report = Engine(
        provider=lambda *_args, **_kwargs: next(responses),
        toolbox=lambda *_args, **_kwargs: FakeToolbox(),
    ).run(RunPromptSpec("create it"), session, output=StringIO())
    assert report.touched == ("touch created.txt",)


def test_engine_marks_failed_run_interrupted_and_never_finishes_it(
    tmp_path: Path,
) -> None:
    from tars.loop.engine import Engine
    from tars.loop.session_store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    session = Session(root=tmp_path, model="fake", store=store)

    def broken(*_args, **_kwargs):
        raise RuntimeError("provider stopped")

    with pytest.raises(RuntimeError, match="provider stopped"):
        Engine(provider=broken).run(RunPromptSpec("task"), session, output=StringIO())
    types = [event["type"] for event in store.load(session.session_id)]
    assert types.index("run_started") < types.index("run_interrupted")
    assert "run_finished" not in types
    assert Session.load(store, session.session_id).interrupted is True


def test_tier_zero_uses_configured_cheaper_model_for_plan_and_act(
    tmp_path: Path,
) -> None:
    from tars.loop.engine import Engine

    models: list[str | None] = []

    def provider(_messages, _tools, model):
        models.append(model)
        return _response("done")

    session = Session(
        root=tmp_path,
        model="flagship",
        tier_models={"0": "openrouter:test-fixture/free"},
    )
    report = Engine(
        provider=provider,
        classify=lambda _task: Classification(0, "low", True, (), "small"),
        verify=lambda **_kwargs: GateReport(root=str(tmp_path), results=()),
    ).run(RunPromptSpec("small task"), session, output=StringIO())
    assert models == ["openrouter:test-fixture/free", "openrouter:test-fixture/free"]
    assert "PLAN model=openrouter:test-fixture/free" in (report.verdict or "")
    assert "ACT model=openrouter:test-fixture/free" in (report.verdict or "")
    assert not report.uncertainties


def test_tier_zero_without_a_resolvable_mapping_falls_back_honestly(
    tmp_path: Path,
) -> None:
    from tars.loop.engine import Engine

    models: list[str | None] = []

    def provider(_messages, _tools, model):
        models.append(model)
        return _response("done")

    session = Session(root=tmp_path, model="flagship", tier_models={"0": None})
    report = Engine(
        provider=provider,
        classify=lambda _task: Classification(0, "low", True, (), "small"),
        verify=lambda **_kwargs: GateReport(root=str(tmp_path), results=()),
    ).run(RunPromptSpec("small task"), session, output=StringIO())
    assert models == ["flagship", "flagship"]
    assert report.uncertainties == (
        "tier 0 requested a cheaper model, but none is configured; used flagship",
    )
    assert "PLAN model=flagship" in (report.verdict or "")
    assert "ACT model=flagship" in (report.verdict or "")


def test_invalid_tier_zero_model_reports_unusable_not_unconfigured(
    tmp_path: Path,
) -> None:
    from tars.loop.engine import Engine

    models: list[str | None] = []

    def provider(_messages, _tools, model):
        models.append(model)
        return _response("done")

    session = Session(
        root=tmp_path,
        model="flagship",
        tier_models={"0": "unknown-provider:model"},
    )
    report = Engine(
        provider=provider,
        classify=lambda _task: Classification(0, "low", True, (), "small"),
        verify=lambda **_kwargs: GateReport(root=str(tmp_path), results=()),
    ).run(RunPromptSpec("small task"), session, output=StringIO())
    assert models == ["flagship", "flagship"]
    assert report.uncertainties[0] == (
        "tier 0 requested a cheaper model, but none was usable; used flagship"
    )


def test_provider_not_configured_on_tier_zero_falls_back_and_continues(
    tmp_path: Path,
) -> None:
    from tars.loop.engine import Engine
    from tars.providers.errors import ProviderNotConfiguredError

    models: list[str | None] = []

    def provider(_messages, _tools, model):
        models.append(model)
        if model == "openrouter:test-fixture/free":
            raise ProviderNotConfiguredError("free unavailable")
        return _response("done")

    session = Session(
        root=tmp_path,
        model="flagship",
        tier_models={"0": "openrouter:test-fixture/free"},
    )
    report = Engine(
        provider=provider,
        classify=lambda _task: Classification(0, "low", True, (), "small"),
        verify=lambda **_kwargs: GateReport(root=str(tmp_path), results=()),
    ).run(RunPromptSpec("small task"), session, output=StringIO())
    assert models == ["openrouter:test-fixture/free", "flagship", "flagship"]
    assert report.uncertainties[0] == (
        "tier 0 requested a cheaper model, but none was usable; used flagship"
    )


def test_agentrouter_transient_provider_error_retries_once_and_records_uncertainty(
    tmp_path: Path,
) -> None:
    from tars.loop.engine import Engine
    from tars.providers.errors import ProviderAPIError

    calls = 0
    sleeps: list[float] = []

    def provider(_messages, _tools, _model):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderAPIError("HTTP 405", provider="agentrouter", status_code=405)
        return _response("done")

    report = Engine(
        provider=provider,
        sleep=sleeps.append,
        verify=lambda **_kwargs: GateReport(root=str(tmp_path), results=()),
    ).run(RunPromptSpec("task"), Session(root=tmp_path, model="agentrouter:fake"))

    assert calls == 3  # PLAN retry succeeds, then one ACT turn
    assert sleeps == [2.0]
    assert report.uncertainties == (
        "provider returned 405; retried once and succeeded",
    )


def test_double_agentrouter_transient_error_is_reported_and_raised(
    tmp_path: Path,
) -> None:
    from tars.loop.engine import Engine
    from tars.providers.errors import ProviderAPIError

    calls = 0

    def provider(_messages, _tools, _model):
        nonlocal calls
        calls += 1
        raise ProviderAPIError("HTTP 405", provider="agentrouter", status_code=405)

    with pytest.raises(ProviderAPIError):
        Engine(provider=provider, sleep=lambda _seconds: None).run(
            RunPromptSpec("task"), Session(root=tmp_path, model="agentrouter:fake")
        )
    assert calls == 2


def test_double_405_act_turn_reaches_report_with_uncertainty(tmp_path: Path) -> None:
    from tars.loop.engine import Engine
    from tars.providers.errors import ProviderAPIError

    calls = 0

    def provider(_messages, _tools, _model):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response("plan")
        raise ProviderAPIError("HTTP 405", provider="agentrouter", status_code=405)

    report = Engine(
        provider=provider,
        sleep=lambda _seconds: None,
        verify=lambda **_kwargs: GateReport(root=str(tmp_path), results=()),
    ).run(
        RunPromptSpec("task", max_turns=2),
        Session(root=tmp_path, model="agentrouter:fake"),
    )
    assert calls == 3
    assert report.uncertainties == (
        "provider returned 405; retried once and failed again",
    )


def test_non_agentrouter_provider_errors_are_not_retried(tmp_path: Path) -> None:
    from tars.loop.engine import Engine
    from tars.providers.errors import ProviderAPIError

    calls = 0

    def provider(_messages, _tools, _model):
        nonlocal calls
        calls += 1
        raise ProviderAPIError("HTTP 503", provider="openrouter", status_code=503)

    with pytest.raises(ProviderAPIError):
        Engine(provider=provider, sleep=lambda _seconds: None).run(
            RunPromptSpec("task"), Session(root=tmp_path, model="openrouter:fake")
        )
    assert calls == 1


def test_tier_zero_fallback_happens_before_flagship_transient_retry(
    tmp_path: Path,
) -> None:
    from tars.loop.engine import Engine
    from tars.providers.errors import ProviderAPIError, ProviderNotConfiguredError

    calls: list[str] = []

    def provider(_messages, _tools, model):
        calls.append(model)
        if model == "cheap":
            raise ProviderNotConfiguredError("cheap unavailable")
        if len([item for item in calls if item == "flagship"]) == 1:
            raise ProviderAPIError("HTTP 503", provider="agentrouter", status_code=503)
        return _response("done")

    Engine(
        provider=provider,
        sleep=lambda _seconds: None,
        classify=lambda _task: Classification(0, "low", True, (), "small"),
        verify=lambda **_kwargs: GateReport(root=str(tmp_path), results=()),
    ).run(
        RunPromptSpec("task"),
        Session(root=tmp_path, model="flagship", tier_models={"0": "cheap"}),
    )
    assert calls == ["cheap", "flagship", "flagship", "flagship"]


def test_agentrouter_auth_error_is_never_retried(tmp_path: Path) -> None:
    from tars.loop.engine import Engine
    from tars.providers.errors import ProviderAPIError

    calls = 0

    def provider(_messages, _tools, _model):
        nonlocal calls
        calls += 1
        raise ProviderAPIError("HTTP 401", provider="agentrouter", status_code=401)

    with pytest.raises(ProviderAPIError):
        Engine(provider=provider, sleep=lambda _seconds: None).run(
            RunPromptSpec("task"), Session(root=tmp_path, model="agentrouter:fake")
        )
    assert calls == 1


def test_agentrouter_auth_error_during_act_still_interrupts(tmp_path: Path) -> None:
    from tars.loop.engine import Engine
    from tars.providers.errors import ProviderAPIError

    calls = 0

    def provider(_messages, _tools, _model):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response("plan")
        raise ProviderAPIError("HTTP 401", provider="agentrouter", status_code=401)

    with pytest.raises(ProviderAPIError):
        Engine(provider=provider, sleep=lambda _seconds: None).run(
            RunPromptSpec("task"), Session(root=tmp_path, model="agentrouter:fake")
        )
    assert calls == 2
