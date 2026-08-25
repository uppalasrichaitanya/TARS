"""Part (a): phase composition, with no live model calls."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from tars.loop.session import Session
from tars.loop.types import RunPromptSpec
from tars.memory import MemoryEntry
from tars.providers import LLMResponse, Message, Usage
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

    def toolbox(root: Path):
        events.append("toolbox")
        assert root == tmp_path
        return object()

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
        assert tools == ()
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
    assert report.verdict == "tier 1 · medium blast · reversible"
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
