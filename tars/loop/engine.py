"""The seven-phase loop, composed from TARS's tested phase modules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TextIO

from .. import memory, providers, router, verification
from ..act import RETRYABLE_REASONS
from ..memory.types import MemoryEntry, ReflectionReport
from ..providers import LLMResponse, Message, ToolDef
from ..providers import ToolCall as ProviderToolCall
from ..router import Classification
from ..verification import GateReport
from .session import Session
from .tools import Toolbox
from .types import RunPromptSpec, RunReport, ToolCall, ToolResult

ProviderCall = Callable[
    [Sequence[Message], Sequence[ToolDef] | None, str | None], LLMResponse
]


class ToolDispatcher(Protocol):
    def dispatch(self, call: ToolCall) -> ToolResult: ...


def _local_toolbox(*args: object, **kwargs: object) -> ToolDispatcher:
    return Toolbox(*args, **kwargs)  # type: ignore[arg-type]


TOOL_DEFS = (
    ToolDef(
        "read_file",
        "Read a UTF-8 text file inside the repository.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    ),
    ToolDef(
        "list_dir",
        "List entries in a repository directory.",
        {"type": "object", "properties": {"path": {"type": "string"}}},
    ),
    ToolDef(
        "apply_edit",
        "Preview, confirm, and apply an anchored edit.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "kind": {"type": "string"},
                "hunks": {"type": "array"},
                "content": {"type": "string"},
            },
            "required": ["path"],
        },
    ),
    ToolDef(
        "run_command",
        "Run an argv command inside the repository.",
        {
            "type": "object",
            "properties": {
                "argv": {"type": "array", "items": {"type": "string"}},
                "cwd": {"type": "string"},
                "timeout": {"type": "number"},
            },
            "required": ["argv"],
        },
    ),
)


class Engine:
    """Run one task through ORIENT, CLASSIFY, PLAN, ACT, VERIFY, REFLECT, REPORT."""

    def __init__(
        self,
        *,
        provider: ProviderCall = providers.generate,
        recall: Callable[..., tuple[MemoryEntry, ...]] = memory.recall,
        classify: Callable[[str], Classification] = router.classify,
        verify: Callable[..., GateReport] = verification.verify,
        propose: Callable[..., ReflectionReport] = memory.propose,
        toolbox: Callable[..., ToolDispatcher] = _local_toolbox,
    ) -> None:
        self.provider = provider
        self.recall = recall
        self.classify = classify
        self.verify = verify
        self.propose = propose
        self.toolbox = toolbox

    def run(
        self,
        spec: RunPromptSpec,
        session: Session,
        *,
        output: TextIO | None = None,
    ) -> RunReport:
        """Execute one complete loop and return its structured report."""
        stream = output
        uncertainties: list[str] = []
        touched: list[str] = []
        declined: list[str] = []

        recalled = self.recall(domains=(), root=session.root)
        verdict = self.classify(spec.task)
        verdict_text = _verdict_text(verdict)
        _write(stream, f"CLASSIFY: {verdict_text}\n")

        session.record(Message.user(spec.task))
        orientation = _orientation(recalled)
        plan = self.provider(
            (*orientation, *session.messages), TOOL_DEFS, session.model
        )
        turns = 1
        session.record(Message.assistant(plan.text or None, tool_calls=plan.tool_calls))

        _write(stream, "ACT\n")
        confirm = (lambda _preview: True) if spec.auto_confirm else None
        toolbox = self.toolbox(session.root, confirm=confirm)

        def dispatch(calls: Sequence[ProviderToolCall]) -> None:
            for call in calls:
                result = _dispatch(toolbox, call)
                session.record(
                    Message.tool_result(call.id, result.text, is_error=not result.ok)
                )
                if result.ok and call.name == "apply_edit":
                    path = call.arguments.get("path")
                    if isinstance(path, str) and path not in touched:
                        touched.append(path)
                elif result.error and "declined" in result.error:
                    declined.append(f"{call.name}: {result.error}")
                elif not result.ok:
                    uncertainties.append(f"tool {call.name} failed: {result.error}")
                for item in result.touched:
                    if item not in touched:
                        touched.append(item)

        if plan.tool_calls:
            dispatch(plan.tool_calls)

        session.record(
            Message.user("Act on the plan above. Use the available tools when needed.")
        )
        action = plan
        while turns < spec.max_turns:
            action = self.provider(
                (*orientation, *session.messages), TOOL_DEFS, session.model
            )
            turns += 1
            session.record(
                Message.assistant(action.text or None, tool_calls=action.tool_calls)
            )
            if not action.tool_calls:
                break
            dispatch(action.tool_calls)
        if action.tool_calls and turns >= spec.max_turns:
            uncertainties.append("max_turns reached while tool calls remained.")

        gate = self.verify(root=session.root)
        session.last_classification = verdict
        session.last_verdict = gate.verdict
        uncertainties.extend(gate.notes)
        reflection = self.propose(session.as_messages(), root=str(session.root))
        uncertainties.extend(reflection.notes)

        return RunReport(
            task=spec.task,
            verdict=verdict_text,
            verification_passed=gate.passed,
            touched=tuple(touched),
            turns=turns,
            proposed=len(reflection.proposed),
            uncertainties=tuple(uncertainties),
            declined=tuple(declined),
        )


def _verdict_text(verdict: Classification) -> str:
    reversible = "reversible" if verdict.reversible else "not reversible"
    return f"tier {verdict.tier} · {verdict.blast} blast · {reversible}"


def _orientation(recalled: Sequence[MemoryEntry]) -> tuple[Message, ...]:
    if not recalled:
        return ()
    facts = "\n".join(f"- {entry.text}" for entry in recalled)
    return (Message.system(f"Relevant project memory:\n{facts}"),)


def _dispatch(toolbox: ToolDispatcher, call: ProviderToolCall) -> ToolResult:
    result = toolbox.dispatch(ToolCall(call.id, call.name, call.arguments))
    reason = result.error.split(":", 1)[0] if result.error else ""
    if not result.ok and reason in RETRYABLE_REASONS:
        return toolbox.dispatch(ToolCall(call.id, call.name, call.arguments))
    return result


def _write(stream: TextIO | None, text: str) -> None:
    if stream is not None:
        stream.write(text)
