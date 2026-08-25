"""The seven-phase loop, composed from TARS's tested phase modules.

This first engine slice owns phase order and reporting. Provider tool calls are
left as an explicit uncertainty until the next slice adds model-to-tool
dispatch; silently treating a requested tool as completed would make REPORT
claim more than ACT established.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from .. import memory, providers, router, verification
from ..memory.types import MemoryEntry, ReflectionReport
from ..providers import LLMResponse, Message, ToolDef
from ..router import Classification
from ..verification import GateReport
from .session import Session
from .tools import Toolbox
from .types import RunPromptSpec, RunReport

ProviderCall = Callable[
    [Sequence[Message], Sequence[ToolDef] | None, str | None], LLMResponse
]


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
        toolbox: Callable[[Path], object] = Toolbox,
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

        recalled = self.recall(domains=(), root=session.root)
        verdict = self.classify(spec.task)
        verdict_text = _verdict_text(verdict)
        _write(stream, f"CLASSIFY: {verdict_text}\n")

        session.record(Message.user(spec.task))
        orientation = _orientation(recalled)
        plan = self.provider((*orientation, *session.messages), (), session.model)
        session.record(Message.assistant(plan.text or ""))

        _write(stream, "ACT\n")
        self.toolbox(session.root)
        session.record(
            Message.user("Act on the plan above. Use the available tools when needed.")
        )
        action = self.provider((*orientation, *session.messages), (), session.model)
        session.record(Message.assistant(action.text or ""))
        if action.tool_calls:
            uncertainties.append(
                "ACT requested tool calls, but model-to-tool dispatch is deferred."
            )

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
            turns=2,
            proposed=len(reflection.proposed),
            uncertainties=tuple(uncertainties),
        )


def _verdict_text(verdict: Classification) -> str:
    reversible = "reversible" if verdict.reversible else "not reversible"
    return f"tier {verdict.tier} · {verdict.blast} blast · {reversible}"


def _orientation(recalled: Sequence[MemoryEntry]) -> tuple[Message, ...]:
    if not recalled:
        return ()
    facts = "\n".join(f"- {entry.text}" for entry in recalled)
    return (Message.system(f"Relevant project memory:\n{facts}"),)


def _write(stream: TextIO | None, text: str) -> None:
    if stream is not None:
        stream.write(text)
