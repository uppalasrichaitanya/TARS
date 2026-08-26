"""The seven-phase loop, composed from TARS's tested phase modules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from time import sleep as _sleep
from typing import Protocol, TextIO

from .. import memory, providers, router, verification
from ..act import RETRYABLE_REASONS
from ..memory.types import MemoryEntry, ReflectionReport
from ..providers import LLMResponse, Message, ToolDef, resolve_model
from ..providers import ToolCall as ProviderToolCall
from ..providers.errors import (
    ProviderAPIError,
    ProviderError,
    ProviderNotConfiguredError,
)
from ..router import Classification
from ..verification import GateReport
from .session import Session
from .tools import Toolbox
from .types import RunPromptSpec, RunReport, ToolCall, ToolResult

ProviderCall = Callable[
    [Sequence[Message], Sequence[ToolDef] | None, str | None], LLMResponse
]
Sleep = Callable[[float], None]


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
        sleep: Sleep = _sleep,
    ) -> None:
        self.provider = provider
        self.recall = recall
        self.classify = classify
        self.verify = verify
        self.propose = propose
        self.toolbox = toolbox
        self.sleep = sleep

    def run(
        self,
        spec: RunPromptSpec,
        session: Session,
        *,
        output: TextIO | None = None,
    ) -> RunReport:
        """Execute one complete loop and return its structured report."""
        session.interrupted = False
        session.record_event("run_started", {"task": spec.task})
        if not session.messages:
            session.record_event("config", {"label": spec.task[:80]})
        try:
            return self._run(spec, session, output=output)
        except BaseException:
            session.interrupted = True
            session.record_event("run_interrupted", {"task": spec.task})
            raise

    def _run(
        self,
        spec: RunPromptSpec,
        session: Session,
        *,
        output: TextIO | None = None,
    ) -> RunReport:
        stream = output
        uncertainties: list[str] = []
        touched: list[str] = []
        declined: list[str] = []

        recalled = self.recall(domains=(), root=session.root)
        verdict = self.classify(spec.task)
        phase_model, fallback = _phase_model(session, verdict.tier)
        if fallback:
            uncertainties.append(fallback)

        session.record(Message.user(spec.task))
        orientation = _orientation(recalled)

        def generate(
            model: str, *, act: bool = False
        ) -> tuple[LLMResponse, str] | None:
            """Call the provider once, retrying AgentRouter infrastructure failures."""
            return self._provider_with_retry(
                (*orientation, *session.messages),
                model,
                uncertainties,
                tolerate_exhausted=act,
            )

        def generate_with_fallback(model: str) -> tuple[LLMResponse, str] | None:
            try:
                return generate(model)
            except ProviderNotConfiguredError:
                if verdict.tier != 0 or model == session.model:
                    raise
                message = (
                    "tier 0 requested a cheaper model, but none was usable; "
                    f"used {session.model}"
                )
                if message not in uncertainties:
                    uncertainties.append(message)
                return generate(session.model)

        plan_result = generate_with_fallback(phase_model)
        assert plan_result is not None
        plan, plan_model = plan_result
        phase_model = plan_model
        verdict_text = _verdict_text(verdict, plan_model, phase_model)
        _write(stream, f"CLASSIFY: {verdict_text}\n")
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
        act_model = phase_model
        act_provider_failed = False
        while turns < spec.max_turns:
            action_result = generate(act_model, act=True)
            turns += 1
            if action_result is None:
                act_provider_failed = True
                break
            action, act_model = action_result
            session.record(
                Message.assistant(action.text or None, tool_calls=action.tool_calls)
            )
            if not action.tool_calls:
                break
            dispatch(action.tool_calls)
        if not act_provider_failed and action.tool_calls and turns >= spec.max_turns:
            uncertainties.append("max_turns reached while tool calls remained.")

        gate = self.verify(root=session.root)
        session.last_classification = verdict
        session.last_verdict = gate.verdict
        session.record_event(
            "classification",
            {
                "tier": verdict.tier,
                "blast": verdict.blast,
                "reversible": verdict.reversible,
                "signals": list(verdict.signals),
                "why": verdict.why,
            },
        )
        session.record_event("verification", {"verdict": gate.verdict})
        uncertainties.extend(gate.notes)
        reflection = self.propose(session.as_messages(), root=str(session.root))
        uncertainties.extend(reflection.notes)

        verdict_text = _verdict_text(verdict, plan_model, act_model)
        report = RunReport(
            task=spec.task,
            verdict=verdict_text,
            verification_passed=gate.passed,
            touched=tuple(touched),
            turns=turns,
            proposed=len(reflection.proposed),
            uncertainties=tuple(uncertainties),
            declined=tuple(declined),
        )
        session.record_event("run_finished", {"task": spec.task})
        return report

    def _provider_with_retry(
        self,
        messages: Sequence[Message],
        model: str,
        uncertainties: list[str],
        *,
        tolerate_exhausted: bool,
    ) -> tuple[LLMResponse, str] | None:
        last_exc: ProviderAPIError | None = None
        for attempt in range(MAX_TRANSIENT_ATTEMPTS):
            try:
                result = self.provider(messages, TOOL_DEFS, model)
            except ProviderAPIError as exc:
                if not _is_transient(exc):
                    raise
                last_exc = exc
                if attempt < MAX_TRANSIENT_ATTEMPTS - 1:
                    # Free-tier hosts (OpenRouter, TokenRouter) throttle (429)
                    # and overload (503) in bursts, so a single retry is not
                    # enough; back off and try again before giving up.
                    self.sleep(2.0 * (attempt + 1))
                continue
            else:
                if last_exc is not None:
                    uncertainties.append(
                        f"provider returned {last_exc.status_code}; retried and "
                        "succeeded"
                    )
                return result, model
        # Every attempt failed with a transient status code.
        status = last_exc.status_code if last_exc is not None else None
        uncertainties.append(
            f"provider returned {status}; retried {MAX_TRANSIENT_ATTEMPTS} "
            "times, all transient"
        )
        if tolerate_exhausted:
            return None
        assert last_exc is not None
        raise last_exc


def _verdict_text(verdict: Classification, plan_model: str, act_model: str) -> str:
    reversible = "reversible" if verdict.reversible else "not reversible"
    return (
        f"tier {verdict.tier} · {verdict.blast} blast · {reversible} · "
        f"PLAN model={plan_model} · ACT model={act_model}"
    )


#: Status codes that mean transient infrastructure, not a bad request. These
#: are retried regardless of which provider answered — AgentRouter's WAF,
#: OpenRouter's throttling, and TokenRouter's overload all surface here.
TRANSIENT_STATUS: tuple[int, ...] = (405, 429, 502, 503)
MAX_TRANSIENT_ATTEMPTS = 3


def _is_transient(exc: ProviderAPIError) -> bool:
    return exc.status_code in TRANSIENT_STATUS


def _phase_model(session: Session, tier: int) -> tuple[str, str | None]:
    """Interpret tier 0 as the cheapest capable model.

    The spec says tier 0 may execute directly. TARS deliberately still selects a
    model because its ACT phase always needs a driver for tool calls; cheap here
    means the cheapest configured model capable of driving that loop.
    """
    configured = session.tier_models.get(str(tier))
    if tier != 0:
        return configured or session.model, None
    if configured:
        try:
            resolve_model(configured)
        except (ProviderError, ValueError):
            pass
        else:
            return configured, None
    reason = "none was usable" if configured else "none is configured"
    return session.model, (
        f"tier 0 requested a cheaper model, but {reason}; used {session.model}"
    )


def _orientation(recalled: Sequence[MemoryEntry]) -> tuple[Message, ...]:
    if not recalled:
        return ()
    facts = "\n".join(f"- {entry.text}" for entry in recalled)
    return (Message.system(f"Relevant project memory:\n{facts}"),)


def _dispatch(toolbox: ToolDispatcher, call: ProviderToolCall) -> ToolResult:
    result = toolbox.dispatch(ToolCall(call.id, call.name, call.arguments))
    reason = result.error.split(":", 1)[0] if result.error else ""
    if (
        not result.ok
        and call.name == "run_command"
        and reason == "not-found"
        and _looks_like_shell_string(call.arguments.get("argv"))
    ):
        return ToolResult.ok_result(
            "run_command",
            "Corrective guidance: run_command accepts an argv list, not a shell string. "
            "Re-issue this call with the program and each argument as separate argv items "
            '(for example: ["mkdir", "-p", "greet"]). Do not use shell syntax.',
        )
    if not result.ok and reason in RETRYABLE_REASONS:
        return toolbox.dispatch(ToolCall(call.id, call.name, call.arguments))
    return result


def _looks_like_shell_string(argv: object) -> bool:
    if not isinstance(argv, list) or len(argv) != 1 or not isinstance(argv[0], str):
        return False
    value = argv[0]
    return any(marker in value for marker in (" ", "&&", "|", ";")) or (
        len(value) >= 2 and value[0] == value[-1] and value[0] in "'\""
    )


def _write(stream: TextIO | None, text: str) -> None:
    if stream is not None:
        stream.write(text)
