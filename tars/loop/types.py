"""The loop's own vocabulary.

These types live here rather than in any phase's module because they describe the
*loop*, which is the product. `Phase` is the taxonomy from
`master_agent_loop.md`; `RunReport` is what one `/run` returns; `ToolCall` and
`ToolResult` are the normalized shape of the model's tool use, so `engine.py`
never touches a provider-specific tool block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

#: How much of a tool's output to show in a one-line log. Full output still
#: reaches the model; this is only what a human sees scroll past.
DISPLAY_WIDTH = 120


class Phase(StrEnum):
    """The seven phases, in the order the spec fixes them.

    A StrEnum so a phase is both a value to branch on and a label to print,
    without a parallel table of display names that can drift.
    """

    ORIENT = "orient"
    CLASSIFY = "classify"
    PLAN = "plan"
    ACT = "act"
    VERIFY = "verify"
    REFLECT = "reflect"
    REPORT = "report"


@dataclass(frozen=True)
class RunPromptSpec:
    """One task, as `/run` received it."""

    task: str
    #: Skip the per-edit confirmation. Set by `--yes` / TARS_NO_CONFIRM.
    auto_confirm: bool = False
    #: Cap on model turns, so a loop that cannot converge still terminates.
    max_turns: int = 12

    def __post_init__(self) -> None:
        if not self.task or not self.task.strip():
            raise ValueError("a run needs a task description")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")


@dataclass(frozen=True)
class ToolCall:
    """One tool the model asked for, provider-neutral."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("a tool call needs an id; it is answered by id")
        if not self.name:
            raise ValueError("a tool call needs a name")


@dataclass(frozen=True)
class ToolResult:
    """What running one tool produced, on its way back to the model."""

    name: str
    output: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def ok_result(cls, name: str, output: str) -> ToolResult:
        return cls(name=name, output=output)

    # `ok` is the property, so the constructors are named for what they carry.
    @classmethod
    def error_result(cls, name: str, error: str) -> ToolResult:
        return cls(name=name, error=error)

    @property
    def text(self) -> str:
        """What the model sees. An error is content, not an exception."""
        return self.output if self.ok else f"error: {self.error}"

    @property
    def display(self) -> str:
        """One line for a human watching, truncated."""
        body = " ".join(self.text.split())
        if len(body) > DISPLAY_WIDTH:
            body = body[: DISPLAY_WIDTH - 1] + "…"
        return f"{self.name}: {body}"


@dataclass(frozen=True)
class RunReport:
    """PHASE 7's content, as data rather than as printed text.

    A report that only exists as stdout cannot be asserted on, and the spec's
    directive 5 ("narrate honestly") is only enforceable if the narration is
    derived from something structured.
    """

    task: str
    #: One-line summary of the router's verdict, as shown before ACT ran.
    verdict: str | None = None
    #: True/False once VERIFY ran; None when it did not run at all.
    verification_passed: bool | None = None
    touched: tuple[str, ...] = field(default_factory=tuple)
    #: Memory candidates proposed by REFLECT — not necessarily written.
    proposed: int = 0
    committed: int = 0
    queued: int = 0
    turns: int = 0
    #: Things the run could not establish. Printed verbatim, never summarized
    #: away: this is the field that keeps REPORT honest.
    uncertainties: tuple[str, ...] = field(default_factory=tuple)
    declined: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("touched", "uncertainties", "declined"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))

    @property
    def changed_anything(self) -> bool:
        return bool(self.touched)
