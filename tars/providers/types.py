"""Provider-neutral types spoken by the TARS agent loop.

Everything above the adapter layer uses only these. A provider adapter's job is
to translate them to/from whatever shape its API wants, so no provider quirk
(Anthropic content blocks, OpenAI `tool_calls`, OpenRouter's passthrough JSON)
ever reaches loop code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

Role = Literal["system", "user", "assistant", "tool"]

VALID_ROLES: frozenset[str] = frozenset({"system", "user", "assistant", "tool"})

# Normalized across providers. Anthropic and OpenAI use different vocabularies
# for the same handful of outcomes; we keep Anthropic's names because they read
# better, and map OpenAI's onto them.
StopReason = Literal[
    "end_turn",  # model finished on its own
    "tool_use",  # model wants tool results before continuing
    "max_tokens",  # hit the output cap — response is truncated
    "stop_sequence",  # hit a caller-supplied stop sequence
    "refusal",  # model declined
    "other",  # provider said something we don't model
]


@dataclass(frozen=True)
class ToolCall:
    """A model's request to run one tool.

    `arguments` is always a decoded dict — providers hand these over as JSON
    strings (OpenAI) or already-decoded objects (Anthropic), and the adapter
    normalizes before it gets here.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """One turn of conversation.

    Roles:
      system     — instructions. Providers that take a dedicated system field
                   get it hoisted there by the adapter.
      user       — input from the human/harness.
      assistant  — model output; may carry `tool_calls`.
      tool       — the result of one tool call; requires `tool_call_id`.
    """

    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    is_error: bool = False

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"unknown role {self.role!r}; expected one of {sorted(VALID_ROLES)}"
            )
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("a tool-role message requires tool_call_id")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages may carry tool_calls")
        # Normalize sequences to tuples so Message stays hashable/immutable even
        # when callers pass a list.
        if not isinstance(self.tool_calls, tuple):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: str | None = None,
        tool_calls: Sequence[ToolCall] = (),
    ) -> Message:
        return cls(role="assistant", content=content, tool_calls=tuple(tool_calls))

    @classmethod
    def tool_result(
        cls,
        tool_call_id: str,
        content: str,
        *,
        is_error: bool = False,
    ) -> Message:
        return cls(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            is_error=is_error,
        )


@dataclass(frozen=True)
class ToolDef:
    """A tool offered to the model. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass(frozen=True)
class Usage:
    """Token accounting, in the units the provider reported.

    Cache fields are zero on providers that don't report them — absence of a
    number is reported as zero, never guessed.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class LLMResponse:
    """One model response, normalized.

    `raw` keeps the provider's own object so callers that genuinely need a
    provider-specific field can reach it without the adapter having to model
    every feature. Loop code should not depend on it.
    """

    text: str
    tool_calls: tuple[ToolCall, ...]
    model: str
    provider: str
    stop_reason: StopReason
    usage: Usage
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        """True when the model asked for tool results before continuing."""
        return bool(self.tool_calls)
