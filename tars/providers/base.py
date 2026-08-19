"""The one interface every provider implements.

`Provider.generate(messages, tools, model, options)` is the whole contract. If
something can't be expressed here, it goes in `GenerateOptions.extra` as a
provider-specific passthrough rather than growing the shared surface.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .errors import ProviderAPIError, ProviderResponseError, is_retryable_status
from .types import LLMResponse, Message, ToolDef

# Big enough that a normal coding-agent turn isn't truncated, small enough to
# stay under non-streaming HTTP timeouts.
DEFAULT_MAX_TOKENS = 16000


@dataclass(frozen=True)
class GenerateOptions:
    """Knobs that mean the same thing on every provider.

    Anything a provider doesn't support is ignored rather than erroring — the
    adapter layer's job is to make one call site work everywhere. Where that
    silence could mislead, the adapter documents it.
    """

    max_tokens: int = DEFAULT_MAX_TOKENS
    system: str | None = None
    temperature: float | None = None
    # low | medium | high | xhigh | max. Maps to Anthropic's output_config.effort
    # and OpenAI's reasoning_effort; ignored by providers without the concept.
    effort: str | None = None
    # "auto" | "none" | "required" | a specific tool name.
    tool_choice: str | None = None
    stop_sequences: tuple[str, ...] = ()
    timeout: float | None = None
    # Merged verbatim into the outbound request. Escape hatch for provider
    # features the shared interface deliberately doesn't model.
    extra: Mapping[str, Any] = field(default_factory=dict)


class Provider(abc.ABC):
    """Base class for provider adapters."""

    #: Registry key / value reported on `LLMResponse.provider`.
    name: str = "unknown"

    @abc.abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDef] | None = None,
        model: str | None = None,
        options: GenerateOptions | None = None,
    ) -> LLMResponse:
        """Run one completion and return it normalized."""

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<{type(self).__name__} name={self.name!r}>"


# ── helpers shared by the adapters ───────────────────────────────────────


def split_system(
    messages: Sequence[Message],
    options_system: str | None = None,
) -> tuple[str | None, list[Message]]:
    """Pull system-role messages out of the history.

    All three providers want system instructions somewhere other than inline in
    the turn list (Anthropic: a top-level `system`; OpenAI/OpenRouter: a leading
    system message). Multiple system messages are joined with blank lines, with
    `options.system` first so a caller-level instruction outranks history.
    """
    parts: list[str] = []
    if options_system:
        parts.append(options_system)
    rest: list[Message] = []
    for msg in messages:
        if msg.role == "system":
            if msg.content:
                parts.append(msg.content)
        else:
            rest.append(msg)
    system = "\n\n".join(parts) if parts else None
    return system, rest


def require_messages(messages: Sequence[Message]) -> None:
    if not messages:
        raise ValueError("generate() needs at least one message")


def decode_arguments(raw: Any, *, provider: str, tool_name: str) -> dict[str, Any]:
    """Normalize a tool call's arguments to a dict.

    OpenAI-shaped APIs send a JSON *string*; Anthropic sends an object. Both
    can send an empty value for a no-argument tool.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                f"{provider} returned unparseable arguments for tool "
                f"{tool_name!r}: {raw[:200]!r}",
                provider=provider,
            ) from exc
        if not isinstance(decoded, dict):
            raise ProviderResponseError(
                f"{provider} returned non-object arguments for tool "
                f"{tool_name!r}: {type(decoded).__name__}",
                provider=provider,
            )
        return decoded
    raise ProviderResponseError(
        f"{provider} returned unsupported argument type for tool "
        f"{tool_name!r}: {type(raw).__name__}",
        provider=provider,
    )


def translate_api_exception(
    exc: BaseException,
    *,
    provider: str,
    model: str | None = None,
) -> ProviderAPIError:
    """Wrap an SDK/HTTP exception as a ProviderAPIError.

    Deliberately duck-typed on `status_code` / `response.status_code` rather
    than importing either SDK's exception tree: both the Anthropic and OpenAI
    clients expose the status the same way, and an exception with no status at
    all is a connection/timeout failure, which is retryable.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    retryable = is_retryable_status(status) if status is not None else True
    detail = getattr(exc, "message", None) or str(exc) or type(exc).__name__
    where = f"{provider} request failed"
    if status is not None:
        where += f" with HTTP {status}"
    return ProviderAPIError(
        f"{where}: {detail}",
        provider=provider,
        model=model,
        status_code=status,
        retryable=retryable,
        original=exc,
    )
