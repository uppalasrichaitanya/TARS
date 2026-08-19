"""Translation shared by every OpenAI-compatible endpoint.

Both the OpenAI SDK adapter and the OpenRouter HTTP adapter speak the Chat
Completions dialect, so the wire-shape translation lives here once. These
functions are pure: dicts in, dicts out, no clients, no I/O.

Normalization reads responses as plain dicts. SDK objects are converted with
`model_dump()` first, so one parser handles both a typed `ChatCompletion` and
OpenRouter's raw JSON.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .base import decode_arguments
from .errors import ProviderResponseError
from .types import LLMResponse, Message, StopReason, ToolCall, ToolDef, Usage

_FINISH_REASONS: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",  # legacy name, same meaning
    "length": "max_tokens",
    "content_filter": "refusal",
}


def to_chat_messages(
    messages: Sequence[Message],
    system: str | None = None,
) -> list[dict[str, Any]]:
    """Render neutral messages as a Chat Completions `messages` array."""
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for msg in messages:
        if msg.role == "user":
            if not msg.content:
                raise ValueError("a user message must have content")
            out.append({"role": "user", "content": msg.content})
        elif msg.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            # Arguments go over the wire as a JSON *string* here,
                            # unlike Anthropic's decoded object.
                            "arguments": json.dumps(call.arguments or {}),
                        },
                    }
                    for call in msg.tool_calls
                ]
            elif not msg.content:
                raise ValueError("an assistant message must have content or tool_calls")
            out.append(entry)
        elif msg.role == "tool":
            # One message per tool call — the mirror image of Anthropic, which
            # wants them merged into a single turn.
            content = msg.content or ""
            if msg.is_error:
                # There is no is_error flag in this dialect, so say it in the
                # text rather than letting a failure read as a success.
                content = f"Error: {content}"
            out.append(
                {"role": "tool", "tool_call_id": msg.tool_call_id, "content": content}
            )
        elif msg.role == "system":
            out.append({"role": "system", "content": msg.content or ""})
        else:  # pragma: no cover - Message validates the role
            raise ValueError(f"unexpected role: {msg.role!r}")

    return out


def to_chat_tools(tools: Sequence[ToolDef]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            },
        }
        for tool in tools
    ]


def to_chat_tool_choice(choice: str | None) -> str | dict[str, Any] | None:
    if not choice:
        return None
    if choice in ("auto", "none", "required"):
        return choice
    return {"type": "function", "function": {"name": choice}}


def normalize_completion(
    raw: Any,
    *,
    provider: str,
    fallback_model: str,
) -> LLMResponse:
    """Turn a Chat Completions response (SDK object or dict) into an LLMResponse."""
    payload = _as_mapping(raw, provider=provider)

    choices = payload.get("choices") or []
    if not choices:
        raise ProviderResponseError(
            f"{provider} returned no choices",
            provider=provider,
            model=payload.get("model") or fallback_model,
        )
    choice = choices[0]
    message = choice.get("message") or {}

    tool_calls: list[ToolCall] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = function.get("name") or ""
        if not name:
            raise ProviderResponseError(
                f"{provider} returned a tool call with no function name",
                provider=provider,
            )
        tool_calls.append(
            ToolCall(
                id=call.get("id") or "",
                name=name,
                arguments=decode_arguments(
                    function.get("arguments"), provider=provider, tool_name=name
                ),
            )
        )

    return LLMResponse(
        text=_extract_text(message.get("content")),
        tool_calls=tuple(tool_calls),
        model=payload.get("model") or fallback_model,
        provider=provider,
        stop_reason=_FINISH_REASONS.get(choice.get("finish_reason") or "", "other"),
        usage=to_usage(payload.get("usage")),
        raw=raw,
    )


def to_usage(usage: Mapping[str, Any] | None) -> Usage:
    if not usage:
        return Usage()
    details = usage.get("prompt_tokens_details") or {}
    return Usage(
        input_tokens=usage.get("prompt_tokens") or 0,
        output_tokens=usage.get("completion_tokens") or 0,
        cache_read_tokens=details.get("cached_tokens") or 0,
        # This dialect has no "tokens written to cache" counter.
        cache_write_tokens=0,
    )


def _extract_text(content: Any) -> str:
    """Read assistant text, tolerating a null or multi-part content field."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, Mapping) and part.get("type") in (None, "text")
        ]
        return "".join(parts)
    return str(content)


def _as_mapping(raw: Any, *, provider: str) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        return dump()
    raise ProviderResponseError(
        f"{provider} returned an unreadable response of type {type(raw).__name__}",
        provider=provider,
    )
