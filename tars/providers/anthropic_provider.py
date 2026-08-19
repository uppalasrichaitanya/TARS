"""Anthropic adapter — direct SDK, first-class per the spec.

Translation notes that are easy to get wrong:

* `system` is a top-level request field, not a turn in `messages`.
* `max_tokens` is required on every request.
* Current Claude models (Opus 5 / 4.8 / 4.7, Sonnet 5, Fable 5) **reject**
  `temperature`, so it is only ever sent when the caller explicitly set it.
* All tool results answering one assistant turn must arrive in a *single* user
  message; splitting them trains the model out of parallel tool calls.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from .base import (
    GenerateOptions,
    Provider,
    decode_arguments,
    require_messages,
    split_system,
    translate_api_exception,
)
from .errors import (
    ProviderNotConfiguredError,
    ProviderNotInstalledError,
    ProviderResponseError,
)
from .registry import resolve_model
from .types import LLMResponse, Message, StopReason, ToolCall, ToolDef, Usage

DEFAULT_MODEL = "claude-opus-5"

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "refusal": "refusal",
}


class AnthropicProvider(Provider):
    """Talks to the Anthropic Messages API through the official SDK."""

    name = "anthropic"

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        env: Mapping[str, str] | None = None,
        default_model: str = DEFAULT_MODEL,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url
        # Injected so credential resolution is deterministic and testable
        # instead of reaching into os.environ from inside the SDK.
        self._env = os.environ if env is None else env
        self.default_model = default_model

    # ── client construction ──────────────────────────────────────────────

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = self._api_key or self._env.get("ANTHROPIC_API_KEY")
        auth_token = None if api_key else self._env.get("ANTHROPIC_AUTH_TOKEN")
        if not api_key and not auth_token:
            raise ProviderNotConfiguredError(
                "no Anthropic credentials found: set ANTHROPIC_API_KEY "
                "(or ANTHROPIC_AUTH_TOKEN) in the environment, or pass "
                "api_key= to AnthropicProvider",
                provider=self.name,
            )

        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on install state
            raise ProviderNotInstalledError(
                "the anthropic SDK is not installed",
                provider=self.name,
                install_hint="pip install anthropic",
            ) from exc

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        else:
            kwargs["auth_token"] = auth_token
        base_url = self._base_url or self._env.get("ANTHROPIC_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**kwargs)
        return self._client

    # ── the interface ────────────────────────────────────────────────────

    def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDef] | None = None,
        model: str | None = None,
        options: GenerateOptions | None = None,
    ) -> LLMResponse:
        require_messages(messages)
        opts = options or GenerateOptions()
        model_id = resolve_model(model or self.default_model).model_id

        system, turns = split_system(messages, opts.system)
        payload: dict[str, Any] = {
            "model": model_id,
            "max_tokens": opts.max_tokens,
            "messages": _to_anthropic_messages(turns),
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [_to_anthropic_tool(t) for t in tools]
            tool_choice = _to_anthropic_tool_choice(opts.tool_choice)
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if opts.temperature is not None:
            payload["temperature"] = opts.temperature
        if opts.effort:
            payload["output_config"] = {"effort": opts.effort}
        if opts.stop_sequences:
            payload["stop_sequences"] = list(opts.stop_sequences)
        if opts.timeout is not None:
            payload["timeout"] = opts.timeout
        if opts.extra:
            payload.update(opts.extra)

        client = self._get_client()
        try:
            raw = client.messages.create(**payload)
        except Exception as exc:  # translated below; genuine bugs re-raised
            if _is_sdk_error(exc):
                raise translate_api_exception(
                    exc, provider=self.name, model=model_id
                ) from exc
            raise

        return self._normalize(raw, model_id)

    # ── response normalization ───────────────────────────────────────────

    def _normalize(self, raw: Any, model_id: str) -> LLMResponse:
        content = getattr(raw, "content", None)
        if content is None:
            raise ProviderResponseError(
                "Anthropic response had no content field",
                provider=self.name,
                model=model_id,
            )

        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                texts.append(block.text)
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=decode_arguments(
                            block.input, provider=self.name, tool_name=block.name
                        ),
                    )
                )
            # thinking / redacted_thinking / server-tool blocks are deliberately
            # not folded into `text`: they are not the model's answer.

        return LLMResponse(
            text="\n".join(texts),
            tool_calls=tuple(tool_calls),
            model=getattr(raw, "model", model_id) or model_id,
            provider=self.name,
            stop_reason=_STOP_REASONS.get(getattr(raw, "stop_reason", None) or "", "other"),
            usage=_to_usage(getattr(raw, "usage", None)),
            raw=raw,
        )


# ── translation helpers ──────────────────────────────────────────────────


def _to_anthropic_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if pending:
            out.append({"role": "user", "content": list(pending)})
            pending.clear()

    for msg in messages:
        if msg.role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": msg.tool_call_id,
                "content": msg.content or "",
            }
            if msg.is_error:
                block["is_error"] = True
            pending.append(block)
            continue

        flush()
        if msg.role == "user":
            if not msg.content:
                raise ValueError("a user message must have content")
            out.append({"role": "user", "content": [{"type": "text", "text": msg.content}]})
        elif msg.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for call in msg.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            if not blocks:
                raise ValueError("an assistant message must have content or tool_calls")
            out.append({"role": "assistant", "content": blocks})
        else:  # pragma: no cover - split_system removed system, Message validated the rest
            raise ValueError(f"unexpected role in history: {msg.role!r}")

    flush()
    return out


def _to_anthropic_tool(tool: ToolDef) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": dict(tool.parameters),
    }


def _to_anthropic_tool_choice(choice: str | None) -> dict[str, Any] | None:
    if not choice or choice == "auto":
        return None
    if choice == "none":
        return {"type": "none"}
    if choice == "required":
        return {"type": "any"}
    return {"type": "tool", "name": choice}


def _to_usage(usage: Any) -> Usage:
    if usage is None:
        return Usage()
    return Usage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


def _is_sdk_error(exc: BaseException) -> bool:
    """True when `exc` came from the Anthropic SDK rather than from our code."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - only when the SDK is absent
        return False
    return isinstance(exc, anthropic.AnthropicError)
