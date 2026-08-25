"""OpenAI adapter — direct SDK, first-class per the spec.

Uses Chat Completions, which is also the dialect OpenRouter speaks, so the
wire translation is shared with the OpenRouter adapter (`_openai_compat`).

Quirks handled here:

* `max_completion_tokens`, not `max_tokens` — current reasoning models reject
  the old parameter.
* `temperature` is only sent when the caller set it; reasoning models reject it.
* `reasoning_effort` is only sent when the caller asked for an effort level,
  since non-reasoning models reject it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from ._openai_compat import (
    normalize_completion,
    to_chat_messages,
    to_chat_tool_choice,
    to_chat_tools,
)
from .base import (
    GenerateOptions,
    Provider,
    require_messages,
    split_system,
    translate_api_exception,
)
from .errors import ProviderNotConfiguredError, ProviderNotInstalledError
from .registry import resolve_model
from .types import LLMResponse, Message, ToolDef

DEFAULT_MODEL = "gpt-5"


class OpenAIProvider(Provider):
    """Talks to the OpenAI Chat Completions API through the official SDK."""

    name = "openai"

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
        self._env = os.environ if env is None else env
        self.default_model = default_model

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = self._api_key or self._env.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderNotConfiguredError(
                "no OpenAI credentials found: set OPENAI_API_KEY in the "
                "environment, or pass api_key= to OpenAIProvider",
                provider=self.name,
            )

        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on install state
            raise ProviderNotInstalledError(
                "the openai SDK is not installed",
                provider=self.name,
                install_hint="pip install openai",
            ) from exc

        kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = self._base_url or self._env.get("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        return self._client

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
            "messages": to_chat_messages(turns, system),
            "max_completion_tokens": opts.max_tokens,
        }
        if tools:
            payload["tools"] = to_chat_tools(tools)
            tool_choice = to_chat_tool_choice(opts.tool_choice)
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if opts.temperature is not None:
            payload["temperature"] = opts.temperature
        if opts.effort:
            payload["reasoning_effort"] = opts.effort
        if opts.stop_sequences:
            payload["stop"] = list(opts.stop_sequences)
        if opts.timeout is not None:
            payload["timeout"] = opts.timeout
        if opts.extra:
            payload.update(opts.extra)

        client = self._get_client()
        try:
            raw = client.chat.completions.create(**payload)
        except Exception as exc:  # translated below; genuine bugs re-raised
            if _is_sdk_error(exc):
                raise translate_api_exception(
                    exc, provider=self.name, model=model_id
                ) from exc
            raise

        return normalize_completion(raw, provider=self.name, fallback_model=model_id)


def _is_sdk_error(exc: BaseException) -> bool:
    """True when `exc` came from the OpenAI SDK rather than from our code."""
    try:
        import openai
    except ImportError:  # pragma: no cover - only when the SDK is absent
        return False
    return isinstance(exc, openai.OpenAIError)
