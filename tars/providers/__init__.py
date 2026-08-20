"""TARS provider adapter layer.

One interface, four providers:

    from tars.providers import Message, ToolDef, generate

    response = generate(
        [Message.user("What files are in this repo?")],
        tools=[ToolDef("ls", "List files", {"type": "object", "properties": {}})],
        model="claude-opus-5",          # or "gpt-5", or "qwen/qwen-3-235b"
    )
    response.text, response.tool_calls, response.usage.total_tokens

Routing is by model string: ``claude-*`` uses the Anthropic SDK, ``gpt-*``/
``o3``-style uses the OpenAI SDK, everything else goes to OpenRouter over plain
HTTP. Prefix a model with ``openrouter/`` to force it through OpenRouter, or
``anthropic/`` / ``openai/`` to be explicit.

``agentrouter:claude-opus-5`` routes to AgentRouter, which serves the Anthropic
Messages API at its own host under a bearer token. It is reachable only by an
explicit prefix — a bare ``claude-*`` keeps going to Anthropic, so adding it
re-routes nothing that already worked.

Loop code should import only from this module. If you find yourself reaching for
`response.raw` or a provider class directly, that's the signal a provider quirk
is leaking upward.
"""

from __future__ import annotations

from typing import Any, Sequence

from .base import DEFAULT_MAX_TOKENS, GenerateOptions, Provider
from .errors import (
    ProviderAPIError,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderNotInstalledError,
    ProviderResponseError,
    UnknownProviderError,
)
from .registry import (
    AGENTROUTER,
    ANTHROPIC,
    KNOWN_PROVIDERS,
    OPENAI,
    OPENROUTER,
    ModelRoute,
    get_provider,
    install_default_providers,
    register_provider,
    reset_providers,
    resolve_model,
)
from .types import (
    LLMResponse,
    Message,
    Role,
    StopReason,
    ToolCall,
    ToolDef,
    Usage,
)

#: Used when a caller doesn't name a model. Deliberately the flagship: the loop's
#: router is what steps down to a cheaper tier, not this default.
DEFAULT_MODEL = "claude-opus-5"

__all__ = [
    "AGENTROUTER",
    "ANTHROPIC",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "GenerateOptions",
    "KNOWN_PROVIDERS",
    "LLMResponse",
    "Message",
    "ModelRoute",
    "OPENAI",
    "OPENROUTER",
    "Provider",
    "ProviderAPIError",
    "ProviderError",
    "ProviderNotConfiguredError",
    "ProviderNotInstalledError",
    "ProviderResponseError",
    "Role",
    "StopReason",
    "ToolCall",
    "ToolDef",
    "UnknownProviderError",
    "Usage",
    "generate",
    "get_provider",
    "install_default_providers",
    "register_provider",
    "reset_providers",
    "resolve_model",
]

install_default_providers()


def generate(
    messages: Sequence[Message] | str,
    tools: Sequence[ToolDef] | None = None,
    model: str | None = None,
    *,
    options: GenerateOptions | None = None,
    **overrides: Any,
) -> LLMResponse:
    """Run one completion against whichever provider owns `model`.

    Args:
        messages: Conversation history, or a bare string treated as one user
            message.
        tools: Tools to offer the model. Omit for a plain completion.
        model: Model string. Routed per this module's docstring; defaults to
            `DEFAULT_MODEL`.
        options: A prebuilt `GenerateOptions`. Mutually exclusive with the
            keyword overrides below.
        **overrides: Any `GenerateOptions` field (`max_tokens`, `system`,
            `temperature`, `effort`, `tool_choice`, `stop_sequences`, `timeout`,
            `extra`) as a shorthand.

    Raises:
        UnknownProviderError: the model named a provider we have no adapter for.
        ProviderNotConfiguredError: credentials for that provider are missing.
        ProviderNotInstalledError: that provider's SDK isn't installed.
        ProviderAPIError: the provider was reached and failed the request.
        ProviderResponseError: the response couldn't be normalized.
    """
    if options is not None and overrides:
        raise ValueError(
            "pass either options= or keyword overrides, not both: "
            f"got options plus {sorted(overrides)}"
        )
    if isinstance(messages, str):
        messages = [Message.user(messages)]

    opts = options if options is not None else GenerateOptions(**overrides)
    model_name = model or DEFAULT_MODEL
    route = resolve_model(model_name)
    provider = get_provider(route.provider)
    # The full model string is forwarded, not `route.model_id`: adapters
    # re-resolve it, which keeps the prefix-stripping rule in exactly one place.
    return provider.generate(messages, tools, model_name, opts)
