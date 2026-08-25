"""Model-string routing and provider construction.

Routing rules, in order:

1. ``provider/model`` or ``provider:model`` where the prefix names a known
   provider -> that provider, prefix stripped. This is the escape hatch for
   forcing a model through a specific route (``openrouter/anthropic/claude-opus-5``
   deliberately goes over OpenRouter rather than the direct Anthropic SDK).
2. ``something:model`` where ``something`` has no slash and is *not* a known
   provider -> error. An explicit namespace that we don't recognize is a typo,
   not something to silently guess at. (A colon *after* a slash is left alone;
   OpenRouter ids like ``deepseek/deepseek-r1:free`` use it as a variant tag.)
3. Bare well-known names: ``claude-*`` -> Anthropic, ``gpt-*``/``o3``-style ->
   OpenAI.
4. Everything else -> OpenRouter, verbatim. That's the spec's rule: the two
   direct SDKs stay first-class, OpenRouter is the catch-all.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from .base import Provider
from .errors import UnknownProviderError

ANTHROPIC = "anthropic"
OPENAI = "openai"
OPENROUTER = "openrouter"
AGENTROUTER = "agentrouter"

KNOWN_PROVIDERS: tuple[str, ...] = (ANTHROPIC, OPENAI, OPENROUTER, AGENTROUTER)

# Bare-name patterns for the two direct-SDK providers.
_ANTHROPIC_BARE = re.compile(r"^claude[-_.]", re.IGNORECASE)
_OPENAI_BARE = re.compile(r"^(gpt[-_.]|chatgpt[-_.]|o[1-9]([-_.]|$))", re.IGNORECASE)


@dataclass(frozen=True)
class ModelRoute:
    """Where a model string should go, and what to call it once there."""

    provider: str
    model_id: str


def resolve_model(model: str) -> ModelRoute:
    """Map a caller-facing model string to (provider, upstream model id)."""
    if model is None:
        raise ValueError("model must be a non-empty string")
    name = model.strip()
    if not name:
        raise ValueError("model must be a non-empty string")

    if "/" in name:
        prefix, _, rest = name.partition("/")
        if prefix.lower() in KNOWN_PROVIDERS and rest:
            return ModelRoute(prefix.lower(), rest)

    if ":" in name:
        prefix, _, rest = name.partition(":")
        if "/" not in prefix:
            if prefix.lower() in KNOWN_PROVIDERS and rest:
                return ModelRoute(prefix.lower(), rest)
            raise UnknownProviderError(
                f"unknown provider {prefix!r} in model {model!r}; "
                f"known providers: {', '.join(KNOWN_PROVIDERS)}"
            )

    if _ANTHROPIC_BARE.match(name):
        return ModelRoute(ANTHROPIC, name)
    if _OPENAI_BARE.match(name):
        return ModelRoute(OPENAI, name)

    # Spec: "OpenRouter via plain HTTP for everything else."
    return ModelRoute(OPENROUTER, name)


# ── provider construction ────────────────────────────────────────────────
# Factories are lazy so that importing this module never imports the SDKs, and
# so a missing `openai` install can't break an Anthropic-only run.

ProviderFactory = Callable[[], Provider]

_FACTORIES: dict[str, ProviderFactory] = {}
_CACHE: dict[str, Provider] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register (or replace) a provider factory. Clears any cached instance."""
    _FACTORIES[name] = factory
    _CACHE.pop(name, None)


def get_provider(name: str) -> Provider:
    """Return the provider adapter registered under `name`, constructing once."""
    if name in _CACHE:
        return _CACHE[name]
    factory = _FACTORIES.get(name)
    if factory is None:
        raise UnknownProviderError(
            f"no adapter registered for provider {name!r}; "
            f"known providers: {', '.join(sorted(_FACTORIES))}"
        )
    provider = factory()
    _CACHE[name] = provider
    return provider


def reset_providers() -> None:
    """Drop cached provider instances (used by tests and after config changes)."""
    _CACHE.clear()


def _default_factories() -> None:
    from .agentrouter_provider import AgentRouterProvider
    from .anthropic_provider import AnthropicProvider
    from .openai_provider import OpenAIProvider
    from .openrouter_provider import OpenRouterProvider

    register_provider(ANTHROPIC, AnthropicProvider)
    register_provider(OPENAI, OpenAIProvider)
    register_provider(OPENROUTER, OpenRouterProvider)
    register_provider(AGENTROUTER, AgentRouterProvider)


def install_default_providers() -> None:
    """Register the four built-in adapters if nothing is registered yet."""
    if not _FACTORIES:
        _default_factories()
