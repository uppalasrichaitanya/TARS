"""AgentRouter adapter — the Anthropic Messages API at a third-party host.

Why this is a separate adapter rather than a base-URL flag on
`AnthropicProvider`:

* AgentRouter authenticates with ``Authorization: Bearer``, which the Anthropic
  SDK sends only for ``auth_token=``. Passing the same string as ``api_key=``
  makes it an ``x-api-key`` header, which AgentRouter refuses.
* Its base URL must **not** end in ``/v1``. The SDK appends ``/v1/messages``
  itself, so a ``/v1`` host requests ``/v1/v1/messages`` and gets a bare 404
  that names neither the cause nor the fix.
* `ANTHROPIC_API_KEY` is a credential for api.anthropic.com. Borrowing it here
  would send a real Anthropic key to a third party, so it is never used.
* A response should record which host answered, not just which wire format.

Everything else — message translation, tool calls, error mapping — is the
Anthropic adapter's, inherited unchanged, because the wire format is identical.

Verified live 2026-08-20: ``claude-opus-5`` answers at this host through the
SDK. AgentRouter admits only clients it recognises — plain ``curl`` and the
OpenAI SDK are both rejected with ``unauthorized_client_error`` even carrying a
valid key — so this adapter deliberately goes through the official Anthropic
SDK rather than raw HTTP.
"""

from __future__ import annotations

from typing import Any

from .anthropic_provider import AnthropicProvider
from .errors import ProviderNotConfiguredError, ProviderNotInstalledError

#: No ``/v1``: the SDK adds it. See the module docstring.
DEFAULT_BASE_URL = "https://agentrouter.org"


class AgentRouterProvider(AnthropicProvider):
    """Talks to AgentRouter's Anthropic-compatible endpoint."""

    name = "agentrouter"

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        token = (
            self._api_key
            or self._env.get("AGENTROUTER_API_KEY")
            or self._env.get("ANTHROPIC_AUTH_TOKEN")
        )
        if not token:
            raise ProviderNotConfiguredError(
                "no AgentRouter credentials found: set AGENTROUTER_API_KEY "
                "(or ANTHROPIC_AUTH_TOKEN, which is the variable Claude Code "
                "itself uses) in the environment, or pass api_key= to "
                "AgentRouterProvider. Note ANTHROPIC_API_KEY is deliberately "
                "not consulted: that key belongs to api.anthropic.com.",
                provider=self.name,
            )

        base_url = (
            self._base_url or self._env.get("AGENTROUTER_BASE_URL") or DEFAULT_BASE_URL
        )
        _reject_versioned_base_url(base_url, provider=self.name)

        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on install state
            raise ProviderNotInstalledError(
                "the anthropic SDK is not installed",
                provider=self.name,
                install_hint="pip install anthropic",
            ) from exc

        # auth_token, never api_key — that choice is the whole adapter.
        self._client = anthropic.Anthropic(auth_token=token, base_url=base_url)
        return self._client


def _reject_versioned_base_url(base_url: str, *, provider: str) -> None:
    """Refuse a base URL the SDK would double-version."""
    if base_url.rstrip("/").endswith("/v1"):
        raise ProviderNotConfiguredError(
            f"base_url {base_url!r} ends in '/v1', which the Anthropic SDK "
            f"appends itself — requests would go to '/v1/v1/messages'. Use "
            f"'{DEFAULT_BASE_URL}' instead. (The '/v1' form is correct only "
            f"for AgentRouter's OpenAI-compatible endpoint, which is a "
            f"different API.)",
            provider=provider,
        )
