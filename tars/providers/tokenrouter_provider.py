"""TokenRouter adapter — an OpenAI-compatible surface at a third-party host.

Why this is a separate adapter rather than a base-URL flag on
`OpenAIProvider`:

* `OPENAI_API_KEY` is a credential for api.openai.com. Borrowing it here would
  send a real OpenAI key to a third party, so it is never consulted; this
  adapter reads `TOKENROUTER_API_KEY` only.
* TokenRouter's OpenAI-compatible surface lives at ``/v1`` and *requires* that
  suffix — the opposite of AgentRouter's Anthropic-compatible host, where a
  `/v1` base URL double-versions the path. Here the SDK adds nothing, so the
  default carries `/v1` itself.
* A response should record which host answered, not just which wire format.

Everything else — message translation, tool calls, error mapping — is the
OpenAI adapter's, inherited unchanged, because the wire format is identical.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import ProviderNotConfiguredError, ProviderNotInstalledError
from .openai_provider import OpenAIProvider

#: The `/v1` is required here: the OpenAI SDK appends nothing of its own.
DEFAULT_BASE_URL = "https://api.tokenrouter.com/v1"

#: Free models are the reason this provider exists; default to its free Qwen.
DEFAULT_MODEL = "qwen/qwen3.8-max-free"


class TokenRouterProvider(OpenAIProvider):
    """Talks to TokenRouter's OpenAI-compatible endpoint."""

    name = "tokenrouter"

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        env: Mapping[str, str] | None = None,
        default_model: str = DEFAULT_MODEL,
    ) -> None:
        super().__init__(
            client=client,
            api_key=api_key,
            base_url=base_url,
            env=env,
            default_model=default_model,
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = self._api_key or self._env.get("TOKENROUTER_API_KEY")
        if not api_key:
            raise ProviderNotConfiguredError(
                "no TokenRouter credentials found: set TOKENROUTER_API_KEY in "
                "the environment, or pass api_key= to TokenRouterProvider. "
                "Note OPENAI_API_KEY is deliberately not consulted: that key "
                "belongs to api.openai.com.",
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

        base_url = (
            self._base_url or self._env.get("TOKENROUTER_BASE_URL") or DEFAULT_BASE_URL
        )
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        return self._client
