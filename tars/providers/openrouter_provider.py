"""OpenRouter adapter — plain HTTP, no SDK.

OpenRouter is an OpenAI-compatible endpoint, so this adapter is a thin stdlib
HTTP client plus the shared `_openai_compat` translation. No third-party
dependency is needed for the catch-all provider, which is the point: TARS can
reach hundreds of models with the two direct SDKs uninstalled.

Two OpenRouter-specific behaviors worth knowing:

* It can return **HTTP 200 with an `error` object in the body** when an upstream
  provider fails. That's checked explicitly — treating it as success would look
  like the model returned nothing.
* Retries live here rather than in the caller, because unlike the two SDKs there
  is no client doing it for us. 429/408/409/5xx and connection failures are
  retried with exponential backoff; other 4xx are not.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

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
from .errors import (
    ProviderAPIError,
    ProviderNotConfiguredError,
    ProviderResponseError,
    is_retryable_status,
)
from .registry import resolve_model
from .types import LLMResponse, Message, ToolDef

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-opus-5"
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 120.0

# Sent so OpenRouter can attribute traffic; it shows up on their dashboards and
# in rankings. Overridable via env for anyone running a fork.
DEFAULT_REFERER = "https://github.com/tars-agent/tars"
DEFAULT_TITLE = "TARS"


@dataclass(frozen=True)
class HTTPResponse:
    """A minimal transport result: status plus decoded JSON (or raw text)."""

    status: int
    json: Any | None = None
    text: str | None = None


#: A transport takes (url, headers, body, timeout) and returns an HTTPResponse.
#: Injectable so the adapter is testable without a network and swappable for
#: httpx/requests in a host application that already has one.
Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], HTTPResponse]


def urllib_transport(
    url: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    timeout: float,
) -> HTTPResponse:
    """Default transport: stdlib only, no third-party HTTP dependency."""
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _read(response.status, response.read())
    except urllib.error.HTTPError as exc:
        # An HTTP error response is data, not an exception, as far as the
        # adapter's retry logic is concerned.
        return _read(exc.code, exc.read())


def _read(status: int, payload: bytes) -> HTTPResponse:
    text = payload.decode("utf-8", errors="replace")
    try:
        return HTTPResponse(status, json.loads(text))
    except json.JSONDecodeError:
        return HTTPResponse(status, None, text=text)


class OpenRouterProvider(Provider):
    """Catch-all provider: anything that isn't a direct-SDK model goes here."""

    name = "openrouter"

    def __init__(
        self,
        *,
        transport: Transport | None = None,
        api_key: str | None = None,
        url: str = OPENROUTER_URL,
        env: Mapping[str, str] | None = None,
        default_model: str = DEFAULT_MODEL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._transport = transport or urllib_transport
        self._api_key = api_key
        self._url = url
        self._env = os.environ if env is None else env
        self.default_model = default_model
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        if sleep is not None:
            self._sleep = sleep
        else:
            import time

            self._sleep = time.sleep

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
        body: dict[str, Any] = {
            "model": model_id,
            "messages": to_chat_messages(turns, system),
            # OpenRouter normalizes to `max_tokens` across its upstreams.
            "max_tokens": opts.max_tokens,
        }
        if tools:
            body["tools"] = to_chat_tools(tools)
            tool_choice = to_chat_tool_choice(opts.tool_choice)
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        if opts.temperature is not None:
            body["temperature"] = opts.temperature
        if opts.effort:
            body["reasoning"] = {"effort": opts.effort}
        if opts.stop_sequences:
            body["stop"] = list(opts.stop_sequences)
        if opts.extra:
            body.update(opts.extra)

        timeout = opts.timeout if opts.timeout is not None else DEFAULT_TIMEOUT_SECONDS
        response = self._post(body, timeout=timeout, model_id=model_id)
        payload = self._payload(response, model_id)
        return normalize_completion(payload, provider=self.name, fallback_model=model_id)

    # ── HTTP ─────────────────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        api_key = self._api_key or self._env.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ProviderNotConfiguredError(
                "no OpenRouter credentials found: set OPENROUTER_API_KEY in the "
                "environment, or pass api_key= to OpenRouterProvider",
                provider=self.name,
            )
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._env.get("OPENROUTER_REFERER", DEFAULT_REFERER),
            "X-Title": self._env.get("OPENROUTER_TITLE", DEFAULT_TITLE),
        }

    def _post(
        self,
        body: Mapping[str, Any],
        *,
        timeout: float,
        model_id: str,
    ) -> HTTPResponse:
        headers = self._headers()
        attempts = self.max_retries + 1
        last_error: ProviderAPIError | None = None

        for attempt in range(attempts):
            try:
                response = self._transport(self._url, headers, body, timeout)
            except Exception as exc:
                # No status at all: connection reset, DNS, timeout. Retryable.
                last_error = translate_api_exception(
                    exc, provider=self.name, model=model_id
                )
            else:
                if response.status < 400:
                    return response
                last_error = ProviderAPIError(
                    f"openrouter request failed with HTTP {response.status}: "
                    f"{_error_message(response)}",
                    provider=self.name,
                    model=model_id,
                    status_code=response.status,
                )

            if not last_error.retryable or attempt == attempts - 1:
                raise last_error
            self._sleep(self.backoff_seconds * (2**attempt))

        raise last_error  # pragma: no cover - loop always returns or raises

    def _payload(self, response: HTTPResponse, model_id: str) -> Mapping[str, Any]:
        if response.json is None:
            raise ProviderResponseError(
                "openrouter returned a non-JSON body "
                f"(HTTP {response.status}): {(response.text or '')[:200]!r}",
                provider=self.name,
                model=model_id,
            )
        if not isinstance(response.json, Mapping):
            raise ProviderResponseError(
                f"openrouter returned JSON that is not an object: "
                f"{type(response.json).__name__}",
                provider=self.name,
                model=model_id,
            )
        error = response.json.get("error")
        if error:
            # HTTP 200 + an error body happens when an upstream provider fails.
            status = _error_status(error)
            raise ProviderAPIError(
                f"openrouter reported an upstream error: {_message_of(error)}",
                provider=self.name,
                model=model_id,
                status_code=status,
                retryable=is_retryable_status(status),
            )
        return response.json


def _error_message(response: HTTPResponse) -> str:
    if isinstance(response.json, Mapping):
        return _message_of(response.json.get("error")) or str(response.json)[:200]
    return (response.text or "")[:200]


def _message_of(error: Any) -> str:
    if isinstance(error, Mapping):
        return str(error.get("message") or error)
    return str(error) if error else ""


def _error_status(error: Any) -> int | None:
    if isinstance(error, Mapping):
        code = error.get("code")
        if isinstance(code, int):
            return code
        if isinstance(code, str) and code.isdigit():
            return int(code)
    return None
