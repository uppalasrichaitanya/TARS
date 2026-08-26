"""TokenRouter adapter: credentials, base URL, and routing.

TokenRouter serves an OpenAI-compatible Chat Completions API at its own host,
so this adapter inherits every bit of request translation from
`OpenAIProvider` (whose tests live in `test_openai_provider.py`). What's
tested here is only what differs:

* the credential is `TOKENROUTER_API_KEY` — `OPENAI_API_KEY` is deliberately
  never consulted, because a real OpenAI key must not leak to a third party
* the base URL defaults to TokenRouter and *does* carry `/v1`, which the
  OpenAI-compatible surface requires (the opposite of the Anthropic one)
* `tokenrouter:` / `tokenrouter/` route here; a slashed model id survives,
  so free models like `tokenrouter:qwen/qwen3.8-max-free` resolve whole
"""

import pytest
from test_openai_provider import RecordingClient, make_response

from tars.providers import registry
from tars.providers.errors import ProviderNotConfiguredError
from tars.providers.tokenrouter_provider import (
    DEFAULT_BASE_URL,
    TokenRouterProvider,
)
from tars.providers.types import Message


class ConstructionSpy:
    """Captures the kwargs the adapter passes to `openai.OpenAI(...)`."""

    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return RecordingClient(response=make_response())


@pytest.fixture
def spy(monkeypatch):
    import openai

    spy = ConstructionSpy()
    monkeypatch.setattr(openai, "OpenAI", spy)
    return spy


# ── credentials ──────────────────────────────────────────────────────────


def test_the_credential_comes_from_tokenrouter_api_key(spy):
    TokenRouterProvider(api_key="tr-test", env={})._get_client()
    assert spy.kwargs["api_key"] == "tr-test"


def test_tokenrouter_api_key_is_read_from_the_environment(spy):
    TokenRouterProvider(env={"TOKENROUTER_API_KEY": "tr-env"})._get_client()
    assert spy.kwargs["api_key"] == "tr-env"


def test_an_explicit_key_outranks_the_environment(spy):
    TokenRouterProvider(
        api_key="tr-explicit",
        env={"TOKENROUTER_API_KEY": "tr-env"},
    )._get_client()
    assert spy.kwargs["api_key"] == "tr-explicit"


def test_the_openai_api_key_is_deliberately_never_consulted():
    """The whole point of a separate adapter.

    A real OpenAI key belongs to api.openai.com; sending it to a third-party
    host would leak it even when that host is innocent.
    """
    with pytest.raises(ProviderNotConfiguredError):
        TokenRouterProvider(env={"OPENAI_API_KEY": "sk-openai-real"})._get_client()


# ── base URL ─────────────────────────────────────────────────────────────


def test_the_base_url_defaults_to_tokenrouter_with_its_v1(spy):
    """The OpenAI-compatible surface *requires* `/v1`; the SDK does not add it."""
    TokenRouterProvider(api_key="tr", env={})._get_client()
    assert spy.kwargs["base_url"] == DEFAULT_BASE_URL
    assert DEFAULT_BASE_URL.endswith("/v1")


def test_an_environment_base_url_outranks_the_default(spy):
    TokenRouterProvider(
        api_key="tr",
        env={"TOKENROUTER_BASE_URL": "https://mirror.example.com/v1"},
    )._get_client()
    assert spy.kwargs["base_url"] == "https://mirror.example.com/v1"


# ── routing ──────────────────────────────────────────────────────────────


def test_tokenrouter_prefixed_models_route_here_with_the_slash_kept():
    route = registry.resolve_model("tokenrouter:qwen/qwen3.8-max-free")
    assert route.provider == "tokenrouter"
    assert route.model_id == "qwen/qwen3.8-max-free"


def test_tokenrouter_slash_form_routes_here_too():
    route = registry.resolve_model("tokenrouter/qwen/qwen3.8-max-free")
    assert route.provider == "tokenrouter"
    assert route.model_id == "qwen/qwen3.8-max-free"


def test_a_generate_call_reaches_the_injected_client():
    client = RecordingClient(
        response=make_response(message={"role": "assistant", "content": "pong"})
    )
    provider = TokenRouterProvider(client=client, env={})
    response = provider.generate([Message.user("ping")], model="qwen/qwen3.8-max-free")
    assert response.text == "pong"
    assert response.provider == "tokenrouter"
