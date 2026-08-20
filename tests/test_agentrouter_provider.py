"""AgentRouter adapter: credentials, base URL, and routing.

AgentRouter serves the Anthropic Messages API at its own host, so this adapter
inherits every bit of request translation from `AnthropicProvider` and the tests
for that live in `test_anthropic_provider.py`. What's tested here is only what
differs, and each difference is something that was verified against the live
service rather than assumed:

* the credential goes out as a *bearer* token, never as `x-api-key`
* the base URL defaults to AgentRouter and must not carry a `/v1` suffix
* `agentrouter:` / `agentrouter/` route here, while bare `claude-*` does not
"""

import pytest
from test_anthropic_provider import RecordingClient, make_response

from tars.providers import registry
from tars.providers.agentrouter_provider import (
    DEFAULT_BASE_URL,
    AgentRouterProvider,
)
from tars.providers.errors import ProviderNotConfiguredError
from tars.providers.types import Message


class ConstructionSpy:
    """Captures the kwargs the adapter passes to `anthropic.Anthropic(...)`."""

    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return RecordingClient()


@pytest.fixture
def spy(monkeypatch):
    import anthropic

    spy = ConstructionSpy()
    monkeypatch.setattr(anthropic, "Anthropic", spy)
    return spy


# ── credentials ──────────────────────────────────────────────────────────


def test_the_credential_is_sent_as_a_bearer_token_not_an_api_key(spy):
    """The whole reason this adapter exists.

    AgentRouter authenticates with `Authorization: Bearer`. The Anthropic SDK
    only sends that when given `auth_token=`; `api_key=` becomes an `x-api-key`
    header, which AgentRouter rejects.
    """
    AgentRouterProvider(api_key="sk-test", env={})._get_client()
    assert spy.kwargs["auth_token"] == "sk-test"
    assert "api_key" not in spy.kwargs


def test_agentrouter_api_key_is_read_from_the_environment(spy):
    AgentRouterProvider(env={"AGENTROUTER_API_KEY": "sk-env"})._get_client()
    assert spy.kwargs["auth_token"] == "sk-env"


def test_anthropic_auth_token_is_accepted_as_a_fallback(spy):
    """The shape Claude Code itself uses, and what's already set on this machine."""
    AgentRouterProvider(env={"ANTHROPIC_AUTH_TOKEN": "sk-claude-code"})._get_client()
    assert spy.kwargs["auth_token"] == "sk-claude-code"


def test_an_explicit_key_outranks_the_environment(spy):
    AgentRouterProvider(
        api_key="sk-explicit",
        env={"AGENTROUTER_API_KEY": "sk-env"},
    )._get_client()
    assert spy.kwargs["auth_token"] == "sk-explicit"


def test_agentrouter_api_key_outranks_the_anthropic_fallback(spy):
    AgentRouterProvider(
        env={"AGENTROUTER_API_KEY": "sk-own", "ANTHROPIC_AUTH_TOKEN": "sk-borrowed"}
    )._get_client()
    assert spy.kwargs["auth_token"] == "sk-own"


def test_missing_credentials_says_agentrouter_and_names_its_variable():
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        AgentRouterProvider(env={})._get_client()
    assert excinfo.value.provider == "agentrouter"
    assert "AGENTROUTER_API_KEY" in str(excinfo.value)


def test_an_anthropic_api_key_alone_is_not_treated_as_a_credential():
    """`ANTHROPIC_API_KEY` is for api.anthropic.com and is not an AgentRouter key.

    Silently borrowing it would send a real Anthropic key to a third-party host.
    """
    with pytest.raises(ProviderNotConfiguredError):
        AgentRouterProvider(env={"ANTHROPIC_API_KEY": "sk-ant-real"})._get_client()


# ── base URL, including the documented /v1 trap ──────────────────────────


def test_the_base_url_defaults_to_agentrouter(spy):
    AgentRouterProvider(api_key="sk-test", env={})._get_client()
    assert spy.kwargs["base_url"] == DEFAULT_BASE_URL == "https://agentrouter.org"


def test_the_default_base_url_carries_no_v1_suffix():
    """The Anthropic SDK appends `/v1/messages` itself."""
    assert not DEFAULT_BASE_URL.rstrip("/").endswith("/v1")


def test_agentrouter_base_url_overrides_the_default(spy):
    AgentRouterProvider(
        api_key="sk-test",
        env={"AGENTROUTER_BASE_URL": "https://staging.example.com"},
    )._get_client()
    assert spy.kwargs["base_url"] == "https://staging.example.com"


def test_anthropic_base_url_is_ignored(spy):
    """This adapter owns its host; it doesn't follow the Anthropic provider's."""
    AgentRouterProvider(
        api_key="sk-test",
        env={"ANTHROPIC_BASE_URL": "https://api.anthropic.com"},
    )._get_client()
    assert spy.kwargs["base_url"] == DEFAULT_BASE_URL


@pytest.mark.parametrize(
    "bad_url",
    [
        "https://agentrouter.org/v1",
        "https://agentrouter.org/v1/",
    ],
)
def test_a_base_url_ending_in_v1_is_refused_with_an_explanation(bad_url):
    """AgentRouter's own docs warn about this, and the failure is obscure.

    The SDK would request `/v1/v1/messages` and report a 404 that names neither
    the cause nor the fix, so the adapter refuses up front instead.
    """
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        AgentRouterProvider(api_key="sk-test", base_url=bad_url, env={})._get_client()
    message = str(excinfo.value)
    assert "/v1" in message
    assert "https://agentrouter.org" in message


def test_an_explicit_base_url_outranks_the_environment(spy):
    AgentRouterProvider(
        api_key="sk-test",
        base_url="https://explicit.example.com",
        env={"AGENTROUTER_BASE_URL": "https://env.example.com"},
    )._get_client()
    assert spy.kwargs["base_url"] == "https://explicit.example.com"


# ── identity and routing ─────────────────────────────────────────────────


def test_responses_are_attributed_to_agentrouter():
    """Not "anthropic" — otherwise a log can't tell you which host answered."""
    prov = AgentRouterProvider(client=RecordingClient())
    assert prov.generate([Message.user("hi")]).provider == "agentrouter"


def test_the_prefixed_form_routes_here():
    for model in ("agentrouter:claude-opus-5", "agentrouter/claude-opus-5"):
        route = registry.resolve_model(model)
        assert route.provider == "agentrouter", model
        assert route.model_id == "claude-opus-5", model


def test_the_prefix_is_stripped_before_the_request_goes_out():
    client = RecordingClient()
    prov = AgentRouterProvider(client=client)
    prov.generate([Message.user("hi")], model="agentrouter:claude-opus-5")
    assert client.last["model"] == "claude-opus-5"


def test_a_bare_claude_name_still_routes_to_anthropic():
    """Adding this provider must not silently re-route existing callers."""
    assert registry.resolve_model("claude-opus-5").provider == "anthropic"


def test_agentrouter_is_a_known_provider():
    assert "agentrouter" in registry.KNOWN_PROVIDERS


def test_the_provider_name_is_exported_from_the_package_surface():
    """Loop code imports from `tars.providers`, not from the submodules."""
    from tars import providers

    assert providers.AGENTROUTER == "agentrouter"
    assert "AGENTROUTER" in providers.__all__


def test_it_is_registered_as_a_default_adapter():
    registry.install_default_providers()
    assert isinstance(registry.get_provider("agentrouter"), AgentRouterProvider)


def test_the_default_model_is_the_one_agentrouter_actually_serves():
    """Verified live: `claude-opus-5` is a valid model id at this host."""
    client = RecordingClient(response=make_response())
    prov = AgentRouterProvider(client=client)
    prov.generate([Message.user("hi")])
    assert client.last["model"] == "claude-opus-5"
