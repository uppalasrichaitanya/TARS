"""The one call site the agent loop uses: generate(messages, tools, model).

Everything here is about dispatch and the public surface — the adapters
themselves are tested in their own files.
"""

import pytest

from tars import providers
from tars.providers import (
    GenerateOptions,
    LLMResponse,
    Message,
    ToolDef,
    Usage,
    generate,
)
from tars.providers.base import Provider
from tars.providers.errors import UnknownProviderError
from tars.providers.registry import get_provider, register_provider, reset_providers


class SpyProvider(Provider):
    """A real Provider implementation that records what it was handed."""

    def __init__(self, name="spy"):
        self.name = name
        self.calls = []

    def generate(self, messages, tools=None, model=None, options=None):
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "model": model,
                "options": options,
            }
        )
        return LLMResponse(
            text="spied",
            tool_calls=(),
            model=model or "?",
            provider=self.name,
            stop_reason="end_turn",
            usage=Usage(),
        )

    @property
    def last(self):
        return self.calls[-1]


@pytest.fixture
def spies():
    """Replace the three real adapters with spies for the duration of a test."""
    installed = {
        name: SpyProvider(name) for name in ("anthropic", "openai", "openrouter")
    }
    for name, spy in installed.items():
        register_provider(name, lambda spy=spy: spy)
    yield installed
    reset_providers()
    for name in installed:
        register_provider(name, _real_factory(name))


def _real_factory(name):
    from tars.providers.anthropic_provider import AnthropicProvider
    from tars.providers.openai_provider import OpenAIProvider
    from tars.providers.openrouter_provider import OpenRouterProvider

    return {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "openrouter": OpenRouterProvider,
    }[name]


def test_claude_model_dispatches_to_the_anthropic_adapter(spies):
    generate([Message.user("hi")], model="claude-opus-5")
    assert len(spies["anthropic"].calls) == 1
    assert spies["openai"].calls == []


def test_gpt_model_dispatches_to_the_openai_adapter(spies):
    generate([Message.user("hi")], model="gpt-5")
    assert len(spies["openai"].calls) == 1


def test_unknown_model_dispatches_to_openrouter(spies):
    generate([Message.user("hi")], model="qwen/qwen-3-235b")
    assert len(spies["openrouter"].calls) == 1


def test_full_model_string_is_forwarded_so_the_adapter_sees_the_prefix(spies):
    # The adapter re-resolves and strips it; forwarding the original keeps the
    # routing decision auditable in one place.
    generate([Message.user("hi")], model="openrouter/anthropic/claude-opus-5")
    assert spies["openrouter"].last["model"] == "openrouter/anthropic/claude-opus-5"


def test_tools_are_forwarded_untouched(spies):
    tools = [ToolDef("ls", "list", {"type": "object", "properties": {}})]
    generate([Message.user("hi")], tools=tools, model="claude-opus-5")
    assert spies["anthropic"].last["tools"] == tools


def test_keyword_options_are_bundled_into_generate_options(spies):
    generate(
        [Message.user("hi")],
        model="claude-opus-5",
        max_tokens=123,
        system="be terse",
        effort="low",
        temperature=0.2,
        tool_choice="required",
    )
    opts = spies["anthropic"].last["options"]
    assert isinstance(opts, GenerateOptions)
    assert opts.max_tokens == 123
    assert opts.system == "be terse"
    assert opts.effort == "low"
    assert opts.temperature == 0.2
    assert opts.tool_choice == "required"


def test_an_explicit_options_object_is_passed_through(spies):
    opts = GenerateOptions(max_tokens=7)
    generate([Message.user("hi")], model="claude-opus-5", options=opts)
    assert spies["anthropic"].last["options"] is opts


def test_options_and_keyword_overrides_cannot_be_mixed(spies):
    with pytest.raises(ValueError, match="options"):
        generate(
            [Message.user("hi")],
            model="claude-opus-5",
            options=GenerateOptions(),
            max_tokens=5,
        )


def test_a_plain_string_prompt_is_accepted_as_one_user_message(spies):
    generate("just do it", model="claude-opus-5")
    sent = spies["anthropic"].last["messages"]
    assert sent == [Message.user("just do it")]


def test_the_response_comes_back_normalized(spies):
    resp = generate([Message.user("hi")], model="claude-opus-5")
    assert isinstance(resp, LLMResponse)
    assert resp.text == "spied"
    assert resp.provider == "anthropic"


def test_default_model_is_used_when_none_is_given(spies):
    generate([Message.user("hi")])
    assert len(spies["anthropic"].calls) == 1


def test_an_unknown_explicit_provider_prefix_is_rejected(spies):
    with pytest.raises(UnknownProviderError):
        generate([Message.user("hi")], model="banana:whatever")


def test_the_three_adapters_are_registered_by_default():
    reset_providers()
    providers.install_default_providers()
    assert get_provider("anthropic").name == "anthropic"
    assert get_provider("openai").name == "openai"
    assert get_provider("openrouter").name == "openrouter"


def test_public_surface_is_importable_from_the_package_root():
    for symbol in (
        "generate",
        "Message",
        "ToolCall",
        "ToolDef",
        "LLMResponse",
        "Usage",
        "GenerateOptions",
        "Provider",
        "ProviderError",
        "resolve_model",
    ):
        assert hasattr(providers, symbol), symbol
