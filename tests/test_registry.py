"""Model-string -> provider routing.

The spec's rule: Claude and OpenAI get direct SDKs and stay first-class,
OpenRouter handles "everything else". These tests pin that down, including the
escape hatch for forcing a model through a specific provider.
"""

import pytest

from tars.providers.errors import UnknownProviderError
from tars.providers.registry import resolve_model


def test_bare_claude_model_routes_to_anthropic():
    route = resolve_model("claude-opus-5")
    assert route.provider == "anthropic"
    assert route.model_id == "claude-opus-5"


def test_bare_gpt_model_routes_to_openai():
    assert resolve_model("gpt-5").provider == "openai"


def test_bare_o_series_model_routes_to_openai():
    assert resolve_model("o3-mini").provider == "openai"


def test_explicit_prefix_wins_and_is_stripped():
    route = resolve_model("anthropic/claude-sonnet-5")
    assert route.provider == "anthropic"
    assert route.model_id == "claude-sonnet-5"


def test_openrouter_prefix_keeps_the_rest_of_the_path_intact():
    # OpenRouter's own model ids contain slashes; only the routing prefix is
    # consumed, so `anthropic/claude-opus-5` survives as the upstream id.
    route = resolve_model("openrouter/anthropic/claude-opus-5")
    assert route.provider == "openrouter"
    assert route.model_id == "anthropic/claude-opus-5"


def test_vendor_slash_model_without_known_prefix_goes_to_openrouter():
    route = resolve_model("meta-llama/llama-3.3-70b-instruct")
    assert route.provider == "openrouter"
    assert route.model_id == "meta-llama/llama-3.3-70b-instruct"


def test_unrecognized_bare_model_falls_back_to_openrouter():
    route = resolve_model("some-new-model-v9")
    assert route.provider == "openrouter"
    assert route.model_id == "some-new-model-v9"


def test_unknown_explicit_prefix_is_an_error_not_a_silent_fallback():
    with pytest.raises(UnknownProviderError, match="banana"):
        resolve_model("banana:llama-3")


def test_colon_syntax_is_accepted_as_a_prefix_too():
    route = resolve_model("openai:gpt-5")
    assert route.provider == "openai"
    assert route.model_id == "gpt-5"


def test_empty_model_is_rejected():
    with pytest.raises(ValueError, match="model"):
        resolve_model("   ")


def test_model_string_is_stripped_of_whitespace():
    assert resolve_model("  claude-opus-5  ").model_id == "claude-opus-5"
