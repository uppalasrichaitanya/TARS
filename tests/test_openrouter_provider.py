"""OpenRouter adapter: plain HTTP, no SDK.

The HTTP transport is injected, so these tests assert on the exact request that
would go on the wire (URL, headers, JSON body) and on how responses and HTTP
failures are handled — without a network call.
"""

import pytest

from tars.providers.base import GenerateOptions
from tars.providers.errors import (
    ProviderAPIError,
    ProviderNotConfiguredError,
    ProviderResponseError,
)
from tars.providers.openrouter_provider import (
    OPENROUTER_URL,
    HTTPResponse,
    OpenRouterProvider,
)
from tars.providers.types import Message, ToolCall, ToolDef


def chat_response(
    content="hi",
    tool_calls=None,
    finish_reason="stop",
    model="meta-llama/llama-3.3-70b-instruct",
):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "gen-1",
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }


class RecordingTransport:
    """Records outbound requests; replays a queued list of HTTPResponses."""

    def __init__(self, *responses):
        self.responses = list(responses) or [HTTPResponse(200, chat_response())]
        self.requests = []

    def __call__(self, url, headers, body, timeout):
        self.requests.append(
            {"url": url, "headers": headers, "body": body, "timeout": timeout}
        )
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]

    @property
    def last(self):
        return self.requests[-1]


def provider(*responses, env=None, **kwargs):
    transport = RecordingTransport(*responses)
    prov = OpenRouterProvider(
        transport=transport,
        env={"OPENROUTER_API_KEY": "sk-or-test"} if env is None else env,
        sleep=lambda _seconds: None,
        **kwargs,
    )
    return prov, transport


# ── request shape ────────────────────────────────────────────────────────


def test_posts_to_the_openrouter_completions_endpoint():
    prov, transport = provider()
    prov.generate([Message.user("hi")], model="meta-llama/llama-3.3-70b-instruct")
    assert transport.last["url"] == OPENROUTER_URL


def test_bearer_token_is_sent():
    prov, transport = provider()
    prov.generate([Message.user("hi")], model="mistralai/mistral-large")
    assert transport.last["headers"]["Authorization"] == "Bearer sk-or-test"
    assert transport.last["headers"]["Content-Type"] == "application/json"


def test_attribution_headers_are_sent_so_usage_is_traceable():
    prov, transport = provider()
    prov.generate([Message.user("hi")], model="mistralai/mistral-large")
    assert "HTTP-Referer" in transport.last["headers"]
    assert transport.last["headers"]["X-Title"]


def test_body_uses_the_chat_completions_dialect():
    prov, transport = provider()
    prov.generate(
        [Message.system("be terse"), Message.user("hi")],
        model="mistralai/mistral-large",
    )
    body = transport.last["body"]
    assert body["model"] == "mistralai/mistral-large"
    assert body["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


def test_openrouter_prefix_is_stripped_but_the_vendor_path_is_kept():
    prov, transport = provider()
    prov.generate([Message.user("hi")], model="openrouter/anthropic/claude-opus-5")
    assert transport.last["body"]["model"] == "anthropic/claude-opus-5"


def test_tools_use_the_function_envelope():
    prov, transport = provider()
    prov.generate(
        [Message.user("hi")],
        tools=[ToolDef("ls", "list files", {"type": "object", "properties": {}})],
        model="mistralai/mistral-large",
    )
    assert transport.last["body"]["tools"][0]["function"]["name"] == "ls"


def test_max_tokens_uses_the_classic_parameter_name():
    # OpenRouter normalizes to `max_tokens` across its upstreams.
    prov, transport = provider()
    prov.generate(
        [Message.user("hi")],
        model="mistralai/mistral-large",
        options=GenerateOptions(max_tokens=64),
    )
    assert transport.last["body"]["max_tokens"] == 64


def test_temperature_is_omitted_unless_set():
    prov, transport = provider()
    prov.generate([Message.user("hi")], model="mistralai/mistral-large")
    assert "temperature" not in transport.last["body"]


def test_extra_options_are_merged_into_the_body():
    prov, transport = provider()
    prov.generate(
        [Message.user("hi")],
        model="mistralai/mistral-large",
        options=GenerateOptions(extra={"provider": {"order": ["Together"]}}),
    )
    assert transport.last["body"]["provider"] == {"order": ["Together"]}


def test_timeout_is_passed_to_the_transport():
    prov, transport = provider()
    prov.generate(
        [Message.user("hi")],
        model="mistralai/mistral-large",
        options=GenerateOptions(timeout=5.0),
    )
    assert transport.last["timeout"] == 5.0


# ── response handling ────────────────────────────────────────────────────


def test_text_response_is_normalized():
    prov, _ = provider(HTTPResponse(200, chat_response(content="the answer")))
    resp = prov.generate([Message.user("hi")], model="meta-llama/llama-3.3-70b-instruct")
    assert resp.text == "the answer"
    assert resp.provider == "openrouter"
    assert resp.model == "meta-llama/llama-3.3-70b-instruct"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 7
    assert resp.usage.output_tokens == 3


def test_tool_calls_are_normalized_with_decoded_arguments():
    prov, _ = provider(
        HTTPResponse(
            200,
            chat_response(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                    }
                ],
            ),
        )
    )
    resp = prov.generate([Message.user("hi")], model="mistralai/mistral-large")
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls == (
        ToolCall(id="c1", name="read_file", arguments={"path": "a.py"}),
    )


def test_a_body_level_error_is_reported_even_with_http_200():
    # OpenRouter can return HTTP 200 with an error object in the body when an
    # upstream provider fails. Treating that as a successful empty response
    # would silently look like the model said nothing.
    prov, _ = provider(
        HTTPResponse(200, {"error": {"message": "upstream is down", "code": 502}})
    )
    with pytest.raises(ProviderAPIError) as excinfo:
        prov.generate([Message.user("hi")], model="mistralai/mistral-large")
    assert "upstream is down" in str(excinfo.value)
    assert excinfo.value.status_code == 502


def test_non_json_body_is_a_response_error():
    prov, _ = provider(HTTPResponse(200, None, text="<html>gateway</html>"))
    with pytest.raises(ProviderResponseError, match="JSON"):
        prov.generate([Message.user("hi")], model="mistralai/mistral-large")


# ── HTTP failure handling ────────────────────────────────────────────────


def test_client_error_is_a_non_retryable_api_error_and_is_not_retried():
    prov, transport = provider(HTTPResponse(400, {"error": {"message": "bad model"}}))
    with pytest.raises(ProviderAPIError) as excinfo:
        prov.generate([Message.user("hi")], model="mistralai/mistral-large")
    assert excinfo.value.status_code == 400
    assert excinfo.value.retryable is False
    assert len(transport.requests) == 1


def test_rate_limit_is_retried_then_succeeds():
    prov, transport = provider(
        HTTPResponse(429, {"error": {"message": "slow down"}}),
        HTTPResponse(200, chat_response(content="ok")),
    )
    resp = prov.generate([Message.user("hi")], model="mistralai/mistral-large")
    assert resp.text == "ok"
    assert len(transport.requests) == 2


def test_retries_are_bounded_and_the_last_failure_is_raised():
    prov, transport = provider(HTTPResponse(503, {"error": {"message": "unavailable"}}), max_retries=2)
    with pytest.raises(ProviderAPIError) as excinfo:
        prov.generate([Message.user("hi")], model="mistralai/mistral-large")
    assert excinfo.value.status_code == 503
    assert excinfo.value.retryable is True
    assert len(transport.requests) == 3  # initial attempt + 2 retries


def test_transport_exceptions_are_retried_then_translated():
    def failing_transport(url, headers, body, timeout):
        raise OSError("connection reset")

    prov = OpenRouterProvider(
        transport=failing_transport,
        env={"OPENROUTER_API_KEY": "sk-or-test"},
        sleep=lambda _s: None,
        max_retries=1,
    )
    with pytest.raises(ProviderAPIError) as excinfo:
        prov.generate([Message.user("hi")], model="mistralai/mistral-large")
    assert excinfo.value.retryable is True


def test_missing_credentials_raise_not_configured():
    prov, _ = provider(env={})
    with pytest.raises(ProviderNotConfiguredError, match="OPENROUTER_API_KEY"):
        prov.generate([Message.user("hi")], model="mistralai/mistral-large")


def test_empty_message_list_is_rejected_before_any_request():
    prov, transport = provider()
    with pytest.raises(ValueError):
        prov.generate([], model="mistralai/mistral-large")
    assert transport.requests == []
