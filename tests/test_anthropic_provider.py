"""Anthropic adapter: request translation and response normalization.

The client is injected, so these tests exercise the real translation code with
no network. Responses are built through the real `anthropic.types.Message`
Pydantic model, so a shape our parser accepts is a shape the SDK would actually
hand us.
"""

import anthropic
import pytest
from anthropic.types import Message as AnthropicMessage

from tars.providers.anthropic_provider import AnthropicProvider
from tars.providers.base import GenerateOptions
from tars.providers.errors import (
    ProviderAPIError,
    ProviderNotConfiguredError,
    ProviderResponseError,
)
from tars.providers.types import Message, ToolCall, ToolDef


def make_response(
    content=None,
    stop_reason="end_turn",
    model="claude-opus-5",
    usage=None,
):
    """Build a real SDK Message object from the wire shape."""
    return AnthropicMessage.model_validate(
        {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content if content is not None else [{"type": "text", "text": "hi"}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage or {"input_tokens": 12, "output_tokens": 5},
        }
    )


class RecordingClient:
    """Stands in for anthropic.Anthropic — records kwargs, returns a canned reply."""

    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response if self._response is not None else make_response()

    @property
    def last(self):
        return self.calls[-1]


def provider(response=None, error=None):
    client = RecordingClient(response=response, error=error)
    return AnthropicProvider(client=client), client


# ── request translation ──────────────────────────────────────────────────


def test_user_messages_pass_through_as_content_blocks():
    prov, client = provider()
    prov.generate([Message.user("hello")], model="claude-opus-5")
    assert client.last["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]


def test_system_messages_are_hoisted_to_the_system_parameter():
    prov, client = provider()
    prov.generate(
        [Message.system("be terse"), Message.user("hi")],
        model="claude-opus-5",
    )
    assert client.last["system"] == "be terse"
    assert [m["role"] for m in client.last["messages"]] == ["user"]


def test_assistant_tool_calls_become_tool_use_blocks():
    prov, client = provider()
    prov.generate(
        [
            Message.user("read a.py"),
            Message.assistant(
                "on it",
                tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"})],
            ),
            Message.tool_result("call_1", "contents"),
        ],
        model="claude-opus-5",
    )
    assistant = client.last["messages"][1]
    assert assistant["content"] == [
        {"type": "text", "text": "on it"},
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "read_file",
            "input": {"path": "a.py"},
        },
    ]


def test_tool_results_become_user_tool_result_blocks():
    prov, client = provider()
    prov.generate(
        [
            Message.user("go"),
            Message.assistant(tool_calls=[ToolCall(id="c1", name="ls", arguments={})]),
            Message.tool_result("c1", "a.py b.py"),
        ],
        model="claude-opus-5",
    )
    last = client.last["messages"][-1]
    assert last["role"] == "user"
    assert last["content"] == [
        {"type": "tool_result", "tool_use_id": "c1", "content": "a.py b.py"}
    ]


def test_consecutive_tool_results_are_merged_into_one_user_turn():
    # Anthropic requires all tool_results for one assistant turn in a single
    # user message; splitting them teaches the model to stop calling in parallel.
    prov, client = provider()
    prov.generate(
        [
            Message.user("go"),
            Message.assistant(
                tool_calls=[
                    ToolCall(id="c1", name="ls", arguments={}),
                    ToolCall(id="c2", name="pwd", arguments={}),
                ]
            ),
            Message.tool_result("c1", "a.py"),
            Message.tool_result("c2", "/tmp"),
        ],
        model="claude-opus-5",
    )
    tool_turns = [m for m in client.last["messages"] if m["role"] == "user"][1:]
    assert len(tool_turns) == 1
    assert [b["tool_use_id"] for b in tool_turns[0]["content"]] == ["c1", "c2"]


def test_failed_tool_result_is_marked_is_error():
    prov, client = provider()
    prov.generate(
        [
            Message.user("go"),
            Message.assistant(tool_calls=[ToolCall(id="c1", name="ls", arguments={})]),
            Message.tool_result("c1", "boom", is_error=True),
        ],
        model="claude-opus-5",
    )
    block = client.last["messages"][-1]["content"][0]
    assert block["is_error"] is True


def test_tools_are_translated_to_anthropic_input_schema():
    prov, client = provider()
    prov.generate(
        [Message.user("hi")],
        tools=[
            ToolDef(
                name="read_file",
                description="Read a file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ],
        model="claude-opus-5",
    )
    assert client.last["tools"] == [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]


def test_no_tools_key_is_sent_when_there_are_no_tools():
    prov, client = provider()
    prov.generate([Message.user("hi")], model="claude-opus-5")
    assert "tools" not in client.last


def test_max_tokens_is_always_sent_because_anthropic_requires_it():
    prov, client = provider()
    prov.generate([Message.user("hi")], model="claude-opus-5")
    assert client.last["max_tokens"] > 0


def test_temperature_is_omitted_unless_the_caller_set_it():
    # Current Claude models reject `temperature` outright, so a default value
    # would break every call.
    prov, client = provider()
    prov.generate([Message.user("hi")], model="claude-opus-5")
    assert "temperature" not in client.last


def test_effort_maps_to_output_config():
    prov, client = provider()
    prov.generate(
        [Message.user("hi")],
        model="claude-opus-5",
        options=GenerateOptions(effort="low"),
    )
    assert client.last["output_config"] == {"effort": "low"}


def test_tool_choice_required_maps_to_any():
    prov, client = provider()
    prov.generate(
        [Message.user("hi")],
        tools=[ToolDef("ls", "list", {"type": "object", "properties": {}})],
        model="claude-opus-5",
        options=GenerateOptions(tool_choice="required"),
    )
    assert client.last["tool_choice"] == {"type": "any"}


def test_named_tool_choice_forces_that_tool():
    prov, client = provider()
    prov.generate(
        [Message.user("hi")],
        tools=[ToolDef("ls", "list", {"type": "object", "properties": {}})],
        model="claude-opus-5",
        options=GenerateOptions(tool_choice="ls"),
    )
    assert client.last["tool_choice"] == {"type": "tool", "name": "ls"}


def test_extra_options_are_merged_into_the_request():
    prov, client = provider()
    prov.generate(
        [Message.user("hi")],
        model="claude-opus-5",
        options=GenerateOptions(extra={"thinking": {"type": "adaptive"}}),
    )
    assert client.last["thinking"] == {"type": "adaptive"}


def test_provider_prefix_is_stripped_before_the_model_reaches_the_sdk():
    prov, client = provider()
    prov.generate([Message.user("hi")], model="anthropic/claude-sonnet-5")
    assert client.last["model"] == "claude-sonnet-5"


# ── response normalization ───────────────────────────────────────────────


def test_text_response_is_normalized():
    prov, _ = provider(
        response=make_response(content=[{"type": "text", "text": "the answer"}])
    )
    resp = prov.generate([Message.user("hi")], model="claude-opus-5")
    assert resp.text == "the answer"
    assert resp.provider == "anthropic"
    assert resp.model == "claude-opus-5"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 12
    assert resp.usage.output_tokens == 5
    assert resp.wants_tools is False


def test_multiple_text_blocks_are_joined():
    prov, _ = provider(
        response=make_response(
            content=[{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]
        )
    )
    assert prov.generate([Message.user("hi")], model="claude-opus-5").text == "one\ntwo"


def test_thinking_blocks_are_not_treated_as_answer_text():
    prov, _ = provider(
        response=make_response(
            content=[
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "answer"},
            ]
        )
    )
    assert prov.generate([Message.user("hi")], model="claude-opus-5").text == "answer"


def test_tool_use_blocks_are_normalized_to_tool_calls():
    prov, _ = provider(
        response=make_response(
            content=[
                {"type": "text", "text": "reading"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "read_file",
                    "input": {"path": "a.py"},
                },
            ],
            stop_reason="tool_use",
        )
    )
    resp = prov.generate([Message.user("hi")], model="claude-opus-5")
    assert resp.stop_reason == "tool_use"
    assert resp.wants_tools is True
    assert resp.tool_calls == (
        ToolCall(id="toolu_1", name="read_file", arguments={"path": "a.py"}),
    )


def test_cache_usage_is_carried_through():
    prov, _ = provider(
        response=make_response(
            usage={
                "input_tokens": 3,
                "output_tokens": 4,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 50,
            }
        )
    )
    usage = prov.generate([Message.user("hi")], model="claude-opus-5").usage
    assert usage.cache_read_tokens == 100
    assert usage.cache_write_tokens == 50


def test_refusal_stop_reason_is_preserved():
    prov, _ = provider(response=make_response(stop_reason="refusal", content=[]))
    assert prov.generate([Message.user("hi")], model="claude-opus-5").stop_reason == "refusal"


def test_max_tokens_stop_reason_is_preserved():
    prov, _ = provider(response=make_response(stop_reason="max_tokens"))
    assert prov.generate([Message.user("hi")], model="claude-opus-5").stop_reason == "max_tokens"


def test_unknown_stop_reason_becomes_other_rather_than_crashing():
    prov, _ = provider(response=make_response(stop_reason="pause_turn"))
    assert prov.generate([Message.user("hi")], model="claude-opus-5").stop_reason == "other"


def test_raw_response_is_attached_for_escape_hatch_access():
    canned = make_response()
    prov, _ = provider(response=canned)
    assert prov.generate([Message.user("hi")], model="claude-opus-5").raw is canned


# ── failure translation ──────────────────────────────────────────────────


def test_sdk_status_errors_become_provider_api_errors_with_status():
    err = anthropic.RateLimitError(
        "slow down",
        response=_fake_httpx_response(429),
        body=None,
    )
    prov, _ = provider(error=err)
    with pytest.raises(ProviderAPIError) as excinfo:
        prov.generate([Message.user("hi")], model="claude-opus-5")
    assert excinfo.value.status_code == 429
    assert excinfo.value.retryable is True
    assert excinfo.value.provider == "anthropic"


def test_bad_request_is_reported_as_not_retryable():
    err = anthropic.BadRequestError(
        "nope", response=_fake_httpx_response(400), body=None
    )
    prov, _ = provider(error=err)
    with pytest.raises(ProviderAPIError) as excinfo:
        prov.generate([Message.user("hi")], model="claude-opus-5")
    assert excinfo.value.retryable is False


def test_connection_errors_are_retryable_provider_api_errors():
    prov, _ = provider(error=anthropic.APIConnectionError(request=_fake_httpx_request()))
    with pytest.raises(ProviderAPIError) as excinfo:
        prov.generate([Message.user("hi")], model="claude-opus-5")
    assert excinfo.value.retryable is True


def test_missing_credentials_raise_not_configured():
    # No injected client and no key anywhere in the environment.
    prov = AnthropicProvider(api_key=None, env={})
    with pytest.raises(ProviderNotConfiguredError, match="ANTHROPIC_API_KEY"):
        prov.generate([Message.user("hi")], model="claude-opus-5")


def test_malformed_tool_input_is_a_response_error():
    # The real SDK model refuses to build a tool_use block with a non-object
    # `input`, so this can only reach us from something duck-typed — a proxy in
    # front of the API, or an SDK shape change. Assert we refuse it rather than
    # handing garbage to a tool.
    class FakeBlock:
        type = "tool_use"
        id = "t1"
        name = "ls"
        input = "not-a-dict"

    class FakeResponse:
        content = [FakeBlock()]
        model = "claude-opus-5"
        stop_reason = "tool_use"
        usage = None

    prov, _ = provider(response=FakeResponse())
    with pytest.raises(ProviderResponseError, match="ls"):
        prov.generate([Message.user("hi")], model="claude-opus-5")


def test_empty_message_list_is_rejected_before_any_api_call():
    prov, client = provider()
    with pytest.raises(ValueError):
        prov.generate([], model="claude-opus-5")
    assert client.calls == []


# ── helpers ──────────────────────────────────────────────────────────────


def _fake_httpx_request():
    import httpx

    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _fake_httpx_response(status_code):
    import httpx

    return httpx.Response(status_code, request=_fake_httpx_request())
