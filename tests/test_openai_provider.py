"""OpenAI adapter: request translation and response normalization.

Same approach as the Anthropic tests — injected client, replies built through
the real `openai.types.chat.ChatCompletion` model so accepted shapes are shapes
the SDK would really return.
"""

import json

import openai
import pytest
from openai.types.chat import ChatCompletion

from tars.providers.base import GenerateOptions
from tars.providers.errors import (
    ProviderAPIError,
    ProviderNotConfiguredError,
    ProviderResponseError,
)
from tars.providers.openai_provider import OpenAIProvider
from tars.providers.types import Message, ToolCall, ToolDef


def make_response(
    message=None,
    finish_reason="stop",
    model="gpt-5",
    usage=None,
):
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": message or {"role": "assistant", "content": "hi"},
                }
            ],
            "usage": usage
            or {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
        }
    )


class RecordingClient:
    """Stands in for openai.OpenAI — records kwargs, returns a canned reply."""

    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.calls = []
        self.chat = self
        self.completions = self

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
    return OpenAIProvider(client=client), client


# ── request translation ──────────────────────────────────────────────────


def test_user_message_becomes_a_plain_chat_message():
    prov, client = provider()
    prov.generate([Message.user("hello")], model="gpt-5")
    assert client.last["messages"] == [{"role": "user", "content": "hello"}]


def test_system_message_leads_the_message_list():
    prov, client = provider()
    prov.generate([Message.system("be terse"), Message.user("hi")], model="gpt-5")
    assert client.last["messages"][0] == {"role": "system", "content": "be terse"}


def test_assistant_tool_calls_are_serialized_with_json_string_arguments():
    prov, client = provider()
    prov.generate(
        [
            Message.user("read a.py"),
            Message.assistant(
                "on it",
                tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})],
            ),
            Message.tool_result("c1", "contents"),
        ],
        model="gpt-5",
    )
    assistant = client.last["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "on it"
    call = assistant["tool_calls"][0]
    assert call["id"] == "c1"
    assert call["type"] == "function"
    assert call["function"]["name"] == "read_file"
    # OpenAI wants arguments as a JSON string, not an object.
    assert json.loads(call["function"]["arguments"]) == {"path": "a.py"}


def test_tool_results_stay_separate_tool_role_messages():
    # The mirror image of Anthropic: OpenAI wants one tool message per call.
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
        model="gpt-5",
    )
    tool_msgs = [m for m in client.last["messages"] if m["role"] == "tool"]
    assert tool_msgs == [
        {"role": "tool", "tool_call_id": "c1", "content": "a.py"},
        {"role": "tool", "tool_call_id": "c2", "content": "/tmp"},
    ]


def test_assistant_message_without_text_omits_content():
    prov, client = provider()
    prov.generate(
        [
            Message.user("go"),
            Message.assistant(tool_calls=[ToolCall(id="c1", name="ls", arguments={})]),
            Message.tool_result("c1", "a.py"),
        ],
        model="gpt-5",
    )
    assistant = client.last["messages"][1]
    assert assistant.get("content") is None


def test_failed_tool_result_is_labeled_in_content_since_openai_has_no_error_flag():
    prov, client = provider()
    prov.generate(
        [
            Message.user("go"),
            Message.assistant(tool_calls=[ToolCall(id="c1", name="ls", arguments={})]),
            Message.tool_result("c1", "boom", is_error=True),
        ],
        model="gpt-5",
    )
    content = [m for m in client.last["messages"] if m["role"] == "tool"][0]["content"]
    assert "boom" in content
    assert "error" in content.lower()


def test_tools_are_wrapped_in_the_function_envelope():
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
        model="gpt-5",
    )
    assert client.last["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]


def test_no_tools_key_is_sent_when_there_are_no_tools():
    prov, client = provider()
    prov.generate([Message.user("hi")], model="gpt-5")
    assert "tools" not in client.last


def test_max_tokens_uses_the_completion_tokens_parameter():
    # `max_tokens` is rejected by current reasoning models; the replacement is
    # max_completion_tokens.
    prov, client = provider()
    prov.generate([Message.user("hi")], model="gpt-5", options=GenerateOptions(max_tokens=99))
    assert client.last["max_completion_tokens"] == 99
    assert "max_tokens" not in client.last


def test_effort_maps_to_reasoning_effort():
    prov, client = provider()
    prov.generate([Message.user("hi")], model="gpt-5", options=GenerateOptions(effort="low"))
    assert client.last["reasoning_effort"] == "low"


def test_temperature_is_omitted_unless_set():
    prov, client = provider()
    prov.generate([Message.user("hi")], model="gpt-5")
    assert "temperature" not in client.last


def test_tool_choice_required_passes_through():
    prov, client = provider()
    prov.generate(
        [Message.user("hi")],
        tools=[ToolDef("ls", "list", {"type": "object", "properties": {}})],
        model="gpt-5",
        options=GenerateOptions(tool_choice="required"),
    )
    assert client.last["tool_choice"] == "required"


def test_named_tool_choice_uses_the_function_form():
    prov, client = provider()
    prov.generate(
        [Message.user("hi")],
        tools=[ToolDef("ls", "list", {"type": "object", "properties": {}})],
        model="gpt-5",
        options=GenerateOptions(tool_choice="ls"),
    )
    assert client.last["tool_choice"] == {"type": "function", "function": {"name": "ls"}}


def test_provider_prefix_is_stripped_before_the_model_reaches_the_sdk():
    prov, client = provider()
    prov.generate([Message.user("hi")], model="openai/gpt-5")
    assert client.last["model"] == "gpt-5"


# ── response normalization ───────────────────────────────────────────────


def test_text_response_is_normalized():
    prov, _ = provider(response=make_response(message={"role": "assistant", "content": "the answer"}))
    resp = prov.generate([Message.user("hi")], model="gpt-5")
    assert resp.text == "the answer"
    assert resp.provider == "openai"
    assert resp.model == "gpt-5"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 12
    assert resp.usage.output_tokens == 5


def test_tool_calls_are_normalized_with_decoded_arguments():
    prov, _ = provider(
        response=make_response(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "a.py"}',
                        },
                    }
                ],
            },
            finish_reason="tool_calls",
        )
    )
    resp = prov.generate([Message.user("hi")], model="gpt-5")
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls == (
        ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"}),
    )
    assert resp.text == ""


def test_empty_argument_string_becomes_an_empty_dict():
    prov, _ = provider(
        response=make_response(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "pwd", "arguments": ""}}
                ],
            },
            finish_reason="tool_calls",
        )
    )
    assert prov.generate([Message.user("hi")], model="gpt-5").tool_calls[0].arguments == {}


def test_length_finish_reason_maps_to_max_tokens():
    prov, _ = provider(response=make_response(finish_reason="length"))
    assert prov.generate([Message.user("hi")], model="gpt-5").stop_reason == "max_tokens"


def test_content_filter_finish_reason_maps_to_refusal():
    prov, _ = provider(response=make_response(finish_reason="content_filter"))
    assert prov.generate([Message.user("hi")], model="gpt-5").stop_reason == "refusal"


def test_cached_prompt_tokens_are_reported():
    prov, _ = provider(
        response=make_response(
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 3,
                "total_tokens": 23,
                "prompt_tokens_details": {"cached_tokens": 16},
            }
        )
    )
    assert prov.generate([Message.user("hi")], model="gpt-5").usage.cache_read_tokens == 16


def test_a_response_with_no_choices_is_a_response_error():
    empty = make_response()
    empty.choices = []
    prov, _ = provider(response=empty)
    with pytest.raises(ProviderResponseError, match="no choices"):
        prov.generate([Message.user("hi")], model="gpt-5")


def test_raw_response_is_attached():
    canned = make_response()
    prov, _ = provider(response=canned)
    assert prov.generate([Message.user("hi")], model="gpt-5").raw is canned


# ── failure translation ──────────────────────────────────────────────────


def test_sdk_status_errors_become_provider_api_errors():
    err = openai.RateLimitError(
        "slow down", response=_fake_httpx_response(429), body=None
    )
    prov, _ = provider(error=err)
    with pytest.raises(ProviderAPIError) as excinfo:
        prov.generate([Message.user("hi")], model="gpt-5")
    assert excinfo.value.status_code == 429
    assert excinfo.value.retryable is True
    assert excinfo.value.provider == "openai"


def test_connection_errors_are_retryable():
    prov, _ = provider(error=openai.APIConnectionError(request=_fake_httpx_request()))
    with pytest.raises(ProviderAPIError) as excinfo:
        prov.generate([Message.user("hi")], model="gpt-5")
    assert excinfo.value.retryable is True


def test_missing_credentials_raise_not_configured():
    prov = OpenAIProvider(api_key=None, env={})
    with pytest.raises(ProviderNotConfiguredError, match="OPENAI_API_KEY"):
        prov.generate([Message.user("hi")], model="gpt-5")


def test_empty_message_list_is_rejected_before_any_api_call():
    prov, client = provider()
    with pytest.raises(ValueError):
        prov.generate([], model="gpt-5")
    assert client.calls == []


# ── helpers ──────────────────────────────────────────────────────────────


def _fake_httpx_request():
    import httpx

    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _fake_httpx_response(status_code):
    import httpx

    return httpx.Response(status_code, request=_fake_httpx_request())
