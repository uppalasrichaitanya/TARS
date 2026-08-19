"""Provider-neutral message/response types — RED phase tests.

These pin down the shape the whole agent loop will speak, so provider quirks
stop at the adapter boundary.
"""

import pytest

from tars.providers.types import (
    LLMResponse,
    Message,
    ToolCall,
    ToolDef,
    Usage,
)


def test_user_message_carries_text():
    msg = Message.user("hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.tool_calls == ()


def test_assistant_message_can_carry_tool_calls():
    call = ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"})
    msg = Message.assistant("I'll read it", tool_calls=[call])
    assert msg.tool_calls == (call,)
    assert msg.content == "I'll read it"


def test_tool_result_message_requires_a_call_id():
    msg = Message.tool_result("call_1", "file contents")
    assert msg.role == "tool"
    assert msg.tool_call_id == "call_1"
    assert msg.content == "file contents"


def test_tool_role_without_call_id_is_rejected():
    with pytest.raises(ValueError, match="tool_call_id"):
        Message(role="tool", content="oops")


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="role"):
        Message(role="wizard", content="hi")


def test_tool_result_can_be_flagged_as_an_error():
    msg = Message.tool_result("call_1", "boom", is_error=True)
    assert msg.is_error is True


def test_tooldef_exposes_json_schema_parameters():
    tool = ToolDef(
        name="read_file",
        description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    assert tool.parameters["type"] == "object"


def test_usage_totals_input_and_output():
    usage = Usage(input_tokens=10, output_tokens=4)
    assert usage.total_tokens == 14


def test_response_text_and_tool_call_accessors():
    call = ToolCall(id="c1", name="ls", arguments={})
    resp = LLMResponse(
        text="done",
        tool_calls=(call,),
        model="claude-opus-5",
        provider="anthropic",
        stop_reason="tool_use",
        usage=Usage(input_tokens=1, output_tokens=2),
    )
    assert resp.wants_tools is True
    assert resp.tool_calls[0].name == "ls"


def test_response_without_tool_calls_does_not_want_tools():
    resp = LLMResponse(
        text="done",
        tool_calls=(),
        model="gpt-5",
        provider="openai",
        stop_reason="end_turn",
        usage=Usage(),
    )
    assert resp.wants_tools is False
