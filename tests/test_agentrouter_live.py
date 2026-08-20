"""AgentRouter, live. Opt-in: `TARS_LIVE=1 pytest tests/test_agentrouter_live.py`.

Skipped by default, and that default is deliberate. The rest of the suite runs
in under five seconds with no network and no spend, which is what makes it worth
running on every change; a live call belongs behind a flag you set on purpose.

What this covers that the mocked tests cannot: that AgentRouter accepts our
client at all. It authenticates only requests from clients it recognises — plain
`curl` and the OpenAI SDK are both refused with `unauthorized_client_error`
carrying a valid key — so "the adapter builds the right request" and "the
adapter can talk to AgentRouter" are genuinely separate claims.
"""

from __future__ import annotations

import os

import pytest

from tars.providers import Message, ToolDef, generate
from tars.providers.types import ToolCall

LIVE = os.environ.get("TARS_LIVE") == "1"

pytestmark = pytest.mark.skipif(
    not LIVE,
    reason="live AgentRouter call; set TARS_LIVE=1 to run (spends tokens)",
)

MODEL = "agentrouter:claude-opus-5"

WEATHER = ToolDef(
    name="get_weather",
    description="Get the current temperature for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


def test_a_plain_completion_comes_back_through_agentrouter():
    response = generate(
        "Reply with exactly: OK",
        model=MODEL,
        max_tokens=64,
    )
    assert "OK" in response.text
    assert response.provider == "agentrouter"
    assert response.model == "claude-opus-5"
    assert response.usage.total_tokens > 0


def test_a_tool_call_round_trip_survives_the_proxy():
    """The half that most often breaks through a proxy.

    A relay can pass plain text through and still mangle `tool_use` blocks or
    drop the ids they have to be answered by, so the assertion is that the
    model's *second* turn uses a number only the tool could have supplied.
    """
    first = generate(
        [Message.user("What is the temperature in Paris? Use the tool.")],
        tools=[WEATHER],
        model=MODEL,
        max_tokens=1024,
    )
    assert first.wants_tools, f"expected a tool call, got {first.stop_reason}"
    call = first.tool_calls[0]
    assert call.name == "get_weather"
    assert call.arguments.get("city")
    assert call.id, "a tool call with no id cannot be answered"

    second = generate(
        [
            Message.user("What is the temperature in Paris? Use the tool."),
            Message.assistant(first.text or None, tool_calls=[ToolCall(**vars(call))]),
            Message.tool_result(call.id, "13C, light rain"),
        ],
        tools=[WEATHER],
        model=MODEL,
        max_tokens=256,
    )
    assert "13" in second.text, second.text
