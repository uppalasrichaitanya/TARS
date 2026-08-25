"""Shared adapter helpers: system hoisting, argument decoding, error mapping."""

import pytest

from tars.providers.base import (
    GenerateOptions,
    decode_arguments,
    require_messages,
    split_system,
    translate_api_exception,
)
from tars.providers.errors import ProviderResponseError
from tars.providers.types import Message


def test_split_system_hoists_and_joins_system_turns():
    system, rest = split_system(
        [Message.system("rule one"), Message.user("hi"), Message.system("rule two")]
    )
    assert system == "rule one\n\nrule two"
    assert [m.role for m in rest] == ["user"]


def test_option_level_system_comes_first():
    system, _ = split_system([Message.system("from history")], "from options")
    assert system == "from options\n\nfrom history"


def test_split_system_returns_none_when_there_is_no_system_text():
    system, rest = split_system([Message.user("hi")])
    assert system is None
    assert len(rest) == 1


def test_require_messages_rejects_an_empty_history():
    with pytest.raises(ValueError, match="at least one message"):
        require_messages([])


def test_decode_arguments_accepts_a_json_string():
    assert decode_arguments(
        '{"path": "a.py"}', provider="openai", tool_name="read"
    ) == {"path": "a.py"}


def test_decode_arguments_accepts_a_dict_unchanged():
    assert decode_arguments({"a": 1}, provider="anthropic", tool_name="t") == {"a": 1}


def test_decode_arguments_treats_empty_as_no_arguments():
    assert decode_arguments("", provider="openai", tool_name="t") == {}
    assert decode_arguments(None, provider="openai", tool_name="t") == {}


def test_decode_arguments_rejects_unparseable_json():
    with pytest.raises(ProviderResponseError, match="unparseable"):
        decode_arguments("{oops", provider="openai", tool_name="read")


def test_decode_arguments_rejects_a_json_scalar():
    with pytest.raises(ProviderResponseError, match="non-object"):
        decode_arguments("42", provider="openai", tool_name="read")


def test_translate_api_exception_reads_status_from_the_exception():
    class Boom(Exception):
        status_code = 503

    err = translate_api_exception(Boom("down"), provider="openai", model="gpt-5")
    assert err.status_code == 503
    assert err.retryable is True
    assert err.provider == "openai"
    assert err.model == "gpt-5"


def test_translate_api_exception_reads_status_from_a_nested_response():
    class Response:
        status_code = 404

    class Boom(Exception):
        response = Response()

    err = translate_api_exception(Boom("gone"), provider="openrouter")
    assert err.status_code == 404
    assert err.retryable is False


def test_statusless_exceptions_are_treated_as_retryable_connection_failures():
    err = translate_api_exception(OSError("connection reset"), provider="anthropic")
    assert err.status_code is None
    assert err.retryable is True
    assert err.original is not None


def test_default_options_send_a_nonzero_max_tokens_and_no_temperature():
    opts = GenerateOptions()
    assert opts.max_tokens > 0
    assert opts.temperature is None
