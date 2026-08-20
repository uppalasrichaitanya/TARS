"""Reading a transcript is where a memory writer can quietly go wrong, so the
parser is tested harder than its size suggests.

The failure mode that matters: **whose voice is this?** A human saying "never
put the API key in the repo" is a constraint. The same sentence echoed back by
a grep result is not evidence of anything. The scorer weights roles, so if the
parser lets tool output arrive wearing a `user` label, every downstream score
is wrong in the direction of over-confidence.

Everything else here is forgiveness: provider-shaped message lists, JSON, JSONL,
and plain phase-logged text all have to parse, because the loop above this
module will hand over whatever it happens to have.
"""

from __future__ import annotations

import json

import pytest

from tars.memory.transcript import (
    PHASES,
    Transcript,
    Turn,
    load_transcript,
    parse_text,
    parse_transcript,
)

# ── provider-shaped message lists ─────────────────────────────────────────


def test_a_plain_role_content_list_becomes_turns() -> None:
    script = parse_transcript(
        [
            {"role": "user", "content": "use stdlib http for openrouter"},
            {"role": "assistant", "content": "agreed, no SDK for that path"},
        ]
    )
    assert [t.role for t in script.turns] == ["user", "assistant"]
    assert script.turns[0].text == "use stdlib http for openrouter"


def test_turns_are_numbered_from_one_because_humans_read_the_numbers() -> None:
    script = parse_transcript([{"role": "user", "content": "hello"}])
    assert script.turns[0].index == 1


def test_content_blocks_are_flattened_into_the_turns_text() -> None:
    script = parse_transcript(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "first part"},
                    {"type": "text", "text": "second part"},
                ],
            }
        ]
    )
    assert script.turns[0].text == "first part\nsecond part"


def test_a_tool_call_is_recorded_as_a_tool_name_not_as_prose() -> None:
    # A tool call's arguments are not something anybody said. Letting them into
    # `text` would put JSON in front of the sentence segmenter and, worse, make
    # a quotable "statement" out of a machine's input.
    script = parse_transcript(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "running the suite"},
                    {"type": "tool_use", "name": "bash", "input": {"cmd": "pytest -q"}},
                ],
            }
        ]
    )
    assert script.turns[0].text == "running the suite"
    assert script.turns[0].tools == ("bash",)


def test_a_message_that_is_only_tool_results_is_not_the_user_speaking() -> None:
    # The one that matters. Anthropic-shaped tool results arrive inside a `user`
    # message; a writer that trusts the envelope treats command output as human
    # instruction, which is the most over-confident mistake available here.
    script = parse_transcript(
        [
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": "1 failed, 461 passed"}],
            }
        ]
    )
    assert script.turns[0].role == "tool"
    assert "461 passed" in script.turns[0].text


def test_a_user_message_mixing_prose_and_tool_results_stays_a_user_turn() -> None:
    script = parse_transcript(
        [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "1 failed"},
                    {"type": "text", "text": "fix that before anything else"},
                ],
            }
        ]
    )
    assert script.turns[0].role == "user"


def test_empty_and_whitespace_only_messages_are_dropped() -> None:
    script = parse_transcript(
        [
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "real"},
        ]
    )
    assert len(script.turns) == 1
    assert script.turns[0].index == 1, "numbering counts kept turns, not raw messages"


def test_an_unrecognised_role_is_kept_rather_than_relabelled() -> None:
    # Relabelling would be a guess about voice, and voice is exactly what the
    # scorer trusts. An unknown role simply earns no role bonus later.
    script = parse_transcript([{"role": "developer", "content": "ship it"}])
    assert script.turns[0].role == "developer"


def test_a_system_prompt_keeps_its_role() -> None:
    script = parse_transcript([{"role": "system", "content": "you are TARS"}])
    assert script.turns[0].role == "system"


def test_a_message_list_that_is_not_a_list_of_mappings_is_refused() -> None:
    with pytest.raises(TypeError):
        parse_transcript(["just a string"])


# ── the object form, and files ────────────────────────────────────────────


def test_a_session_labelled_object_form_carries_its_label() -> None:
    script = parse_transcript(
        {"session": "phase/memory-writer", "turns": [{"role": "user", "content": "go"}]}
    )
    assert script.session == "phase/memory-writer"
    assert len(script.turns) == 1


def test_a_bare_message_list_gets_no_invented_session_label() -> None:
    assert parse_transcript([{"role": "user", "content": "go"}]).session == ""


def test_a_turn_can_declare_its_own_phase() -> None:
    script = parse_transcript(
        [{"role": "assistant", "content": "running checks", "phase": "VERIFY"}]
    )
    assert script.turns[0].phase == "VERIFY"


def test_a_json_file_of_messages_loads(tmp_path) -> None:
    path = tmp_path / "session.json"
    path.write_text(json.dumps([{"role": "user", "content": "use stdlib http"}]))
    script = load_transcript(path)
    assert script.turns[0].text == "use stdlib http"
    assert script.session == "session", "the filename is the honest default label"


def test_a_jsonl_file_loads_one_message_per_line(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(
        '{"role": "user", "content": "first"}\n\n{"role": "assistant", "content": "second"}\n'
    )
    assert [t.text for t in load_transcript(path).turns] == ["first", "second"]


def test_a_plain_text_log_loads_as_text_rather_than_failing(tmp_path) -> None:
    path = tmp_path / "phase.log"
    path.write_text("## PHASE 4 — ACT\n\nwrote the parser\n")
    script = load_transcript(path)
    assert script.turns[0].phase == "ACT"


def test_a_missing_file_says_which_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="nope.json"):
        load_transcript(tmp_path / "nope.json")


def test_malformed_json_is_reported_as_malformed_not_parsed_as_prose(tmp_path) -> None:
    # Silently falling back to text on a broken .json would turn a typo into a
    # transcript that parses but means nothing.
    path = tmp_path / "session.json"
    path.write_text("{not json at all")
    with pytest.raises(ValueError, match="session.json"):
        load_transcript(path)


# ── plain text ────────────────────────────────────────────────────────────


def test_phase_headings_split_plain_text_into_phase_tagged_turns() -> None:
    script = parse_text(
        "## PHASE 3 — PLAN\nwill add a parser\n\n### VERIFY\nran the suite\n"
    )
    assert [(t.phase, t.text) for t in script.turns] == [
        ("PLAN", "will add a parser"),
        ("VERIFY", "ran the suite"),
    ]


def test_prose_that_merely_mentions_a_phase_word_is_not_a_heading() -> None:
    # "I will act on that" must not open an ACT section, or phase tagging
    # becomes noise wherever the loop's own vocabulary appears in a sentence.
    script = parse_text("I will act on that later and then report back.")
    assert script.turns[0].phase is None


def test_text_before_the_first_heading_is_kept_untagged() -> None:
    script = parse_text("preamble line\n\n## ACT\ndid the thing")
    assert script.turns[0].phase is None
    assert script.turns[0].text == "preamble line"


def test_role_prefixed_lines_are_attributed_to_that_role() -> None:
    script = parse_text("user: never commit the key\nassistant: understood")
    assert [(t.role, t.text) for t in script.turns] == [
        ("user", "never commit the key"),
        ("assistant", "understood"),
    ]


def test_a_colon_in_ordinary_prose_is_not_a_role_marker() -> None:
    script = parse_text("note: the retry backoff doubles each attempt")
    assert script.turns[0].role == "assistant"
    assert script.turns[0].text == "note: the retry backoff doubles each attempt"


def test_terminal_colour_codes_are_stripped_from_captured_logs() -> None:
    # `claude --print > phase.log` captures ANSI. Left in, the escape bytes end
    # up inside a quoted piece of evidence.
    script = parse_text("\x1b[32mtests passed\x1b[0m")
    assert script.turns[0].text == "tests passed"


def test_an_empty_transcript_is_an_empty_transcript_not_an_error() -> None:
    script = parse_text("   \n\n  ")
    assert script.turns == ()
    assert not script


# ── Transcript helpers ────────────────────────────────────────────────────


def script_of(*specs: tuple[str, str, str | None]) -> Transcript:
    return Transcript(
        session="s",
        turns=tuple(
            Turn(index=i, role=role, text=text, phase=phase)
            for i, (role, text, phase) in enumerate(specs, start=1)
        ),
    )


def test_a_transcript_reports_which_phases_it_saw_in_loop_order() -> None:
    script = script_of(
        ("assistant", "ran it", "VERIFY"),
        ("assistant", "planned it", "PLAN"),
        ("assistant", "untagged", None),
    )
    assert script.phases_seen() == ("PLAN", "VERIFY")


def test_a_transcript_can_hand_back_one_phases_turns() -> None:
    script = script_of(
        ("assistant", "ran it", "VERIFY"), ("assistant", "planned", "PLAN")
    )
    assert [t.text for t in script.by_phase("VERIFY")] == ["ran it"]


def test_phase_lookup_is_case_insensitive_because_logs_are_inconsistent() -> None:
    script = script_of(("assistant", "ran it", "VERIFY"))
    assert len(script.by_phase("verify")) == 1


def test_the_seven_loop_phases_are_the_taxonomy() -> None:
    assert PHASES == (
        "ORIENT",
        "CLASSIFY",
        "PLAN",
        "ACT",
        "VERIFY",
        "REFLECT",
        "REPORT",
    )


def test_a_turn_knows_whether_it_is_machine_output() -> None:
    assert Turn(index=1, role="tool", text="exit 1").is_machine
    assert not Turn(index=1, role="user", text="fix it").is_machine


def test_a_transcript_serialises_for_the_cli() -> None:
    data = script_of(("user", "go", "ACT")).to_dict()
    assert data["session"] == "s"
    assert data["turns"][0]["role"] == "user"
    assert data["phases_seen"] == ["ACT"]
