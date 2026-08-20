"""What in a transcript looks like a fact worth keeping, and what only looks busy.

Core directive 4 draws the line: memory-worthy is "a decision, a constraint, a
rejected approach, a non-obvious gotcha"; not memory-worthy is "routine
execution, anything visible by just reading the code". Everything in this file
is that sentence turned into detectors, so the tests are mostly real transcript
prose with the answer next to it.

The asymmetry is deliberate. A missed entry costs a future session one
re-derivation. A wrong entry gets believed without being re-derived at all,
which is the failure the guardrail *never fabricate a memory entry to fill a
gap* is about. So routine narration has to be recognised even when it wears a
decision verb ("I decided to run the formatter"), and code narration has to be
recognised even though it is perfectly true.
"""

from __future__ import annotations

from tars.memory.signals import (
    kind_of,
    read_signals,
    reason_clause,
    routine_markers,
    segment,
    specific_tokens,
    supersede_markers,
    weak_markers,
)
from tars.memory.transcript import Turn


def kind(text: str) -> str | None:
    return kind_of(read_signals(text))


def names(text: str) -> set[str]:
    return {s.name for s in read_signals(text)}


# ── segmentation ──────────────────────────────────────────────────────────


def statements(text: str, **over: object) -> list[str]:
    fields: dict[str, object] = {"index": 1, "role": "assistant", "text": text}
    fields.update(over)
    return [s.text for s in segment(Turn(**fields))]  # type: ignore[arg-type]


def test_a_paragraph_is_split_into_sentences() -> None:
    assert statements("We chose sqlite. It is enough for this.") == [
        "We chose sqlite.",
        "It is enough for this.",
    ]


def test_an_abbreviation_does_not_end_a_sentence() -> None:
    # "e.g." and friends are where naive sentence splitting produces fragments,
    # and a fragment quoted as evidence is a misquote.
    assert statements("Use a real DB, e.g. postgres, for anything shared.") == [
        "Use a real DB, e.g. postgres, for anything shared."
    ]


def test_a_version_number_does_not_end_a_sentence() -> None:
    assert statements("We pin python 3.12 because 3.13 breaks the wheel.") == [
        "We pin python 3.12 because 3.13 breaks the wheel."
    ]


def test_bullets_are_separate_statements() -> None:
    assert statements("Decisions:\n- use sqlite\n- skip the ORM") == [
        "Decisions:",
        "use sqlite",
        "skip the ORM",
    ]


def test_a_fenced_code_block_is_not_prose() -> None:
    # Code in a transcript is not a statement anybody made, and the spec puts
    # "anything visible by just reading the code" outside memory on purpose.
    assert statements("We chose this shape:\n```python\nx = 1\n```\nIt is fine.") == [
        "We chose this shape:",
        "It is fine.",
    ]


def test_a_shell_transcript_line_is_not_prose() -> None:
    assert statements("$ pytest -q\n1 failed, 461 passed") == []


def test_a_traceback_is_not_prose() -> None:
    body = 'Traceback (most recent call last):\n  File "x.py", line 1\nValueError: nope'
    assert statements(body) == []


def test_an_empty_turn_yields_nothing() -> None:
    assert statements("   ") == []


def test_statements_carry_the_turn_they_came_from() -> None:
    found = segment(
        Turn(index=7, role="user", text="Never commit the key.", phase="ACT")
    )
    assert found[0].turn == 7
    assert found[0].role == "user"
    assert found[0].phase == "ACT"


# ── the four memory-worthy kinds ──────────────────────────────────────────


def test_an_explicit_decision_reads_as_a_decision() -> None:
    assert kind("We decided to use stdlib http for OpenRouter.") == "decision"


def test_a_choice_between_options_reads_as_a_decision() -> None:
    assert (
        kind("Going with sqlite over postgres for the structured side.") == "decision"
    )


def test_a_standing_rule_reads_as_a_constraint() -> None:
    assert kind("Never put the auth token in the repo.") == "constraint"


def test_a_hard_requirement_reads_as_a_constraint() -> None:
    assert kind("The adapter must not import the SDK at module import time.") == (
        "constraint"
    )


def test_an_approach_that_was_tried_and_dropped_reads_as_rejected() -> None:
    assert (
        kind("We ruled out Chroma because the vector DB already exists.") == "rejected"
    )


def test_deciding_against_something_reads_as_rejected() -> None:
    assert kind("Decided against bolting on Qdrant.") == "rejected"


def test_a_non_obvious_cause_reads_as_a_gotcha() -> None:
    assert kind("Turns out the formatter rewrites README fences too.") == "gotcha"


def test_a_silent_failure_reads_as_a_gotcha() -> None:
    # "Silently" is the highest-value word in a transcript: it marks exactly the
    # thing a future session cannot discover by reading the code.
    assert kind("The exclude is silently ignored when the path is named.") == "gotcha"


def test_a_rejection_beats_a_bare_decision_when_both_fire() -> None:
    # "instead of" makes a decision *and* records what was turned down; the
    # rejected approach is the part a future session can't recover from code.
    assert kind("We used stdlib http instead of the openai SDK.") == "rejected"


# ── what is not memory-worthy ─────────────────────────────────────────────


def test_plain_narration_is_not_any_kind() -> None:
    assert kind("Now let me look at the readme module.") is None


def test_a_passing_test_run_is_not_any_kind() -> None:
    assert kind("Ran the suite: 461 passed.") is None


def test_a_question_is_not_a_fact() -> None:
    assert kind("Should we use sqlite or postgres here?") is None


def test_routine_work_is_marked_even_when_it_wears_a_decision_verb() -> None:
    # The one that keeps the corpus honest. "I decided to run the formatter" is
    # a decision in grammar only.
    assert routine_markers("I decided to run the formatter over the file")


def test_a_real_decision_is_not_marked_as_routine() -> None:
    assert not routine_markers("We decided to use stdlib http for OpenRouter.")


def test_describing_code_that_can_simply_be_read_is_marked_as_routine() -> None:
    # "Anything visible by just reading the code" — true, useful in a diff,
    # worthless in memory.
    assert routine_markers("Added a helper called _clean to transcript.py")
    assert routine_markers("Renamed the fixture to script_of")


def test_hedged_language_is_reported_as_weak() -> None:
    assert weak_markers("Maybe we should probably use sqlite")


def test_a_flat_assertion_is_not_weak() -> None:
    assert not weak_markers("We use sqlite for the structured side.")


# ── specificity ───────────────────────────────────────────────────────────


def test_a_path_counts_as_something_concrete() -> None:
    assert specific_tokens("the retry lives in tars/providers/base.py")


def test_an_env_var_counts_as_something_concrete() -> None:
    assert "ANTHROPIC_AUTH_TOKEN" in specific_tokens("use ANTHROPIC_AUTH_TOKEN instead")


def test_a_product_name_counts_as_something_concrete() -> None:
    assert "OpenRouter" in specific_tokens("stdlib http is enough for OpenRouter")


def test_a_backticked_identifier_counts_as_something_concrete() -> None:
    assert specific_tokens("`force-exclude` has to be on")


def test_a_version_counts_as_something_concrete() -> None:
    assert specific_tokens("pin python 3.12")


def test_an_entirely_vague_sentence_has_nothing_concrete() -> None:
    # This is the shape of a memory entry that rots without anyone noticing:
    # true, agreeable, and impossible to act on.
    assert specific_tokens("we should be careful with the provider layer") == ()


def test_an_ordinary_capitalised_first_word_is_not_an_identifier() -> None:
    assert specific_tokens("Never commit the token") == ()


# ── supersede cues ────────────────────────────────────────────────────────


def test_replacing_an_earlier_decision_is_flagged() -> None:
    assert supersede_markers("We no longer use the openai SDK for this path")


def test_a_changed_mind_is_flagged() -> None:
    assert supersede_markers("This replaces the earlier decision to use Chroma")


def test_a_first_time_decision_is_not_a_supersede() -> None:
    assert not supersede_markers("We use sqlite for the structured side")


# ── sourced reasons ───────────────────────────────────────────────────────


def test_a_because_clause_is_a_reason_the_transcript_actually_gave() -> None:
    # `why it matters` is required by the schema. Where the transcript supplies
    # a reason, it beats anything the writer could compose.
    assert reason_clause(
        "We ruled out Chroma because the vector DB already exists"
    ) == ("the vector DB already exists")


def test_a_since_clause_counts_as_a_reason() -> None:
    assert reason_clause("Use stdlib http since it is just an HTTP endpoint") == (
        "it is just an HTTP endpoint"
    )


def test_a_so_that_clause_counts_as_a_reason() -> None:
    assert reason_clause(
        "Keep the SDK import lazy so that import never needs a key"
    ) == ("import never needs a key")


def test_a_statement_with_no_stated_reason_says_so() -> None:
    assert reason_clause("We use sqlite for the structured side") is None


def test_because_inside_the_first_words_is_not_a_reason_clause() -> None:
    # "Because the gate ran, we shipped" has its reason first; treating the tail
    # as the reason inverts the sentence.
    assert reason_clause("Because of that we use sqlite") is None


# ── signals are auditable ─────────────────────────────────────────────────


def test_every_signal_says_what_fired_and_how_much_it_counts() -> None:
    signals = read_signals("We decided to use stdlib http instead of the openai SDK")
    assert signals
    for signal in signals:
        assert signal.name
        assert signal.kind in {"decision", "constraint", "rejected", "gotcha"}
        assert 0.0 < signal.weight <= 1.0


def test_the_matched_text_is_kept_so_a_score_can_be_traced() -> None:
    signals = read_signals("Turns out the formatter rewrites fences")
    assert any(s.matched.lower().startswith("turns out") for s in signals)


def test_two_cues_of_the_same_kind_are_both_reported() -> None:
    found = names("We decided to use stdlib http, and agreed to skip the SDK")
    assert len(found) >= 2
