"""One REFLECT pass, end to end: transcript in, scored candidates out.

This is where the spec's two hardest sentences meet. *"Low-confidence or
ambiguous entries go to a human review queue; high-confidence, clearly
decision-shaped entries can commit directly"* means a number has to decide who
sees an entry first — so most of this file is about which transcript facts move
that number, and in which direction.

*"Never fabricate a memory entry to fill a gap. An empty memory is safer than a
wrong one"* means the scorer is deliberately lopsided. Every test here that
checks something is *not* committed is testing that guardrail, and there are more
of those than the other kind on purpose.

The one to read first is
`test_the_same_words_from_a_tool_result_do_not_commit`. A grep hit that echoes
"never put the API key in the repo" contains a perfect constraint and is evidence
of nothing at all, because a machine repeating a string is not somebody deciding
something. It is the most over-confident mistake this module can make, and the
only defence is that voice is scored.
"""

from __future__ import annotations

from tars.memory.domains import Domain, DomainMap
from tars.memory.propose import propose
from tars.memory.transcript import Transcript, Turn
from tars.memory.types import (
    COMMIT_THRESHOLD,
    Candidate,
    MemoryEntry,
    ReflectionReport,
)

Spec = tuple[str, str] | tuple[str, str, str]


def script(*specs: Spec, session: str = "s") -> Transcript:
    turns = []
    for index, spec in enumerate(specs, start=1):
        role, text = spec[0], spec[1]
        phase = spec[2] if len(spec) > 2 else None
        turns.append(Turn(index=index, role=role, text=text, phase=phase))
    return Transcript(session=session, turns=tuple(turns))


def report(*specs: Spec, **kwargs: object) -> ReflectionReport:
    return propose(script(*specs), **kwargs)  # type: ignore[arg-type]


def only(*specs: Spec, **kwargs: object) -> Candidate:
    found = report(*specs, **kwargs).candidates
    assert len(found) == 1, [c.text for c in found]
    return found[0]


DECISION = "We decided to use stdlib http for OpenRouter."


# ── the pass as a whole ───────────────────────────────────────────────────


def test_a_clear_decision_becomes_one_tagged_candidate() -> None:
    cand = only(("user", DECISION, "REFLECT"))
    assert cand.kind == "decision"
    assert cand.domains == ("providers",)
    assert cand.text == DECISION


def test_the_report_carries_the_session_it_reflected_on() -> None:
    body = script(("user", DECISION), session="phase/memory")
    assert propose(body).session == "phase/memory"


def test_every_candidate_quotes_the_turn_it_came_from() -> None:
    # An entry a future session acts on without re-deriving has to be traceable
    # to somebody actually saying it.
    cand = only(("user", DECISION, "REFLECT"))
    assert cand.evidence
    assert cand.evidence[0].turn == 1
    assert cand.evidence[0].role == "user"


def test_the_quote_is_verbatim_rather_than_tidied() -> None:
    # The moment the writer paraphrases, the turn number stops being checkable.
    turn_text = f"Right, so. {DECISION} Moving on."
    cand = only(("user", turn_text))
    assert cand.evidence[0].quote in turn_text


def test_a_transcript_of_pure_narration_proposes_nothing() -> None:
    # "If nothing is memory-worthy, say so" — the writer has to be able to
    # produce zero without that looking like a failure.
    result = report(
        ("assistant", "Now let me look at the readme module.", "ACT"),
        ("assistant", "Ran the suite: 461 passed.", "VERIFY"),
        ("assistant", "Committed that and moved on.", "ACT"),
    )
    assert result.nothing_worth_remembering
    assert result.proposed == ()


def test_an_empty_transcript_says_so_rather_than_inventing_something() -> None:
    result = propose(script())
    assert result.nothing_worth_remembering
    assert any("no turns" in note for note in result.notes)


def test_a_provider_shaped_message_list_can_be_reflected_on_directly() -> None:
    # The CLI and the loop both hand over whatever they happen to have.
    result = propose([{"role": "user", "content": DECISION}])
    assert result.proposed


# ── voice, and the over-confidence guard ─────────────────────────────────


def test_a_specific_domain_tagged_decision_from_a_person_commits() -> None:
    cand = only(("user", DECISION, "REFLECT"))
    assert cand.confidence >= COMMIT_THRESHOLD
    assert cand.disposition == "commit"


def test_the_same_words_from_a_tool_result_do_not_commit() -> None:
    # The one that matters most in this file. Identical text, different voice:
    # a machine echoing a decision is not a decision being made.
    cand = only(("tool", DECISION))
    assert cand.disposition != "commit"


def test_a_hedged_decision_goes_to_a_human_instead_of_the_store() -> None:
    cand = only(("user", "I think we decided to use sqlite for the structured side."))
    assert cand.disposition == "review"


def test_a_gotcha_goes_to_review_however_strong_the_evidence() -> None:
    # A gotcha is a causal claim about what happened — the easiest kind of claim
    # to hold confidently and wrongly. Confidence alone never promotes one.
    cand = only(
        (
            "user",
            "Turns out the formatter silently rewrites tars/memory/signals.py fences.",
            "REFLECT",
        )
    )
    assert cand.kind == "gotcha"
    assert cand.confidence >= COMMIT_THRESHOLD
    assert cand.disposition == "review"


def test_a_candidate_no_domain_claims_cannot_commit_however_confident() -> None:
    # An untagged entry is one a future ORIENT can never retrieve. Writing it
    # into the store unseen is the worst of both outcomes.
    cand = only(
        ("user", "We decided to keep the wording exactly as it was.", "REFLECT")
    )
    assert cand.domains == ()
    assert cand.confidence >= COMMIT_THRESHOLD
    assert cand.disposition == "review"


def test_routine_work_in_decision_grammar_does_not_survive_the_floor() -> None:
    result = report(
        ("assistant", "I decided to run the formatter over tars/memory/signals.py.")
    )
    assert result.proposed == ()
    assert result.dropped, "dropped, not silently absent — the score is auditable"


def test_confidence_never_leaves_the_zero_to_one_range() -> None:
    result = report(
        (
            "user",
            "Turns out the formatter silently rewrites tars/memory/signals.py fences.",
            "REFLECT",
        ),
        ("tool", "never never never must must always"),
    )
    for cand in result.candidates:
        assert 0.0 <= cand.confidence <= 1.0


def test_the_signals_behind_a_score_are_recorded() -> None:
    # A score nobody can take apart is a score nobody can correct.
    cand = only(("user", DECISION, "REFLECT"))
    assert cand.signals
    assert any("role:user" == s for s in cand.signals)


# ── why it matters ───────────────────────────────────────────────────────


def test_a_stated_reason_is_used_verbatim_as_the_why() -> None:
    cand = only(("user", "We ruled out Chroma because the vector DB already exists."))
    assert cand.why == "the vector DB already exists"


def test_an_unstated_reason_is_templated_and_flagged_as_such() -> None:
    # The schema requires a `why`. Where the transcript didn't give one, the
    # writer supplies a generic one and says that it did — the alternative is a
    # reason that reads sourced and isn't.
    cand = only(("user", "Never put the auth token in the repo."))
    assert cand.why.strip()
    assert "why:template" in cand.signals


# ── saying the same thing twice ──────────────────────────────────────────


def test_the_same_decision_said_twice_is_one_candidate() -> None:
    result = report(
        ("user", DECISION), ("assistant", "Understood."), ("user", DECISION)
    )
    assert len(result.candidates) == 1
    assert [e.turn for e in result.candidates[0].evidence] == [1, 3]


def test_being_said_twice_counts_for_slightly_more_than_once() -> None:
    once = only(("user", "We decided to keep the wording exactly as it was."))
    twice = only(
        ("user", "We decided to keep the wording exactly as it was."),
        ("user", "We decided to keep the wording exactly as it was."),
    )
    assert twice.confidence > once.confidence


# ── superseding ──────────────────────────────────────────────────────────


def existing_entry() -> MemoryEntry:
    return MemoryEntry(
        id="decision-abcd1234",
        text="The provider layer uses the openai SDK for OpenRouter.",
        kind="decision",
        domains=("providers",),
        why="it was the fastest path at the time",
        confidence=0.9,
    )


def test_an_explicit_replacement_marks_the_entry_it_replaces() -> None:
    prior = existing_entry()
    cand = only(
        ("user", "We no longer use the openai SDK for OpenRouter."),
        existing=(prior,),
    )
    assert cand.supersedes == (prior.id,)


def test_a_contradiction_with_no_explicit_cue_supersedes_nothing() -> None:
    # "Mark superseded entries so contradicting entries aren't both live" is not
    # a licence to guess which of two entries won. Inferring it would retire a
    # fact nobody retired.
    prior = existing_entry()
    cand = only(("user", DECISION, "REFLECT"), existing=(prior,))
    assert cand.supersedes == ()


def test_a_replacement_cue_about_an_unrelated_domain_supersedes_nothing() -> None:
    prior = existing_entry()
    cand = only(
        ("user", "We no longer use the Dockerfile healthcheck."), existing=(prior,)
    )
    assert cand.supersedes == ()


def test_superseding_needs_a_shared_identifier_not_just_a_shared_domain() -> None:
    # Same partition, replacement cue present, but the two statements are about
    # different things: `providers` is a big enough domain to hold plenty of
    # unrelated facts, so a domain match alone would retire the wrong one.
    prior = existing_entry()
    cand = only(
        ("user", "The provider adapter no longer uses ANTHROPIC_AUTH_TOKEN."),
        existing=(prior,),
    )
    assert cand.domains == ("providers",), "shares the prior entry's domain"
    assert cand.supersedes == ()


def test_a_vague_replacement_has_nothing_to_match_on_and_supersedes_nothing() -> None:
    prior = existing_entry()
    cand = only(
        ("user", "The provider adapter no longer uses a retry budget."),
        existing=(prior,),
    )
    assert cand.supersedes == ()


# ── determinism and honest caps ──────────────────────────────────────────


def test_candidates_come_back_strongest_first() -> None:
    result = report(
        ("tool", "The build must not import the SDK."),
        ("user", DECISION, "REFLECT"),
    )
    scores = [c.confidence for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_two_passes_over_one_transcript_agree_exactly() -> None:
    # A writer whose output wobbles between runs makes every diff of the store
    # unreadable, and the store dedupes on content.
    body = script(("user", DECISION, "REFLECT"), ("user", "Never commit the key."))
    assert propose(body).to_dict() == propose(body).to_dict()


def test_a_cap_says_what_it_left_out() -> None:
    # A truncated list that doesn't mention truncating reads as "that's
    # everything", which is a lie by omission the loop can't detect.
    result = report(
        ("user", "We decided to use sqlite for the structured side.", "REFLECT"),
        ("user", "We decided to use ruff for tars/memory/signals.py.", "REFLECT"),
        ("user", "We decided to use pytest for the test suite.", "REFLECT"),
        ("user", "We decided to use Dockerfile pinning for infra.", "REFLECT"),
        max_candidates=2,
    )
    assert len(result.candidates) == 2
    assert any("2" in note and "cap" in note.lower() for note in result.notes)


# ── the domain map is the project's ──────────────────────────────────────


def test_a_projects_own_domain_map_is_what_tags_the_candidates() -> None:
    tiny = DomainMap(
        domains=(Domain(name="billing", description="invoices", keywords=("invoice",)),)
    )
    cand = only(("user", "We decided to use stripe for the invoice run."), domains=tiny)
    assert cand.domains == ("billing",)
