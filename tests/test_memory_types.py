"""The schema is where PHASE 6's prose rules become executable, so it gets the
strictest tests in this phase.

Four rules from the spec are encoded as properties of the types rather than as
behaviour of the proposer, because a rule enforced in one call site is a rule
that the next call site forgets:

* An entry with no domain tag can never auto-commit. ORIENT retrieves *by
  domain*; an untagged entry is unreachable, so committing it is a silent drop.
* Only decision-shaped kinds auto-commit. "High-confidence, clearly
  decision-shaped entries can commit directly" — a gotcha is a causal claim,
  which is the easiest kind of claim to be confidently wrong about.
* Low confidence flags for review; it never vanishes and never commits.
* A superseded entry stops being live. "Don't leave contradicting entries both
  live" is a property of the store, and this is the flag it reads.
"""

from __future__ import annotations

import pytest

from tars.memory.types import (
    COMMIT_THRESHOLD,
    REVIEW_FLOOR,
    Candidate,
    Evidence,
    MemoryEntry,
    ReflectionReport,
)


def candidate(**over: object) -> Candidate:
    """A high-confidence, committable candidate; override one field per test."""
    fields: dict[str, object] = {
        "text": "Use stdlib http for OpenRouter instead of adding an SDK.",
        "kind": "decision",
        "domains": ("providers",),
        "why": "A future session would otherwise add a dependency we rejected.",
        "confidence": 0.9,
    }
    fields.update(over)
    return Candidate(**fields)  # type: ignore[arg-type]


# ── Evidence ──────────────────────────────────────────────────────────────


def test_evidence_carries_the_turn_it_came_from() -> None:
    ev = Evidence(turn=3, role="user", quote="we're going with stdlib http")
    assert ev.turn == 3
    assert "stdlib" in ev.quote


def test_evidence_locates_itself_the_way_a_reader_could_check_it() -> None:
    # The readme phase carries `path:line` for the same reason: a claim with no
    # checkable origin is indistinguishable from a fabricated one.
    ev = Evidence(turn=3, role="user", quote="we're going with stdlib http")
    assert ev.location == "turn 3 (user)"


def test_evidence_refuses_an_empty_quote() -> None:
    with pytest.raises(ValueError):
        Evidence(turn=1, role="user", quote="   ")


# ── Candidate: validation ─────────────────────────────────────────────────


def test_candidate_refuses_empty_text() -> None:
    with pytest.raises(ValueError):
        candidate(text="  ")


def test_candidate_refuses_a_kind_outside_the_taxonomy() -> None:
    with pytest.raises(ValueError):
        candidate(kind="vibes")


def test_candidate_refuses_a_confidence_outside_zero_to_one() -> None:
    with pytest.raises(ValueError):
        candidate(confidence=1.4)


def test_candidate_requires_a_reason_it_matters() -> None:
    # "and why it matters" is part of the spec's entry shape, not a nicety: an
    # entry that can't say why it matters can't be argued down later either.
    with pytest.raises(ValueError):
        candidate(why="")


def test_candidate_normalises_sequences_to_tuples() -> None:
    cand = candidate(domains=["providers", "infra"])
    assert cand.domains == ("providers", "infra")


# ── Candidate: disposition ────────────────────────────────────────────────


def test_a_high_confidence_decision_with_a_domain_commits_directly() -> None:
    assert candidate(confidence=COMMIT_THRESHOLD).disposition == "commit"


def test_an_untagged_candidate_never_commits_however_confident() -> None:
    assert candidate(confidence=1.0, domains=()).disposition == "review"


def test_a_gotcha_goes_to_review_even_at_full_confidence() -> None:
    assert candidate(kind="gotcha", confidence=1.0).disposition == "review"


def test_a_constraint_is_decision_shaped_enough_to_commit() -> None:
    assert candidate(kind="constraint", confidence=0.9).disposition == "commit"


def test_a_rejected_approach_is_decision_shaped_enough_to_commit() -> None:
    assert candidate(kind="rejected", confidence=0.9).disposition == "commit"


def test_a_middling_candidate_is_flagged_for_review_not_committed() -> None:
    assert (
        candidate(confidence=(COMMIT_THRESHOLD + REVIEW_FLOOR) / 2).disposition
        == "review"
    )


def test_a_candidate_below_the_floor_is_dropped() -> None:
    assert candidate(confidence=REVIEW_FLOOR - 0.01).disposition == "drop"


def test_the_floor_itself_still_earns_a_review() -> None:
    # Boundaries are where "flag it for a human" quietly turns into "delete it".
    assert candidate(confidence=REVIEW_FLOOR).disposition == "review"


def test_thresholds_leave_room_between_dropping_and_committing() -> None:
    assert 0.0 < REVIEW_FLOOR < COMMIT_THRESHOLD < 1.0


# ── Candidate: shape of the entry itself ──────────────────────────────────


def test_a_candidate_knows_how_many_sentences_it_spends() -> None:
    # The spec asks for one or two sentences. Length is a quality signal the
    # scorer uses, so the type has to be able to report it.
    assert candidate(text="One thing. Two things. Three things.").sentences == 3
    assert candidate(text="Just the one thing.").sentences == 1


def test_a_candidate_serialises_with_its_disposition_and_evidence() -> None:
    cand = candidate(
        evidence=(Evidence(turn=2, role="assistant", quote="going with stdlib http"),),
        signals=("decision-cue",),
    )
    data = cand.to_dict()
    assert data["disposition"] == "commit"
    assert data["evidence"][0]["turn"] == 2
    assert data["signals"] == ["decision-cue"]


# ── MemoryEntry ───────────────────────────────────────────────────────────


def test_an_entry_gets_a_stable_id_from_its_own_content() -> None:
    # Ids have to survive a re-run: the store dedupes on them, and a random id
    # would make every re-reflection look like a new fact.
    first = MemoryEntry.from_candidate(candidate(), session="s1")
    second = MemoryEntry.from_candidate(candidate(), session="s2")
    assert first.id == second.id


def test_entries_with_different_text_get_different_ids() -> None:
    other = candidate(text="Use the anthropic SDK for the Claude path.")
    assert MemoryEntry.from_candidate(candidate(), session="s").id != (
        MemoryEntry.from_candidate(other, session="s").id
    )


def test_an_entry_id_says_what_kind_of_entry_it_is() -> None:
    entry = MemoryEntry.from_candidate(candidate(kind="gotcha"), session="s")
    assert entry.id.startswith("gotcha-")


def test_an_entry_keeps_the_candidates_evidence_and_reason() -> None:
    cand = candidate(evidence=(Evidence(turn=1, role="user", quote="stdlib http"),))
    entry = MemoryEntry.from_candidate(
        cand, session="s1", created_at="2026-08-20T05:00:00Z"
    )
    assert entry.evidence == cand.evidence
    assert entry.why == cand.why
    assert entry.session == "s1"
    assert entry.created_at == "2026-08-20T05:00:00Z"


def test_a_fresh_entry_is_live() -> None:
    assert MemoryEntry.from_candidate(candidate(), session="s").live


def test_an_entry_pointed_at_by_a_successor_is_no_longer_live() -> None:
    entry = MemoryEntry.from_candidate(candidate(), session="s")
    replaced = entry.superseded_by_entry("decision-abc12345")
    assert not replaced.live
    assert replaced.superseded_by == "decision-abc12345"
    assert entry.live, "superseding must not mutate the original in place"


def test_an_entry_round_trips_through_its_dict_form() -> None:
    entry = MemoryEntry.from_candidate(
        candidate(evidence=(Evidence(turn=4, role="tool", quote="exit 1"),)),
        session="s1",
        created_at="2026-08-20T05:00:00Z",
    )
    assert MemoryEntry.from_dict(entry.to_dict()) == entry


# ── ReflectionReport ──────────────────────────────────────────────────────


def report(*cands: Candidate, **over: object) -> ReflectionReport:
    fields: dict[str, object] = {"session": "s1", "candidates": cands}
    fields.update(over)
    return ReflectionReport(**fields)  # type: ignore[arg-type]


def test_a_report_partitions_candidates_by_disposition() -> None:
    keep = candidate(confidence=0.9)
    look = candidate(text="Maybe the retry backoff should be longer.", confidence=0.5)
    junk = candidate(text="Ran the test suite.", confidence=0.1)
    rep = report(keep, look, junk)
    assert rep.to_commit == (keep,)
    assert rep.to_review == (look,)
    assert rep.dropped == (junk,)


def test_a_report_with_nothing_above_the_floor_says_so_outright() -> None:
    # "If nothing is memory-worthy this cycle, say so." A report that can only
    # express findings has no way to say that.
    assert report(candidate(confidence=0.1)).nothing_worth_remembering
    assert report().nothing_worth_remembering


def test_a_report_holding_one_review_item_is_not_empty_handed() -> None:
    assert not report(candidate(confidence=0.5)).nothing_worth_remembering


def test_a_report_summarises_itself_in_one_readable_line() -> None:
    line = report(
        candidate(confidence=0.9), candidate(text="Hm.", confidence=0.5)
    ).summary_line()
    assert "1 to commit" in line
    assert "1 for review" in line


def test_an_empty_report_summarises_as_nothing_rather_than_zeroes() -> None:
    assert "nothing memory-worthy" in report().summary_line().lower()


def test_a_report_lists_the_domains_it_touched() -> None:
    rep = report(
        candidate(domains=("providers",)), candidate(domains=("infra", "providers"))
    )
    assert rep.domains_touched() == ("infra", "providers")


def test_a_report_only_counts_domains_it_would_actually_write_to() -> None:
    # A dropped candidate's domain tag is not a domain this cycle learned about.
    rep = report(candidate(domains=("providers",), confidence=0.1))
    assert rep.domains_touched() == ()


def test_a_report_serialises_notes_and_candidates_for_the_cli() -> None:
    data = report(candidate(), notes=("transcript had no VERIFY phase",)).to_dict()
    assert data["session"] == "s1"
    assert data["notes"] == ["transcript had no VERIFY phase"]
    assert data["candidates"][0]["kind"] == "decision"
    assert data["summary"] == data["summary"].strip()
