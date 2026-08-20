"""Where a kept entry lives, and what a future ORIENT can get back out.

The store is small on purpose — domain-partitioned JSON under `.tars/memory/`,
one file per partition, plus one review queue. Everything interesting about it is
a consequence of two rules from the spec.

*"Low-confidence or ambiguous entries go to a human review queue."* A queue only
means something if what is in it cannot be recalled as fact. The test that keeps
this module honest is
`test_a_queued_candidate_is_never_returned_by_recall`: if pending entries leaked
into recall, the queue would be decoration and the review step would be a lie.

*"Mark superseded entries so contradicting entries aren't both live."* Superseding
here never deletes. The old entry stays in its file, readable, with a pointer to
what replaced it — because "we used to think X" is itself worth knowing, and
because a store that quietly drops rows is one nobody can audit after a bad
session.

Time is a parameter, not a call to the clock: the tests need to compare files
byte for byte, and a store that stamps its own writes can't be compared.
"""

from __future__ import annotations

import json

import pytest

from tars.memory.store import STORE_DIR, MemoryStore
from tars.memory.types import Candidate, Evidence, MemoryEntry, ReflectionReport

STAMP = "2026-08-20T02:00:00+00:00"


def candidate(
    text: str = "We decided to use stdlib http for OpenRouter.",
    kind: str = "decision",
    domains: tuple[str, ...] = ("providers",),
    confidence: float = 0.9,
    **over: object,
) -> Candidate:
    fields: dict[str, object] = {
        "text": text,
        "kind": kind,
        "domains": domains,
        "why": "so a later session doesn't re-litigate it",
        "confidence": confidence,
        "evidence": (Evidence(turn=1, role="user", quote=text),),
    }
    fields.update(over)
    return Candidate(**fields)  # type: ignore[arg-type]


def store(tmp_path: object) -> MemoryStore:
    return MemoryStore(root=tmp_path)  # type: ignore[arg-type]


# ── committing ────────────────────────────────────────────────────────────


def test_a_committed_entry_lands_in_its_domains_partition(tmp_path) -> None:
    kept = store(tmp_path)
    kept.write(
        ReflectionReport(session="s", candidates=(candidate(),)), created_at=STAMP
    )
    assert (tmp_path / STORE_DIR / "providers.json").is_file()


def test_a_committed_entry_comes_back_out_of_recall(tmp_path) -> None:
    kept = store(tmp_path)
    kept.write(
        ReflectionReport(session="s", candidates=(candidate(),)), created_at=STAMP
    )
    found = kept.recall(domains=("providers",))
    assert [e.text for e in found] == ["We decided to use stdlib http for OpenRouter."]


def test_an_entry_records_which_session_wrote_it(tmp_path) -> None:
    # So a session that turned out to be wrong can have its writes found again.
    kept = store(tmp_path)
    kept.write(
        ReflectionReport(session="phase/memory-writer", candidates=(candidate(),)),
        created_at=STAMP,
    )
    assert kept.recall()[0].session == "phase/memory-writer"


def test_the_caller_supplies_the_timestamp(tmp_path) -> None:
    kept = store(tmp_path)
    kept.write(
        ReflectionReport(session="s", candidates=(candidate(),)), created_at=STAMP
    )
    assert kept.recall()[0].created_at == STAMP


def test_a_cross_domain_entry_is_filed_under_every_domain_it_claims(
    tmp_path,
) -> None:
    kept = store(tmp_path)
    cand = candidate(domains=("providers", "verification"))
    kept.write(ReflectionReport(session="s", candidates=(cand,)), created_at=STAMP)
    assert kept.recall(domains=("providers",))
    assert kept.recall(domains=("verification",))


def test_a_cross_domain_entry_is_recalled_once_not_once_per_partition(
    tmp_path,
) -> None:
    kept = store(tmp_path)
    cand = candidate(domains=("providers", "verification"))
    kept.write(ReflectionReport(session="s", candidates=(cand,)), created_at=STAMP)
    assert len(kept.recall(domains=("providers", "verification"))) == 1


def test_recall_with_no_domain_reads_every_partition(tmp_path) -> None:
    kept = store(tmp_path)
    kept.write(
        ReflectionReport(
            session="s",
            candidates=(
                candidate(),
                candidate(
                    text="Never commit ANTHROPIC_AUTH_TOKEN to the repo.",
                    kind="constraint",
                    domains=("security",),
                ),
            ),
        ),
        created_at=STAMP,
    )
    assert len(kept.recall()) == 2


def test_recall_can_ask_for_one_kind(tmp_path) -> None:
    # ORIENT asking "what am I not allowed to do here" is a different question
    # from "what was decided here".
    kept = store(tmp_path)
    kept.write(
        ReflectionReport(
            session="s",
            candidates=(
                candidate(),
                candidate(
                    text="Never commit ANTHROPIC_AUTH_TOKEN to the repo.",
                    kind="constraint",
                ),
            ),
        ),
        created_at=STAMP,
    )
    assert [e.kind for e in kept.recall(kinds=("constraint",))] == ["constraint"]


def test_recall_returns_the_most_confident_first(tmp_path) -> None:
    kept = store(tmp_path)
    kept.write(
        ReflectionReport(
            session="s",
            candidates=(
                candidate(confidence=0.78),
                candidate(
                    text="We decided to pin ruff in pyproject.toml.", confidence=0.95
                ),
            ),
        ),
        created_at=STAMP,
    )
    scores = [e.confidence for e in kept.recall()]
    assert scores == sorted(scores, reverse=True)


def test_an_empty_store_recalls_nothing_rather_than_failing(tmp_path) -> None:
    assert store(tmp_path).recall() == ()


# ── the review queue ──────────────────────────────────────────────────────


def test_a_low_confidence_candidate_goes_to_the_queue_not_a_partition(
    tmp_path,
) -> None:
    kept = store(tmp_path)
    result = kept.write(
        ReflectionReport(session="s", candidates=(candidate(confidence=0.5),)),
        created_at=STAMP,
    )
    assert result.committed == ()
    assert len(result.queued) == 1
    assert not (tmp_path / STORE_DIR / "providers.json").exists()


def test_a_queued_candidate_is_never_returned_by_recall(tmp_path) -> None:
    # The load-bearing test of this file. A future session recalling a pending
    # entry as fact is exactly the human review step being skipped.
    kept = store(tmp_path)
    kept.write(
        ReflectionReport(session="s", candidates=(candidate(confidence=0.5),)),
        created_at=STAMP,
    )
    assert kept.recall() == ()
    assert len(kept.pending()) == 1


def test_a_dropped_candidate_is_neither_stored_nor_queued(tmp_path) -> None:
    # Below the floor it isn't a proposal at all; the report is where it stays
    # readable, not the queue.
    kept = store(tmp_path)
    result = kept.write(
        ReflectionReport(session="s", candidates=(candidate(confidence=0.1),)),
        created_at=STAMP,
    )
    assert result.committed == ()
    assert result.queued == ()
    assert store(tmp_path).pending() == ()


def test_an_untagged_candidate_can_only_ever_be_queued(tmp_path) -> None:
    kept = store(tmp_path)
    result = kept.write(
        ReflectionReport(session="s", candidates=(candidate(domains=()),)),
        created_at=STAMP,
    )
    assert result.committed == ()
    assert len(result.queued) == 1


def test_approving_a_queued_entry_moves_it_into_the_partition(tmp_path) -> None:
    # A queue nobody can drain is theatre, so approval is part of the surface.
    kept = store(tmp_path)
    cand = candidate(confidence=0.5)
    kept.write(ReflectionReport(session="s", candidates=(cand,)), created_at=STAMP)
    entry_id = MemoryEntry.from_candidate(cand).id
    approved = kept.approve(entry_id, session="human", created_at=STAMP)
    assert approved is not None
    assert kept.recall()[0].id == entry_id
    assert kept.pending() == ()


def test_rejecting_a_queued_entry_removes_it_without_storing_it(tmp_path) -> None:
    kept = store(tmp_path)
    cand = candidate(confidence=0.5)
    kept.write(ReflectionReport(session="s", candidates=(cand,)), created_at=STAMP)
    assert kept.reject(MemoryEntry.from_candidate(cand).id)
    assert kept.pending() == ()
    assert kept.recall() == ()


def test_approving_something_that_is_not_queued_says_so(tmp_path) -> None:
    assert store(tmp_path).approve("decision-deadbeef", session="h") is None


def test_the_same_candidate_queued_twice_is_one_queue_entry(tmp_path) -> None:
    kept = store(tmp_path)
    body = ReflectionReport(session="s", candidates=(candidate(confidence=0.5),))
    kept.write(body, created_at=STAMP)
    kept.write(body, created_at=STAMP)
    assert len(kept.pending()) == 1


# ── superseding ───────────────────────────────────────────────────────────


def test_committing_a_replacement_marks_what_it_replaced(tmp_path) -> None:
    kept = store(tmp_path)
    old = candidate(text="The provider layer uses the openai SDK for OpenRouter.")
    kept.write(ReflectionReport(session="s1", candidates=(old,)), created_at=STAMP)
    old_id = MemoryEntry.from_candidate(old).id

    new = candidate(
        text="We no longer use the openai SDK for OpenRouter.",
        kind="rejected",
        supersedes=(old_id,),
    )
    result = kept.write(
        ReflectionReport(session="s2", candidates=(new,)), created_at=STAMP
    )

    assert old_id in result.superseded
    assert [e.id for e in kept.recall()] == [MemoryEntry.from_candidate(new).id]


def test_a_superseded_entry_stays_readable_rather_than_being_deleted(
    tmp_path,
) -> None:
    # "We used to think X" is worth knowing, and a store that drops rows is one
    # nobody can audit after a bad session.
    kept = store(tmp_path)
    old = candidate(text="The provider layer uses the openai SDK for OpenRouter.")
    kept.write(ReflectionReport(session="s1", candidates=(old,)), created_at=STAMP)
    old_id = MemoryEntry.from_candidate(old).id
    new = candidate(
        text="We no longer use the openai SDK for OpenRouter.",
        kind="rejected",
        supersedes=(old_id,),
    )
    kept.write(ReflectionReport(session="s2", candidates=(new,)), created_at=STAMP)

    archived = {e.id: e for e in kept.entries()}
    assert old_id in archived
    assert not archived[old_id].live
    assert archived[old_id].superseded_by == MemoryEntry.from_candidate(new).id


def test_superseding_something_that_was_never_stored_is_noted_not_fatal(
    tmp_path,
) -> None:
    kept = store(tmp_path)
    new = candidate(supersedes=("decision-deadbeef",))
    result = kept.write(
        ReflectionReport(session="s", candidates=(new,)), created_at=STAMP
    )
    assert result.committed
    assert any("deadbeef" in note for note in result.notes)


# ── writing the same thing twice ──────────────────────────────────────────


def test_re_reflecting_on_one_session_does_not_duplicate_entries(tmp_path) -> None:
    # Ids are content-derived precisely so this is safe; an overnight run that
    # reflects twice must not double every fact.
    kept = store(tmp_path)
    body = ReflectionReport(session="s", candidates=(candidate(),))
    kept.write(body, created_at=STAMP)
    second = kept.write(body, created_at="2026-08-21T02:00:00+00:00")
    assert len(kept.recall()) == 1
    assert second.committed == ()
    assert second.unchanged


def test_the_first_write_of_an_entry_keeps_its_original_timestamp(tmp_path) -> None:
    kept = store(tmp_path)
    body = ReflectionReport(session="s", candidates=(candidate(),))
    kept.write(body, created_at=STAMP)
    kept.write(body, created_at="2026-08-21T02:00:00+00:00")
    assert kept.recall()[0].created_at == STAMP


# ── the files themselves ──────────────────────────────────────────────────


def test_a_partition_is_readable_json_a_person_can_edit(tmp_path) -> None:
    # A memory a human can't open and correct is one they have to trust blindly.
    kept = store(tmp_path)
    kept.write(
        ReflectionReport(session="s", candidates=(candidate(),)), created_at=STAMP
    )
    data = json.loads((tmp_path / STORE_DIR / "providers.json").read_text())
    assert data["domain"] == "providers"
    assert data["entries"][0]["why"]


def test_a_corrupt_partition_is_reported_rather_than_recalled_around(
    tmp_path,
) -> None:
    # Silently recalling the partitions that happen to parse would make "memory
    # before opinion" quietly false for whichever domain broke.
    path = tmp_path / STORE_DIR / "providers.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    with pytest.raises(ValueError, match="providers.json"):
        store(tmp_path).recall()


def test_a_domain_name_that_could_escape_the_store_is_refused(tmp_path) -> None:
    # Domains reaching the store as plain strings — from a CLI flag, say — get
    # checked here too, not only where Domain objects are built.
    with pytest.raises(ValueError):
        store(tmp_path).recall(domains=("../../etc",))


def test_two_writes_of_the_same_content_produce_identical_files(tmp_path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    body = ReflectionReport(session="s", candidates=(candidate(),))
    MemoryStore(root=first).write(body, created_at=STAMP)
    MemoryStore(root=second).write(body, created_at=STAMP)
    assert (first / STORE_DIR / "providers.json").read_text() == (
        (second / STORE_DIR / "providers.json").read_text()
    )


def test_the_write_result_says_what_it_did(tmp_path) -> None:
    kept = store(tmp_path)
    result = kept.write(
        ReflectionReport(
            session="s",
            candidates=(
                candidate(),
                candidate(text="Maybe use sqlite.", confidence=0.5),
            ),
        ),
        created_at=STAMP,
    )
    line = result.summary_line()
    assert "1 committed" in line
    assert "1 queued" in line
