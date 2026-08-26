"""The `tars-memory` command: the only surface a human or the loop actually
touches, so the phase's two dangerous behaviours get pinned here as well as
downstream.

`main()` takes argv and both streams, so nearly everything drives the real entry
point in-process. One test spawns `python -m tars.memory` for real, because
"runnable as a command" is a claim about the module being executable and only a
process proves it.

Three properties are load-bearing:

* **Reflecting writes nothing without `--write`.** A memory entry is a claim a
  future session acts on without re-deriving it, so the default has to be the
  one that can't be wrong: print what it would keep, touch nothing.
* **`recall` never surfaces a queued candidate.** The store already refuses
  this; it is pinned again here because the CLI is where a loop asks "what do we
  know", and a pending entry answering that question is the human review step
  being skipped in the one place it would go unnoticed.
* **The exit code separates "nothing to remember" from "kept something" from "a
  human is needed".** An unattended run reads the code, not the prose. Folding
  "nothing memory-worthy" into 0 would let a cycle that learned nothing report
  as a productive one.
"""

import json
import subprocess
import sys
from io import StringIO

from tars.memory.cli import (
    EXIT_NOTHING,
    EXIT_OK,
    EXIT_REVIEW,
    EXIT_USAGE,
    main,
)
from tars.memory.propose import propose
from tars.memory.store import STORE_DIR, MemoryStore
from tars.memory.transcript import load_transcript
from tars.memory.types import Candidate, Evidence, MemoryEntry, ReflectionReport

STAMP = "2026-08-20T02:00:00+00:00"

#: One clean decision, naming a real path — scores high enough to commit, and
#: produces nothing for review.
DECISION = "We decided to use stdlib http for OpenRouter instead of adding an SDK.\n"

#: A causal claim. Decision-shaped kinds can auto-commit; a gotcha never does,
#: so this transcript's only outcome is a review item.
GOTCHA = "It turns out `tars/providers/base.py` silently retries on a 429.\n"

#: Routine execution. The spec puts this outside memory outright.
ROUTINE = "Ran the tests. All 461 passed. Committed the change.\n"

#: A pair that supersedes: the second uses a replacement phrase, lands in the
#: same domain, and names the same identifier — the three conditions the proposer
#: requires before it will claim one entry retires another.
OLD = (
    "We decided to use ANTHROPIC_AUTH_TOKEN for the Claude path in "
    "`tars/providers/anthropic_provider.py`.\n"
)
REPLACEMENT = (
    "The adapter in `tars/providers/anthropic_provider.py` no longer uses "
    "ANTHROPIC_AUTH_TOKEN.\n"
)

#: The same replacement, hedged, so it scores below the commit threshold and can
#: only reach the review queue.
HEDGED_REPLACEMENT = (
    "Maybe the adapter in `tars/providers/anthropic_provider.py` no longer uses "
    "ANTHROPIC_AUTH_TOKEN.\n"
)


def run(*argv, stdin=None):
    out, err = StringIO(), StringIO()
    code = main([*argv], stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def transcript(tmp_path, text=DECISION, name="session.md"):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def queued(tmp_path, text="It turns out the loader silently skips empty files."):
    """Put one candidate in the review queue and return its id."""
    cand = Candidate(
        text=text,
        kind="gotcha",
        domains=("providers",),
        why="so it isn't rediscovered the hard way",
        confidence=0.6,
        evidence=(Evidence(turn=1, role="user", quote=text),),
    )
    from tars.memory.types import ReflectionReport

    MemoryStore(root=tmp_path).write(
        ReflectionReport(session="s", candidates=(cand,)), created_at=STAMP
    )
    return MemoryEntry.from_candidate(cand).id


# ── reflect: what it reports ──────────────────────────────────────────────


def test_reflect_prints_the_candidate_it_found(tmp_path) -> None:
    code, out, _ = run("reflect", transcript(tmp_path), "--root", str(tmp_path))
    assert code == EXIT_OK
    assert "stdlib http" in out


def test_reflect_shows_the_confidence_behind_each_candidate(tmp_path) -> None:
    # A score a reader can't see is a score nobody can argue with. The number
    # comes from the scorer rather than being hard-coded here: this pins that the
    # report shows it, not what the scorer currently thinks.
    path = transcript(tmp_path)
    _, out, _ = run("reflect", path, "--root", str(tmp_path))
    expected = propose(load_transcript(path)).candidates[0].confidence
    assert f"{expected:.2f}" in out


def test_reflect_shows_why_each_candidate_matters(tmp_path) -> None:
    _, out, _ = run("reflect", transcript(tmp_path), "--root", str(tmp_path))
    assert "why" in out.lower()


def test_reflect_shows_which_domains_a_candidate_was_tagged_with(tmp_path) -> None:
    _, out, _ = run("reflect", transcript(tmp_path), "--root", str(tmp_path))
    assert "providers" in out


def test_reflect_cites_the_turn_a_candidate_came_from(tmp_path) -> None:
    # Provenance is the difference between a memory entry and an assertion.
    _, out, _ = run("reflect", transcript(tmp_path), "--root", str(tmp_path))
    assert "turn 1" in out


def test_reflect_says_so_outright_when_nothing_is_memory_worthy(tmp_path) -> None:
    code, out, _ = run(
        "reflect", transcript(tmp_path, ROUTINE), "--root", str(tmp_path)
    )
    assert code == EXIT_NOTHING
    assert "nothing memory-worthy" in out.lower()


# ── reflect: the exit code ────────────────────────────────────────────────


def test_a_reflection_with_nothing_for_a_human_exits_zero(tmp_path) -> None:
    code, _, _ = run("reflect", transcript(tmp_path), "--root", str(tmp_path))
    assert code == EXIT_OK


def test_a_reflection_holding_something_for_review_says_so_in_its_code(
    tmp_path,
) -> None:
    # The overnight run reads this to decide whether to leave a note behind.
    code, _, _ = run("reflect", transcript(tmp_path, GOTCHA), "--root", str(tmp_path))
    assert code == EXIT_REVIEW


def test_an_empty_reflection_is_not_reported_as_a_successful_one(tmp_path) -> None:
    code, _, _ = run("reflect", transcript(tmp_path, ROUTINE), "--root", str(tmp_path))
    assert code != EXIT_OK


# ── reflect: writing, and not writing ─────────────────────────────────────


def test_reflecting_without_write_leaves_the_store_untouched(tmp_path) -> None:
    # The load-bearing test of this file. A dry run that quietly commits is a
    # tool that can put a wrong fact into memory while reporting a preview.
    run("reflect", transcript(tmp_path), "--root", str(tmp_path))
    assert not (tmp_path / STORE_DIR).exists()


def test_a_dry_run_says_that_it_wrote_nothing(tmp_path) -> None:
    _, out, err = run("reflect", transcript(tmp_path), "--root", str(tmp_path))
    assert "--write" in out + err


def test_reflect_write_commits_a_high_confidence_decision(tmp_path) -> None:
    code, _, _ = run(
        "reflect", transcript(tmp_path), "--root", str(tmp_path), "--write"
    )
    assert code == EXIT_OK
    kept = MemoryStore(root=tmp_path).recall()
    assert [e.text for e in kept] == [DECISION.strip()]


def test_reflect_write_queues_a_gotcha_rather_than_committing_it(tmp_path) -> None:
    run("reflect", transcript(tmp_path, GOTCHA), "--root", str(tmp_path), "--write")
    kept = MemoryStore(root=tmp_path)
    assert kept.recall() == ()
    assert len(kept.pending()) == 1


def test_reflect_write_reports_what_it_did_to_the_store(tmp_path) -> None:
    _, out, err = run(
        "reflect", transcript(tmp_path), "--root", str(tmp_path), "--write"
    )
    assert "1 committed" in out + err


def test_the_caller_can_pin_the_timestamp_entries_are_written_with(tmp_path) -> None:
    # Same reason the store takes time as a parameter: two runs over the same
    # content have to produce comparable files.
    run(
        "reflect",
        transcript(tmp_path),
        "--root",
        str(tmp_path),
        "--write",
        "--now",
        STAMP,
    )
    assert MemoryStore(root=tmp_path).recall()[0].created_at == STAMP


def test_an_entry_records_the_session_it_was_reflected_out_of(tmp_path) -> None:
    run(
        "reflect",
        transcript(tmp_path),
        "--root",
        str(tmp_path),
        "--write",
        "--session",
        "phase/memory-writer",
    )
    assert MemoryStore(root=tmp_path).recall()[0].session == "phase/memory-writer"


def test_the_session_defaults_to_the_transcripts_own_name(tmp_path) -> None:
    run("reflect", transcript(tmp_path), "--root", str(tmp_path), "--write")
    assert MemoryStore(root=tmp_path).recall()[0].session == "session"


def test_reflecting_twice_over_one_transcript_does_not_double_the_entry(
    tmp_path,
) -> None:
    args = ("reflect", transcript(tmp_path), "--root", str(tmp_path), "--write")
    run(*args)
    run(*args)
    assert len(MemoryStore(root=tmp_path).recall()) == 1


def test_reflect_reads_the_existing_store_so_a_replacement_can_supersede(
    tmp_path,
) -> None:
    # Supersede detection needs to see what is already known; a reflect that
    # ignores the store can only ever add contradicting entries beside it.
    run(
        "reflect",
        transcript(tmp_path, OLD, "first.md"),
        "--root",
        str(tmp_path),
        "--write",
    )
    old_id = MemoryStore(root=tmp_path).recall()[0].id

    _, out, err = run(
        "reflect",
        transcript(tmp_path, REPLACEMENT, "second.md"),
        "--root",
        str(tmp_path),
        "--write",
    )
    archived = {e.id: e for e in MemoryStore(root=tmp_path).entries()}
    assert not archived[old_id].live, out + err


def test_a_replacement_still_awaiting_review_does_not_retire_the_old_entry(
    tmp_path,
) -> None:
    # The whole point of the queue is that nothing acts on an entry until a human
    # has. A queued replacement that silently retires a live fact would make the
    # store disagree with memory nobody approved.
    run(
        "reflect",
        transcript(tmp_path, OLD, "first.md"),
        "--root",
        str(tmp_path),
        "--write",
    )
    old_id = MemoryStore(root=tmp_path).recall()[0].id

    run(
        "reflect",
        transcript(tmp_path, HEDGED_REPLACEMENT, "second.md"),
        "--root",
        str(tmp_path),
        "--write",
    )
    kept = MemoryStore(root=tmp_path)
    assert kept.recall()[0].id == old_id
    assert len(kept.pending()) == 1


def test_reflect_names_the_entry_a_candidate_claims_to_replace(tmp_path) -> None:
    run(
        "reflect",
        transcript(tmp_path, OLD, "first.md"),
        "--root",
        str(tmp_path),
        "--write",
    )
    old_id = MemoryStore(root=tmp_path).recall()[0].id
    _, out, _ = run(
        "reflect",
        transcript(tmp_path, REPLACEMENT, "second.md"),
        "--root",
        str(tmp_path),
    )
    assert old_id in out


# ── reflect: --json ───────────────────────────────────────────────────────


def test_reflect_json_is_machine_readable(tmp_path) -> None:
    _, out, _ = run("reflect", transcript(tmp_path), "--root", str(tmp_path), "--json")
    data = json.loads(out)
    assert data["counts"]["commit"] == 1


def test_reflect_json_carries_the_verdict_it_exited_with(tmp_path) -> None:
    # A caller reading only stdout would otherwise have to re-derive the verdict
    # from the shape of the payload.
    code, out, _ = run(
        "reflect", transcript(tmp_path, ROUTINE), "--root", str(tmp_path), "--json"
    )
    assert json.loads(out)["exit_code"] == code


def test_reflect_json_reports_what_was_written_when_it_wrote(tmp_path) -> None:
    _, out, _ = run(
        "reflect",
        transcript(tmp_path),
        "--root",
        str(tmp_path),
        "--write",
        "--json",
    )
    assert json.loads(out)["write"]["committed"][0]["id"].startswith("decision-")


# ── recall ────────────────────────────────────────────────────────────────


def test_recall_prints_what_a_previous_session_kept(tmp_path) -> None:
    run("reflect", transcript(tmp_path), "--root", str(tmp_path), "--write")
    code, out, _ = run("recall", "--root", str(tmp_path))
    assert code == EXIT_OK
    assert "stdlib http" in out


def test_recall_can_ask_for_one_domain(tmp_path) -> None:
    run("reflect", transcript(tmp_path), "--root", str(tmp_path), "--write")
    code, out, _ = run("recall", "--root", str(tmp_path), "--domain", "verification")
    assert code == EXIT_NOTHING
    assert "stdlib http" not in out


def test_recall_can_ask_for_one_kind(tmp_path) -> None:
    run("reflect", transcript(tmp_path), "--root", str(tmp_path), "--write")
    code, _, _ = run("recall", "--root", str(tmp_path), "--kind", "constraint")
    assert code == EXIT_NOTHING


def test_recall_on_an_empty_store_says_nothing_rather_than_failing(tmp_path) -> None:
    code, out, _ = run("recall", "--root", str(tmp_path))
    assert code == EXIT_NOTHING
    assert out.strip()


def test_recall_never_surfaces_a_queued_candidate(tmp_path) -> None:
    # The other load-bearing test here. This is the exact call a future ORIENT
    # makes, and a pending entry answering it is the review step skipped.
    queued(tmp_path)
    code, out, _ = run("recall", "--root", str(tmp_path))
    assert code == EXIT_NOTHING
    assert "silently skips" not in out


def test_recall_json_is_machine_readable(tmp_path) -> None:
    run("reflect", transcript(tmp_path), "--root", str(tmp_path), "--write")
    _, out, _ = run("recall", "--root", str(tmp_path), "--json")
    assert json.loads(out)["entries"][0]["kind"] == "decision"


def test_recall_refuses_a_domain_name_that_could_escape_the_store(tmp_path) -> None:
    code, _, err = run("recall", "--root", str(tmp_path), "--domain", "../../etc")
    assert code == EXIT_USAGE
    assert "etc" in err


# ── the review queue ──────────────────────────────────────────────────────


def test_pending_lists_the_queue_with_ids_a_human_can_act_on(tmp_path) -> None:
    entry_id = queued(tmp_path)
    code, out, _ = run("pending", "--root", str(tmp_path))
    assert code == EXIT_REVIEW
    assert entry_id in out


def test_an_empty_queue_is_reported_as_nothing_waiting(tmp_path) -> None:
    code, out, _ = run("pending", "--root", str(tmp_path))
    assert code == EXIT_NOTHING
    assert out.strip()


def test_approving_a_queued_entry_makes_it_recallable(tmp_path) -> None:
    entry_id = queued(tmp_path)
    code, _, _ = run("approve", entry_id, "--root", str(tmp_path), "--now", STAMP)
    assert code == EXIT_OK
    kept = MemoryStore(root=tmp_path)
    assert [e.id for e in kept.recall()] == [entry_id]
    assert kept.pending() == ()


def test_rejecting_a_queued_entry_drops_it_without_storing_it(tmp_path) -> None:
    entry_id = queued(tmp_path)
    code, _, _ = run("reject", entry_id, "--root", str(tmp_path))
    assert code == EXIT_OK
    kept = MemoryStore(root=tmp_path)
    assert kept.pending() == ()
    assert kept.recall() == ()


def test_approving_an_id_that_is_not_queued_is_an_error_not_a_silent_no_op(
    tmp_path,
) -> None:
    code, _, err = run("approve", "gotcha-deadbeef", "--root", str(tmp_path))
    assert code == EXIT_USAGE
    assert "deadbeef" in err


def test_rejecting_an_id_that_is_not_queued_is_an_error_too(tmp_path) -> None:
    code, _, err = run("reject", "gotcha-deadbeef", "--root", str(tmp_path))
    assert code == EXIT_USAGE
    assert "deadbeef" in err


def test_correct_marks_a_stored_entry_wrong(tmp_path) -> None:
    candidate = Candidate(
        text="Use frobnicator cache.",
        kind="decision",
        domains=("backend",),
        why="Future sessions would rely on it.",
        confidence=0.80,
    )
    store = MemoryStore(tmp_path)
    entry = store.write(ReflectionReport("test", (candidate,))).committed[0]

    code, out, err = run(
        "correct",
        entry.id,
        "was wrong because it corrupts data",
        "--root",
        str(tmp_path),
    )

    assert code == EXIT_OK, err
    assert "corrected" in out
    assert store.recall(("backend",)) == ()


# ── invocation ────────────────────────────────────────────────────────────


def test_no_subcommand_is_a_usage_error_rather_than_a_default_action(tmp_path) -> None:
    # Every verb here either writes or answers a question; guessing which one was
    # meant is not a service.
    code, _, err = run()
    assert code == EXIT_USAGE
    assert err.strip()


def test_help_exits_cleanly() -> None:
    code, _, _ = run("--help")
    assert code == EXIT_OK


def test_a_missing_transcript_is_reported_not_raised(tmp_path) -> None:
    code, _, err = run("reflect", str(tmp_path / "nope.md"), "--root", str(tmp_path))
    assert code == EXIT_USAGE
    assert "nope.md" in err


def test_a_directory_given_as_a_transcript_is_reported_not_raised(tmp_path) -> None:
    code, _, err = run("reflect", str(tmp_path), "--root", str(tmp_path))
    assert code == EXIT_USAGE
    assert err.strip()


def test_a_transcript_of_unreadable_json_is_reported_not_raised(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    code, _, err = run("reflect", str(path), "--root", str(tmp_path))
    assert code == EXIT_USAGE
    assert "broken.json" in err


def test_a_cap_of_zero_candidates_is_refused_rather_than_reported_as_empty(
    tmp_path,
) -> None:
    # `--max 0` would print "nothing memory-worthy" over a transcript that had
    # something. Refusing it keeps that sentence true.
    code, _, err = run(
        "reflect", transcript(tmp_path), "--root", str(tmp_path), "--max", "0"
    )
    assert code == EXIT_USAGE
    assert err.strip()


def test_an_unknown_kind_is_refused_by_the_parser(tmp_path) -> None:
    code, _, err = run("recall", "--root", str(tmp_path), "--kind", "vibes")
    assert code == EXIT_USAGE
    assert "vibes" in err


def test_a_negative_limit_is_refused(tmp_path) -> None:
    code, _, err = run("recall", "--root", str(tmp_path), "--limit", "-1")
    assert code == EXIT_USAGE
    assert err.strip()


def test_the_module_is_runnable_as_a_command(tmp_path) -> None:
    # Only a real process proves `python -m tars.memory` works; an import in the
    # test runner would pass even with a broken entry point.
    path = tmp_path / "session.md"
    path.write_text(DECISION)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "tars.memory",
            "reflect",
            str(path),
            "--root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == EXIT_OK, done.stderr
    assert "stdlib http" in done.stdout
