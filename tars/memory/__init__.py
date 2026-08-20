"""TARS memory writer — PHASE 6 (REFLECT), isolated.

Takes a session transcript and proposes the memory entries that session earned:
each one domain-tagged, scored, quoted from the turn it came from, and carrying
one sentence on why a future session would care.

    from tars.memory import reflect_on

    report = reflect_on("run/session.jsonl")
    report.summary_line()
    report.to_commit    # high-confidence, decision-shaped, domain-tagged
    report.to_review    # what a human should see before anything believes it
    report.dropped      # considered, scored too low, kept so you can see it

Nothing here decides to remember something because it looks important. The four
kinds are exactly the ones core directive 4 names — a decision, a constraint, a
rejected approach, a non-obvious gotcha — and the same sentence puts routine
execution and anything visible by reading the code outside memory entirely. A
transcript full of "ran the tests, all passed" produces an empty report, and an
empty report says so rather than reaching for something:

    from tars.memory import reflect_on

    reflect_on("run/session.jsonl").nothing_worth_remembering

Every score is a sum of named signals, and the names ship with the candidate.
A confidence nobody can take apart is a confidence nobody can correct:

    from tars.memory import propose, parse_text

    candidate = propose(parse_text("We decided to use stdlib http.")).candidates[0]
    candidate.signals      # ("cue:decided-to", "role:assistant", ...)
    candidate.evidence     # the turns it was read out of, quoted verbatim
    candidate.disposition  # "commit", "review" or "drop"

Writing is a separate act from proposing, and the split is not cosmetic. A
missed entry costs one re-derivation; a wrong entry gets believed without one.
So `remember` is something a caller does on purpose, and only entries that are
high-confidence *and* decision-shaped *and* domain-tagged go straight in — the
rest wait in a review queue that `recall` cannot see:

    from tars.memory import reflect_on, remember

    remember(reflect_on("run/session.jsonl"))

`recall` is the other end, and the one ORIENT uses: live entries for a domain,
strongest first. Superseded entries stay on disk and readable but stop coming
back from here, because "we used to think X" is worth keeping and worth not
acting on:

    from tars.memory import recall

    recall(domains=("providers",))
    recall(kinds=("constraint",), limit=5)

Domains are tagged only from explicit evidence — a path, a keyword, a name the
statement actually contains. Cross-cutting entries get every domain they name and
nothing they merely imply, which is the spec's rule verbatim: tag explicitly, or
not at all. A project whose vocabulary differs from the built-in map says so in
`.tars/memory/domains.json` rather than hoping the defaults generalise:

    {"domains": [{"name": "billing", "keywords": ["invoice", "stripe"]}]}

Nothing here talks to a model, reads a credential, or imports the provider
layer. A reflection pass is text in, scored text out, which is what makes it
runnable on a transcript from a session that has already ended.

Also a standalone command:

    tars-memory reflect run/session.jsonl
    tars-memory reflect run/session.jsonl --write
    tars-memory pending
    tars-memory recall --domain providers
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .domains import DEFAULT_DOMAINS, DOMAINS_PATH, Domain, DomainMap, load_domain_map
from .propose import MAX_CANDIDATES, propose, score
from .store import STORE_DIR, MemoryStore, WriteResult
from .transcript import (
    PHASES,
    Transcript,
    Turn,
    load_transcript,
    parse_messages,
    parse_text,
    parse_transcript,
)
from .types import (
    COMMIT_THRESHOLD,
    REVIEW_FLOOR,
    Candidate,
    Evidence,
    MemoryEntry,
    ReflectionReport,
)


def reflect_on(
    transcript: str | Path,
    *,
    root: str | Path = ".",
    session: str | None = None,
    max_candidates: int = MAX_CANDIDATES,
) -> ReflectionReport:
    """Read a transcript from disk and score everything in it.

    Takes a path, not text: `parse_text` and `propose` are right there for an
    in-memory transcript, and a function that guessed which one a string was
    would eventually read a file path as prose.

    What is already stored is passed to the proposer, so a statement that
    replaces an earlier entry can name it. Raises `FileNotFoundError` rather
    than returning an empty report — a caller acting on the result would
    otherwise read "this session learned nothing" off a typo.
    """
    store = MemoryStore(root=Path(root))
    return propose(
        load_transcript(transcript, session=session),
        existing=store.entries(),
        max_candidates=max_candidates,
        root=str(root),
    )


def remember(
    report: ReflectionReport,
    *,
    root: str | Path = ".",
    created_at: str | None = None,
) -> WriteResult:
    """Apply a report: commit what may commit, queue the rest.

    The split is the candidate's own call, not this function's, so the rule
    cannot drift between here, the store and the CLI. `created_at` is a
    parameter because two runs over the same transcript have to produce
    comparable files.
    """
    return MemoryStore(root=Path(root)).write(report, created_at=created_at)


def recall(
    domains: Sequence[str] = (),
    kinds: Sequence[str] = (),
    limit: int | None = None,
    root: str | Path = ".",
) -> tuple[MemoryEntry, ...]:
    """What a session should read before forming an opinion.

    Live entries only, strongest first. Queued candidates and superseded entries
    are both deliberately unreachable from here: one hasn't been reviewed, the
    other has been replaced, and neither is something to act on without being
    told.
    """
    return MemoryStore(root=Path(root)).recall(
        domains=domains, kinds=kinds, limit=limit
    )


__all__ = [
    "COMMIT_THRESHOLD",
    "DEFAULT_DOMAINS",
    "DOMAINS_PATH",
    "MAX_CANDIDATES",
    "PHASES",
    "REVIEW_FLOOR",
    "STORE_DIR",
    "Candidate",
    "Domain",
    "DomainMap",
    "Evidence",
    "MemoryEntry",
    "MemoryStore",
    "ReflectionReport",
    "Transcript",
    "Turn",
    "WriteResult",
    "load_domain_map",
    "load_transcript",
    "parse_messages",
    "parse_text",
    "parse_transcript",
    "propose",
    "recall",
    "reflect_on",
    "remember",
    "score",
]
