"""Types the memory writer speaks: what a candidate is, and what makes one
trustworthy enough to keep.

One shape per concern:
  Evidence         — the turn a claim came from, quoted.
  Candidate        — a proposed entry, scored, with its disposition.
  MemoryEntry      — a candidate that made it into the store.
  ReflectionReport — everything one REFLECT pass produced.

Three fields are load-bearing and worth reading carefully.

``Candidate.evidence`` exists because a memory entry is a claim a future session
will act on without re-deriving it. When the writer says "this project decided to
use stdlib HTTP", the next question is always "who said that, and where?" —
carrying the quoted turn makes the answer checkable instead of a matter of faith.
The guardrail it serves is *never fabricate a memory entry to fill a gap*.

``Candidate.confidence`` is not a decoration on the text; it is what decides
whether a human sees the entry before a future session believes it. The mapping
from confidence to action lives in :attr:`Candidate.disposition` so that every
call site gets the same answer.

``MemoryEntry.superseded_by`` is how "don't leave contradicting entries both
live" is enforced. Superseding never deletes: the old entry stays readable, it
just stops being retrievable as current.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Literal

#: What kind of thing is worth remembering. These are exactly the four the spec
#: names as memory-worthy — a decision, a constraint, a rejected approach, a
#: non-obvious gotcha. Anything that isn't one of these four is, by the same
#: sentence, routine execution.
MemoryKind = Literal["decision", "constraint", "rejected", "gotcha"]

VALID_KINDS: frozenset[str] = frozenset(
    {"decision", "constraint", "rejected", "gotcha"}
)

#: Kinds that may auto-commit. "High-confidence, clearly decision-shaped entries
#: can commit directly" — a decision, a constraint and a rejected approach are
#: all records of a choice someone made, and a transcript either contains that
#: choice or it doesn't. A gotcha is a causal claim *about* what happened, which
#: is the easiest kind of claim to hold confidently and wrongly, so it always
#: gets a human look.
DECISION_SHAPED: frozenset[str] = frozenset({"decision", "constraint", "rejected"})

#: At or above this, a decision-shaped, domain-tagged candidate commits directly.
COMMIT_THRESHOLD = 0.75

#: Below this, a candidate isn't proposed at all. It stays in the report as
#: `drop` with its score, so "the writer considered and rejected this" is
#: readable, but it never reaches a human's review queue as noise.
REVIEW_FLOOR = 0.35

#: What the writer decided to do with a candidate.
Disposition = Literal["commit", "review", "drop"]

#: Who produced a turn. `tool` covers command output fed back into the loop.
Role = Literal["user", "assistant", "tool", "system"]

_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


@dataclass(frozen=True)
class Evidence:
    """Where a claim came from, quoted verbatim.

    `quote` is never paraphrased. The moment the writer starts tidying the
    source sentence, `turn` stops being something a reader can check.
    """

    turn: int
    role: str
    quote: str

    def __post_init__(self) -> None:
        if not self.quote.strip():
            raise ValueError(f"evidence at turn {self.turn} has no quote")

    @property
    def location(self) -> str:
        """The origin as a human would cite it."""
        return f"turn {self.turn} ({self.role})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "role": self.role,
            "quote": self.quote,
            "location": self.location,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        return cls(
            turn=int(data["turn"]), role=str(data["role"]), quote=str(data["quote"])
        )


@dataclass(frozen=True)
class Candidate:
    """A proposed memory entry, with the score that decides its fate."""

    text: str
    kind: MemoryKind
    #: Domain partitions this belongs to, tagged only from explicit evidence.
    #: Empty is a legitimate answer — "tag explicitly, or not at all" — and it
    #: caps the disposition at `review`, because an untagged entry is one ORIENT
    #: can never retrieve.
    domains: tuple[str, ...]
    #: Why a future session would care. One sentence, and required: an entry
    #: that can't say why it matters can't be argued down later either.
    why: str
    confidence: float
    #: The turns this was read out of.
    evidence: tuple[Evidence, ...] = ()
    #: Which scoring signals fired, so a score can be audited rather than
    #: trusted. Reads like `("decision-cue", "specific:tars/providers")`.
    signals: tuple[str, ...] = ()
    #: Ids of prior entries this replaces, when the transcript says so.
    supersedes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("candidate has no text")
        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"unknown memory kind {self.kind!r}; expected one of {sorted(VALID_KINDS)}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence!r} is outside 0..1")
        if not self.why.strip():
            raise ValueError(f"candidate {self.text!r} does not say why it matters")
        for name in ("domains", "evidence", "signals", "supersedes"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))

    @property
    def disposition(self) -> Disposition:
        """Commit it, show a human, or leave it in the report as considered.

        Encoded here rather than at the call site so that the rule can't drift
        between the CLI, the store and the loop.
        """
        if self.confidence < REVIEW_FLOOR:
            return "drop"
        if (
            self.confidence >= COMMIT_THRESHOLD
            and self.kind in DECISION_SHAPED
            and self.domains
        ):
            return "commit"
        return "review"

    @property
    def sentences(self) -> int:
        """How many sentences the entry spends.

        The spec asks for one or two. Longer usually means the writer copied a
        transcript span instead of stating a fact, which is what the scorer
        penalises.
        """
        return (
            len([s for s in _SENTENCE_END.split(self.text.strip()) if s.strip()]) or 1
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind,
            "domains": list(self.domains),
            "why": self.why,
            "confidence": round(self.confidence, 3),
            "disposition": self.disposition,
            "evidence": [e.to_dict() for e in self.evidence],
            "signals": list(self.signals),
            "supersedes": list(self.supersedes),
        }


def _entry_id(kind: str, text: str) -> str:
    """A stable id derived from content.

    Deterministic on purpose: the store dedupes on id, so re-reflecting on the
    same transcript has to produce the same id or every re-run looks like a new
    fact.
    """
    digest = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:8]
    return f"{kind}-{digest}"


@dataclass(frozen=True)
class MemoryEntry:
    """A candidate that made it into the store."""

    id: str
    text: str
    kind: MemoryKind
    domains: tuple[str, ...]
    why: str
    confidence: float
    #: Which session produced it, so a bad session's writes can be found.
    session: str = ""
    created_at: str | None = None
    evidence: tuple[Evidence, ...] = ()
    supersedes: tuple[str, ...] = ()
    #: Set when a later entry replaces this one. Not a deletion: the entry stays
    #: readable, it just stops being current.
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        for name in ("domains", "evidence", "supersedes"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))

    @classmethod
    def from_candidate(
        cls,
        candidate: Candidate,
        session: str = "",
        created_at: str | None = None,
    ) -> MemoryEntry:
        return cls(
            id=_entry_id(candidate.kind, candidate.text),
            text=candidate.text,
            kind=candidate.kind,
            domains=candidate.domains,
            why=candidate.why,
            confidence=candidate.confidence,
            session=session,
            created_at=created_at,
            evidence=candidate.evidence,
            supersedes=candidate.supersedes,
        )

    @property
    def live(self) -> bool:
        return self.superseded_by is None

    def superseded_by_entry(self, entry_id: str) -> MemoryEntry:
        """This entry, marked as replaced by `entry_id`."""
        return MemoryEntry(
            id=self.id,
            text=self.text,
            kind=self.kind,
            domains=self.domains,
            why=self.why,
            confidence=self.confidence,
            session=self.session,
            created_at=self.created_at,
            evidence=self.evidence,
            supersedes=self.supersedes,
            superseded_by=entry_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
            "domains": list(self.domains),
            "why": self.why,
            "confidence": round(self.confidence, 3),
            "session": self.session,
            "created_at": self.created_at,
            "evidence": [e.to_dict() for e in self.evidence],
            "supersedes": list(self.supersedes),
            "superseded_by": self.superseded_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            kind=data["kind"],
            domains=tuple(data.get("domains", ())),
            why=str(data.get("why", "")),
            confidence=float(data.get("confidence", 0.0)),
            session=str(data.get("session", "")),
            created_at=data.get("created_at"),
            evidence=tuple(Evidence.from_dict(e) for e in data.get("evidence", ())),
            supersedes=tuple(data.get("supersedes", ())),
            superseded_by=data.get("superseded_by"),
        )


@dataclass(frozen=True)
class ReflectionReport:
    """Everything one REFLECT pass produced.

    Dropped candidates are kept rather than discarded. "I considered this and
    scored it 0.2" is a different, more useful statement than silence, and it is
    the only way to tell a writer that found nothing from one that found things
    and threw them away.
    """

    session: str
    candidates: tuple[Candidate, ...] = ()
    #: Anything the caller should know about how this pass went: a transcript
    #: with no ACT phase, a domain map that matched nothing, text that was
    #: truncated.
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("candidates", "notes"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))

    def _with(self, disposition: str) -> tuple[Candidate, ...]:
        return tuple(c for c in self.candidates if c.disposition == disposition)

    @property
    def to_commit(self) -> tuple[Candidate, ...]:
        return self._with("commit")

    @property
    def to_review(self) -> tuple[Candidate, ...]:
        return self._with("review")

    @property
    def dropped(self) -> tuple[Candidate, ...]:
        return self._with("drop")

    @property
    def proposed(self) -> tuple[Candidate, ...]:
        """Everything that survived the floor, in either direction."""
        return tuple(c for c in self.candidates if c.disposition != "drop")

    @property
    def nothing_worth_remembering(self) -> bool:
        """True when this cycle has nothing to write.

        The spec asks the loop to say this out loud when it happens, which means
        the report needs a way to express it that isn't just an empty list.
        """
        return not self.proposed

    def domains_touched(self) -> tuple[str, ...]:
        """Domains this cycle would actually write into, sorted."""
        seen: set[str] = set()
        for cand in self.proposed:
            seen.update(cand.domains)
        return tuple(sorted(seen))

    def summary_line(self) -> str:
        if self.nothing_worth_remembering:
            considered = len(self.dropped)
            tail = f" ({considered} considered, none kept)" if considered else ""
            return f"nothing memory-worthy this cycle{tail}"
        parts = [
            f"{len(self.to_commit)} to commit",
            f"{len(self.to_review)} for review",
        ]
        if self.dropped:
            parts.append(f"{len(self.dropped)} dropped")
        domains = ", ".join(self.domains_touched()) or "no domain"
        return f"{'; '.join(parts)} — {domains}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session,
            "summary": self.summary_line(),
            "nothing_worth_remembering": self.nothing_worth_remembering,
            "domains_touched": list(self.domains_touched()),
            "counts": {
                "commit": len(self.to_commit),
                "review": len(self.to_review),
                "dropped": len(self.dropped),
            },
            "candidates": [c.to_dict() for c in self.candidates],
            "notes": list(self.notes),
        }
