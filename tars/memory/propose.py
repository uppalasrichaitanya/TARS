"""The REFLECT pass: read a transcript, propose what is worth remembering.

Three questions get answered here, in order.

**Is this a claim at all?** :mod:`tars.memory.signals` answers that — a statement
with no cue for any of the four memory-worthy kinds is not a candidate, and most
of a transcript is not.

**How much should anyone trust it?** That is :func:`score`, and it is the part
worth arguing with. The inputs are all facts about the transcript rather than
judgements about the world: who said it, in which phase, whether it names
anything concrete, whether it hedges, whether it was said more than once. The
weights are coarse and deliberately lopsided against writing — a missed entry
costs a future session one re-derivation, while a wrong entry gets believed
without being re-derived at all.

**Who sees it first?** That is not decided here. :attr:`Candidate.disposition`
owns it, so the CLI, the store and the loop cannot drift apart on what 0.8 means.

Two things this module refuses to do, both because the spec says so and both
tested:

*It does not infer supersedes.* Replacing a prior entry needs an explicit cue in
the transcript ("no longer", "replaces the earlier..."), a shared domain, and a
shared identifier. Two entries that merely disagree are left both live for a
human, because guessing which one won retires a fact nobody retired.

*It does not invent a reason.* `why` comes from the sentence's own because/since
clause when there is one. When there isn't, the templated fallback is flagged
`why:template` in the candidate's signals, so a reviewer can see that the reason
is the writer's and not the transcript's.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .domains import DomainMap, load_domain_map
from .signals import (
    Statement,
    kind_of,
    read_signals,
    reason_clause,
    routine_markers,
    segment,
    specific_tokens,
    strength_of,
    supersede_markers,
    weak_markers,
)
from .transcript import Transcript, parse_transcript
from .types import Candidate, Evidence, MemoryEntry, ReflectionReport

#: How many candidates one pass will propose. A REFLECT pass that hands over
#: forty entries is not reflecting, and a human review queue that long gets
#: skimmed — which is worse than a short one. Whatever the cap leaves out is
#: named in the report's notes.
MAX_CANDIDATES = 12

#: Who said it. The spread between `user` and `tool` is the widest single
#: adjustment in the table, and the one doing the most work: a person stating a
#: constraint is deciding something, while command output containing the same
#: sentence is a machine repeating a string.
ROLE_WEIGHTS: dict[str, float] = {
    "user": 0.20,
    "assistant": 0.10,
    "system": 0.10,
    "tool": -0.25,
}

#: Where in the loop it was said. REFLECT is where the loop is explicitly asked
#: what it learned, so a claim made there is likelier to be a considered one.
PHASE_WEIGHTS: dict[str, float] = {
    "REFLECT": 0.10,
    "ACT": 0.05,
    "VERIFY": 0.05,
    "REPORT": 0.05,
}

#: Naming something concrete — a path, a flag, an env var, a version. This is
#: the difference between "be careful with the provider layer" and "the retry
#: budget lives in tars/providers/base.py".
SPECIFIC_BONUS = 0.12
SPECIFIC_EXTRA = 0.05

#: Tagged into at least one declared partition, so a future ORIENT can find it.
DOMAIN_BONUS = 0.05

#: Each hedge, floored: three "maybe"s are not three times as uncertain as one.
HEDGE_PENALTY = -0.25
HEDGE_FLOOR = -0.35

#: Routine execution or code narration. Large enough to sink a statement on its
#: own, because "I decided to run the formatter" otherwise scores like a
#: decision.
ROUTINE_PENALTY = -0.60

#: Said more than once across the session.
CORROBORATION_BONUS = 0.08

#: Each additional cue of the same kind, capped.
EXTRA_CUE_BONUS = 0.05
EXTRA_CUE_CAP = 0.10

#: A one-line fragment is rarely a fact anyone can use a month later.
SHORT_TEXT_CHARS = 25
SHORT_TEXT_PENALTY = -0.15

#: A wall of text usually means the writer copied a transcript span instead of
#: stating something.
LONG_TEXT_CHARS = 300
LONG_TEXT_PENALTY = -0.10

#: Used only when the transcript gives no reason of its own, and always flagged.
_WHY_TEMPLATES: dict[str, str] = {
    "decision": "a later session that doesn't know this will re-litigate the choice.",
    "constraint": "a later session that breaks this finds out the hard way.",
    "rejected": "nothing in the tree records what was tried and turned down.",
    "gotcha": "this is not discoverable by reading the code.",
}

_WHY_TEMPLATE_SIGNAL = "why:template"


def _normalise(text: str) -> str:
    """The key two mentions of the same claim have to agree on."""
    return " ".join(text.lower().split()).strip(" .!?,;:")


def score(
    statement: Statement,
    *,
    domains: tuple[str, ...],
    mentions: int = 1,
) -> tuple[float, tuple[str, ...]]:
    """How much to trust `statement`, and every reason for the number.

    The signal list is returned alongside the score rather than derived later,
    because a score nobody can take apart is a score nobody can correct.
    """
    signals = read_signals(statement.text)
    kind = kind_of(signals)
    if kind is None:
        return 0.0, ()

    total = strength_of(signals, kind)
    reasons = [f"cue:{s.name}" for s in signals if s.kind == kind]

    extra = len([s for s in signals if s.kind == kind]) - 1
    if extra > 0:
        bonus = min(extra * EXTRA_CUE_BONUS, EXTRA_CUE_CAP)
        total += bonus
        reasons.append(f"cues:{extra + 1}")

    role_weight = ROLE_WEIGHTS.get(statement.role, 0.0)
    if role_weight:
        total += role_weight
        reasons.append(f"role:{statement.role}")

    if statement.phase:
        phase_weight = PHASE_WEIGHTS.get(statement.phase.upper(), 0.0)
        if phase_weight:
            total += phase_weight
            reasons.append(f"phase:{statement.phase.upper()}")

    tokens = specific_tokens(statement.text)
    if tokens:
        total += SPECIFIC_BONUS
        if len(tokens) >= 2:
            total += SPECIFIC_EXTRA
        reasons.append(f"specific:{tokens[0]}")

    if domains:
        total += DOMAIN_BONUS
        reasons.append("domain-tagged")

    hedges = weak_markers(statement.text)
    if hedges:
        total += max(len(hedges) * HEDGE_PENALTY, HEDGE_FLOOR)
        reasons.append(f"hedged:{hedges[0]}")

    routine = routine_markers(statement.text)
    if routine:
        total += ROUTINE_PENALTY
        reasons.append(f"routine:{routine[0]}")

    length = len(statement.text)
    if length < SHORT_TEXT_CHARS:
        total += SHORT_TEXT_PENALTY
        reasons.append("very-short")
    elif length > LONG_TEXT_CHARS:
        total += LONG_TEXT_PENALTY
        reasons.append("very-long")

    if mentions > 1:
        total += CORROBORATION_BONUS
        reasons.append(f"said-{mentions}-times")

    return max(0.0, min(1.0, total)), tuple(reasons)


def _supersedes(
    text: str,
    domains: tuple[str, ...],
    existing: Sequence[MemoryEntry],
) -> tuple[str, ...]:
    """Which prior entries this statement says outright that it replaces.

    Three conditions, all required: the transcript uses a replacement phrase, the
    statement lands in a domain the prior entry is in, and the two name something
    in common. Any two of the three would be a guess, and a wrong supersede
    silently retires a fact that is still true.
    """
    if not existing or not supersede_markers(text):
        return ()
    tokens = {t.lower() for t in specific_tokens(text)}
    if not tokens:
        return ()
    hits: list[str] = []
    for entry in existing:
        if not entry.live:
            continue
        if domains and not set(domains) & set(entry.domains):
            continue
        if tokens & {t.lower() for t in specific_tokens(entry.text)}:
            hits.append(entry.id)
    return tuple(hits)


def _why(statement: Statement, kind: str) -> tuple[str, bool]:
    """The reason to keep this, and whether the transcript actually gave it."""
    stated = reason_clause(statement.text)
    if stated:
        return stated, False
    return _WHY_TEMPLATES[kind], True


def _as_transcript(
    source: Transcript | Sequence[Any] | Mapping[str, Any],
) -> Transcript:
    if isinstance(source, Transcript):
        return source
    return parse_transcript(source)


def _statements(script: Transcript) -> list[Statement]:
    found: list[Statement] = []
    for turn in script.turns:
        found.extend(segment(turn))
    return found


def propose(
    source: Transcript | Sequence[Any] | Mapping[str, Any],
    *,
    domains: DomainMap | None = None,
    existing: Iterable[MemoryEntry] = (),
    max_candidates: int = MAX_CANDIDATES,
    root: str = ".",
) -> ReflectionReport:
    """Everything one REFLECT pass would propose from `source`.

    Dropped candidates stay in the report with their scores: "considered this,
    scored it 0.2" is a more useful thing to be able to read than silence, and it
    is the only way to tell a pass that found nothing from one that found things
    and threw them away.
    """
    script = _as_transcript(source)
    domain_map = domains if domains is not None else load_domain_map(root)
    prior = tuple(existing)
    notes: list[str] = []

    if not script.turns:
        return ReflectionReport(
            session=script.session, notes=("transcript has no turns to reflect on",)
        )

    statements = _statements(script)
    if not statements:
        notes.append(f"no prose statements in {len(script.turns)} turns")

    counts: dict[str, int] = {}
    for statement in statements:
        key = _normalise(statement.text)
        counts[key] = counts.get(key, 0) + 1

    # Group by claim first: the same sentence said in two turns is one candidate
    # with two pieces of evidence, not two entries that will collide in the store.
    grouped: dict[str, list[Statement]] = {}
    for statement in statements:
        if kind_of(read_signals(statement.text)) is None:
            continue
        grouped.setdefault(_normalise(statement.text), []).append(statement)

    candidates: list[Candidate] = []
    for key, group in grouped.items():
        scored = []
        for statement in group:
            tags = domain_map.domains_for(statement.text)
            confidence, reasons = score(statement, domains=tags, mentions=counts[key])
            scored.append((confidence, reasons, tags, statement))
        # The best-supported reading of a claim wins; every mention is evidence.
        confidence, reasons, tags, best = max(scored, key=lambda item: item[0])
        kind = kind_of(read_signals(best.text))
        if kind is None:  # pragma: no cover - grouped on a non-None kind
            continue
        why, templated = _why(best, kind)
        signals = reasons + ((_WHY_TEMPLATE_SIGNAL,) if templated else ())
        candidates.append(
            Candidate(
                text=best.text,
                kind=kind,  # type: ignore[arg-type]
                domains=tags,
                why=why,
                confidence=confidence,
                evidence=tuple(
                    Evidence(turn=s.turn, role=s.role, quote=s.text)
                    for s in sorted(group, key=lambda s: s.turn)
                ),
                signals=signals,
                supersedes=_supersedes(best.text, tags, prior),
            )
        )

    # Strongest first, then by text, so two passes over one transcript agree.
    candidates.sort(key=lambda c: (-c.confidence, c.text))
    if len(candidates) > max_candidates:
        cut = candidates[max_candidates:]
        candidates = candidates[:max_candidates]
        left_out = "; ".join(f"{c.kind} {c.confidence:.2f} {c.text[:48]}" for c in cut)
        notes.append(
            f"cap of {max_candidates} reached: {len(cut)} lower-scoring "
            f"candidates were not proposed — {left_out}"
        )

    proposed = [c for c in candidates if c.disposition != "drop"]
    if proposed and not any(c.domains for c in proposed):
        notes.append(
            "no candidate matched a declared domain, so none can commit directly; "
            "a human has to place them"
        )

    return ReflectionReport(
        session=script.session, candidates=tuple(candidates), notes=tuple(notes)
    )
