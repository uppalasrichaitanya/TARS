"""Detectors for the sentence in core directive 4.

Memory-worthy: "a decision, a constraint, a rejected approach, a non-obvious
gotcha". Not memory-worthy: "routine execution, anything visible by just reading
the code". This file turns that into cue tables and the vetoes next to them, and
nothing here is clever — every match is a literal phrase a person actually
writes, kept in one place so the corpus can argue with it.

Three design notes worth knowing before changing anything:

**Cues carry weights, and the strongest cue per kind wins.** Summing would let a
verbose sentence out-score a precise one. A cue can also fire for two kinds at
once — "instead of" is evidence of a decision *and* of a rejected approach — and
the heavier reading wins, which is why "we used X instead of Y" is filed as
`rejected`: what was turned down is the part a future session cannot recover by
reading the code.

**Routine detection has to survive decision grammar.** "I decided to run the
formatter" carries a decision verb and means nothing. Every cue table in this
file is worthless without the routine veto next to it.

**A veto is not a low score.** Three shapes of statement fire no cue at all — a
question, a retraction, and a rule that expires with the session — because none
of them makes a claim about the project that could be more or less true. Routine
narration and hedges score instead of vetoing, since those *are* claims, just
ones a reviewer should see ranked last rather than never.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .transcript import Turn

#: Highest-weight kind wins; this order breaks a tie. `rejected` first because a
#: rejected approach is the least recoverable fact in a codebase — nothing in
#: the tree records what was tried and dropped.
KIND_PRIORITY: tuple[str, ...] = ("rejected", "gotcha", "constraint", "decision")


@dataclass(frozen=True)
class Signal:
    """One cue that fired, with the text that fired it."""

    name: str
    kind: str
    weight: float
    matched: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "weight": self.weight,
            "matched": self.matched,
        }


@dataclass(frozen=True)
class Statement:
    """One sentence out of one turn, with where it came from."""

    text: str
    turn: int
    role: str
    phase: str | None = None

    @property
    def words(self) -> int:
        return len(self.text.split())


# ── cue tables ────────────────────────────────────────────────────────────
#
# (name, kind, weight, pattern). Weights are coarse on purpose: 0.5 for a phrase
# that means the thing outright, ~0.4 for one that usually does, ~0.3 for one
# that only suggests it. Fine-grained weights would imply a precision this
# approach doesn't have.

_CUES: tuple[tuple[str, str, float, str], ...] = (
    # decisions
    ("decided-to", "decision", 0.50, r"\bdecided (?:to|on|that)\b"),
    ("decision-is", "decision", 0.50, r"\bthe decision (?:is|was)\b"),
    ("going-with", "decision", 0.50, r"\bgoing with\b"),
    ("settled-on", "decision", 0.50, r"\bsettled on\b"),
    (
        "we-will-use",
        "decision",
        0.50,
        r"\bwe(?:'ll| will| are going to)? (?:use|keep|go with)\b",
    ),
    ("agreed-to", "decision", 0.48, r"\bagreed (?:to|on|that)\b"),
    ("chose", "decision", 0.48, r"\b(?:chose|chosen|picked)\b"),
    ("from-now-on", "decision", 0.46, r"\bfrom now on\b"),
    ("standardise-on", "decision", 0.46, r"\bstandardi[sz]e(?:d)? on\b"),
    ("switching-to", "decision", 0.40, r"\bswitch(?:ed|ing)? to\b"),
    ("prefer-over", "decision", 0.40, r"\bprefer(?:red|ring)? .{1,40}\bover\b"),
    ("adopted", "decision", 0.38, r"\badopt(?:ed|ing)\b"),
    ("convention", "decision", 0.32, r"\b(?:the )?convention (?:is|here is)\b"),
    ("for-consistency", "decision", 0.30, r"\bfor consistency\b"),
    # constraints
    ("never", "constraint", 0.50, r"\bnever\b"),
    ("must-not", "constraint", 0.50, r"\bmust not\b"),
    ("must", "constraint", 0.48, r"\bmust\b"),
    ("has-to", "constraint", 0.46, r"\b(?:has|have|had) to\b"),
    ("only-works-if", "constraint", 0.50, r"\bonly works? (?:if|when|with|on)\b"),
    ("not-allowed", "constraint", 0.50, r"\b(?:not allowed|forbidden|off limits)\b"),
    ("requires", "constraint", 0.44, r"\brequire(?:s|d)\b"),
    ("always", "constraint", 0.42, r"\balways\b"),
    ("cannot", "constraint", 0.40, r"\b(?:cannot|can't|won't work|will not work)\b"),
    (
        "do-not",
        "constraint",
        0.35,
        r"\b(?:do not|don't) (?:use|put|commit|call|import|add)\b",
    ),
    # rejected approaches
    ("ruled-out", "rejected", 0.50, r"\b(?:ruled out|rejected|abandoned|discarded)\b"),
    ("decided-against", "rejected", 0.50, r"\bdecided against\b"),
    (
        "wont-use",
        "rejected",
        0.48,
        r"\b(?:won't|will not|not going to) (?:use|be using|add)\b",
    ),
    ("dropped-approach", "rejected", 0.48, r"\bdropped the (?:idea|approach|plan)\b"),
    (
        "no-longer",
        "rejected",
        0.46,
        r"\bno longer (?:use|uses|using|need|needs|call)\b",
    ),
    ("backed-out", "rejected", 0.46, r"\b(?:backed out|reverted) (?:of |the )?\b"),
    ("avoid-using", "rejected", 0.44, r"\bavoid(?:ed|ing)? (?:using|adding)\b"),
    (
        "tried-but-failed",
        "rejected",
        0.46,
        r"\btried\b.{1,60}\b(?:failed|didn't work|broke)\b",
    ),
    ("instead-of", "rejected", 0.42, r"\binstead of\b"),
    ("rather-than", "rejected", 0.40, r"\brather than\b"),
    # the same two phrases also record that a choice was made
    ("instead-of-choice", "decision", 0.34, r"\binstead of\b"),
    ("rather-than-choice", "decision", 0.32, r"\brather than\b"),
    # gotchas
    ("turns-out", "gotcha", 0.50, r"\bturns out\b"),
    ("silently", "gotcha", 0.50, r"\bsilently\b"),
    ("gotcha", "gotcha", 0.50, r"\b(?:gotcha|footgun|pitfall)\b"),
    ("the-catch", "gotcha", 0.50, r"\bthe catch (?:is|was)\b"),
    ("beware", "gotcha", 0.48, r"\b(?:beware|watch out)\b"),
    ("root-cause", "gotcha", 0.48, r"\b(?:root cause|the real (?:reason|cause))\b"),
    (
        "fails-without",
        "gotcha",
        0.48,
        r"\b(?:fails?|breaks?|errors?) (?:without|unless)\b",
    ),
    ("counterintuitive", "gotcha", 0.46, r"\b(?:counter-?intuitive|surprisingly)\b"),
    ("easy-to-miss", "gotcha", 0.44, r"\b(?:easy to miss|not obvious|non-obvious)\b"),
    ("caveat", "gotcha", 0.42, r"\bcaveat\b"),
    ("undocumented", "gotcha", 0.42, r"\bundocumented\b"),
    ("quirk", "gotcha", 0.38, r"\bquirk\b"),
    ("note-that", "gotcha", 0.30, r"\bnote that\b"),
)

_COMPILED_CUES = tuple(
    (name, kind, weight, re.compile(pattern, re.IGNORECASE))
    for name, kind, weight, pattern in _CUES
)

#: Routine execution, and code narration. Both are true statements about the
#: session; neither would change how a future session behaves, because the tree
#: and the git log already say them.
_ROUTINE: tuple[str, ...] = (
    r"\b(?:ran|run|running|re-?ran) (?:the )?(?:tests?|suite|linter|formatter|checks?|gate)\b",
    r"\ball (?:the )?tests? (?:pass|passed)\b",
    r"\b\d+ (?:passed|failed)\b",
    r"\b(?:installed|committed|staged|pushed|bumped|reformatted|formatted)\b",
    r"\blet me\b",
    r"\bI'?ll (?:now|start|go|run|open|read)\b",
    r"^(?:now|next|first|then|finally)\b",
    r"\b(?:looking at|checking|reading) (?:the )?\w+",
    r"\badded (?:a|an|the) (?:helper|function|class|method|field|test|fixture|module|file)\b",
    r"\b(?:renamed|deleted|removed the file|moved) \w+",
    r"\bextracted \w+ into\b",
    r"\bwrote (?:the )?(?:tests?|function|class|module|file)\b",
    r"\bfixed a typo\b",
)

_COMPILED_ROUTINE = tuple(re.compile(p, re.IGNORECASE) for p in _ROUTINE)

#: Withdrawals. A retraction takes a claim back, so — exactly like a question —
#: there is no claim left to be confident about, and the veto belongs in the same
#: place for the same reason.
#:
#: Two absences are deliberate. "I was wrong about X" is not here: a correction
#: replaces a claim with a better one, and the better one is usually the most
#: memory-worthy sentence in the transcript. And the phrases that take an object
#: require that object to be a previous utterance ("ignore that *last* one"),
#: because a bare "ignore the lockfile" is a constraint, and vetoing it would
#: throw away the strongest kind of entry there is.
_RETRACTION: tuple[str, ...] = (
    r"\bnever mind\b",
    r"\b(?:scratch|strike) that\b",
    (
        r"\b(?:ignore|disregard|forget) (?:that|this|the|my) "
        r"(?:last|previous|prior|earlier)\b"
    ),
    r"\bforget what I (?:said|wrote)\b",
    # Not "actually no longer use X", which is a supersede and worth keeping.
    r"\bactually,?\s+no\b(?!\s+\w)",
)

_COMPILED_RETRACTION = tuple(re.compile(p, re.IGNORECASE) for p in _RETRACTION)

#: Rules that expire when the session does. Memory-worthy means "would change
#: how a FUTURE session behaves", and "this has to be the last commit tonight"
#: cannot, however imperative its grammar — by tomorrow there is nothing left to
#: obey. So this is a veto, not a penalty: it isn't a weak claim about the
#: project, it is not a claim about the project at all.
#:
#: Every pattern requires the time word to be doing scope work. A bare "tonight"
#: is not enough, because "tonight we settled on sqlite" only *dates* a decision
#: — it puts no expiry on it.
_EPHEMERAL: tuple[str, ...] = (
    r"\blast (?:\w+ ){0,2}(?:tonight|today|this session|this run)\b",
    r"\bstop (?:for|at) (?:tonight|today|now|the (?:day|night|evening))\b",
    (
        r"\bwait (?:until|till|for) "
        r"(?:tomorrow|the morning|next (?:session|run|time)|later)\b"
    ),
    (
        r"\bbefore (?:I|we) "
        r"(?:go to bed|sleep|call it a night|stop for the (?:day|night))\b"
    ),
    r"\b(?:rest|remainder) of (?:tonight|today|this session)\b",
    (
        r"\bpick (?:this|it|that) up (?:again )?"
        r"(?:tomorrow|in the morning|next session)\b"
    ),
)

_COMPILED_EPHEMERAL = tuple(re.compile(p, re.IGNORECASE) for p in _EPHEMERAL)

#: Hedges. Not a veto — a hedged gotcha is often the most valuable thing in a
#: transcript — but a hedge is the transcript saying "don't bank on this yet",
#: which is exactly what a review flag is for.
_WEAK: tuple[str, ...] = (
    r"\bmaybe\b",
    r"\bprobably\b",
    r"\bperhaps\b",
    r"\bpossibly\b",
    r"\bmight\b",
    r"\bi think\b",
    r"\bi believe\b",
    r"\bnot sure\b",
    r"\bunclear\b",
    r"\bseems? (?:to|like)?\b",
    r"\bappears? to\b",
    r"\bcould be\b",
    r"\blikely\b",
    r"\bpresumably\b",
    r"\barguably\b",
    r"\bi guess\b",
    r"\bworth (?:considering|a look)\b",
)

_COMPILED_WEAK = tuple(re.compile(p, re.IGNORECASE) for p in _WEAK)

#: Phrases that say a prior belief is being replaced. Superseding is only ever
#: claimed from one of these: inferring it from two entries that merely disagree
#: in tone would retire facts nobody retired.
_SUPERSEDE: tuple[str, ...] = (
    r"\bno longer\b",
    r"\bsupersede(?:s|d)?\b",
    r"\breplaces (?:the|our|that)\b",
    r"\bused to\b",
    r"\bpreviously (?:we|the)\b",
    r"\bwe now\b",
    r"\bchanged (?:our|my) mind\b",
    r"\boverrides the (?:earlier|previous)\b",
    r"\binstead of the (?:earlier|previous|old)\b",
    r"\bswitching from\b",
    r"\breverses (?:the|our)\b",
)

_COMPILED_SUPERSEDE = tuple(re.compile(p, re.IGNORECASE) for p in _SUPERSEDE)

#: Tokens concrete enough to act on: a path, a file, an env var, a flag, a
#: version, a backticked identifier, a CamelCase or snake_case name. A statement
#: with none of these is the shape of an entry that rots unnoticed — true,
#: agreeable, and impossible to act on.
_SPECIFIC: tuple[str, ...] = (
    r"`[^`]+`",
    r"\b[\w.-]+/[\w./-]+\b",
    r"\b[\w.-]+\.(?:py|toml|json|jsonl|md|ya?ml|txt|js|ts|tsx|sh|cfg|ini|lock|log)\b",
    r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b",
    r"\b[A-Z]{3,}\b",
    r"--[\w-]{2,}",
    r"\b\d+\.\d+(?:\.\d+)?\b",
    r"\b[A-Za-z]+[a-z]+[A-Z][A-Za-z]*\b",
    r"\b_?[a-z][a-z0-9]*_[a-z0-9_]+\b",
)

_COMPILED_SPECIFIC = tuple(re.compile(p) for p in _SPECIFIC)

#: A stated reason, when the sentence gives one. Preferred over anything the
#: writer could compose, because it is the transcript's own words.
_REASON = re.compile(r"\b(because(?! of)|since|so that|as it)\b", re.IGNORECASE)

#: How many words must precede a reason connective. "Because of that we use
#: sqlite" puts its reason first; taking the tail would invert the sentence.
_REASON_MIN_WORDS = 3

_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "cf.",
    "no.",
    "fig.",
    "al.",
    "approx.",
    "resp.",
    "cases.",
)

_BULLET = re.compile(r"^\s*(?:[-*•+]|\(?\d+[.)])\s+")

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])["\')\]]?\s+(?=[A-Z"\'(\[])')

#: Lines that are machine output rather than something someone said.
_NOT_PROSE = (
    re.compile(r"^\s*[$>#]\s"),
    re.compile(r"^\s*Traceback\b"),
    re.compile(r'^\s*File "'),
    re.compile(r"^\s*[A-Za-z_.]*(?:Error|Exception|Warning)\b.*:"),
    re.compile(r"^\s*\d"),
    re.compile(r"^\s*[-=─]{3,}\s*$"),
    re.compile(r"^\s{4,}\S"),
    # A lone token on its own line — a path, a command name, a symbol — unless it
    # ends in punctuation, which is how a label like "Decisions:" introducing a
    # list of them is written.
    re.compile(r"^\s*\S*[^\s:.!?]\s*$"),
)

_FENCE = re.compile(r"^\s*(?:```|~~~)")

#: Words that can only open a question. `do` and `have` are deliberately absent:
#: "Do not use the SDK" is an imperative constraint, and vetoing it would lose
#: the strongest kind of entry there is.
_INTERROGATIVES = (
    "should",
    "shall",
    "could",
    "would",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "can",
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "who",
    "whether",
)

_OPENS_A_QUESTION = re.compile(
    r"^\s*(?:" + "|".join(_INTERROGATIVES) + r")\b", re.IGNORECASE
)


def is_question(text: str) -> bool:
    """Whether `text` asks rather than asserts.

    A question is not a weaker fact, it is the absence of one — "should we use
    sqlite or postgres here?" contains the grammar of a decision and none of its
    content. Committing it would fabricate the answer.
    """
    stripped = text.strip()
    if not stripped:
        return False
    return stripped.endswith("?") or bool(_OPENS_A_QUESTION.match(stripped))


def _is_prose(line: str) -> bool:
    return not any(pattern.match(line) for pattern in _NOT_PROSE)


def _split_sentences(paragraph: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(paragraph)
    merged: list[str] = []
    for part in parts:
        if merged and merged[-1].lower().endswith(_ABBREVIATIONS):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return [p.strip() for p in merged if p.strip()]


def segment(turn: Turn) -> tuple[Statement, ...]:
    """Split one turn into the statements it makes.

    Fenced code, shell transcripts and tracebacks are dropped: they are not
    statements anybody made, and the spec puts what is visible by reading the
    code outside memory anyway. Bullets are statements in their own right, since
    that is how transcripts list decisions.
    """
    statements: list[Statement] = []
    in_fence = False
    for raw in turn.text.splitlines():
        if _FENCE.match(raw):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        bullet = _BULLET.match(raw)
        line = raw[bullet.end() :] if bullet else raw
        if not line.strip() or not _is_prose(line):
            continue
        for sentence in _split_sentences(line.strip()):
            statements.append(
                Statement(
                    text=sentence,
                    turn=turn.index,
                    role=turn.role,
                    phase=turn.phase,
                )
            )
    return tuple(statements)


def read_signals(text: str) -> tuple[Signal, ...]:
    """Every cue that fires on `text`, with what matched.

    Three kinds of statement fire nothing at all: a question, a retraction, and a
    rule scoped to this session. The vetoes belong here rather than in the scorer
    because none of them is low-confidence evidence for a claim — a question has
    no claim yet, a retraction has taken its claim back, and a rule that expires
    tonight makes no claim about the project. Scoring them low would still leave
    them in the queue for a human to read and reject one at a time.
    """
    if is_question(text) or retraction_markers(text) or ephemeral_markers(text):
        return ()
    found: list[Signal] = []
    for name, kind, weight, pattern in _COMPILED_CUES:
        match = pattern.search(text)
        if match:
            found.append(
                Signal(name=name, kind=kind, weight=weight, matched=match.group(0))
            )
    return tuple(found)


def kind_of(signals: tuple[Signal, ...]) -> str | None:
    """Which of the four kinds this statement is, if any.

    Strongest single cue per kind, not the sum: otherwise a rambling sentence
    with four weak cues outranks a flat "never commit the token".
    """
    if not signals:
        return None
    best: dict[str, float] = {}
    for signal in signals:
        best[signal.kind] = max(best.get(signal.kind, 0.0), signal.weight)
    top = max(best.values())
    for kind in KIND_PRIORITY:
        if best.get(kind, 0.0) == top:
            return kind
    return None


def strength_of(signals: tuple[Signal, ...], kind: str) -> float:
    """The strongest cue weight backing `kind`."""
    weights = [s.weight for s in signals if s.kind == kind]
    return max(weights) if weights else 0.0


def _matches(patterns: tuple[re.Pattern[str], ...], text: str) -> tuple[str, ...]:
    found: list[str] = []
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            found.append(match.group(0).strip())
    return tuple(found)


def routine_markers(text: str) -> tuple[str, ...]:
    """Phrases that mark this as routine execution or code narration."""
    return _matches(_COMPILED_ROUTINE, text)


def ephemeral_markers(text: str) -> tuple[str, ...]:
    """Phrases that scope a statement to this session rather than the project."""
    return _matches(_COMPILED_EPHEMERAL, text)


def retraction_markers(text: str) -> tuple[str, ...]:
    """Phrases that withdraw a claim instead of making one."""
    return _matches(_COMPILED_RETRACTION, text)


def weak_markers(text: str) -> tuple[str, ...]:
    """Hedges — the transcript saying "don't bank on this yet"."""
    return _matches(_COMPILED_WEAK, text)


def supersede_markers(text: str) -> tuple[str, ...]:
    """Phrases that claim a prior belief is being replaced."""
    return _matches(_COMPILED_SUPERSEDE, text)


def specific_tokens(text: str) -> tuple[str, ...]:
    """Concrete things named in `text`, deduped, in the order they appear."""
    found: list[str] = []
    for pattern in _COMPILED_SPECIFIC:
        for match in pattern.finditer(text):
            token = match.group(0).strip()
            if token and token not in found:
                found.append(token)
    return tuple(found)


def reason_clause(text: str) -> str | None:
    """The reason the statement gives for itself, if it gives one.

    Only a trailing clause counts. A leading "Because ..." states the cause of
    the sentence's subject, not the reason for a decision, and lifting it would
    invert the meaning.
    """
    match = _REASON.search(text)
    if not match:
        return None
    if len(text[: match.start()].split()) < _REASON_MIN_WORDS:
        return None
    clause = text[match.end() :].strip(" ,;:")
    clause = clause.rstrip(".!?").strip()
    return clause or None
