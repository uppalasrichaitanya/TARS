"""PHASE 2 (CLASSIFY), isolated: a task description in, a routing verdict out.

Three independent answers, because they answer different questions:

    tier        how much machinery the loop should spend deciding
    blast       what it costs if the change is wrong
    reversible  whether `git revert` alone puts the world back

Nothing here calls a model. The router's whole value is being cheap enough to
run before the expensive decision, so it is cue matching over the task text and
nothing else. That has a real ceiling, and the confusion matrix in
`tests/test_router_benchmark.py` is where that ceiling is reported rather than
hidden.

Every verdict ships the signals that produced it. A tier nobody can take apart
is a tier nobody can correct — and per the spec, the classification is a signal
to the loop, not a cage: the loop is expected to overrule it and say so.

The cues below are written from the rubric in `data/labeled_tasks.json`, not
from individual cases. Where a rule would only exist to fix one labeled example,
the miss was left in place and shows up in the matrix instead.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar

#: low < medium < high, so "the worse of two cues wins" is just max().
BLAST_ORDER: tuple[str, ...] = ("low", "medium", "high")

#: Tier 1 is the honest default: wrong by at most one step in either direction.
DEFAULT_TIER = 1

#: What a tier implies about reach when no blast cue fires at all.
BLAST_FROM_TIER: dict[int, str] = {0: "low", 1: "medium", 2: "high"}


@dataclass(frozen=True)
class Classification:
    """One routing verdict, with its reasoning attached."""

    tier: int
    blast: str
    reversible: bool
    signals: tuple[str, ...]
    why: str

    @property
    def needs_care(self) -> bool:
        """True when a mistake here is expensive to undo or wide-reaching."""
        return not self.reversible or self.blast == "high"


# ── cue tables ───────────────────────────────────────────────────────────
# Word-boundary anchored on purpose: `design` must not match `designation`, or
# every cue becomes a coin flip.

_TIER_CUES: tuple[tuple[str, str, int], ...] = (
    # tier 0 — no decision left in it
    (r"\btypos?\b", "cue:typo", 0),
    (r"\bdocstrings?\b|\bcomments?\b|\bcopyright\b", "cue:prose", 0),
    (r"\brename\b(?=.{0,40}\b(local|variable|parameter)\b)", "cue:local-rename", 0),
    (
        r"\breorder\b(?=.{0,30}\bimports?\b)|\bimports?\b(?=.{0,30}\bsort)",
        "cue:import-order",
        0,
    ),
    (r"\bwhitespace\b|\bindent\w*|\bblank line\b|\btrailing\b", "cue:whitespace", 0),
    (
        r"\bbump\b(?=.{0,30}\bversion\b)|\bversion\b(?=.{0,20}\bbump\b)",
        "cue:version-bump",
        0,
    ),
    (r"\bpin\b|\bpinned\b", "cue:dependency-pin", 0),
    (r"\bsilenc\w*", "cue:silence-warning", 0),
    (r"\bhelp text\b|\bhelp string\b", "cue:help-text", 0),
    (r"\bconstants?\b", "cue:constant", 0),
    (r"\bf-strings?\b", "cue:mechanical-rewrite", 0),
    (r"\bcommented[- ]out\b|\bdead code\b", "cue:dead-code", 0),
    (r"\btrace\b|\bwhich module\b|\breport the file\b", "cue:lookup", 0),
    (r"\blabels?\b|\bwording\b", "cue:label", 0),
    # tier 1 — contained, an existing pattern to follow
    (
        r"\b(add|adds|adding|support|accept)\b(?=.{0,40}\b(flags?|options?)\b)",
        "cue:add-flag",
        1,
    ),
    (r"\bbugs?\b|\bcrash\w*|\braises? \w*error\b", "cue:bugfix", 1),
    (
        (
            r"\b(add|adds|adding|write|writing|extend)\b(?=.{0,40}\btests?\b)"
            r"|\bunit tests?\b|\btest coverage\b"
        ),
        "cue:tests",
        1,
    ),
    (r"\bextract\w*", "cue:extract", 1),
    (r"\bport\b|\bports\b", "cue:port", 1),
    (r"\binstrument\w*", "cue:instrument", 1),
    (r"\btighten\w*|\btolerat\w*", "cue:robustness", 1),
    (r"\breproduc\w*", "cue:reproduce", 1),
    (r"\bwire\w*", "cue:wire-up", 1),
    (r"\bexpose\w*", "cue:expose", 1),
    (
        r"\binvestigat\w*|\bfind out why\b|\bwhy .{0,30}\b(dropped|regressed|broke)\b",
        "cue:investigate",
        1,
    ),
    (r"\bsplit\b(?=.{0,40}\binto\b)", "cue:split-module", 1),
    (r"\brefactor\w*", "cue:refactor", 1),
    (r"\benabl\w*", "cue:enable", 1),
    # tier 2 — the approach is the hard part, or it crosses subsystems
    (r"\bdesign(s|ed|ing)?\b", "cue:design", 2),
    (r"\barchitect\w*", "cue:architecture", 2),
    (r"\bmigrat\w*", "cue:migration", 2),
    (r"\bschema\b|\bsqlite\b|\bvector db\b", "cue:datastore", 2),
    (
        r"\bconcurren\w*|\brace condition\b|\bthread[- ]saf\w*|\bparalleli[sz]\w*",
        "cue:concurrency",
        2,
    ),
    (r"\bcredential\w*|\bsecrets?\b|\bapi keys?\b|\btokens?\b", "cue:credentials", 2),
    (r"\brotat\w*", "cue:rotate-secret", 2),
    (r"\bpublish\w*|\breleas\w*|\bpypi\b", "cue:publish", 2),
    (r"\bdeprecat\w*", "cue:deprecate", 2),
    (r"\bbackfill\w*", "cue:backfill", 2),
    (r"\brestor\w*", "cue:restore", 2),
    (r"\baudit\w*", "cue:audit", 2),
    (r"\bsecurity\b|\bharden\w*", "cue:security", 2),
    (r"\bintroduc\w*", "cue:new-subsystem", 2),
    (r"\borchestrat\w*", "cue:orchestration", 2),
    (
        r"\bevery\b|\ball three\b|\bwhole\b|\bacross all\b|\bcross[- ]cutting\b|\bentire\b",
        "cue:breadth",
        2,
    ),
    (r"\binterfaces?\b", "cue:shared-interface", 2),
)

_BLAST_CUES: tuple[tuple[str, str, str], ...] = (
    # low — wrong is cheap and local
    (
        r"\btypos?\b|\bdocstrings?\b|\bcomments?\b|\bcopyright\b",
        "reach:prose-only",
        "low",
    ),
    (
        r"\bwhitespace\b|\bindent\w*|\bblank line\b|\btrailing\b",
        "reach:cosmetic",
        "low",
    ),
    (r"\b(local|private) (variable|helper|function)\b", "reach:file-local", "low"),
    (r"\bunit tests?\b|\badd\b(?=.{0,30}\btests?\b)", "reach:test-only", "low"),
    (
        r"\btrace\b|\bwhich module\b|\breport the file\b|\binvestigat\w*|\baudit\w*",
        "reach:read-only",
        "low",
    ),
    (r"\bhelp text\b|\bhelp string\b|\blabels?\b", "reach:one-string", "low"),
    (r"\bflags?\b", "reach:additive-flag", "low"),
    (r"\bcommented[- ]out\b|\bsilenc\w*|\bf-strings?\b", "reach:one-file", "low"),
    # medium — wrong breaks callers inside this repo
    (r"\bconstants?\b|\bdefault\b", "reach:shared-default", "medium"),
    (r"\bpin\b|\bpinned\b|\bextra\b|\bmanifest\b", "reach:dependency", "medium"),
    (r"\bsubsystem\b|\bevery caller\b", "reach:subsystem", "medium"),
    # high — wrong escapes the repo, or tests cannot catch it
    (r"\bpublish\w*|\breleas\w*|\bpypi\b", "reach:published", "high"),
    (
        r"\bcredential\w*|\bsecrets?\b|\bapi keys?\b|\brotat\w*",
        "reach:credentials",
        "high",
    ),
    (
        r"\bmigrat\w*|\bschema\b|\bsqlite\b|\bbackfill\w*|\brestor\w*",
        "reach:stored-data",
        "high",
    ),
    (
        r"\bproduction\b|\bpublic api\b|\busers?\b|\bsubscribers?\b|\bemails?\b",
        "reach:user-visible",
        "high",
    ),
    (r"\bci\b|\bpipeline\b", "reach:ci", "high"),
    (
        r"\bevery\b|\ball three\b|\bwhole\b|\bacross all\b|\bentire\b",
        "reach:repo-wide",
        "high",
    ),
    (r"\binterfaces?\b|\borchestrat\w*|\bintroduc\w*", "reach:shared-surface", "high"),
    (
        r"\bconcurren\w*|\bthread[- ]saf\w*|\brace condition\b",
        "reach:corruption",
        "high",
    ),
    (r"\bdeprecat\w*", "reach:removes-a-route", "high"),
)

#: Reversibility is about the world, not the diff.
_IRREVERSIBLE_CUES: tuple[tuple[str, str], ...] = (
    (r"\bpublish\w*|\bpypi\b|\breleas\w*(?=.{0,30}\b(to|on)\b)", "world:published"),
    (
        r"\brotat\w*(?=.{0,40}\b(key|credential|secret|token)\w*)",
        "world:secret-rotated",
    ),
    (r"\bmigrat\w*", "world:schema-migrated"),
    (r"\bbackfill\w*", "world:records-rewritten"),
    (r"\brestor\w*(?=.{0,40}\b(backup|store|data)\w*)", "world:state-overwritten"),
    (r"\bdelet\w*|\bpurg\w*|\btruncat\w*|\bdrop\b", "world:data-destroyed"),
    (
        r"\bsend\w*(?=.{0,40}\b(email|mail|message|notification)\w*)",
        "world:message-sent",
    ),
    (r"\bmove\w*(?=.{0,40}\blogs?\b)", "world:records-relocated"),
)

_DATA_OBJECT = re.compile(
    r"\b(entr(y|ies)|records?|rows?|tables?|files?|partitions?|logs?|data|entries)\b"
)

#: A file path or a filename is a *reference*, not a description of the work.
#: `tests/test_registry.py` spells "test" three times, and a cue that reads those
#: would classify every edit to a test file as test-writing. Stripped before any
#: cue runs; prose survives, so "the run/ logs" keeps its "logs".
_PATH_TOKEN = re.compile(
    r"\S*(?:/\S+|\.(?:py|pyi|json|toml|md|ya?ml|txt|cfg|ini|lock|sh|jsonl))\b\S*"
)


_CueValue = TypeVar("_CueValue")


def _compile(
    table: Iterable[tuple[str, str, _CueValue]],
) -> tuple[tuple[re.Pattern[str], str, _CueValue], ...]:
    return tuple(
        (re.compile(p, re.IGNORECASE), signal, value) for p, signal, value in table
    )


_TIER_TABLE = _compile(_TIER_CUES)
_BLAST_TABLE = _compile(_BLAST_CUES)
_IRREVERSIBLE_TABLE = tuple(
    (re.compile(p, re.IGNORECASE), signal) for p, signal in _IRREVERSIBLE_CUES
)


def classify(task: str) -> Classification:
    """Score one task description.

    Raises:
        ValueError: the description is empty. A blank task would otherwise
            silently become the default tier, and "we didn't know" is a
            different answer from "it's a medium job".
    """
    if task is None or not task.strip():
        raise ValueError("task description must be a non-empty string")
    text = _PATH_TOKEN.sub(" ", task.strip())

    signals: list[str] = []

    tier_hits = [
        (value, signal)
        for pattern, signal, value in _TIER_TABLE
        if pattern.search(text)
    ]
    blast_hits = [
        (value, signal)
        for pattern, signal, value in _BLAST_TABLE
        if pattern.search(text)
    ]
    irreversible_hits = [
        signal
        for pattern, signal in _IRREVERSIBLE_TABLE
        if _fires(pattern, signal, text)
    ]

    # The rubric's tie rule: take the higher tier. Under-spending on a hard task
    # costs a wrong answer; over-spending costs tokens.
    if tier_hits:
        tier = max(value for value, _ in tier_hits)
        signals.extend(signal for _, signal in tier_hits)
    else:
        tier = DEFAULT_TIER
        signals.append("default:tier-1-no-cue")

    if blast_hits:
        blast = max((value for value, _ in blast_hits), key=BLAST_ORDER.index)
        signals.extend(signal for _, signal in blast_hits)
    else:
        blast = BLAST_FROM_TIER[tier]
        signals.append(f"default:blast-from-tier-{tier}")

    reversible = not irreversible_hits
    signals.extend(irreversible_hits)

    # An irreversible task is never tier 0 and never blast low, however small
    # its diff. This is the one place a floor is applied rather than a cue.
    if not reversible:
        if tier < 1:
            tier = 1
            signals.append("floor:irreversible-is-not-trivial")
        if blast == "low":
            blast = "medium"
            signals.append("floor:irreversible-is-not-local")

    return Classification(
        tier=tier,
        blast=blast,
        reversible=reversible,
        signals=tuple(dict.fromkeys(signals)),
        why=_explain(tier, blast, reversible),
    )


def _fires(pattern: re.Pattern[str], signal: str, text: str) -> bool:
    """Irreversibility cues, with one guard.

    Destructive verbs are only irreversible when there is something to destroy:
    "remove the commented-out block" deletes code, which a revert restores,
    while "delete the archived entries" does not.
    """
    if not pattern.search(text):
        return False
    if signal == "world:data-destroyed":
        return bool(_DATA_OBJECT.search(text))
    return True


def _explain(tier: int, blast: str, reversible: bool) -> str:
    tier_says = {
        0: "one obvious edit, no design choice left",
        1: "contained work with a pattern to follow",
        2: "the approach is the hard part, or it crosses subsystems",
    }[tier]
    blast_says = {
        "low": "wrong is cheap and local",
        "medium": "wrong breaks callers inside this repo",
        "high": "wrong escapes the repo or tests cannot catch it",
    }[blast]
    tail = "" if reversible else "; irreversible, so a revert will not undo it"
    return f"tier {tier}: {tier_says}. Blast {blast}: {blast_says}{tail}."
