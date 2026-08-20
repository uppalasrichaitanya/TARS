"""The corpus benchmark: what this phase's judgement is actually worth.

Every other test file here asserts a property. This one measures an accuracy,
against `tests/corpus/memory_statements.json` — statements hand-labelled from
core directive 4, not from the cue tables. `signals.py` says its tables are
"kept in one place so the corpus can argue with it"; this is the corpus.

Two layers get scored, because they fail differently:

* **the detector** — `kind_of(read_signals(text))`, which decides whether a
  statement is memory-worthy at all and which of the four kinds it is;
* **the disposition** — what `propose` would actually do with it: commit it,
  queue it for a human, or drop it.

The floors below are not symmetric, and the asymmetry is the point. A missed
entry costs one re-derivation. A wrong entry gets believed without one. So
precision is held higher than recall, and one rule has no tolerance at all:
**nothing the corpus calls not-memory-worthy may auto-commit** — checked in the
most generous configuration the scorer offers, not the average one.

Run it as a script for the scorecard, including every disagreement:

    python tests/test_memory_benchmark.py
"""

import json
from pathlib import Path

import pytest

from tars.memory.domains import load_domain_map
from tars.memory.propose import propose
from tars.memory.signals import kind_of, read_signals
from tars.memory.transcript import Transcript, Turn

CORPUS_PATH = Path(__file__).resolve().parent / "corpus" / "memory_statements.json"

#: Held higher than recall on purpose: see the module docstring. These are floors
#: a regression trips, not targets — the scorecard prints what is actually
#: measured, and the commit that moves a floor should say why.
MIN_DETECTOR_PRECISION = 0.80
MIN_DETECTOR_RECALL = 0.70
MIN_KIND_ACCURACY = 0.85
MIN_STORE_PRECISION = 0.80


def _load() -> list[dict]:
    with CORPUS_PATH.open() as handle:
        return json.load(handle)["statements"]


CORPUS = _load()


def _expected(item: dict) -> tuple[str, ...]:
    """The kinds a person would accept for this statement; empty if none."""
    expect = item["expect"]
    if expect is None:
        return ()
    return (expect,) if isinstance(expect, str) else tuple(expect)


@pytest.fixture(scope="module")
def domains(tmp_path_factory):
    # The built-in map, read through the same loader the CLI uses, from a
    # directory with no project override — so the score is the shipped default's
    # and not this repo's.
    return load_domain_map(str(tmp_path_factory.mktemp("no-overrides")))


def _disposition(text: str, domains, *, generous: bool = False) -> str:
    """What `propose` would do with this one statement, on its own.

    One statement per transcript, so nothing is scored by its neighbours.
    `generous=True` is the most favourable reading the scorer offers — a phase
    heading and the role weight that pays most — which is the setting the
    no-false-commit rule is checked under.
    """
    turn = Turn(
        index=1,
        role="user",
        text=text,
        phase="REFLECT" if generous else None,
    )
    report = propose(Transcript(session="corpus", turns=(turn,)), domains=domains)
    if not report.candidates:
        return "none"
    return report.candidates[0].disposition


def _score(domains) -> dict:
    """The whole corpus, one pass, both layers."""
    rows = []
    for item in CORPUS:
        wanted = _expected(item)
        predicted = kind_of(read_signals(item["text"]))
        rows.append(
            {
                "text": item["text"],
                "note": item.get("note", ""),
                "expect": wanted,
                "predicted": predicted,
                "disposition": _disposition(item["text"], domains),
            }
        )

    gold = [r for r in rows if r["expect"]]
    junk = [r for r in rows if not r["expect"]]
    detected = [r for r in rows if r["predicted"] is not None]
    true_detections = [r for r in gold if r["predicted"] is not None]
    right_kind = [r for r in true_detections if r["predicted"] in r["expect"]]
    stored = [r for r in rows if r["disposition"] in ("commit", "review")]
    stored_gold = [r for r in stored if r["expect"]]

    return {
        "rows": rows,
        "total": len(rows),
        "gold": len(gold),
        "junk": len(junk),
        "detector_precision": len(true_detections) / len(detected) if detected else 1.0,
        "detector_recall": len(true_detections) / len(gold),
        "kind_accuracy": (
            len(right_kind) / len(true_detections) if true_detections else 1.0
        ),
        "store_precision": len(stored_gold) / len(stored) if stored else 1.0,
        "store_recall": len(stored_gold) / len(gold),
        "misses": [r for r in gold if r["predicted"] is None],
        "false_positives": [r for r in junk if r["predicted"] is not None],
        "wrong_kind": [r for r in true_detections if r["predicted"] not in r["expect"]],
        "false_stores": [r for r in stored if not r["expect"]],
    }


@pytest.fixture(scope="module")
def scored(domains):
    return _score(domains)


# ── the corpus itself ─────────────────────────────────────────────────────


def test_the_corpus_is_big_enough_to_mean_something():
    # A dozen statements can be satisfied by a dozen cue phrases; a measurement
    # that small measures the tables, not the judgement.
    assert len(CORPUS) >= 60


def test_the_corpus_labels_both_outcomes_in_bulk():
    # A corpus of nothing but memory-worthy statements can't catch a detector
    # that says yes to everything, which is the failure that matters here.
    gold = sum(1 for item in CORPUS if _expected(item))
    assert gold >= 25
    assert len(CORPUS) - gold >= 25


def test_every_label_is_one_of_the_four_kinds_or_nothing():
    allowed = {"decision", "constraint", "rejected", "gotcha"}
    for item in CORPUS:
        assert set(_expected(item)) <= allowed, item["text"]


def test_no_statement_is_labelled_twice():
    texts = [item["text"] for item in CORPUS]
    assert len(texts) == len(set(texts))


def test_the_corpus_includes_statements_carrying_no_cue_phrase():
    # The point of a corpus is to measure what the detector misses. One written
    # only from the cue tables would score 100% and mean nothing, so at least
    # some memory-worthy statements have to be ones no cue fires on.
    uncued = [
        item
        for item in CORPUS
        if _expected(item) and kind_of(read_signals(item["text"])) is None
    ]
    assert uncued, "every labelled statement matches a cue — the corpus is circular"


# ── the safety property ───────────────────────────────────────────────────


def test_nothing_the_corpus_calls_routine_can_auto_commit(domains):
    # The one rule with no tolerance, checked in the scorer's most generous
    # setting rather than its average one: a false commit is a wrong fact that
    # a future session acts on without re-deriving it.
    committed = [
        item["text"]
        for item in CORPUS
        if not _expected(item)
        and _disposition(item["text"], domains, generous=True) == "commit"
    ]
    assert committed == []


def test_a_question_is_never_stored_however_it_is_phrased(domains):
    questions = [item for item in CORPUS if item["text"].rstrip().endswith("?")]
    assert questions, "the corpus should include questions"
    for item in questions:
        text = item["text"]
        assert _disposition(text, domains, generous=True) == "none", text


# ── the measurement ───────────────────────────────────────────────────────


def test_the_detector_is_precise_enough_to_be_believed(scored):
    assert scored["detector_precision"] >= MIN_DETECTOR_PRECISION, _table(scored)


def test_the_detector_finds_most_of_what_a_person_would_keep(scored):
    assert scored["detector_recall"] >= MIN_DETECTOR_RECALL, _table(scored)


def test_a_detected_statement_is_filed_under_the_right_kind(scored):
    # Recall for the wrong kind is worse than a miss: `recall(kinds=...)` is how
    # ORIENT asks a narrow question, and a constraint filed as a decision answers
    # the wrong one.
    assert scored["kind_accuracy"] >= MIN_KIND_ACCURACY, _table(scored)


def test_most_of_what_reaches_the_store_deserves_to_be_there(scored):
    # Queued entries cost a human's attention even when they never commit, so
    # this is measured over commit *and* review.
    assert scored["store_precision"] >= MIN_STORE_PRECISION, _table(scored)


# ── the scorecard ─────────────────────────────────────────────────────────


def _table(scored: dict) -> str:
    lines = [
        "",
        (
            f"corpus: {scored['total']} statements "
            f"({scored['gold']} memory-worthy, {scored['junk']} not)"
        ),
        "",
        (
            f"  detector precision  {scored['detector_precision']:.2f}"
            f"   (floor {MIN_DETECTOR_PRECISION:.2f})"
        ),
        (
            f"  detector recall     {scored['detector_recall']:.2f}"
            f"   (floor {MIN_DETECTOR_RECALL:.2f})"
        ),
        (
            f"  kind accuracy       {scored['kind_accuracy']:.2f}"
            f"   (floor {MIN_KIND_ACCURACY:.2f})"
        ),
        (
            f"  store precision     {scored['store_precision']:.2f}"
            f"   (floor {MIN_STORE_PRECISION:.2f})"
        ),
        f"  store recall        {scored['store_recall']:.2f}",
        "",
    ]
    for label, key in (
        ("missed (labelled memory-worthy, detected nothing)", "misses"),
        ("false positives (labelled routine, detected something)", "false_positives"),
        ("wrong kind", "wrong_kind"),
        ("reached the store but should not have", "false_stores"),
    ):
        rows = scored[key]
        lines.append(f"{label}: {len(rows)}")
        for row in rows:
            expect = "/".join(row["expect"]) or "-"
            lines.append(
                f"    [{expect} → {row['predicted']}/{row['disposition']}] "
                f"{row['text'][:76]}"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - a scorecard, not a test
    import tempfile

    with tempfile.TemporaryDirectory() as empty:
        print(_table(_score(load_domain_map(empty))))
