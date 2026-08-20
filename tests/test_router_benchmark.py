"""What the router actually gets right, measured, with the matrix printed.

    python tests/test_router_benchmark.py     # the scorecard, by split

**An honest caveat about the holdout.** The labeled set and the classifier's
cues were written in the same sitting by the same author, so `holdout_split()`
is not a set the rules have never seen — it is a *structural* holdout: balanced,
disjoint, and recorded per case. Its value is forward-looking. The next person to
tune a cue can move dev and check holdout to find out whether they generalised
or just fitted. Read today's holdout number as "the rules aren't overfit to
individual dev cases", not as "unseen-data performance".

The floors below were fixed before any number was measured, and they are floors
rather than targets: a cue-only classifier has a real ceiling, and the point of
printing the matrix is to show where it sits instead of quietly tuning until a
number looks good. Where a rule would only exist to fix one case, the miss was
left in and appears below.
"""

from __future__ import annotations

import pytest

from tars.router.classify import classify
from tars.router.dataset import dev_split, holdout_split, load_dataset

# ── floors, fixed in advance ─────────────────────────────────────────────
# The majority-class baseline is ~0.33 for tier and ~0.42 for blast, so these
# are set where a rule-based classifier has to be clearly better than guessing
# the biggest class, without pretending cue matching can reach the high 90s.

MIN_TIER_ACCURACY = 0.70
MIN_TIER_WITHIN_ONE = 0.95
MIN_BLAST_ACCURACY = 0.65
MIN_REVERSIBLE_ACCURACY = 0.90
MIN_TIER_LIFT_OVER_MAJORITY = 0.25

TIERS = (0, 1, 2)
BLASTS = ("low", "medium", "high")

SPLITS = {"dev": dev_split, "holdout": holdout_split}


def score(cases) -> dict:
    """Predict every case once and total up the ways it can be wrong."""
    rows = [(case, classify(case.task)) for case in cases]
    total = len(rows)

    tier_right = sum(1 for c, p in rows if p.tier == c.tier)
    within_one = sum(1 for c, p in rows if abs(p.tier - c.tier) <= 1)
    blast_right = sum(1 for c, p in rows if p.blast == c.blast)
    rev_right = sum(1 for c, p in rows if p.reversible == c.reversible)

    majority = max(sum(1 for c, _ in rows if c.tier == t) for t in TIERS) / total

    return {
        "total": total,
        "tier_accuracy": tier_right / total,
        "tier_within_one": within_one / total,
        "blast_accuracy": blast_right / total,
        "reversible_accuracy": rev_right / total,
        "tier_majority_baseline": majority,
        "tier_matrix": {
            actual: {
                predicted: sum(
                    1 for c, p in rows if c.tier == actual and p.tier == predicted
                )
                for predicted in TIERS
            }
            for actual in TIERS
        },
        "blast_matrix": {
            actual: {
                predicted: sum(
                    1 for c, p in rows if c.blast == actual and p.blast == predicted
                )
                for predicted in BLASTS
            }
            for actual in BLASTS
        },
        "tier_misses": [
            {
                "id": c.id,
                "task": c.task,
                "expected": c.tier,
                "got": p.tier,
                "signals": p.signals,
            }
            for c, p in rows
            if p.tier != c.tier
        ],
        "blast_misses": [
            {"id": c.id, "task": c.task, "expected": c.blast, "got": p.blast}
            for c, p in rows
            if p.blast != c.blast
        ],
        "reversible_misses": [
            {"id": c.id, "task": c.task, "expected": c.reversible, "got": p.reversible}
            for c, p in rows
            if p.reversible != c.reversible
        ],
    }


@pytest.fixture(scope="module")
def scored() -> dict:
    return {name: score(fn()) for name, fn in SPLITS.items()}


# ── the floors ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("split", list(SPLITS))
def test_tier_is_predicted_better_than_guessing_the_biggest_class(scored, split):
    s = scored[split]
    lift = s["tier_accuracy"] - s["tier_majority_baseline"]
    assert lift >= MIN_TIER_LIFT_OVER_MAJORITY, _table(split, s)


@pytest.mark.parametrize("split", list(SPLITS))
def test_tier_accuracy_clears_its_floor(scored, split):
    assert scored[split]["tier_accuracy"] >= MIN_TIER_ACCURACY, _table(
        split, scored[split]
    )


@pytest.mark.parametrize("split", list(SPLITS))
def test_the_router_is_almost_never_wrong_by_two_whole_tiers(scored, split):
    """Off-by-one wastes tokens. Off-by-two skips planning on a migration."""
    assert scored[split]["tier_within_one"] >= MIN_TIER_WITHIN_ONE, _table(
        split, scored[split]
    )


@pytest.mark.parametrize("split", list(SPLITS))
def test_blast_accuracy_clears_its_floor(scored, split):
    assert scored[split]["blast_accuracy"] >= MIN_BLAST_ACCURACY, _table(
        split, scored[split]
    )


@pytest.mark.parametrize("split", list(SPLITS))
def test_irreversibility_is_the_label_it_can_least_afford_to_miss(scored, split):
    """Held highest on purpose: a missed irreversible task is the one failure
    the loop cannot recover from by trying again."""
    assert scored[split]["reversible_accuracy"] >= MIN_REVERSIBLE_ACCURACY, _table(
        split, scored[split]
    )


def test_the_two_splits_do_not_diverge_wildly():
    """A large dev/holdout gap is the signature of cues fitted to single cases."""
    dev, hold = score(dev_split()), score(holdout_split())
    assert abs(dev["tier_accuracy"] - hold["tier_accuracy"]) <= 0.25


def test_the_matrix_accounts_for_every_case(scored):
    for split, s in scored.items():
        counted = sum(sum(row.values()) for row in s["tier_matrix"].values())
        assert counted == s["total"], split


def test_the_whole_corpus_is_covered_by_the_two_splits():
    assert len(dev_split()) + len(holdout_split()) == len(load_dataset())


# ── the scorecard ────────────────────────────────────────────────────────


def _matrix_block(title: str, matrix: dict, labels) -> list[str]:
    width = max(len(str(label)) for label in labels) + 2
    lines = [
        f"  {title}",
        "    actual \\ predicted   "
        + "".join(str(label).rjust(width) for label in labels),
    ]
    for actual in labels:
        row = matrix[actual]
        cells = "".join(str(row[p]).rjust(width) for p in labels)
        lines.append(f"    {str(actual).ljust(20)}  {cells}")
    return lines


def _table(split: str, s: dict) -> str:
    lines = [
        "",
        f"split '{split}' — {s['total']} cases",
        "",
        (
            f"  tier accuracy        {s['tier_accuracy']:.2f}   "
            f"(floor {MIN_TIER_ACCURACY:.2f}, "
            f"majority baseline {s['tier_majority_baseline']:.2f})"
        ),
        f"  tier within one      {s['tier_within_one']:.2f}   (floor {MIN_TIER_WITHIN_ONE:.2f})",
        f"  blast accuracy       {s['blast_accuracy']:.2f}   (floor {MIN_BLAST_ACCURACY:.2f})",
        (
            f"  reversible accuracy  {s['reversible_accuracy']:.2f}   "
            f"(floor {MIN_REVERSIBLE_ACCURACY:.2f})"
        ),
        "",
    ]
    lines += _matrix_block("tier", s["tier_matrix"], TIERS) + [""]
    lines += _matrix_block("blast", s["blast_matrix"], BLASTS) + [""]

    for label, key in (
        ("tier misses", "tier_misses"),
        ("blast misses", "blast_misses"),
        ("reversibility misses", "reversible_misses"),
    ):
        rows = s[key]
        lines.append(f"  {label}: {len(rows)}")
        for row in rows:
            lines.append(
                f"    [{row['expected']} -> {row['got']}] {row['id']}  {row['task'][:68]}"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - a scorecard, not a test
    for _name, _fn in SPLITS.items():
        print(_table(_name, score(_fn())))
