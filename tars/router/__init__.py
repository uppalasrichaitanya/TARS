"""TARS router-classifier — PHASE 2 (CLASSIFY), isolated.

Given a task description, three answers before any model is called:

    from tars.router import classify

    verdict = classify("Migrate the memory store from JSON to SQLite.")
    verdict.tier        # 0 | 1 | 2 — how much machinery to spend deciding
    verdict.blast       # low | medium | high — what a wrong answer costs
    verdict.reversible  # False: no diff un-migrates data
    verdict.signals     # the cues that produced all three
    verdict.why         # one sentence, in the rubric's own words

The three are scored separately because they answer different questions. A
one-token change to a shared default is the smallest possible diff with a
repo-wide reach — tier 0, blast medium — and collapsing them into one number
loses exactly the case worth catching:

    from tars.router import classify

    classify("Change the default max_tokens constant from 16000 to 8192.").blast

Nothing here calls a model, reads a credential, or imports the provider layer.
That is the point: CLASSIFY runs before the expensive decision, on every task
including the trivial ones, so it has to cost nothing. It is cue matching over
text, which has a real ceiling — `tests/test_router_benchmark.py` prints the
confusion matrix per split rather than reporting a single number:

    python tests/test_router_benchmark.py

The labeled set it is measured against ships with the rubric that produced it,
so a label can be argued with without reading any Python:

    from tars.router import load_dataset, rubric

    len(load_dataset())
    rubric()["ties"]

Per the spec's guardrail, this is a signal to the loop and not a cage: when the
router and the loop's own judgment disagree, the loop is expected to trust its
judgment and say so.
"""

from __future__ import annotations

from .classify import (
    BLAST_FROM_TIER,
    BLAST_ORDER,
    DEFAULT_TIER,
    Classification,
    classify,
)
from .dataset import (
    VALID_BLAST,
    VALID_SPLITS,
    VALID_TIERS,
    DatasetError,
    LabeledTask,
    dev_split,
    holdout_split,
    load_dataset,
    rubric,
    split,
)

__all__ = [
    "BLAST_FROM_TIER",
    "BLAST_ORDER",
    "DEFAULT_TIER",
    "VALID_BLAST",
    "VALID_SPLITS",
    "VALID_TIERS",
    "Classification",
    "DatasetError",
    "LabeledTask",
    "classify",
    "dev_split",
    "holdout_split",
    "load_dataset",
    "rubric",
    "split",
]
