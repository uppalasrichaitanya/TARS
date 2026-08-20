"""The labeled set is the measuring stick, so it gets checked like one.

These tests never assert an accuracy number. They assert the dataset is
well-formed, that the two splits are usable as splits, and that the labels obey
the rubric they claim to follow. Accuracy lives in test_router_benchmark.py.
"""

from __future__ import annotations

import json

import pytest

from tars.router.dataset import (
    LabeledTask,
    dev_split,
    holdout_split,
    load_dataset,
)


@pytest.fixture(scope="module")
def cases() -> tuple[LabeledTask, ...]:
    return load_dataset()


def test_dataset_loads_labeled_tasks(cases: tuple[LabeledTask, ...]) -> None:
    assert len(cases) >= 40, "too few cases to report a meaningful accuracy"
    assert all(isinstance(c, LabeledTask) for c in cases)


def test_every_case_carries_a_task_description(cases: tuple[LabeledTask, ...]) -> None:
    for case in cases:
        assert case.task.strip(), f"{case.id} has an empty task description"


def test_labels_are_inside_the_taxonomy(cases: tuple[LabeledTask, ...]) -> None:
    for case in cases:
        assert case.tier in (0, 1, 2), f"{case.id} has tier {case.tier}"
        assert case.blast in ("low", "medium", "high"), f"{case.id} blast {case.blast}"
        assert isinstance(case.reversible, bool)


def test_case_ids_are_unique(cases: tuple[LabeledTask, ...]) -> None:
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_every_case_explains_its_label(cases: tuple[LabeledTask, ...]) -> None:
    # A label with no stated reason can't be argued with, which means it can't
    # be corrected either.
    for case in cases:
        assert case.why.strip(), f"{case.id} has no rationale for its label"


def test_splits_partition_the_dataset(cases: tuple[LabeledTask, ...]) -> None:
    dev, holdout = dev_split(), holdout_split()
    assert len(dev) + len(holdout) == len(cases)
    assert not {c.id for c in dev} & {c.id for c in holdout}


def test_both_splits_are_big_enough_to_mean_something() -> None:
    assert len(dev_split()) >= 20
    assert len(holdout_split()) >= 20


def test_every_tier_appears_in_both_splits() -> None:
    for split in (dev_split(), holdout_split()):
        tiers = {c.tier for c in split}
        assert tiers == {0, 1, 2}, (
            f"split is missing tiers: {sorted({0, 1, 2} - tiers)}"
        )


def test_every_blast_level_appears_in_both_splits() -> None:
    for split in (dev_split(), holdout_split()):
        levels = {c.blast for c in split}
        assert levels == {"low", "medium", "high"}


def test_both_splits_contain_an_irreversible_case() -> None:
    # Irreversibility is the label most likely to be tested by nothing at all,
    # because irreversible tasks are rare in any honestly-sampled set.
    for split in (dev_split(), holdout_split()):
        assert any(not c.reversible for c in split)


def test_no_tier_dominates_the_dataset(cases: tuple[LabeledTask, ...]) -> None:
    # A set that is 80% tier 1 makes "always guess 1" look like a good
    # classifier. The majority-class baseline is reported alongside accuracy
    # for the same reason, but skew is worth preventing at the source too.
    for tier in (0, 1, 2):
        share = sum(1 for c in cases if c.tier == tier) / len(cases)
        assert 0.2 <= share <= 0.45, f"tier {tier} is {share:.0%} of the dataset"


def test_the_rubric_ships_with_the_data() -> None:
    # The labels are only reproducible if the rules used to write them are
    # readable next to them.
    raw = json.loads(_raw_text())
    assert raw["rubric"]["tier"].keys() == {"0", "1", "2"}
    assert raw["rubric"]["blast"].keys() == {"low", "medium", "high"}
    assert raw["rubric"]["reversible"]
    assert raw["rubric"]["ties"]


def test_task_descriptions_are_not_all_phrased_the_same_way(
    cases: tuple[LabeledTask, ...],
) -> None:
    # If every tier-0 case starts with "Run" and every tier-2 case starts with
    # "Design", the classifier is being handed the answer by the sampling.
    first_words = [c.task.split()[0].lower() for c in cases]
    assert len(set(first_words)) >= 15, "descriptions are too templated to test on"


def _raw_text() -> str:
    from importlib.resources import files

    return (files("tars.router") / "data" / "labeled_tasks.json").read_text("utf-8")
