"""The classifier's rules, one claim at a time.

These tests assert *rules*, not accuracy. Accuracy over the labeled set lives in
test_router_benchmark.py, and the two are kept apart on purpose: a rule that is
right for a stated reason survives a corpus change, and a rule that only exists
to move a number does not.

Every claim here is traceable to the rubric shipped in
`tars/router/data/labeled_tasks.json`.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tars.router.classify import Classification, classify


def tier_of(task: str) -> int:
    return classify(task).tier


def blast_of(task: str) -> str:
    return classify(task).blast


# ── tier: the amount of machinery the loop should spend ──────────────────


@pytest.mark.parametrize(
    "task",
    [
        "Fix the typo 'recieve' in the docs.",
        "Add a docstring to the helper.",
        "Reorder the imports so ruff stops complaining.",
        "Rename the local variable tmp to payload.",
    ],
)
def test_edits_with_no_decision_left_are_tier_zero(task):
    assert tier_of(task) == 0


@pytest.mark.parametrize(
    "task",
    [
        "Design the loop orchestrator that wires the modules together.",
        "Migrate the store from JSON to SQLite.",
        "Harden the store against two runs writing concurrently.",
        "Rotate the API key and update every place that reads it.",
    ],
)
def test_design_bearing_and_cross_cutting_work_is_tier_two(task):
    assert tier_of(task) == 2


@pytest.mark.parametrize(
    "task",
    [
        "Add a --limit flag to the recall command.",
        "Fix the bug where an empty file raises IndexError.",
        "Add unit tests for the supersede path.",
    ],
)
def test_contained_work_with_an_obvious_shape_is_tier_one(task):
    assert tier_of(task) == 1


def test_an_unrecognised_task_lands_in_the_middle_tier():
    """Tier 1 is the honest default.

    Guessing 0 tells the loop to skip planning on something it has not
    understood; guessing 2 burns a full plan on a one-line edit. The middle is
    wrong by at most one step in either direction.
    """
    assert tier_of("Handle the frobnicator's quarterly reconciliation.") == 1


def test_the_higher_tier_wins_when_cues_disagree():
    """The rubric's tie rule: under-spending costs a wrong answer."""
    result = classify("Fix the typo in the migration that rewrites every stored entry.")
    assert result.tier == 2


# ── blast radius: what it costs if the change is wrong ───────────────────


@pytest.mark.parametrize(
    "task",
    [
        "Publish version 0.2.0 to PyPI.",
        "Rotate the AgentRouter credential.",
        "Delete the superseded entries from every partition file.",
        "Migrate the store's schema.",
    ],
)
def test_work_that_escapes_the_repo_is_high_blast(task):
    assert blast_of(task) == "high"


@pytest.mark.parametrize(
    "task",
    [
        "Fix the typo in the README.",
        "Add a docstring to the private helper.",
        "Add unit tests for the store's supersede path.",
    ],
)
def test_local_and_test_only_work_is_low_blast(task):
    assert blast_of(task) == "low"


def test_a_read_only_investigation_is_low_blast():
    """Nothing is written, so nothing can break."""
    assert (
        blast_of("Trace which module defines DEFAULT_MODEL and report the file.")
        == "low"
    )


def test_tier_and_blast_are_scored_independently():
    """The rubric's worked example, and the reason there are two axes.

    A one-token edit to a default every adapter reads is the smallest possible
    diff with a repo-wide reach.
    """
    result = classify("Change the default max_tokens constant from 16000 to 8192.")
    assert result.tier == 0
    assert result.blast == "medium"


# ── reversibility: about the world, not the diff ─────────────────────────


@pytest.mark.parametrize(
    "task",
    [
        "Publish the release to PyPI.",
        "Rotate the API key.",
        "Delete the archived entries from disk.",
        "Backfill domain tags on the stored entries.",
        "Migrate the schema to SQLite.",
    ],
)
def test_changes_the_world_keeps_are_marked_irreversible(task):
    assert classify(task).reversible is False


@pytest.mark.parametrize(
    "task",
    [
        "Fix the typo in the README.",
        "Extract the duplicated helper into one function.",
        "Design the loop orchestrator.",
    ],
)
def test_code_only_changes_stay_reversible(task):
    assert classify(task).reversible is True


def test_an_irreversible_task_is_never_tier_zero_however_small():
    """The rubric's floor. `git revert` does not un-send or un-delete."""
    result = classify("Delete the stored entries older than a year.")
    assert result.reversible is False
    assert result.tier >= 1


def test_an_irreversible_task_is_never_low_blast():
    result = classify("Send the summary email to every subscriber.")
    assert result.reversible is False
    assert result.blast in ("medium", "high")


# ── the output is meant to be argued with ────────────────────────────────


def test_every_classification_names_the_signals_behind_it():
    """A tier nobody can take apart is a tier nobody can correct."""
    result = classify("Migrate the memory store to SQLite.")
    assert result.signals
    assert all(isinstance(s, str) and s for s in result.signals)
    assert any("migrat" in s for s in result.signals)


def test_the_default_says_it_is_a_default_rather_than_inventing_a_reason():
    result = classify("Handle the frobnicator's quarterly reconciliation.")
    assert "default" in " ".join(result.signals)


def test_the_rationale_is_a_sentence_not_an_empty_string():
    assert classify("Design the orchestrator.").why.strip()


def test_a_classification_is_immutable():
    result = classify("Fix the typo.")
    with pytest.raises(FrozenInstanceError):
        result.tier = 2  # type: ignore[misc]


# ── input handling ───────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_an_empty_task_is_rejected_rather_than_classified(bad):
    """A blank description would otherwise silently become the default tier."""
    with pytest.raises(ValueError, match="task"):
        classify(bad)


def test_classification_is_case_insensitive():
    assert (
        classify("MIGRATE THE SCHEMA TO SQLITE").tier
        == classify("migrate the schema to sqlite").tier
    )


def test_cues_do_not_fire_on_substrings_of_unrelated_words():
    """`design` must not match inside `designation`, or every cue is a coin flip."""
    result = classify("Update the designation field's label in the form.")
    assert result.tier == 0


def test_the_same_task_always_classifies_the_same_way():
    """The router is a signal to the loop; a nondeterministic one is noise."""
    task = "Refactor every adapter onto a streaming interface."
    assert classify(task) == classify(task)


def test_the_result_is_the_documented_type():
    assert isinstance(classify("Fix the typo."), Classification)


# ── words inside paths and identifiers are not prose ─────────────────────


def test_a_filename_does_not_trigger_the_cues_spelled_inside_it():
    """Real task descriptions name files constantly.

    `tests/test_registry.py` contains "test" three times, which would otherwise
    make every edit to a test file look like test-writing work.
    """
    result = classify("Silence the unused-import warning in tests/test_registry.py.")
    assert result.tier == 0, result.signals


@pytest.mark.parametrize(
    "task",
    [
        "Rename the local variable tmp to payload in tars/memory/store.py.",
        "Remove the commented-out block at the bottom of tars/verification/detect.py.",
        "Reorder the imports in tars/readme/cli.py so ruff stops complaining.",
    ],
)
def test_a_path_does_not_drag_a_trivial_edit_up_a_tier(task):
    assert tier_of(task) == 0


def test_naming_a_flag_is_not_the_same_as_adding_one():
    """ "Document the X flag" is prose work; "add a --limit flag" is a feature."""
    assert (
        tier_of("Document the TARS_LIVE flag in the live test's module docstring.") == 0
    )
    assert tier_of("Add a --limit flag to the recall command.") == 1


def test_mentioning_a_test_file_is_not_the_same_as_writing_tests():
    assert tier_of("Add unit tests for the supersede path.") == 1
    assert tier_of("Fix the typo in the docstring of tests/test_store.py.") == 0


def test_stripping_paths_does_not_lose_a_cue_that_only_appears_in_prose():
    """`run/` is a path, but "logs" next to "move" is what makes this irreversible."""
    result = classify("Move the run/ logs into a proper SQLite run history.")
    assert result.reversible is False
    assert result.tier == 2
