"""The ACT phase's vocabulary: what to do, and what happened.

These types exist so PHASE 4 has something to say other than "it worked". The
spec's rule is that a failure must carry *why* — `master_agent_loop.md:119`,
"capture WHY (not-found vs timeout vs assertion mismatch) … never blind-retry" —
and a result type that can hold `ok=False` with no reason makes that rule
unenforceable. So the invariant is pinned here rather than trusted to callers.
"""

from __future__ import annotations

import pytest

from tars.act.types import (
    DEFAULT_COMMAND_TIMEOUT_S,
    ActionResult,
    Command,
    Edit,
    Hunk,
)

# ── Edit: four shapes, one type ──────────────────────────────────────────


def test_a_replace_edit_carries_its_hunks():
    edit = Edit.replace("a.py", Hunk("old", "new"))
    assert edit.kind == "replace"
    assert edit.path == "a.py"
    assert edit.hunks == (Hunk("old", "new"),)
    assert edit.content is None


def test_a_create_edit_carries_whole_content():
    edit = Edit.create("new.py", "print(1)\n")
    assert edit.kind == "create"
    assert edit.content == "print(1)\n"
    assert edit.hunks == ()


def test_an_overwrite_edit_is_distinct_from_a_create():
    """Different intents: one asserts absence, the other asserts presence.

    Collapsing them means a typo'd path silently creates a file instead of
    failing, which is the kind of mistake nobody notices for a week.
    """
    assert Edit.overwrite("a.py", "x = 1\n").kind == "overwrite"
    assert Edit.create("a.py", "x = 1\n").kind == "create"


def test_a_delete_edit_needs_neither_hunks_nor_content():
    edit = Edit.delete("gone.py")
    assert edit.kind == "delete"
    assert edit.hunks == ()
    assert edit.content is None


def test_a_replace_with_no_hunks_is_rejected():
    """An edit that changes nothing is a caller bug, not a no-op to absorb."""
    with pytest.raises(ValueError, match="hunk"):
        Edit(path="a.py", kind="replace")


def test_a_create_with_no_content_is_rejected():
    with pytest.raises(ValueError, match="content"):
        Edit(path="a.py", kind="create")


def test_a_delete_carrying_content_is_rejected():
    with pytest.raises(ValueError, match="content"):
        Edit(path="a.py", kind="delete", content="x")


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="kind"):
        Edit(path="a.py", kind="mangle")  # type: ignore[arg-type]


def test_an_empty_path_is_rejected():
    with pytest.raises(ValueError, match="path"):
        Edit.delete("   ")


def test_hunks_are_normalized_to_a_tuple_so_an_edit_stays_hashable():
    edit = Edit(path="a.py", kind="replace", hunks=[Hunk("a", "b")])
    assert isinstance(edit.hunks, tuple)
    assert hash(edit)


def test_a_hunk_that_changes_nothing_is_rejected():
    with pytest.raises(ValueError, match="identical"):
        Hunk("same", "same")


def test_a_hunk_with_an_empty_anchor_is_rejected():
    """An empty anchor matches everywhere, which is the worst kind of match."""
    with pytest.raises(ValueError, match="anchor"):
        Hunk("", "new")


def test_deleting_a_snippet_is_expressed_as_an_empty_replacement():
    assert Hunk("old\n", "").new == ""


# ── Command: argv, never a shell string ──────────────────────────────────


def test_a_command_keeps_its_argv_as_a_tuple():
    command = Command(["pytest", "-q"])
    assert command.argv == ("pytest", "-q")


def test_a_bare_string_command_is_rejected():
    """`Command("ls -la")` would otherwise iterate into single characters.

    Refusing it here is what keeps the argv-only rule from being a convention
    that one careless call site quietly breaks.
    """
    with pytest.raises(TypeError, match="argv"):
        Command("ls -la")  # type: ignore[arg-type]


def test_an_empty_command_is_rejected():
    with pytest.raises(ValueError, match="argv"):
        Command([])


def test_a_command_has_a_default_timeout():
    assert Command(["pytest"]).timeout == DEFAULT_COMMAND_TIMEOUT_S
    assert DEFAULT_COMMAND_TIMEOUT_S > 0


def test_a_command_timeout_can_be_removed_deliberately():
    assert Command(["sleep", "1"], timeout=None).timeout is None


def test_a_non_positive_timeout_is_rejected():
    with pytest.raises(ValueError, match="timeout"):
        Command(["pytest"], timeout=0)


def test_a_command_renders_itself_for_a_log_without_pretending_to_be_a_shell():
    assert Command(["pytest", "-k", "not slow"]).display == "pytest -k 'not slow'"


# ── ActionResult: a failure always says why ──────────────────────────────


def test_a_failure_must_name_a_reason():
    """The invariant the whole type exists for."""
    with pytest.raises(ValueError, match="reason"):
        ActionResult(ok=False, action="replace a.py")


def test_a_success_must_not_claim_a_reason():
    with pytest.raises(ValueError, match="reason"):
        ActionResult(ok=True, action="replace a.py", reason="timeout")


def test_an_unknown_reason_is_rejected():
    with pytest.raises(ValueError, match="reason"):
        ActionResult(ok=False, action="x", reason="vibes")  # type: ignore[arg-type]


def test_a_result_reports_whether_it_is_worth_retrying():
    """not-found and ambiguous mean the plan was wrong; retrying repeats it."""
    assert ActionResult(ok=False, action="x", reason="timeout").retryable is True
    assert ActionResult(ok=False, action="x", reason="not-found").retryable is False
    assert ActionResult(ok=False, action="x", reason="ambiguous").retryable is False
    assert ActionResult(ok=True, action="x").retryable is False


def test_a_command_that_ran_and_reported_failure_is_distinct_from_one_that_crashed():
    """A test suite exiting 1 did not crash — it worked and said no.

    Collapsing the two would make `retryable` wrong in the common case: you
    cannot fix a failing suite by running it again.
    """
    failed = ActionResult(ok=False, action="pytest", reason="failed", exit_code=1)
    assert failed.retryable is False
    assert ActionResult(ok=False, action="x", reason="crashed").retryable is False


def test_a_result_is_immutable():
    from dataclasses import FrozenInstanceError

    result = ActionResult(ok=True, action="x")
    with pytest.raises(FrozenInstanceError):
        result.ok = False  # type: ignore[misc]


def test_a_successful_command_result_keeps_its_streams_and_exit_code():
    result = ActionResult(
        ok=True, action="pytest", stdout="1 passed", stderr="", exit_code=0
    )
    assert result.stdout == "1 passed"
    assert result.exit_code == 0
