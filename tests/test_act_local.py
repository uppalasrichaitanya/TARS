"""LocalExecutor: the built-in hands. Stdlib only, no shell, no partial writes.

The tests worth reading here are the failure ones. Applying a clean edit is easy;
what makes an executor safe to hand to an unattended loop is that a bad edit
leaves the file exactly as it was, and that every refusal says which kind of
refusal it is.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
from pathlib import Path

import pytest

from tars.act.base import Executor
from tars.act.local import LocalExecutor
from tars.act.types import Command, Edit, Hunk


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def ex(root: Path) -> LocalExecutor:
    return LocalExecutor(root)


def write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ── identity ─────────────────────────────────────────────────────────────


def test_the_local_executor_is_an_executor(ex):
    assert isinstance(ex, Executor)
    assert ex.name == "local"


# ── replace: the anchored happy path ─────────────────────────────────────


def test_a_single_hunk_is_applied(ex, root):
    write(root, "a.py", "x = 1\ny = 2\n")
    result = ex.apply(Edit.replace("a.py", Hunk("x = 1", "x = 99")))
    assert result.ok, result.detail
    assert (root / "a.py").read_text() == "x = 99\ny = 2\n"
    assert result.touched == ("a.py",)


def test_every_hunk_in_one_edit_is_applied(ex, root):
    write(root, "a.py", "one\ntwo\nthree\n")
    result = ex.apply(
        Edit.replace("a.py", Hunk("one", "1"), Hunk("two", "2"), Hunk("three", "3"))
    )
    assert result.ok, result.detail
    assert (root / "a.py").read_text() == "1\n2\n3\n"


def test_hunks_apply_in_order_so_a_later_one_can_anchor_on_an_earlier_one(ex, root):
    write(root, "a.py", "value = OLD\n")
    result = ex.apply(
        Edit.replace("a.py", Hunk("OLD", "MIDDLE"), Hunk("MIDDLE", "FINAL"))
    )
    assert result.ok, result.detail
    assert (root / "a.py").read_text() == "value = FINAL\n"


def test_a_snippet_is_removed_by_replacing_it_with_nothing(ex, root):
    write(root, "a.py", "keep\ndrop\nkeep2\n")
    result = ex.apply(Edit.replace("a.py", Hunk("drop\n", "")))
    assert result.ok, result.detail
    assert (root / "a.py").read_text() == "keep\nkeep2\n"


def test_a_successful_edit_reports_a_unified_diff(ex, root):
    write(root, "a.py", "x = 1\n")
    result = ex.apply(Edit.replace("a.py", Hunk("x = 1", "x = 2")))
    assert "-x = 1" in result.diff
    assert "+x = 2" in result.diff


# ── replace: the refusals that matter ────────────────────────────────────


def test_an_anchor_matching_twice_is_refused_as_ambiguous(ex, root):
    """Applying to the first match is a guess, and a guess here corrupts code."""
    write(root, "a.py", "log()\nlog()\n")
    before = (root / "a.py").read_bytes()

    result = ex.apply(Edit.replace("a.py", Hunk("log()", "log(1)")))

    assert not result.ok
    assert result.reason == "ambiguous"
    assert "2" in result.detail, result.detail
    assert (root / "a.py").read_bytes() == before


def test_a_missing_anchor_is_not_found_and_the_file_is_untouched(ex, root):
    write(root, "a.py", "x = 1\n")
    before = (root / "a.py").read_bytes()

    result = ex.apply(Edit.replace("a.py", Hunk("nowhere", "here")))

    assert not result.ok
    assert result.reason == "not-found"
    assert (root / "a.py").read_bytes() == before


def test_a_failing_middle_hunk_leaves_the_file_byte_identical(ex, root):
    """The rule from master_agent_loop.md:105 — a failure is localized, never a
    half-rewritten file. Nothing is written until every hunk has resolved."""
    write(root, "a.py", "one\ntwo\nthree\n")
    before = (root / "a.py").read_bytes()

    result = ex.apply(
        Edit.replace(
            "a.py",
            Hunk("one", "1"),
            Hunk("MISSING", "x"),
            Hunk("three", "3"),
        )
    )

    assert not result.ok
    assert result.reason == "not-found"
    assert (root / "a.py").read_bytes() == before


def test_a_failure_says_which_hunk_failed(ex, root):
    write(root, "a.py", "one\n")
    result = ex.apply(Edit.replace("a.py", Hunk("one", "1"), Hunk("MISSING", "x")))
    assert "2" in result.detail, result.detail


def test_replacing_in_a_file_that_is_not_there_is_not_found(ex):
    result = ex.apply(Edit.replace("ghost.py", Hunk("a", "b")))
    assert not result.ok
    assert result.reason == "not-found"


def test_a_failed_edit_reports_nothing_as_touched(ex, root):
    write(root, "a.py", "x\n")
    result = ex.apply(Edit.replace("a.py", Hunk("nope", "y")))
    assert result.touched == ()


# ── create / overwrite / delete ──────────────────────────────────────────


def test_a_create_writes_a_new_file(ex, root):
    result = ex.apply(Edit.create("new.py", "print(1)\n"))
    assert result.ok, result.detail
    assert (root / "new.py").read_text() == "print(1)\n"


def test_a_create_makes_missing_parent_directories(ex, root):
    result = ex.apply(Edit.create("deep/nested/new.py", "x\n"))
    assert result.ok, result.detail
    assert (root / "deep/nested/new.py").read_text() == "x\n"


def test_a_create_over_an_existing_file_is_a_conflict(ex, root):
    """create asserts absence. Silently clobbering is how work disappears."""
    write(root, "a.py", "precious\n")
    result = ex.apply(Edit.create("a.py", "replacement\n"))
    assert not result.ok
    assert result.reason == "conflict"
    assert (root / "a.py").read_text() == "precious\n"


def test_an_overwrite_replaces_the_whole_file(ex, root):
    write(root, "a.py", "old\n")
    result = ex.apply(Edit.overwrite("a.py", "new\n"))
    assert result.ok, result.detail
    assert (root / "a.py").read_text() == "new\n"


def test_an_overwrite_of_a_missing_file_is_not_found(ex):
    """overwrite asserts presence, so a mistyped path fails instead of creating."""
    result = ex.apply(Edit.overwrite("ghost.py", "x\n"))
    assert not result.ok
    assert result.reason == "not-found"


def test_a_delete_removes_the_file(ex, root):
    write(root, "a.py", "x\n")
    result = ex.apply(Edit.delete("a.py"))
    assert result.ok, result.detail
    assert not (root / "a.py").exists()


def test_deleting_a_file_that_is_not_there_is_not_found(ex):
    result = ex.apply(Edit.delete("ghost.py"))
    assert not result.ok
    assert result.reason == "not-found"


# ── staying inside the root ──────────────────────────────────────────────


def test_an_edit_escaping_the_root_is_denied(ex, root):
    result = ex.apply(Edit.create("../escaped.py", "x\n"))
    assert not result.ok
    assert result.reason == "denied"
    assert not (root.parent / "escaped.py").exists()


def test_a_symlink_out_of_the_root_is_denied(ex, root, tmp_path):
    outside = tmp_path.parent / "outside_target.txt"
    outside.write_text("original\n", encoding="utf-8")
    try:
        (root / "link.txt").symlink_to(outside)
    except OSError:  # pragma: no cover - platforms without symlink permission
        pytest.skip("symlinks not permitted here")

    result = ex.apply(Edit.overwrite("link.txt", "hijacked\n"))

    assert not result.ok
    assert result.reason == "denied"
    assert outside.read_text() == "original\n"


def test_an_absolute_path_inside_the_root_is_allowed(ex, root):
    write(root, "a.py", "x = 1\n")
    result = ex.apply(Edit.replace(str(root / "a.py"), Hunk("x = 1", "x = 2")))
    assert result.ok, result.detail


# ── dry run ──────────────────────────────────────────────────────────────


def test_a_dry_run_returns_the_diff_and_writes_nothing(root):
    write(root, "a.py", "x = 1\n")
    before = (root / "a.py").read_bytes()

    result = LocalExecutor(root, dry_run=True).apply(
        Edit.replace("a.py", Hunk("x = 1", "x = 2"))
    )

    assert result.ok, result.detail
    assert "+x = 2" in result.diff
    assert (root / "a.py").read_bytes() == before


def test_a_dry_run_create_does_not_create(root):
    result = LocalExecutor(root, dry_run=True).apply(Edit.create("new.py", "x\n"))
    assert result.ok, result.detail
    assert not (root / "new.py").exists()


def test_a_dry_run_touches_nothing(root):
    write(root, "a.py", "x = 1\n")
    result = LocalExecutor(root, dry_run=True).apply(
        Edit.replace("a.py", Hunk("x = 1", "x = 2"))
    )
    assert result.touched == ()


def test_a_dry_run_still_detects_an_ambiguous_anchor(root):
    """Otherwise a preview would approve an edit that then fails on apply."""
    write(root, "a.py", "log()\nlog()\n")
    result = LocalExecutor(root, dry_run=True).apply(
        Edit.replace("a.py", Hunk("log()", "log(1)"))
    )
    assert not result.ok
    assert result.reason == "ambiguous"


# ── running commands ─────────────────────────────────────────────────────


def test_a_successful_command_reports_its_output(ex):
    result = ex.run(Command([sys.executable, "-c", "print('hello')"]))
    assert result.ok, result.detail
    assert "hello" in result.stdout
    assert result.exit_code == 0


def test_a_non_zero_exit_is_a_failure_that_is_not_worth_retrying(ex):
    result = ex.run(Command([sys.executable, "-c", "raise SystemExit(3)"]))
    assert not result.ok
    assert result.reason == "failed"
    assert result.exit_code == 3
    assert result.retryable is False


def test_a_program_that_is_not_installed_is_not_found_rather_than_crashed(ex):
    result = ex.run(Command(["tars-definitely-not-installed-xyz"]))
    assert not result.ok
    assert result.reason == "not-found"


def test_a_command_that_overruns_its_deadline_is_a_timeout(ex):
    result = ex.run(
        Command([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
    )
    assert not result.ok
    assert result.reason == "timeout"
    assert result.retryable is True


def test_a_command_runs_in_the_executors_root_by_default(ex, root):
    result = ex.run(Command([sys.executable, "-c", "import os; print(os.getcwd())"]))
    assert result.ok, result.detail
    assert str(root.resolve()) in result.stdout


def test_a_command_can_override_its_working_directory(ex, root):
    (root / "sub").mkdir()
    result = ex.run(
        Command([sys.executable, "-c", "import os; print(os.getcwd())"], cwd="sub")
    )
    assert result.ok, result.detail
    assert "sub" in result.stdout


def test_arguments_reach_the_program_literally_because_no_shell_is_involved(ex):
    """`$(echo hi)` and `;` are data. If a shell ran, this test would notice."""
    payload = "$(echo pwned); rm -rf /"
    result = ex.run(
        Command([sys.executable, "-c", "import sys; print(sys.argv[1])", payload])
    )
    assert result.ok, result.detail
    assert payload in result.stdout


def test_a_command_prefers_the_repos_own_tools_over_path(ex, root):
    """Reuses the verification gate's resolver, so the gate and ACT agree on
    which `ruff` is *this* repo's ruff."""
    local_bin = root / ".venv" / "bin"
    local_bin.mkdir(parents=True)
    tool = local_bin / "faketool"
    tool.write_text("#!/bin/sh\necho local-copy\n", encoding="utf-8")
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    result = ex.run(Command(["faketool"]))

    assert result.ok, result.detail
    assert "local-copy" in result.stdout


def test_a_command_records_how_long_it_took(ex):
    result = ex.run(Command([sys.executable, "-c", "pass"]))
    assert result.duration_s >= 0.0


def test_a_dry_run_does_not_execute_commands(root):
    """A preview that runs the command is not a preview."""
    marker = root / "ran.txt"
    result = LocalExecutor(root, dry_run=True).run(
        Command([sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"])
    )
    assert not marker.exists()
    assert result.ok
    assert "dry run" in result.detail.lower()


# ── stale bytecode: the edit has to actually take effect ──────────────────


def test_editing_a_module_invalidates_its_cached_bytecode(ex, root):
    """CPython validates a `.pyc` against the source's mtime *in whole seconds*
    plus its size. An edit that changes neither — `a - b` to `a + b` is the same
    length — leaves the stale `.pyc` looking valid, so the next process imports
    pre-edit code. VERIFY would then grade code ACT already replaced.
    """
    source = write(root, "calc.py", "def add(a, b):\n    return a - b\n")
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"stale bytecode")

    result = ex.apply(Edit.replace("calc.py", Hunk("return a - b", "return a + b")))

    assert result.ok, result.detail
    assert not cache.exists(), "a stale .pyc survived an edit to its source"


def test_a_deleted_module_takes_its_bytecode_with_it(ex, root):
    source = write(root, "gone.py", "x = 1\n")
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"stale bytecode")

    assert ex.apply(Edit.delete("gone.py")).ok
    assert not cache.exists()


def test_bytecode_from_another_interpreter_version_is_invalidated_too(ex, root):
    """One stale tag is enough to poison a run under that interpreter."""
    write(root, "calc.py", "x = 1\n")
    other = root / "__pycache__" / "calc.cpython-39.pyc"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(b"stale bytecode")

    assert ex.apply(Edit.replace("calc.py", Hunk("x = 1", "x = 2"))).ok
    assert not other.exists()


def test_an_unrelated_modules_bytecode_is_left_alone(ex, root):
    """Only the edited module's cache is dropped, not the whole directory."""
    write(root, "calc.py", "x = 1\n")
    write(root, "other.py", "y = 1\n")
    keep = root / "__pycache__" / "other.cpython-312.pyc"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_bytes(b"someone else's bytecode")

    assert ex.apply(Edit.replace("calc.py", Hunk("x = 1", "x = 2"))).ok
    assert keep.exists()


def test_a_dry_run_leaves_cached_bytecode_alone(root):
    source = write(root, "calc.py", "x = 1\n")
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"stale bytecode")

    LocalExecutor(root, dry_run=True).apply(
        Edit.replace("calc.py", Hunk("x = 1", "x = 2"))
    )

    assert cache.exists(), "a dry run must change nothing at all"


def test_a_following_process_sees_the_edit_even_when_size_and_mtime_match(ex, root):
    """The regression test for the whole class of bug, end to end.

    The source's mtime is deliberately restored after the edit, which is the
    worst case a fast loop hits naturally: same size, same second, stale cache.
    """
    source = write(root, "value.py", "VALUE = 1\n")
    read_it = Command([sys.executable, "-c", "import value; print(value.VALUE)"])

    first = ex.run(read_it)
    assert first.ok and "1" in first.stdout

    stat_before = source.stat()
    assert ex.apply(Edit.replace("value.py", Hunk("VALUE = 1", "VALUE = 2"))).ok
    assert source.stat().st_size == stat_before.st_size, "test needs an equal-size edit"
    os.utime(source, (stat_before.st_atime, stat_before.st_mtime))

    second = ex.run(read_it)
    assert second.ok, second.detail
    assert "2" in second.stdout, f"stale bytecode was imported: {second.stdout!r}"
