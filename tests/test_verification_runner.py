"""Running a check uniformly, whatever the tool is.

These tests use real processes, not mocks: the whole value of this module is
what actually happens to a subprocess (does a timeout leak a grandchild? does
undecodable output crash the gate?), and a mocked Popen can't answer that.
"""

import os
import time
from pathlib import Path

import pytest

from tars.verification.runner import resolve_executable, run_check
from tars.verification.types import Check


def make_check(command, kind="test", tool="fake"):
    return Check(
        kind=kind,
        tool=tool,
        command=tuple(command),
        ecosystem="test-fixture",
        evidence="handwritten in a test",
    )


def write_executable(path: Path, body: str = "#!/bin/sh\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


# ── resolve_executable ────────────────────────────────────────────────────


def test_resolves_a_tool_on_path_to_an_absolute_path(tmp_path):
    resolved = resolve_executable("sh", tmp_path)
    assert resolved is not None
    assert Path(resolved).is_absolute()


def test_prefers_the_repos_venv_over_path(tmp_path):
    # A repo's pinned ruff must win over whatever ruff is on the system PATH,
    # or the gate reports results the repo's own CI would never produce.
    local = write_executable(tmp_path / ".venv" / "bin" / "sh")
    assert resolve_executable("sh", tmp_path) == str(local)


def test_prefers_node_modules_bin(tmp_path):
    local = write_executable(tmp_path / "node_modules" / ".bin" / "eslint")
    assert resolve_executable("eslint", tmp_path) == str(local)


def test_returns_none_for_a_tool_that_is_not_installed(tmp_path):
    assert resolve_executable("definitely-not-a-real-tool-xyz", tmp_path) is None


def test_resolves_a_relative_script_path_against_the_repo_root(tmp_path):
    script = write_executable(tmp_path / "scripts" / "check.sh")
    assert resolve_executable("scripts/check.sh", tmp_path) == str(script)


def test_does_not_resolve_a_non_executable_file(tmp_path):
    # Deliberately a name that cannot also exist on PATH, so this test can't
    # pass for the wrong reason when run inside an activated venv.
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "tars-probe-tool").write_text("not executable")
    assert resolve_executable("tars-probe-tool", tmp_path) is None


# ── run_check ─────────────────────────────────────────────────────────────


def test_passing_command_reports_passed(tmp_path):
    result = run_check(make_check(["sh", "-c", "echo hello"]), tmp_path)
    assert result.status == "passed"
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert result.duration_s >= 0


def test_nonzero_exit_reports_failed_and_keeps_the_code(tmp_path):
    result = run_check(make_check(["sh", "-c", "exit 3"]), tmp_path)
    assert result.status == "failed"
    assert result.exit_code == 3


def test_stderr_is_captured_separately(tmp_path):
    result = run_check(make_check(["sh", "-c", "echo oops >&2; exit 1"]), tmp_path)
    assert "oops" in result.stderr
    assert "oops" not in result.stdout


def test_missing_tool_is_unavailable_not_a_crash(tmp_path):
    result = run_check(make_check(["definitely-not-a-real-tool-xyz"]), tmp_path)
    assert result.status == "unavailable"
    assert result.exit_code is None
    assert "definitely-not-a-real-tool-xyz" in (result.detail or "")


def test_command_runs_in_the_repo_root(tmp_path):
    (tmp_path / "marker.txt").write_text("here")
    result = run_check(make_check(["sh", "-c", "ls"]), tmp_path)
    assert "marker.txt" in result.stdout


def test_records_the_resolved_command_so_a_human_can_reproduce_it(tmp_path):
    local = write_executable(tmp_path / ".venv" / "bin" / "sh")
    result = run_check(make_check(["sh", "-c", "true"]), tmp_path)
    assert result.resolved_command[0] == str(local)
    assert result.resolved_command[1:] == ("-c", "true")


def test_child_env_disables_watch_modes_and_color(tmp_path):
    # Bare `vitest`/`jest` in a repo script would sit in watch mode forever and
    # blow the timeout; CI=1 is what makes them run once and exit.
    script = 'printf \'CI=%s NO_COLOR=%s\' "$CI" "$NO_COLOR"'
    result = run_check(make_check(["sh", "-c", script]), tmp_path)
    assert result.stdout == "CI=1 NO_COLOR=1"


def test_child_inherits_the_parent_environment(tmp_path):
    os.environ["TARS_RUNNER_PROBE"] = "inherited"
    try:
        result = run_check(
            make_check(["sh", "-c", 'echo "$TARS_RUNNER_PROBE"']), tmp_path
        )
    finally:
        del os.environ["TARS_RUNNER_PROBE"]
    assert "inherited" in result.stdout


def test_explicit_env_overrides_win(tmp_path):
    result = run_check(
        make_check(["sh", "-c", 'echo "$CI"']),
        tmp_path,
        env={"CI": "0"},
    )
    assert result.stdout.strip() == "0"


def test_timeout_reports_timeout_with_the_deadline_in_the_detail(tmp_path):
    result = run_check(make_check(["sh", "-c", "sleep 10"]), tmp_path, timeout=0.5)
    assert result.status == "timeout"
    assert result.exit_code is None
    assert "0.5" in (result.detail or "")
    assert result.duration_s < 5  # killed at the deadline, not waited out


def test_timeout_kills_the_whole_process_tree(tmp_path):
    # A plain proc.kill() reaps the shell and orphans its children, which is how
    # a timed-out test runner keeps holding a port after the gate moves on.
    pidfile = tmp_path / "grandchild.pid"
    check = make_check(
        ["sh", "-c", f"sleep 30 & echo $! > {pidfile}; wait"],
    )
    result = run_check(check, tmp_path, timeout=1.0)
    assert result.status == "timeout"

    pid = int(pidfile.read_text().strip())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return  # grandchild is gone: correct
        time.sleep(0.05)
    os.kill(pid, 9)  # don't leak it out of the test run
    pytest.fail(f"grandchild {pid} survived the timeout")


def test_undecodable_output_does_not_crash_the_gate(tmp_path):
    result = run_check(make_check(["sh", "-c", "printf '\\377\\376'"]), tmp_path)
    assert result.status == "passed"
    assert result.stdout  # replaced, not raised


def test_no_timeout_means_wait_as_long_as_it_takes(tmp_path):
    result = run_check(make_check(["sh", "-c", "exit 0"]), tmp_path, timeout=None)
    assert result.status == "passed"
