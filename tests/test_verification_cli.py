"""The standalone CLI.

`main()` takes argv and streams, so these tests exercise the real entry point
without spawning a process — except for one test that does spawn `python -m
tars.verification`, because "standalone CLI-callable" is a claim about the
module actually being executable, and only a real process proves it.

The exit-code contract is the part worth pinning hardest: `no-checks` must not
exit 0 by default. A CI step that reads 0 as "verified" would be reading a repo
with no check surface as a green build.
"""

import json
import subprocess
import sys
from io import StringIO

from tars.verification.cli import main


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def overrides(tmp_path, checks):
    return write(tmp_path / ".tars" / "verification.json", json.dumps({"checks": checks}))


def run(*argv):
    out, err = StringIO(), StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


# ── exit codes ────────────────────────────────────────────────────────────


def test_a_passing_repo_exits_zero(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'exit 0'"})
    code, out, _ = run(str(tmp_path))
    assert code == 0
    assert "PASS" in out


def test_a_failing_check_exits_one(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'exit 1'"})
    code, out, _ = run(str(tmp_path))
    assert code == 1
    assert "FAIL" in out


def test_nothing_to_verify_exits_three_not_zero(tmp_path):
    # The contract that keeps a bare directory from reading as a green build.
    code, out, _ = run(str(tmp_path))
    assert code == 3
    assert "NO-CHECKS" in out


def test_allow_no_checks_downgrades_that_to_zero(tmp_path):
    code, _, _ = run(str(tmp_path), "--allow-no-checks")
    assert code == 0


def test_a_bad_kind_name_is_a_usage_error(tmp_path):
    code, _, err = run(str(tmp_path), "--only", "vibes")
    assert code == 2
    assert "vibes" in err


def test_a_missing_path_is_a_usage_error_not_a_traceback(tmp_path):
    code, _, err = run(str(tmp_path / "nope"))
    assert code == 2
    assert "nope" in err
    assert "Traceback" not in err


# ── output ────────────────────────────────────────────────────────────────


def test_each_check_is_reported_with_its_command(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'exit 0'"})
    _, out, _ = run(str(tmp_path))
    assert "test" in out
    assert "sh -c 'exit 0'" in out


def noisy_script(tmp_path):
    """A failing check whose output text does not appear in its command line.

    Necessary so that "is the output body shown?" can't be answered by the
    command string that's echoed in the status row.
    """
    script = write(tmp_path / "noisy.sh", "#!/bin/sh\necho FIRST_BODY_LINE\necho LAST_BODY_LINE\nexit 1\n")
    script.chmod(0o755)
    overrides(tmp_path, {"test": "sh noisy.sh"})


def test_failing_output_is_shown_as_a_tail(tmp_path):
    noisy_script(tmp_path)
    _, out, _ = run(str(tmp_path))
    assert "LAST_BODY_LINE" in out


def test_output_lines_zero_suppresses_the_body(tmp_path):
    noisy_script(tmp_path)
    _, out, _ = run(str(tmp_path), "--output-lines", "0")
    assert "LAST_BODY_LINE" not in out
    assert "FAIL" in out


def test_quiet_prints_only_the_verdict_line(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'exit 0'"})
    _, out, _ = run(str(tmp_path), "--quiet")
    assert len(out.strip().splitlines()) == 1
    assert "PASS" in out


def test_missing_kinds_are_named_in_the_report(tmp_path):
    # "tests passed" means less in a repo with no type-checker at all.
    overrides(tmp_path, {"test": "sh -c 'exit 0'"})
    _, out, _ = run(str(tmp_path))
    assert "typecheck" in out


def test_an_unavailable_tool_is_reported_as_such_not_as_a_failure(tmp_path):
    overrides(
        tmp_path,
        {"lint": "tars-nonexistent-linter-xyz", "test": "sh -c 'exit 0'"},
    )
    code, out, _ = run(str(tmp_path))
    assert "UNAVAILABLE" in out
    assert code == 0  # a coverage gap, not a failure


def test_notes_from_a_broken_override_file_reach_the_user(tmp_path):
    write(tmp_path / ".tars" / "verification.json", "{not json")
    code, out, _ = run(str(tmp_path))
    assert "verification.json" in out
    assert code == 3  # nothing left to run, so: nothing verified


# ── --list ────────────────────────────────────────────────────────────────


def test_list_shows_detected_checks_with_evidence_and_runs_nothing(tmp_path):
    marker = tmp_path / "ran.txt"
    overrides(tmp_path, {"test": f"sh -c 'touch {marker}'"})
    code, out, _ = run(str(tmp_path), "--list")
    assert code == 0
    assert "verification.json" in out  # the evidence
    assert not marker.exists()


def test_list_marks_a_tool_that_is_not_installed(tmp_path):
    overrides(tmp_path, {"lint": "tars-nonexistent-linter-xyz"})
    _, out, _ = run(str(tmp_path), "--list")
    assert "not installed" in out


def test_list_on_a_repo_with_no_checks_exits_three(tmp_path):
    code, out, _ = run(str(tmp_path), "--list")
    assert code == 3
    assert "no checks" in out.lower()


# ── --json ────────────────────────────────────────────────────────────────


def test_json_output_is_a_single_parseable_object(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'echo hi'"})
    code, out, _ = run(str(tmp_path), "--json")
    payload = json.loads(out)
    assert payload["verdict"] == "pass"
    assert payload["exit_code"] == 0
    assert payload["results"][0]["status"] == "passed"


def test_json_output_carries_the_failure_detail(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'echo why; exit 4'"})
    _, out, _ = run(str(tmp_path), "--json")
    payload = json.loads(out)
    assert payload["results"][0]["exit_code"] == 4
    assert "why" in payload["results"][0]["stdout"]


def test_json_list_output_is_parseable(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'exit 0'"})
    _, out, _ = run(str(tmp_path), "--list", "--json")
    payload = json.loads(out)
    assert payload["checks"][0]["kind"] == "test"
    assert payload["checks"][0]["selected"] is True


# ── kind selection ────────────────────────────────────────────────────────


def test_only_narrows_to_the_named_kinds(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'exit 0'", "lint": "sh -c 'exit 0'"})
    _, out, _ = run(str(tmp_path), "--only", "lint", "--json")
    payload = json.loads(out)
    assert [r["kind"] for r in payload["results"]] == ["lint"]


def test_only_accepts_repeated_flags(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'exit 0'", "lint": "sh -c 'exit 0'"})
    _, out, _ = run(str(tmp_path), "--only", "lint", "--only", "test", "--json")
    payload = json.loads(out)
    assert {r["kind"] for r in payload["results"]} == {"lint", "test"}


def test_include_adds_the_opt_in_build_kind(tmp_path):
    overrides(tmp_path, {"build": "sh -c 'exit 0'"})
    _, without, _ = run(str(tmp_path), "--json", "--allow-no-checks")
    assert json.loads(without)["results"] == []

    _, with_build, _ = run(str(tmp_path), "--include", "build", "--json")
    assert [r["kind"] for r in json.loads(with_build)["results"]] == ["build"]


def test_fail_fast_stops_after_the_first_failure(tmp_path):
    overrides(tmp_path, {"lint": "sh -c 'exit 1'", "test": "sh -c 'exit 0'"})
    _, out, _ = run(str(tmp_path), "--fail-fast", "--json")
    statuses = {r["kind"]: r["status"] for r in json.loads(out)["results"]}
    assert statuses == {"lint": "failed", "test": "skipped"}


def test_timeout_is_honored(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'sleep 10'"})
    code, out, _ = run(str(tmp_path), "--timeout", "0.5", "--json")
    assert code == 1
    assert json.loads(out)["results"][0]["status"] == "timeout"


def test_defaults_to_the_current_directory(tmp_path, monkeypatch):
    overrides(tmp_path, {"test": "sh -c 'exit 0'"})
    monkeypatch.chdir(tmp_path)
    code, _, _ = run()
    assert code == 0


# ── real process ──────────────────────────────────────────────────────────


def test_the_module_is_executable_as_a_standalone_command(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'exit 0'"})
    proc = subprocess.run(
        [sys.executable, "-m", "tars.verification", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["verdict"] == "pass"


def test_help_works_as_a_standalone_command():
    proc = subprocess.run(
        [sys.executable, "-m", "tars.verification", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "--only" in proc.stdout
