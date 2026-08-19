"""The data shapes the verification gate reports in.

The load-bearing test here is the three-valued verdict: a report with zero
executed checks must NOT read as a pass. Core directive #2 says an unverifiable
task gets said out loud, so "we found nothing to run" has to be its own answer,
not a green checkmark.
"""

import json

import pytest

from tars.verification.types import Check, CheckResult, GateReport


def make_check(kind="test", tool="pytest", command=("pytest",)):
    return Check(
        kind=kind,
        tool=tool,
        command=command,
        ecosystem="python",
        evidence="pyproject.toml [tool.pytest.ini_options]",
    )


def make_result(status="passed", check=None, **kw):
    defaults = {
        "exit_code": 0 if status == "passed" else 1,
        "duration_s": 0.5,
        "stdout": "",
        "stderr": "",
    }
    defaults.update(kw)
    return CheckResult(check=check or make_check(), status=status, **defaults)


def test_check_normalizes_command_to_a_tuple():
    # Callers naturally pass lists; Check stays frozen and hashable.
    check = make_check(command=["pytest", "-q"])
    assert check.command == ("pytest", "-q")
    assert hash(check)


def test_check_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        make_check(kind="vibes")


def test_check_rejects_an_empty_command():
    with pytest.raises(ValueError, match="command"):
        make_check(command=())


def test_check_command_line_is_shell_quoted_for_humans_to_rerun():
    check = make_check(command=("pytest", "-k", "not slow"))
    assert check.command_line == "pytest -k 'not slow'"


def test_passed_result_is_ok_and_counts_as_executed():
    result = make_result("passed")
    assert result.ok
    assert result.executed


def test_failed_result_is_not_ok():
    assert not make_result("failed").ok


def test_unavailable_result_neither_passed_nor_failed():
    # A configured-but-not-installed tool is a coverage gap, not a defect.
    result = make_result("unavailable", exit_code=None, detail="ruff not found")
    assert not result.ok
    assert not result.failed
    assert not result.executed


def test_timeout_counts_as_a_failure():
    assert make_result("timeout", exit_code=None).failed


def test_launch_error_counts_as_a_failure():
    assert make_result("error", exit_code=None).failed


def test_verdict_is_pass_when_every_check_passed():
    report = GateReport(root="/repo", results=(make_result("passed"),))
    assert report.verdict == "pass"
    assert report.passed


def test_verdict_is_fail_when_any_check_failed():
    report = GateReport(
        root="/repo",
        results=(make_result("passed"), make_result("failed")),
    )
    assert report.verdict == "fail"
    assert not report.passed


def test_verdict_is_no_checks_when_nothing_was_detected():
    report = GateReport(root="/repo", results=())
    assert report.verdict == "no-checks"
    # The whole point: absence of failure is not success.
    assert not report.passed


def test_verdict_is_no_checks_when_every_check_was_unavailable():
    report = GateReport(root="/repo", results=(make_result("unavailable"),))
    assert report.verdict == "no-checks"
    assert not report.passed


def test_passing_checks_alongside_unavailable_ones_still_pass():
    report = GateReport(
        root="/repo",
        results=(make_result("passed"), make_result("unavailable")),
    )
    assert report.verdict == "pass"


def test_report_partitions_results_by_outcome():
    passed = make_result("passed")
    failed = make_result("failed")
    skipped = make_result("skipped")
    report = GateReport(root="/repo", results=(passed, failed, skipped))
    assert report.failures == (failed,)
    assert report.executed == (passed, failed)
    assert report.skipped == (skipped,)


def test_report_to_dict_is_json_serializable():
    report = GateReport(root="/repo", results=(make_result("failed", stderr="boom"),))
    payload = json.dumps(report.to_dict())
    assert '"verdict": "fail"' in payload
    assert "boom" in payload


def test_to_dict_truncates_long_output_and_says_so():
    # The JSON form feeds an agent loop's context, so it can't carry a 5MB
    # pytest log. Keep the tail — that's where the summary lives.
    result = make_result("failed", stdout="x" * 50 + "TAIL")
    report = GateReport(root="/repo", results=(result,))
    payload = report.to_dict(max_output_chars=20)
    captured = payload["results"][0]["stdout"]
    assert captured.endswith("TAIL")
    assert len(captured) <= 20
    assert payload["results"][0]["stdout_truncated"] is True


def test_to_dict_keeps_short_output_whole():
    report = GateReport(root="/repo", results=(make_result("passed", stdout="fine"),))
    payload = report.to_dict(max_output_chars=100)
    assert payload["results"][0]["stdout"] == "fine"
    assert payload["results"][0]["stdout_truncated"] is False
