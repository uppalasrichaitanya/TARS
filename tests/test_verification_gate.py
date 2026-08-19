"""The gate: detect, select, run, report.

Selection is where "candidates" collapse to "what we ran". The rule under test:
per (ecosystem, kind), run the first candidate that is actually installed — but
if none are, keep the first anyway so the report says `unavailable` with its
evidence, instead of quietly shrinking the check surface.

Tests that need a real command use the project-override file rather than a real
toolchain, so they exercise orchestration with real processes and no dependency
on pytest/ruff/npm being present.
"""

import json

from tars.verification.gate import select_checks, verify
from tars.verification.types import DEFAULT_KINDS


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def fake_tool(root, name, body="#!/bin/sh\nexit 0\n"):
    binary = root / ".venv" / "bin" / name
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(body)
    binary.chmod(0o755)
    return binary


def overrides(tmp_path, checks):
    return write(tmp_path / ".tars" / "verification.json", json.dumps({"checks": checks}))


#: An empty PATH makes availability tests hermetic: only tools the fixture puts
#: in the repo's own .venv/bin can resolve, so a globally-installed flake8 on the
#: developer's machine can't change the result.
NO_PATH = {"PATH": ""}


# ── selection ─────────────────────────────────────────────────────────────


def test_selects_the_first_available_candidate_not_the_first_detected(tmp_path):
    # .flake8 is stronger evidence, but flake8 isn't installed and ruff is.
    write(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    write(tmp_path / ".flake8", "[flake8]\n")
    fake_tool(tmp_path, "ruff")
    selected, rejected = select_checks(tmp_path, kinds=("lint",), env=NO_PATH)
    assert [c.tool for c in selected] == ["ruff"]
    assert [c.tool for c in rejected] == ["flake8"]


def test_keeps_an_unavailable_check_when_nothing_for_that_kind_is_installed(tmp_path):
    # Dropping it would shrink the reported check surface silently; the repo
    # configures a linter and the human should hear that it couldn't run.
    write(tmp_path / "pyproject.toml", "[tool.ruff]\n")
    selected, _ = select_checks(tmp_path, kinds=("lint",), env=NO_PATH)
    assert [c.tool for c in selected] == ["ruff"]


def test_selects_one_check_per_kind_per_ecosystem(tmp_path):
    write(tmp_path / "pyproject.toml", "[tool.pytest.ini_options]\n")
    write(tmp_path / "package.json", json.dumps({"scripts": {"test": "vitest run"}}))
    selected, _ = select_checks(tmp_path, kinds=("test",), env=NO_PATH)
    assert {c.ecosystem for c in selected} == {"python", "node"}


def test_kinds_filter_is_respected(tmp_path):
    write(tmp_path / "pyproject.toml", "[tool.ruff]\n[tool.mypy]\n")
    selected, _ = select_checks(tmp_path, kinds=("typecheck",), env=NO_PATH)
    assert [c.kind for c in selected] == ["typecheck"]


def test_build_is_not_selected_by_default(tmp_path):
    write(tmp_path / "pyproject.toml", "[build-system]\nrequires=['setuptools']\n")
    selected, _ = select_checks(tmp_path, kinds=DEFAULT_KINDS, env=NO_PATH)
    assert [c.kind for c in selected] == []


# ── running ───────────────────────────────────────────────────────────────


def test_a_passing_check_yields_a_pass_verdict(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'exit 0'"})
    report = verify(tmp_path)
    assert report.verdict == "pass"
    assert [r.status for r in report.results] == ["passed"]
    assert report.duration_s >= 0


def test_a_failing_check_yields_a_fail_verdict_with_the_exit_code(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'echo broken; exit 7'"})
    report = verify(tmp_path)
    assert report.verdict == "fail"
    assert report.failures[0].exit_code == 7
    assert "broken" in report.failures[0].stdout


def test_an_empty_directory_reports_no_checks_and_says_why(tmp_path):
    report = verify(tmp_path)
    assert report.verdict == "no-checks"
    assert not report.passed
    assert any("no checks" in note.lower() for note in report.notes)


def test_a_configured_but_uninstalled_tool_is_not_a_pass(tmp_path):
    # The dangerous case: nothing failed, so a boolean gate would say "pass".
    write(tmp_path / "pyproject.toml", "[tool.ruff]\n")
    report = verify(tmp_path, kinds=("lint",), env=NO_PATH)
    assert [r.status for r in report.results] == ["unavailable"]
    assert report.verdict == "no-checks"


def test_checks_run_in_kind_order_cheapest_first(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'exit 0'", "lint": "sh -c 'exit 0'"})
    report = verify(tmp_path, kinds=("lint", "test"))
    assert [r.check.kind for r in report.results] == ["lint", "test"]


def test_fail_fast_skips_the_remaining_checks(tmp_path):
    marker = tmp_path / "test-ran.txt"
    overrides(
        tmp_path,
        {"lint": "sh -c 'exit 1'", "test": f"sh -c 'touch {marker}'"},
    )
    report = verify(tmp_path, fail_fast=True)
    statuses = {r.check.kind: r.status for r in report.results}
    assert statuses == {"lint": "failed", "test": "skipped"}
    assert not marker.exists()
    skipped = [r for r in report.results if r.status == "skipped"][0]
    assert "fail-fast" in (skipped.detail or "")


def test_without_fail_fast_every_check_runs(tmp_path):
    marker = tmp_path / "test-ran.txt"
    overrides(
        tmp_path,
        {"lint": "sh -c 'exit 1'", "test": f"sh -c 'touch {marker}'"},
    )
    report = verify(tmp_path)
    assert marker.exists()
    assert report.verdict == "fail"


def test_an_unavailable_check_does_not_trigger_fail_fast(tmp_path):
    # A missing tool is a coverage gap, not a failure — it must not stop the run.
    write(tmp_path / "pyproject.toml", "[tool.ruff]\n")
    overrides(tmp_path, {"test": "sh -c 'exit 0'"})
    report = verify(tmp_path, kinds=("lint", "test"), fail_fast=True)
    statuses = {r.check.kind: r.status for r in report.results}
    assert statuses["lint"] == "unavailable"
    assert statuses["test"] == "passed"


def test_callbacks_fire_around_each_check(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'exit 0'", "lint": "sh -c 'exit 0'"})
    events = []
    verify(
        tmp_path,
        on_start=lambda check: events.append(("start", check.kind)),
        on_result=lambda result: events.append(("done", result.check.kind)),
    )
    assert events == [("start", "lint"), ("done", "lint"), ("start", "test"), ("done", "test")]


def test_timeouts_are_reported_not_raised(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'sleep 10'"})
    report = verify(tmp_path, timeout=0.5)
    assert [r.status for r in report.results] == ["timeout"]
    assert report.verdict == "fail"


def test_report_records_rejected_candidates(tmp_path):
    write(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    write(tmp_path / ".flake8", "[flake8]\n")
    fake_tool(tmp_path, "ruff")
    report = verify(tmp_path, kinds=("lint",), env=NO_PATH)
    assert [c.tool for c in report.not_selected] == ["flake8"]


def test_report_names_the_kinds_nothing_covered(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'exit 0'"})
    report = verify(tmp_path, kinds=DEFAULT_KINDS)
    assert set(report.missing_kinds()) == {"lint", "format", "typecheck"}


def test_root_is_recorded_as_an_absolute_path(tmp_path):
    overrides(tmp_path, {"test": "sh -c 'exit 0'"})
    report = verify(str(tmp_path))
    assert report.root == str(tmp_path.resolve())


# ── project overrides ─────────────────────────────────────────────────────


def test_an_override_replaces_detection_for_that_kind(tmp_path):
    write(tmp_path / "pyproject.toml", "[tool.pytest.ini_options]\n")
    overrides(tmp_path, {"test": "sh -c 'exit 0'"})
    selected, _ = select_checks(tmp_path, kinds=("test",), env=NO_PATH)
    assert [c.command for c in selected] == [("sh", "-c", "exit 0")]
    assert selected[0].ecosystem == "override"
    assert "verification.json" in selected[0].evidence


def test_an_override_leaves_other_kinds_to_detection(tmp_path):
    write(tmp_path / "pyproject.toml", "[tool.pytest.ini_options]\n[tool.ruff]\n")
    overrides(tmp_path, {"test": "sh -c 'exit 0'"})
    selected, _ = select_checks(tmp_path, kinds=("lint", "test"), env=NO_PATH)
    tools = {c.kind: c.tool for c in selected}
    assert tools["lint"] == "ruff"


def test_an_override_accepts_an_argv_list(tmp_path):
    overrides(tmp_path, {"test": ["sh", "-c", "exit 0"]})
    selected, _ = select_checks(tmp_path, kinds=("test",), env=NO_PATH)
    assert selected[0].command == ("sh", "-c", "exit 0")


def test_a_null_override_disables_a_kind(tmp_path):
    # The escape hatch for "this repo's linter is broken, stop running it".
    write(tmp_path / "pyproject.toml", "[tool.ruff]\n")
    overrides(tmp_path, {"lint": None})
    selected, _ = select_checks(tmp_path, kinds=("lint",), env=NO_PATH)
    assert selected == ()


def test_a_malformed_override_file_is_reported_not_fatal(tmp_path):
    write(tmp_path / ".tars" / "verification.json", "{not json")
    write(tmp_path / "pyproject.toml", "[tool.pytest.ini_options]\n")
    report = verify(tmp_path, kinds=("test",), env=NO_PATH)
    assert any("verification.json" in note for note in report.notes)
    # Detection still worked, so the run isn't lost.
    assert [r.check.tool for r in report.results] == ["pytest"]


def test_an_unknown_kind_in_the_override_file_is_reported(tmp_path):
    overrides(tmp_path, {"vibes": "sh -c 'exit 0'"})
    report = verify(tmp_path)
    assert any("vibes" in note for note in report.notes)
