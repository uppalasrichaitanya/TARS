"""Web target flows through VERIFY and influences the report."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from tars.loop.session import Session
from tars.loop.types import RunPromptSpec, WebTarget
from tars.providers import LLMResponse, Usage
from tars.router import Classification
from tars.verification import GateReport
from tars.verification.types import Check, CheckResult


def _response(text: str = "plan") -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=(),
        model="fake",
        provider="fake",
        stop_reason="end_turn",
        usage=Usage(),
    )


def test_web_verify_failure_makes_report_fail(tmp_path: Path) -> None:
    from tars.loop.engine import Engine

    def verify(root: Path, **_kwargs):
        return GateReport(root=str(root), results=())

    def verify_web(url: str, steps, expect, *, root: Path, **_kwargs):
        check = Check(
            kind="test",
            tool="browser",
            command=("tars-browser", "check", url),
            ecosystem="browser",
            evidence=f"web target {url}",
        )
        return GateReport(
            root=str(root),
            results=(
                CheckResult(
                    check=check,
                    status="failed",
                    exit_code=1,
                    duration_s=0.1,
                    detail="expectation mismatch",
                ),
            ),
        )

    engine = Engine(
        provider=lambda *_a, **_k: _response("done"),
        classify=lambda _t: Classification(1, "medium", True, (), "why"),
        verify=verify,
        verify_web=verify_web,
    )
    spec = RunPromptSpec(
        "verify web",
        web=WebTarget(url="http://example.com", steps=(), expect=({"text": "x"},)),
    )
    report = engine.run(spec, Session(root=tmp_path, model="fake"), output=StringIO())
    assert report.verification_passed is False
    assert report.verdict is not None


def test_web_verify_pass_keeps_report_passing(tmp_path: Path) -> None:
    from tars.loop.engine import Engine

    def verify(root: Path, **_kwargs):
        return GateReport(root=str(root), results=())

    def verify_web(url: str, steps, expect, *, root: Path, **_kwargs):
        check = Check(
            kind="test",
            tool="browser",
            command=("tars-browser", "check", url),
            ecosystem="browser",
            evidence=f"web target {url}",
        )
        return GateReport(
            root=str(root),
            results=(
                CheckResult(
                    check=check,
                    status="passed",
                    exit_code=0,
                    duration_s=0.1,
                    stdout="[ref=1] button",
                ),
            ),
        )

    # repo gate passes (has one passing check)
    def verify_pass(root: Path, **_kwargs):
        check = Check(
            kind="test",
            tool="pytest",
            command=("pytest",),
            ecosystem="python",
            evidence="pyproject.toml",
        )
        return GateReport(
            root=str(root),
            results=(
                CheckResult(check=check, status="passed", exit_code=0, duration_s=0.1),
            ),
        )

    engine = Engine(
        provider=lambda *_a, **_k: _response("done"),
        classify=lambda _t: Classification(1, "medium", True, (), "why"),
        verify=verify_pass,
        verify_web=verify_web,
    )
    spec = RunPromptSpec(
        "verify web",
        web=WebTarget(url="http://example.com", steps=(), expect=({"text": "x"},)),
    )
    report = engine.run(spec, Session(root=tmp_path, model="fake"), output=StringIO())
    assert report.verification_passed is True


def test_no_web_target_does_not_call_verify_web(tmp_path: Path) -> None:
    from tars.loop.engine import Engine

    called = False

    def verify_web(*_a, **_k):
        nonlocal called
        called = True
        return GateReport(root=str(tmp_path), results=())

    engine = Engine(
        provider=lambda *_a, **_k: _response("done"),
        classify=lambda _t: Classification(1, "medium", True, (), "why"),
        verify=lambda **_k: GateReport(root=str(tmp_path), results=()),
        verify_web=verify_web,
    )
    engine.run(
        RunPromptSpec("plain task"),
        Session(root=tmp_path, model="fake"),
        output=StringIO(),
    )
    assert called is False
