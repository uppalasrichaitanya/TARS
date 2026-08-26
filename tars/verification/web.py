"""Web verification via the lazy, optional browser module.

The core verification gate is for repo check surfaces. This module adds a
parallel, equally lazy check for web surfaces: drive an accessibility tree
and grade expectations. The browser dependency stays optional — a missing
``playwright`` is an unavailable coverage gap, never a crash.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .types import Check, CheckResult, GateReport


def verify_web(
    url: str,
    steps: list[dict[str, Any]] | None,
    expect: list[dict[str, Any]] | None,
    *,
    session_factory=None,
    root: str | Path = ".",
) -> GateReport:
    root_path = Path(root).resolve()
    check = Check(
        kind="test",
        tool="browser",
        command=("tars-browser", "check", url),
        ecosystem="browser",
        evidence=f"web target {url}",
    )
    try:
        from tars.browser.cli import run_browser_task
    except ImportError as exc:  # pragma: no cover - optional dependency
        return GateReport(
            root=str(root_path),
            results=(
                CheckResult(
                    check=check,
                    status="unavailable",
                    exit_code=None,
                    duration_s=0.0,
                    detail=f"browser extra not installed: {exc}",
                ),
            ),
            notes=("browser extra not installed; install tars[browser]",),
        )
    import time

    started = time.monotonic()
    try:
        factory = session_factory  # type: ignore[assignment]
        kwargs: dict[str, Any] = {}
        if factory is not None:
            kwargs["session_factory"] = factory
        code, snapshot = run_browser_task(url, steps, expect, **kwargs)
    except RuntimeError as exc:
        if "Playwright is required" in str(exc):
            return GateReport(
                root=str(root_path),
                results=(
                    CheckResult(
                        check=check,
                        status="unavailable",
                        exit_code=None,
                        duration_s=time.monotonic() - started,
                        detail=str(exc),
                    ),
                ),
                notes=("browser extra not installed; install tars[browser]",),
            )
        return GateReport(
            root=str(root_path),
            results=(
                CheckResult(
                    check=check,
                    status="error",
                    exit_code=None,
                    duration_s=time.monotonic() - started,
                    detail=str(exc),
                    stderr=str(exc),
                ),
            ),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return GateReport(
            root=str(root_path),
            results=(
                CheckResult(
                    check=check,
                    status="error",
                    exit_code=None,
                    duration_s=time.monotonic() - started,
                    detail=str(exc),
                    stderr=str(exc),
                ),
            ),
        )
    duration = time.monotonic() - started
    if code == 0:
        return GateReport(
            root=str(root_path),
            results=(
                CheckResult(
                    check=check,
                    status="passed",
                    exit_code=0,
                    duration_s=duration,
                    stdout=snapshot,
                ),
            ),
        )
    if code == 1:
        return GateReport(
            root=str(root_path),
            results=(
                CheckResult(
                    check=check,
                    status="failed",
                    exit_code=1,
                    duration_s=duration,
                    stdout=snapshot,
                    detail="expectation mismatch",
                ),
            ),
        )
    return GateReport(
        root=str(root_path),
        results=(
            CheckResult(
                check=check,
                status="error",
                exit_code=code,
                duration_s=duration,
                stdout=snapshot,
                detail=f"browser task exited {code}",
            ),
        ),
    )
