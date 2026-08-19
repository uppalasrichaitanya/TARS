"""The gate itself: detect → select → run → report.

Selection is the step that turns "candidates" into "what we ran". Per
(ecosystem, kind) it takes the first candidate that is actually installed; if
none are, it keeps the first anyway so the report can say `unavailable` with the
evidence that made it look for that tool. Silently dropping it would shrink the
check surface without telling anyone, which is the same lie as reporting a pass.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from .detect import OVERRIDES_PATH, detect_candidates, load_overrides
from .runner import DEFAULT_TIMEOUT_S, build_child_env, resolve_executable, run_check
from .types import DEFAULT_KINDS, Check, CheckResult, GateReport

OnStart = Callable[[Check], None]
OnResult = Callable[[CheckResult], None]


def select_checks(
    root: Path | str,
    *,
    kinds: Iterable[str] = DEFAULT_KINDS,
    env: Mapping[str, str] | None = None,
    overrides: dict[str, tuple[str, ...] | None] | None = None,
) -> tuple[tuple[Check, ...], tuple[Check, ...]]:
    """Decide what to run. Returns (selected, rejected).

    Rejected candidates are kept so a report can explain the road not taken —
    "we used ruff, not the flake8 your setup.cfg configures, because flake8
    isn't installed" is a much more useful line than silence.
    """
    root = Path(root)
    candidates = detect_candidates(root, kinds=kinds, overrides=overrides)
    child_env = build_child_env(env)

    # dict preserves insertion order, and candidates arrive sorted by kind, so
    # groups come out cheapest-signal-first.
    groups: dict[tuple[str, str], list[Check]] = {}
    for candidate in candidates:
        key = (candidate.check.kind, candidate.check.ecosystem)
        groups.setdefault(key, []).append(candidate.check)

    selected: list[Check] = []
    rejected: list[Check] = []
    for group in groups.values():
        winner = next(
            (c for c in group if resolve_executable(c.command[0], root, env=child_env)),
            group[0],
        )
        selected.append(winner)
        rejected.extend(c for c in group if c is not winner)
    return tuple(selected), tuple(rejected)


def verify(
    root: Path | str = ".",
    *,
    kinds: Iterable[str] = DEFAULT_KINDS,
    timeout: float | None = DEFAULT_TIMEOUT_S,
    fail_fast: bool = False,
    env: Mapping[str, str] | None = None,
    on_start: OnStart | None = None,
    on_result: OnResult | None = None,
) -> GateReport:
    """Run this repo's check surface and report what happened.

    Raises FileNotFoundError for a root that isn't a directory. Every other
    outcome — no checks found, a missing tool, a failing suite, a timeout — is a
    report, not an exception: callers need the partial picture, not a traceback.
    """
    root_path = Path(root).resolve()
    kinds = tuple(kinds)
    overrides, notes = load_overrides(root_path)
    selected, rejected = select_checks(
        root_path, kinds=kinds, env=env, overrides=overrides
    )

    if not selected:
        notes = (
            *notes,
            (
                f"no checks detected in {root_path} for kinds: {', '.join(kinds) or '(none)'}"
                f" — add {OVERRIDES_PATH} to declare this repo's check surface"
            ),
        )

    started = time.monotonic()
    results: list[CheckResult] = []
    stop = False
    for check in selected:
        if stop:
            results.append(_skipped(check))
            continue
        if on_start is not None:
            on_start(check)
        result = run_check(check, root_path, timeout=timeout, env=env)
        results.append(result)
        if on_result is not None:
            on_result(result)
        # Only a real failure short-circuits. An unavailable tool is a coverage
        # gap; stopping on it would hide the checks that could still have run.
        if fail_fast and result.failed:
            stop = True

    return GateReport(
        root=str(root_path),
        results=tuple(results),
        duration_s=time.monotonic() - started,
        not_selected=tuple(rejected),
        notes=tuple(notes),
    )


def _skipped(check: Check) -> CheckResult:
    return CheckResult(
        check=check,
        status="skipped",
        exit_code=None,
        duration_s=0.0,
        detail="not run: an earlier check failed (fail-fast)",
    )


def unavailable_tools(results: Sequence[CheckResult]) -> tuple[str, ...]:
    """Tools a repo asked for that aren't installed here.

    Surfaced separately because it's the one gate outcome a human can fix in a
    single command.
    """
    return tuple(r.check.tool for r in results if r.status == "unavailable")
