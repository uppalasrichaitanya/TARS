"""Running one check, uniformly.

Every tool the gate knows about — pytest, ruff, tsc, cargo, make — goes through
this one function and comes back as the same `CheckResult`. That uniformity is
the module's whole point: callers above never branch on which tool ran.

Two behaviours here are load-bearing and easy to get wrong:

1. Tool resolution prefers the *repo's* tools (`.venv/bin`, `node_modules/.bin`)
   over whatever happens to be on PATH. A gate that lints with a different ruff
   than the repo pins reports results the repo's own CI would never produce.

2. A timeout kills the process *group*, not just the direct child. Shelling out
   to `npm test` spawns a tree; killing the shell alone orphans a test runner
   that keeps holding its port long after the gate has moved on.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

from .types import Check, CheckResult

#: Ten minutes. Long enough for a real test suite, short enough that a hung
#: watch-mode runner doesn't strand an unattended agent loop overnight.
DEFAULT_TIMEOUT_S = 600.0

#: Where a repo keeps its own tools, in preference order. Scripts/ is the
#: Windows layout of the same idea.
_LOCAL_BIN_DIRS = (
    Path(".venv") / "bin",
    Path(".venv") / "Scripts",
    Path("venv") / "bin",
    Path("venv") / "Scripts",
    Path("env") / "bin",
    Path("node_modules") / ".bin",
)

#: Forced into the child environment unless the caller says otherwise.
#: CI=1 is what makes vitest/jest run once instead of entering watch mode and
#: sitting there until the timeout; the color vars keep ANSI escapes out of
#: captured output that an agent has to read.
_CHILD_ENV_DEFAULTS = {
    "CI": "1",
    "NO_COLOR": "1",
    "FORCE_COLOR": "0",
    "PYTHONUNBUFFERED": "1",
}


def resolve_executable(
    program: str,
    root: Path | str,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Find `program` for the repo at `root`, or None if it isn't installed.

    Order: an explicit path (relative to the repo root) > the repo's own bin
    dirs > PATH > this interpreter, for `python` only.
    """
    root = Path(root)

    # An explicit path means the caller already decided; don't second-guess it.
    if os.sep in program or (os.altsep and os.altsep in program):
        candidate = Path(program)
        resolved = candidate if candidate.is_absolute() else root / candidate
        return str(resolved) if _is_executable(resolved) else None

    for rel in _LOCAL_BIN_DIRS:
        found = shutil.which(program, path=str(root / rel))
        if found:
            return found

    path = None if env is None else env.get("PATH")
    found = shutil.which(program, path=path)
    if found:
        return found

    # Last resort for Python itself: the interpreter running the gate always
    # exists, even in a stripped container with no `python` on PATH.
    if program in ("python", "python3") and sys.executable:
        return sys.executable
    return None


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def build_child_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The parent environment, plus the safety defaults, plus caller overrides.

    Inheriting matters: tools need PATH, HOME, and language-specific caches to
    behave the way they do for the human running them by hand.
    """
    child = dict(os.environ)
    child.update(_CHILD_ENV_DEFAULTS)
    if env:
        child.update(env)
    return child


def run_check(
    check: Check,
    root: Path | str,
    *,
    timeout: float | None = DEFAULT_TIMEOUT_S,
    env: Mapping[str, str] | None = None,
) -> CheckResult:
    """Run one check and translate whatever happens into a `CheckResult`.

    Never raises for an expected outcome. A missing tool, a non-zero exit, a
    timeout, and a failed exec are all statuses, because the gate's job is to
    report the state of the repo's check surface, not to abort on the first
    surprise.
    """
    root = Path(root)
    child_env = build_child_env(env)
    program = check.command[0]
    resolved = resolve_executable(program, root, env=child_env)
    if resolved is None:
        return CheckResult(
            check=check,
            status="unavailable",
            exit_code=None,
            duration_s=0.0,
            detail=(
                f"{program!r} is not installed for this repo "
                f"(looked in the repo's venv/node_modules, then PATH)"
            ),
        )

    argv = (resolved, *check.command[1:])
    started = time.monotonic()
    try:
        proc = _spawn(argv, root, child_env)
    except OSError as exc:
        return CheckResult(
            check=check,
            status="error",
            exit_code=None,
            duration_s=time.monotonic() - started,
            detail=f"could not launch {check.command_line!r}: {exc}",
            resolved_command=argv,
        )

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        stdout, stderr = proc.communicate()
    duration = time.monotonic() - started

    if timed_out:
        return CheckResult(
            check=check,
            status="timeout",
            exit_code=None,
            duration_s=duration,
            stdout=stdout or "",
            stderr=stderr or "",
            detail=f"killed after exceeding the {timeout}s timeout",
            resolved_command=argv,
        )

    return CheckResult(
        check=check,
        status="passed" if proc.returncode == 0 else "failed",
        exit_code=proc.returncode,
        duration_s=duration,
        stdout=stdout or "",
        stderr=stderr or "",
        detail=None if proc.returncode == 0 else f"exit code {proc.returncode}",
        resolved_command=argv,
    )


def _spawn(
    argv: Sequence[str],
    root: Path,
    env: Mapping[str, str],
) -> subprocess.Popen[str]:
    kwargs: dict[str, object] = {}
    if hasattr(os, "setsid"):
        # New session => the child leads its own process group, so a timeout can
        # take out the whole tree in one signal.
        kwargs["start_new_session"] = True
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):  # pragma: no cover - Windows
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    return subprocess.Popen(  # noqa: S603 - argv is a resolved path + fixed args
        list(argv),
        cwd=str(root),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # Tools emit whatever their locale gives them; a stray byte in a test
        # name must not turn into a gate crash.
        errors="replace",
        **kwargs,  # type: ignore[arg-type]
    )


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """SIGKILL the child's whole process group, falling back to the child."""
    if hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), 9)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except OSError:  # pragma: no cover - already dead
        pass
