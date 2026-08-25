"""The built-in executor: this machine, this repo, stdlib only.

Two properties make it safe to hand to a loop nobody is watching.

**Nothing is written until every hunk has resolved.** The spec asks for
incremental application "so a failure is localized, not a full-file corruption"
(`master_agent_loop.md:105`). This goes one better than applying hunk by hunk:
hunks are resolved against an in-memory copy and the file is replaced once,
atomically. There is no window in which the file on disk is half-edited, so a
failure on hunk 4 of 5 is not a rollback — it is a write that never happened.

**Anchors must be unique.** A `Hunk` whose anchor appears twice is refused as
`ambiguous` rather than applied to the first match. Guessing which occurrence was
meant is how an agent corrupts working code while reporting success.

Nothing here imports a third-party package, and nothing here invokes a shell.
Command lookup is delegated to the verification gate's own resolver so that ACT
and VERIFY agree on which `ruff` is this repo's ruff.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from ..verification import build_child_env, resolve_executable
from .base import (
    Executor,
    atomic_write,
    failure,
    invalidate_bytecode,
    resolve_inside,
    unified_diff,
)
from .types import ActionResult, Command, Edit


class LocalExecutor(Executor):
    """Applies edits and runs commands directly, inside one repo root."""

    name = "local"

    def __init__(
        self,
        root: Path | str = ".",
        *,
        dry_run: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.root = Path(root)
        #: When set, every action reports what it *would* do and changes nothing.
        self.dry_run = dry_run
        self._env = env

    # ── edits ────────────────────────────────────────────────────────────

    def apply(self, edit: Edit) -> ActionResult:
        action = edit.display
        target = resolve_inside(self.root, edit.path)
        if target is None:
            return failure(
                action,
                "denied",
                f"{edit.path!r} resolves outside the repo root {str(self.root)!r}",
            )

        if edit.kind == "delete":
            return self._delete(action, edit, target)
        if edit.kind == "create":
            return self._create(action, edit, target)
        if edit.kind == "overwrite":
            return self._overwrite(action, edit, target)
        return self._replace(action, edit, target)

    def _replace(self, action: str, edit: Edit, target: Path) -> ActionResult:
        if not target.is_file():
            return failure(action, "not-found", f"{edit.path} is not a file")

        before = self._read(target)
        if before is None:
            return failure(action, "crashed", f"{edit.path} is not readable as text")

        after = before
        for position, hunk in enumerate(edit.hunks, start=1):
            found = after.count(hunk.old)
            if found == 0:
                return failure(
                    action,
                    "not-found",
                    f"hunk {position} of {len(edit.hunks)}: anchor not present in "
                    f"{edit.path}. Nothing was written.",
                )
            if found > 1:
                return failure(
                    action,
                    "ambiguous",
                    f"hunk {position} of {len(edit.hunks)}: anchor appears {found} "
                    f"times in {edit.path}; include more surrounding context so it "
                    f"matches once. Nothing was written.",
                )
            after = after.replace(hunk.old, hunk.new, 1)

        return self._commit(action, edit.path, target, before, after)

    def _create(self, action: str, edit: Edit, target: Path) -> ActionResult:
        if target.exists():
            return failure(
                action,
                "conflict",
                f"{edit.path} already exists; use an overwrite edit to replace it "
                f"on purpose",
            )
        return self._commit(action, edit.path, target, "", edit.content or "")

    def _overwrite(self, action: str, edit: Edit, target: Path) -> ActionResult:
        if not target.is_file():
            return failure(
                action,
                "not-found",
                f"{edit.path} does not exist; use a create edit to make a new file",
            )
        before = self._read(target)
        if before is None:
            return failure(action, "crashed", f"{edit.path} is not readable as text")
        return self._commit(action, edit.path, target, before, edit.content or "")

    def _delete(self, action: str, edit: Edit, target: Path) -> ActionResult:
        if not target.exists():
            return failure(action, "not-found", f"{edit.path} does not exist")
        if target.is_dir():
            return failure(action, "denied", f"{edit.path} is a directory")

        before = self._read(target) or ""
        diff = unified_diff(edit.path, before, "")
        if self.dry_run:
            return ActionResult(
                ok=True, action=action, detail="dry run: nothing written", diff=diff
            )
        try:
            target.unlink()
        except OSError as exc:
            return failure(action, "crashed", f"could not delete {edit.path}: {exc}")
        invalidate_bytecode(target)
        return ActionResult(ok=True, action=action, diff=diff, touched=(edit.path,))

    def _commit(
        self,
        action: str,
        display_path: str,
        target: Path,
        before: str,
        after: str,
    ) -> ActionResult:
        diff = unified_diff(display_path, before, after)
        if self.dry_run:
            return ActionResult(
                ok=True, action=action, detail="dry run: nothing written", diff=diff
            )
        try:
            atomic_write(target, after)
        except OSError as exc:
            return failure(action, "crashed", f"could not write {display_path}: {exc}")
        invalidate_bytecode(target)
        return ActionResult(ok=True, action=action, diff=diff, touched=(display_path,))

    @staticmethod
    def _read(target: Path) -> str | None:
        try:
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    # ── commands ─────────────────────────────────────────────────────────

    def run(self, command: Command) -> ActionResult:
        action = command.display
        cwd = (
            self.root if command.cwd is None else resolve_inside(self.root, command.cwd)
        )
        if cwd is None:
            return failure(
                action,
                "denied",
                f"cwd {command.cwd!r} resolves outside the repo root",
            )

        if self.dry_run:
            return ActionResult(
                ok=True, action=action, detail="dry run: command not executed"
            )

        env = build_child_env(self._env)
        program = resolve_executable(command.argv[0], self.root, env=env)
        if program is None:
            return failure(
                action,
                "not-found",
                f"{command.argv[0]!r} is not installed for this repo "
                f"(looked in the repo's venv/node_modules, then PATH)",
            )

        argv = (program, *command.argv[1:])
        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                list(argv),
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                # The child leads its own process group, so a timeout can take
                # out a whole tree rather than orphaning what the shell spawned.
                start_new_session=hasattr(os, "setsid"),
            )
        except OSError as exc:
            return failure(
                action,
                "crashed",
                f"could not launch {action}: {exc}",
                duration_s=time.monotonic() - started,
            )

        try:
            stdout, stderr = proc.communicate(timeout=command.timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            stdout, stderr = proc.communicate()
            return failure(
                action,
                "timeout",
                f"killed after exceeding the {command.timeout}s timeout",
                stdout=stdout or "",
                stderr=stderr or "",
                duration_s=time.monotonic() - started,
            )

        duration = time.monotonic() - started
        if proc.returncode == 0:
            return ActionResult(
                ok=True,
                action=action,
                stdout=stdout or "",
                stderr=stderr or "",
                exit_code=0,
                duration_s=duration,
            )
        return failure(
            action,
            "failed",
            f"exit code {proc.returncode}",
            stdout=stdout or "",
            stderr=stderr or "",
            exit_code=proc.returncode,
            duration_s=duration,
        )


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """SIGKILL the child's process group, falling back to the child alone."""
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
