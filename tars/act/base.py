"""The one interface every executor implements.

`Executor.apply(edit)` and `Executor.run(command)` are the whole contract — the
same shape as `tars.providers.base.Provider`, which has exactly one method for
the same reason: a narrow seam is one a second implementation can actually
satisfy.

Why this seam exists at all. ACT is the only phase whose work is mechanical
rather than judgment, so it is the only phase an outside tool could do for us.
Keeping it behind an interface means an external harness can be adopted later as
one more `Executor` without the loop above it changing, and dropped again just as
cheaply. Everything else — ORIENT, CLASSIFY, PLAN, VERIFY, REFLECT, REPORT —
stays TARS's own, because that discipline is the point of TARS.

Executors never raise for an expected outcome. A missing anchor, an ambiguous
one, a timeout, a program that isn't installed: all of those are `ActionResult`s
with a reason, because the loop's job is to read what happened and decide, not to
catch six exception types.
"""

from __future__ import annotations

import abc
import difflib
import os
from pathlib import Path

from .types import ActionResult, Command, Edit, FailureReason


class Executor(abc.ABC):
    """Base class for anything that can carry out ACT's mechanical work."""

    #: Registry key, and what a report names as having done the work.
    name: str = "unknown"

    @abc.abstractmethod
    def apply(self, edit: Edit) -> ActionResult:
        """Make one change to one file."""

    @abc.abstractmethod
    def run(self, command: Command) -> ActionResult:
        """Run one program to completion."""

    def __repr__(self) -> str:  # pragma: no cover - debugging affordance
        return f"<{type(self).__name__} name={self.name!r}>"


# ── helpers shared by executors ──────────────────────────────────────────


def failure(
    action: str,
    reason: FailureReason,
    detail: str,
    **extra: object,
) -> ActionResult:
    """A refusal that says which kind of refusal it is."""
    return ActionResult(ok=False, action=action, reason=reason, detail=detail, **extra)  # type: ignore[arg-type]


def unified_diff(path: str, before: str, after: str) -> str:
    """A readable diff of one file's change.

    Produced for successes *and* for dry runs, so a preview and a report are the
    same artifact rather than two things that can drift.
    """
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def resolve_inside(root: Path, candidate: str) -> Path | None:
    """Resolve `candidate` under `root`, or None if it escapes.

    Symlinks are resolved *before* the containment check, so a link inside the
    root pointing outside it is caught. An executor that can be aimed at any path
    on the machine is not something to hand an unattended loop.
    """
    root = root.resolve()
    target = Path(candidate)
    if not target.is_absolute():
        target = root / target
    resolved = _resolve_existing(target)
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _resolve_existing(target: Path) -> Path:
    """`Path.resolve()`, but tolerant of the file not existing yet.

    A create names a path that is not there, so the nearest existing ancestor is
    resolved and the missing tail re-attached — enough to catch `../` and a
    symlinked parent directory.
    """
    if target.exists() or target.is_symlink():
        return target.resolve()

    tail = [target.name]
    parent = target.parent
    while not parent.exists() and parent.parent != parent:
        tail.append(parent.name)
        parent = parent.parent

    resolved = parent.resolve()
    for part in reversed(tail):
        resolved = resolved / part
    return resolved


def atomic_write(path: Path, text: str) -> None:
    """Write `text` to `path` without ever leaving a half-written file.

    Same technique as `tars.memory.store`'s partition writer, and for the same
    reason: an interrupted run is the expected case here, not a rare one, and a
    truncated source file is worse than an unapplied edit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tars-tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def invalidate_bytecode(path: Path) -> None:
    """Drop cached bytecode for a source file just written or removed.

    Not housekeeping — correctness. CPython validates a `.pyc` against the
    source's mtime *in whole seconds* plus its size, so an edit that changes
    neither leaves the stale cache looking valid and the next process imports
    pre-edit code. `a - b` to `a + b` is exactly such an edit, and a loop that
    edits then immediately runs its tests hits this constantly: VERIFY grades
    code ACT already replaced, and reports a fix as failed.

    Every interpreter tag is removed, not only this interpreter's, because one
    stale tag is enough to poison a run under that version.
    """
    if path.suffix != ".py":
        return
    cache_dir = path.parent / "__pycache__"
    if not cache_dir.is_dir():
        return
    for stale in cache_dir.glob(f"{path.stem}.*.pyc"):
        try:
            stale.unlink()
        except OSError:  # pragma: no cover - a cache we can't clear is not fatal
            pass
