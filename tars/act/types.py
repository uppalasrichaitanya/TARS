"""What ACT was asked to do, and what came of it.

Two rules are encoded in these types rather than left to callers, because both
are the kind of discipline that erodes one call site at a time:

* **A failure names why.** `ActionResult` refuses to hold ``ok=False`` without a
  `FailureReason`. The spec's rule is "capture WHY (not-found vs timeout vs
  assertion mismatch) … never blind-retry" (`master_agent_loop.md:119`), and a
  result type that permits a reasonless failure makes that unenforceable.
* **Commands are argv, never a shell string.** `Command("ls -la")` is a
  `TypeError`, not a string that iterates into single characters. Refusing it
  here is what stops the no-shell rule from being a convention.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

#: Ten minutes, matching the verification gate's own default: long enough for a
#: real test suite, short enough that a hung watcher can't strand a loop.
DEFAULT_COMMAND_TIMEOUT_S = 600.0

EditKind = Literal["replace", "create", "overwrite", "delete"]

FailureReason = Literal[
    "not-found",  # the anchor, file or program isn't there
    "ambiguous",  # the anchor matched more than once; guessing is not allowed
    "conflict",  # the file changed underneath us, or already exists
    "timeout",  # exceeded its deadline and was killed
    "denied",  # refused before running: outside the root, or not permitted
    "failed",  # ran to completion and reported failure (non-zero exit)
    "crashed",  # could not be attempted, or raised on the way out
]

VALID_EDIT_KINDS: tuple[str, ...] = ("replace", "create", "overwrite", "delete")
VALID_FAILURE_REASONS: tuple[str, ...] = (
    "not-found",
    "ambiguous",
    "conflict",
    "timeout",
    "denied",
    "failed",
    "crashed",
)

#: Reasons where trying the identical action again could plausibly work — which
#: is only a timeout, and only because the machine may have been loaded.
#: Everything else is a fact about the plan or the world: a missing anchor, an
#: ambiguous one, a suite that legitimately fails. Retrying those is the blind
#: retry `master_agent_loop.md:135` calls a wasted cycle.
RETRYABLE_REASONS: frozenset[str] = frozenset({"timeout"})


@dataclass(frozen=True)
class Hunk:
    """One anchored replacement: find `old` exactly once, put `new` there.

    Anchored rather than line-numbered on purpose. Line numbers go stale the
    moment anything above them moves, and a stale line number applies a change
    to the wrong place silently, where a stale anchor fails loudly.
    """

    old: str
    new: str

    def __post_init__(self) -> None:
        if not self.old:
            raise ValueError("a hunk needs a non-empty anchor to find")
        if self.old == self.new:
            raise ValueError(
                "a hunk whose old and new text are identical changes nothing"
            )


@dataclass(frozen=True)
class Edit:
    """One requested change to one file.

    Four kinds in one type, validated on construction — the same shape
    `tars.providers.types.Message` uses for roles. `create` and `overwrite` stay
    distinct because one asserts the file is absent and the other asserts it is
    present; collapsing them lets a mistyped path create a file instead of
    failing.
    """

    path: str
    kind: EditKind = "replace"
    hunks: tuple[Hunk, ...] = ()
    content: str | None = None

    def __post_init__(self) -> None:
        if not self.path or not self.path.strip():
            raise ValueError("an edit needs a path")
        if self.kind not in VALID_EDIT_KINDS:
            raise ValueError(
                f"unknown edit kind {self.kind!r}; expected one of {VALID_EDIT_KINDS}"
            )
        if not isinstance(self.hunks, tuple):
            object.__setattr__(self, "hunks", tuple(self.hunks))

        if self.kind == "replace":
            if not self.hunks:
                raise ValueError("a replace edit needs at least one hunk")
            if self.content is not None:
                raise ValueError("a replace edit carries hunks, not content")
        elif self.kind in ("create", "overwrite"):
            if self.content is None:
                raise ValueError(f"a {self.kind} edit needs content")
            if self.hunks:
                raise ValueError(f"a {self.kind} edit carries content, not hunks")
        else:  # delete
            if self.content is not None:
                raise ValueError("a delete edit carries no content")
            if self.hunks:
                raise ValueError("a delete edit carries no hunks")

    # ── constructors, so call sites read as intent ───────────────────────

    @classmethod
    def replace(cls, path: str, *hunks: Hunk) -> Edit:
        return cls(path=path, kind="replace", hunks=tuple(hunks))

    @classmethod
    def create(cls, path: str, content: str) -> Edit:
        return cls(path=path, kind="create", content=content)

    @classmethod
    def overwrite(cls, path: str, content: str) -> Edit:
        return cls(path=path, kind="overwrite", content=content)

    @classmethod
    def delete(cls, path: str) -> Edit:
        return cls(path=path, kind="delete")

    @property
    def display(self) -> str:
        return f"{self.kind} {self.path}"


@dataclass(frozen=True)
class Command:
    """One program to run. No shell, ever."""

    argv: tuple[str, ...]
    cwd: str | None = None
    timeout: float | None = DEFAULT_COMMAND_TIMEOUT_S

    def __init__(
        self,
        argv: Sequence[str],
        cwd: str | None = None,
        timeout: float | None = DEFAULT_COMMAND_TIMEOUT_S,
    ) -> None:
        if isinstance(argv, (str, bytes)):
            raise TypeError(
                "argv must be a sequence of arguments, not a string: "
                f"{argv!r} would be read one character at a time. "
                "Pass ['program', 'arg'] — this layer never invokes a shell."
            )
        argv = tuple(argv)
        if not argv or not argv[0]:
            raise ValueError("argv must name a program to run")
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive or None, got {timeout!r}")
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "timeout", timeout)

    @property
    def display(self) -> str:
        """Readable in a log. Not for feeding back to a shell."""
        return shlex.join(self.argv)


@dataclass(frozen=True)
class ActionResult:
    """What happened, in the one shape every executor returns."""

    ok: bool
    action: str
    reason: FailureReason | None = None
    detail: str = ""
    #: Unified diff of what changed — or would change, under dry_run.
    diff: str = ""
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_s: float = 0.0
    #: Paths this action wrote. Empty under dry_run, and empty on failure.
    touched: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.reason is not None and self.reason not in VALID_FAILURE_REASONS:
            raise ValueError(
                f"unknown failure reason {self.reason!r}; "
                f"expected one of {VALID_FAILURE_REASONS}"
            )
        if not self.ok and self.reason is None:
            raise ValueError(
                "a failed action must name a reason: "
                f"{VALID_FAILURE_REASONS}. Silence here is what turns a "
                "diagnosable failure into a blind retry."
            )
        if self.ok and self.reason is not None:
            raise ValueError(f"a successful action cannot carry reason {self.reason!r}")
        if not isinstance(self.touched, tuple):
            object.__setattr__(self, "touched", tuple(self.touched))

    @property
    def retryable(self) -> bool:
        """True when trying the same action again could plausibly work."""
        return self.reason in RETRYABLE_REASONS
