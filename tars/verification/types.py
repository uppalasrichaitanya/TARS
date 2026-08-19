"""Types the verification gate speaks: what to run, what happened, what it means.

One shape per concern:
  Check       — a command we decided to run, and the evidence that made us pick it.
  CheckResult — what happened when we ran it.
  GateReport  — the aggregate, plus the verdict.

The verdict is deliberately three-valued (`pass` / `fail` / `no-checks`).
A gate that returns a boolean has no way to say "I couldn't verify anything",
so it says "pass" — which is the exact failure mode the loop's second core
directive exists to prevent.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

#: What a check is checking. `format` is always run in check-only mode, never
#: as a rewrite. `build` writes artifacts, so it's opt-in (see DEFAULT_KINDS).
CheckKind = Literal["test", "lint", "typecheck", "format", "build"]

#: Ordering is the display and execution order: cheap-and-specific first, so a
#: fail-fast run surfaces the fastest signal instead of waiting on the suite.
ALL_KINDS: tuple[CheckKind, ...] = ("lint", "format", "typecheck", "test", "build")

#: What runs when the caller doesn't say. `build` is excluded on purpose: it
#: writes into the working tree (dist/, target/, .next/), and a verification
#: gate that mutates the repo it's inspecting is a footgun, not a gate.
DEFAULT_KINDS: tuple[CheckKind, ...] = ("lint", "format", "typecheck", "test")

VALID_KINDS: frozenset[str] = frozenset(ALL_KINDS)

CheckStatus = Literal[
    "passed",  # ran, exit 0
    "failed",  # ran, non-zero exit
    "timeout",  # ran, killed at the deadline
    "error",  # could not be launched at all (bad exec, permissions)
    "unavailable",  # the tool is configured but not installed here
    "skipped",  # deliberately not run (filtered out, or fail-fast stopped us)
]

#: Statuses that mean the tool actually executed to completion or deadline.
_EXECUTED: frozenset[str] = frozenset({"passed", "failed", "timeout"})

#: Statuses that mean something is wrong and a human/agent must look.
_FAILED: frozenset[str] = frozenset({"failed", "timeout", "error"})

Verdict = Literal["pass", "fail", "no-checks"]


@dataclass(frozen=True)
class Check:
    """One command the gate intends to run.

    `evidence` is not decoration. When the gate says "I ran mypy", the next
    question is always "why did you think this repo uses mypy?" — recording the
    marker that triggered detection makes the answer auditable instead of a
    black box.
    """

    kind: CheckKind
    tool: str
    command: tuple[str, ...]
    ecosystem: str
    evidence: str

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"unknown check kind {self.kind!r}; expected one of {sorted(VALID_KINDS)}"
            )
        if not isinstance(self.command, tuple):
            object.__setattr__(self, "command", tuple(self.command))
        if not self.command:
            raise ValueError(f"check {self.tool!r} has an empty command")

    @property
    def command_line(self) -> str:
        """The command as a human could paste it back into a shell."""
        return shlex.join(self.command)

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.tool}"


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one check.

    `detail` explains any status the exit code can't: why a tool was
    unavailable, what timed out, which launch error occurred.
    """

    check: Check
    status: CheckStatus
    exit_code: int | None
    duration_s: float
    stdout: str = ""
    stderr: str = ""
    detail: str | None = None
    #: The argv actually executed, after resolving the tool to a real path.
    resolved_command: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "passed"

    @property
    def failed(self) -> bool:
        """True only for real problems — a missing tool is not a failure."""
        return self.status in _FAILED

    @property
    def executed(self) -> bool:
        return self.status in _EXECUTED

    @property
    def output(self) -> str:
        """stdout and stderr in one stream, which is how tools mean them."""
        parts = [p for p in (self.stdout, self.stderr) if p.strip()]
        return "\n".join(parts)

    def tail(self, lines: int = 40) -> str:
        """Last `lines` lines of output — where tool summaries live."""
        if lines <= 0:
            return ""
        collected = self.output.splitlines()
        return "\n".join(collected[-lines:])


def _truncate_tail(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    return text[-limit:], True


@dataclass(frozen=True)
class GateReport:
    """Everything one gate run produced."""

    root: str
    results: tuple[CheckResult, ...] = ()
    duration_s: float = 0.0
    #: Checks that were detected but filtered out before running, kept so the
    #: report can explain what it chose not to do.
    not_selected: tuple[Check, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("results", "not_selected", "notes"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))

    @property
    def verdict(self) -> Verdict:
        if any(r.failed for r in self.results):
            return "fail"
        if any(r.executed for r in self.results):
            return "pass"
        return "no-checks"

    @property
    def passed(self) -> bool:
        """True only when something ran and nothing failed."""
        return self.verdict == "pass"

    @property
    def executed(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.executed)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.failed)

    @property
    def skipped(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if not r.executed)

    def kinds_covered(self) -> tuple[str, ...]:
        seen: list[str] = []
        for result in self.executed:
            if result.check.kind not in seen:
                seen.append(result.check.kind)
        return tuple(seen)

    def missing_kinds(self, expected: Iterable[str] = DEFAULT_KINDS) -> tuple[str, ...]:
        """Kinds the caller wanted that no check actually covered.

        This is the honest-reporting hook: "tests passed" means much less when
        the repo has no type-checker at all, and the loop should be able to say
        which parts of the surface simply don't exist here.
        """
        covered = set(self.kinds_covered())
        return tuple(k for k in expected if k not in covered)

    def summary_line(self) -> str:
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        parts = [f"{counts[s]} {s}" for s in sorted(counts)]
        detail = ", ".join(parts) if parts else "no checks detected"
        return f"{self.verdict.upper()} — {detail} in {self.duration_s:.1f}s"

    def to_dict(self, max_output_chars: int = 4000) -> dict[str, Any]:
        """JSON-ready form.

        Output is tail-truncated by default: this payload is meant to be read
        back by an agent, and an untruncated test log would swamp its context.
        """
        results: list[dict[str, Any]] = []
        for result in self.results:
            stdout, stdout_cut = _truncate_tail(result.stdout, max_output_chars)
            stderr, stderr_cut = _truncate_tail(result.stderr, max_output_chars)
            results.append(
                {
                    "kind": result.check.kind,
                    "tool": result.check.tool,
                    "ecosystem": result.check.ecosystem,
                    "evidence": result.check.evidence,
                    "command": list(result.check.command),
                    "command_line": result.check.command_line,
                    "resolved_command": list(result.resolved_command),
                    "status": result.status,
                    "exit_code": result.exit_code,
                    "duration_s": round(result.duration_s, 3),
                    "detail": result.detail,
                    "stdout": stdout,
                    "stdout_truncated": stdout_cut,
                    "stderr": stderr,
                    "stderr_truncated": stderr_cut,
                }
            )
        return {
            "root": self.root,
            "verdict": self.verdict,
            "passed": self.passed,
            "duration_s": round(self.duration_s, 3),
            "summary": self.summary_line(),
            "kinds_covered": list(self.kinds_covered()),
            "missing_kinds": list(self.missing_kinds()),
            "results": results,
            "not_selected": [
                {
                    "kind": c.kind,
                    "tool": c.tool,
                    "command_line": c.command_line,
                    "evidence": c.evidence,
                }
                for c in self.not_selected
            ],
            "notes": list(self.notes),
        }


def sort_checks(checks: Sequence[Check]) -> tuple[Check, ...]:
    """Order checks by kind, cheapest signal first, stable within a kind."""
    order = {kind: i for i, kind in enumerate(ALL_KINDS)}
    return tuple(sorted(checks, key=lambda c: order.get(c.kind, len(order))))
