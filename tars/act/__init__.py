"""TARS ACT — PHASE 4, isolated: the phase that actually touches the repo.

Edits are anchored, not line-numbered, and an anchor must match exactly once:

    from tars.act import Edit, Hunk, LocalExecutor

    executor = LocalExecutor(".")
    result = executor.apply(Edit.replace("a.py", Hunk("x = 1", "x = 2")))
    result.ok, result.diff, result.touched

Line numbers go stale the moment anything above them moves, and a stale line
number edits the wrong place *silently*. A stale anchor fails loudly. An anchor
that matches twice is refused as ``ambiguous`` rather than applied to the first
hit, because guessing which one was meant is how working code gets corrupted
while the report says success.

Nothing is written until every hunk in an edit has resolved, so a five-hunk edit
that fails on the fourth leaves the file byte-identical — the file is replaced
once, atomically, or not at all. That is stronger than the spec's "apply
incrementally so a failure is localized": there is no partial state to localize.

Every failure names its kind, so the loop can tell a wrong plan from bad luck:

    from tars.act import Command, LocalExecutor

    LocalExecutor(".").run(Command(["pytest", "-q"])).reason

``not-found`` and ``ambiguous`` mean the plan was wrong and a retry would repeat
it; ``failed`` means the program ran and said no; only ``timeout`` is worth
trying again. That distinction is what keeps `master_agent_loop.md`'s "never
blind-retry" enforceable rather than aspirational.

Commands are argv, never a shell string — ``Command("ls -la")`` is a TypeError —
and program lookup is delegated to the verification gate's resolver, so ACT and
VERIFY agree on which ``ruff`` is this repo's ruff.

Pass ``dry_run=True`` to get the diff and change nothing, which is what an
approval prompt needs:

    from tars.act import Edit, Hunk, LocalExecutor

    preview = LocalExecutor(".", dry_run=True)
    preview.apply(Edit.replace("a.py", Hunk("x = 1", "x = 2"))).diff

The executor is chosen by name rather than imported, so a different backend can
be swapped in without the loop above it changing:

    from tars.act import get_executor, known_executors

    get_executor("local")
    known_executors()

Nothing here calls a model, reads a credential, or imports the provider layer.
Deciding *what* to do needs a model; doing it does not.
"""

from __future__ import annotations

from .base import (
    Executor,
    atomic_write,
    failure,
    invalidate_bytecode,
    resolve_inside,
    unified_diff,
)
from .local import LocalExecutor
from .registry import (
    LOCAL,
    ExecutorFactory,
    UnknownExecutorError,
    get_executor,
    install_default_executors,
    known_executors,
    register_executor,
    reset_executors,
)
from .types import (
    DEFAULT_COMMAND_TIMEOUT_S,
    RETRYABLE_REASONS,
    VALID_EDIT_KINDS,
    VALID_FAILURE_REASONS,
    ActionResult,
    Command,
    Edit,
    EditKind,
    FailureReason,
    Hunk,
)

__all__ = [
    "DEFAULT_COMMAND_TIMEOUT_S",
    "LOCAL",
    "RETRYABLE_REASONS",
    "VALID_EDIT_KINDS",
    "VALID_FAILURE_REASONS",
    "ActionResult",
    "Command",
    "Edit",
    "EditKind",
    "Executor",
    "ExecutorFactory",
    "FailureReason",
    "Hunk",
    "LocalExecutor",
    "UnknownExecutorError",
    "atomic_write",
    "failure",
    "get_executor",
    "install_default_executors",
    "invalidate_bytecode",
    "known_executors",
    "register_executor",
    "reset_executors",
    "resolve_inside",
    "unified_diff",
]

install_default_executors()
