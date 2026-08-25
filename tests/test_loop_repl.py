"""The persistent REPL dispatches commands without terminal globals."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from tars.loop.session import Session
from tars.loop.types import RunReport
from tars.memory import Candidate, MemoryEntry, ReflectionReport
from tars.router import Classification
from tars.verification import GateReport


def test_repl_dispatches_status_route_help_and_exit(tmp_path: Path) -> None:
    from tars.loop.repl import Repl

    out = StringIO()
    routed: list[str] = []

    def classify(task: str) -> Classification:
        routed.append(task)
        return Classification(0, "low", True, (), "small")

    repl = Repl(
        Session(root=tmp_path, model="fake"),
        stdin=StringIO("/status\n/route fix typo\n/help\n/exit\n"),
        stdout=out,
        classify=classify,
    )
    assert repl.run() == 0
    text = out.getvalue()
    assert "fake" in text and "tier 0" in text
    assert routed == ["fix typo"]


def test_repl_runs_engine_and_handles_unknown_command(tmp_path: Path) -> None:
    from tars.loop.repl import Repl

    out = StringIO()
    seen: list[str] = []

    class FakeEngine:
        def run(self, spec, session, *, output):
            seen.append(spec.task)
            return RunReport(
                task=spec.task,
                verdict="tier 1",
                verification_passed=True,
                touched=("a.py",),
                proposed=3,
                committed=1,
                queued=2,
                declined=("apply_edit: edit declined",),
                uncertainties=("uncertainty verbatim",),
            )

    repl = Repl(
        Session(root=tmp_path, model="fake"),
        stdin=StringIO("/run fix it\n/wat\n/exit\n"),
        stdout=out,
        engine=FakeEngine(),
    )
    assert repl.run() == 0
    assert seen == ["fix it"]
    assert "unknown command" in out.getvalue()
    text = out.getvalue()
    assert "task: fix it" in text
    assert "touched: a.py" in text
    assert "memory: proposed=3 committed=1 queued=2" in text
    assert "declined: apply_edit: edit declined" in text
    assert "uncertainty verbatim" in text


def test_repl_handles_eof_and_ctrl_c_cleanly(tmp_path: Path) -> None:
    from tars.loop.repl import Repl

    out = StringIO()
    assert Repl(Session(root=tmp_path), stdin=StringIO(""), stdout=out).run() == 0

    class Interrupting:
        def __init__(self) -> None:
            self.interrupted = False

        def readline(self) -> str:
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt
            return ""

        def isatty(self) -> bool:
            return False

    out = StringIO()
    assert Repl(Session(root=tmp_path), stdin=Interrupting(), stdout=out).run() == 0
    assert "cancelled" in out.getvalue()


def test_repl_dispatches_verify_memory_approve_and_pillars(tmp_path: Path) -> None:
    from tars.loop.repl import Repl

    candidate = Candidate(
        text="Use stdlib only.",
        kind="decision",
        domains=("loop",),
        why="Keeps the gate green.",
        confidence=0.95,
    )
    entry = MemoryEntry.from_candidate(candidate)
    proposed: list[object] = []

    def propose(source, **_kwargs):
        proposed.append(source)
        return ReflectionReport(session="repl", candidates=(candidate,))

    class Store:
        def approve(self, entry_id: str, session: str = "") -> MemoryEntry | None:
            assert entry_id == entry.id
            assert session == "repl"
            return entry

    out = StringIO()
    session = Session(root=tmp_path, model="fake")
    repl = Repl(
        session,
        stdin=StringIO(
            f"/verify\n/memory\n/memory approve {entry.id}\n/pillars\n/exit\n"
        ),
        stdout=out,
        verify=lambda **_kwargs: GateReport(root=str(tmp_path)),
        propose=propose,
        store=Store(),
    )
    assert repl.run() == 0
    assert session.last_verdict == "no-checks"
    assert proposed
    text = out.getvalue()
    assert entry.id in text
    assert "approved" in text
    assert "ORIENT" in text and "REPORT" in text
