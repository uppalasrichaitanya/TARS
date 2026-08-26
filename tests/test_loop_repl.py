"""The persistent REPL dispatches commands without terminal globals."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import subprocess

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
        Session(root=tmp_path, model="fake", tier_models={"0": "cheap-fake"}),
        stdin=StringIO("/status\n/route fix typo\n/help\n/exit\n"),
        stdout=out,
        classify=classify,
    )
    assert repl.run() == 0
    text = out.getvalue()
    assert "fake" in text and "tier 0" in text
    assert '"tier_0_model": "cheap-fake"' in text
    assert "precedence: explicit config > TARS_TIER0_MODEL env > unset" in text
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


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_undo_restores_a_clean_tree_edit_from_checkpoint(tmp_path: Path) -> None:
    from tars.loop.repl import Repl
    from tars.loop.tools import Toolbox
    from tars.loop.types import ToolCall

    _git(tmp_path, "init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    _git(tmp_path, "add", "code.py")
    _git(
        tmp_path,
        "-c",
        "user.name=TARS Test",
        "-c",
        "user.email=tars@example.invalid",
        "commit",
        "-qm",
        "base",
    )
    session = Session(root=tmp_path, model="fake")
    session.ensure_checkpoint()
    tools = Toolbox(
        tmp_path, confirm=lambda _preview: True, on_mutation=session.note_mutation
    )
    assert tools.dispatch(
        ToolCall(
            "1",
            "apply_edit",
            {"path": "code.py", "hunks": [{"old": "x = 1", "new": "x = 2"}]},
        )
    ).ok

    out = StringIO()
    Repl(session, stdin=StringIO("/undo\n/exit\n"), stdout=out).run()

    assert session.checkpoint_sha
    assert _git(tmp_path, "cat-file", "-t", session.checkpoint_sha) == "commit"
    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "x = 1\n"


def test_undo_preserves_unrelated_dirty_worktree_change(tmp_path: Path) -> None:
    from tars.loop.repl import Repl
    from tars.loop.tools import Toolbox
    from tars.loop.types import ToolCall

    _git(tmp_path, "init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("committed\n", encoding="utf-8")
    _git(tmp_path, "add", "code.py", "notes.txt")
    _git(
        tmp_path,
        "-c",
        "user.name=TARS Test",
        "-c",
        "user.email=tars@example.invalid",
        "commit",
        "-qm",
        "base",
    )
    (tmp_path / "notes.txt").write_text("user's unsaved work\n", encoding="utf-8")
    session = Session(root=tmp_path, model="fake")
    session.ensure_checkpoint()
    tools = Toolbox(
        tmp_path, confirm=lambda _preview: True, on_mutation=session.note_mutation
    )
    assert tools.dispatch(
        ToolCall(
            "1",
            "apply_edit",
            {"path": "code.py", "hunks": [{"old": "x = 1", "new": "x = 2"}]},
        )
    ).ok

    Repl(session, stdin=StringIO("/undo\n/exit\n"), stdout=StringIO()).run()

    assert (tmp_path / "code.py").read_text(encoding="utf-8") == "x = 1\n"
    assert (tmp_path / "notes.txt").read_text(
        encoding="utf-8"
    ) == "user's unsaved work\n"


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


def test_repl_reports_unconfigured_provider_and_returns_to_prompt(
    tmp_path: Path,
) -> None:
    from tars.loop.repl import Repl
    from tars.providers.errors import ProviderNotConfiguredError

    class BrokenEngine:
        def run(self, spec, session, *, output):
            raise ProviderNotConfiguredError("missing")

    out = StringIO()
    repl = Repl(
        Session(root=tmp_path),
        stdin=StringIO("/run fix it\n/status\n/exit\n"),
        stdout=out,
        engine=BrokenEngine(),
    )
    assert repl.run() == 0
    text = out.getvalue()
    assert "no Anthropic credentials" in text
    assert '"model"' in text
    assert "Traceback" not in text


def test_resume_lists_same_root_then_loads_explicit_id(tmp_path: Path) -> None:
    from tars.loop.repl import Repl
    from tars.loop.session_store import SessionStore
    from tars.providers import Message

    store = SessionStore(tmp_path / "sessions")
    saved = Session(root=tmp_path, model="saved-model", session_id="saved", store=store)
    saved.record(Message.user("continue this"))
    out = StringIO()
    repl = Repl(
        Session(root=tmp_path, model="fake", store=store),
        stdin=StringIO("/resume\n/resume saved\n/exit\n"),
        stdout=out,
        session_store=store,
    )
    assert repl.run() == 0
    text = out.getvalue()
    assert "saved" in text and "explicit ID" in text
    assert "saved-model" in text and "turns=1" in text


def test_resume_does_not_silently_replace_nonempty_session(tmp_path: Path) -> None:
    from tars.loop.repl import Repl
    from tars.loop.session_store import SessionStore
    from tars.providers import Message

    store = SessionStore(tmp_path / "sessions")
    Session(root=tmp_path, session_id="saved", store=store)
    current = Session(root=tmp_path, store=store)
    current.record(Message.user("unsaved context"))
    out = StringIO()
    Repl(
        current,
        stdin=StringIO("/resume saved\n/exit\n"),
        stdout=out,
        session_store=store,
    ).run()
    assert "refusing to replace non-empty session" in out.getvalue()


def test_resume_treats_a_last_verdict_as_nonempty_state(tmp_path: Path) -> None:
    from tars.loop.repl import Repl
    from tars.loop.session_store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    Session(root=tmp_path, session_id="saved", store=store)
    current = Session(root=tmp_path, store=store)
    current.last_verdict = "pass"
    out = StringIO()
    Repl(
        current,
        stdin=StringIO("/resume saved\n/exit\n"),
        stdout=out,
        session_store=store,
    ).run()
    assert "refusing to replace non-empty session" in out.getvalue()


def test_clear_allows_resume_and_status_lists_metadata(tmp_path: Path) -> None:
    from tars.loop.repl import Repl
    from tars.loop.session_store import SessionStore
    from tars.providers import Message

    store = SessionStore(tmp_path / "sessions")
    saved = Session(root=tmp_path, model="saved-model", session_id="saved", store=store)
    saved.record_event("run_started", {"task": "Fix the parser"})
    saved.record(Message.user("hello"))
    saved.record_event("verification", {"verdict": "pass"})
    current = Session(root=tmp_path, store=store)
    current.record(Message.user("discard me"))
    out = StringIO()
    Repl(
        current,
        stdin=StringIO("/clear\n/resume saved\n/status\n/exit\n"),
        stdout=out,
        session_store=store,
    ).run()
    text = out.getvalue()
    assert "resumed saved" in text
    assert "label=Fix the parser" in text
    assert "updated=" in text and "verdict=pass" in text
