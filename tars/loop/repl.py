"""The persistent slash-command interface over one live :class:`Session`."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from typing import Protocol, TextIO

from .. import memory, router, verification
from ..memory.store import MemoryStore
from ..providers.errors import ProviderError, ProviderNotConfiguredError
from ..router import Classification
from ._io import color_enabled, is_tty, paint
from .engine import Engine
from .session import Session
from .types import RunPromptSpec, RunReport


class EngineRunner(Protocol):
    def run(
        self,
        spec: RunPromptSpec,
        session: Session,
        *,
        output: TextIO,
    ) -> RunReport: ...


class InputStream(Protocol):
    def readline(self) -> str: ...

    def isatty(self) -> bool: ...


class MemoryApprover(Protocol):
    def approve(
        self, entry_id: str, session: str = ""
    ) -> memory.MemoryEntry | None: ...


class Repl:
    """Read commands until exit, EOF, or Ctrl-C."""

    def __init__(
        self,
        session: Session | None = None,
        *,
        stdin: InputStream | None = None,
        stdout: TextIO | None = None,
        engine: EngineRunner | None = None,
        classify: Callable[[str], Classification] = router.classify,
        verify: Callable[..., verification.GateReport] = verification.verify,
        propose: Callable[..., memory.ReflectionReport] = memory.propose,
        store: MemoryApprover | None = None,
    ) -> None:
        self.session = session or Session()
        self.stdin = stdin if stdin is not None else sys.stdin
        self.stdout = stdout if stdout is not None else sys.stdout
        self.engine = engine or Engine()
        self.classify = classify
        self.verify = verify
        self.propose = propose
        self.store = store or MemoryStore(root=self.session.root)
        self._color = color_enabled(self.stdout)
        self._prompt = is_tty(self.stdout)

    def run(self) -> int:
        while True:
            try:
                if self._prompt:
                    self._write("tars> ")
                line = self.stdin.readline()
            except KeyboardInterrupt:
                self._write("cancelled\n")
                continue
            if not line:
                return 0
            if self._dispatch(line.strip()):
                return 0

    def _dispatch(self, line: str) -> bool:
        if not line:
            return False
        if not line.startswith("/"):
            self._write("commands start with /; try /help\n")
            return False
        command, _, argument = line.partition(" ")
        argument = argument.strip()
        try:
            if command in ("/exit", "/quit"):
                return True
            if command == "/help":
                self._help()
            elif command == "/run":
                self._run(argument)
            elif command == "/route":
                self._route(argument)
            elif command == "/verify":
                self._verify()
            elif command == "/memory":
                self._memory(argument)
            elif command == "/status":
                self._status()
            elif command == "/pillars":
                self._pillars()
            else:
                self._write(f"unknown command: {command}; try /help\n")
        except ProviderError as exc:
            self._write(_provider_message(exc) + "\n")
        except (OSError, TypeError, ValueError, AttributeError) as exc:
            self._write(f"error: {exc}\n")
        return False

    def _run(self, task: str) -> None:
        if not task:
            self._write("usage: /run <task>\n")
            return
        report = self.engine.run(
            RunPromptSpec(task, auto_confirm=self.session.auto_confirm),
            self.session,
            output=self.stdout,
        )
        write_report(report, self.stdout)

    def _route(self, task: str) -> None:
        if not task:
            self._write("usage: /route <task>\n")
            return
        verdict = self.classify(task)
        self.session.last_classification = verdict
        self._write(f"{_verdict(verdict)}\n")

    def _verify(self) -> None:
        report = self.verify(root=self.session.root)
        self.session.last_verdict = report.verdict
        self._write(f"VERIFY: {report.summary_line()}\n")

    def _memory(self, argument: str) -> None:
        parts = argument.split()
        if parts and parts[0] == "approve":
            if len(parts) != 2:
                self._write("usage: /memory approve <id>\n")
                return
            entry = self.store.approve(parts[1], session="repl")
            self._write(
                f"approved {entry.id}\n" if entry else f"not pending: {parts[1]}\n"
            )
            return
        if parts:
            self._write("usage: /memory [approve <id>]\n")
            return
        report = self.propose(self.session.as_messages(), root=str(self.session.root))
        self._write(report.summary_line() + "\n")
        for candidate in report.proposed:
            entry_id = memory.MemoryEntry.from_candidate(candidate).id
            self._write(f"{entry_id}: {candidate.text}\n")

    def _status(self) -> None:
        self._write(json.dumps(self.session.snapshot(), sort_keys=True) + "\n")

    def _pillars(self) -> None:
        self._write("ORIENT → CLASSIFY → PLAN → ACT → VERIFY → REFLECT → REPORT\n")

    def _help(self) -> None:
        self._write(
            "/run <task>  /route <task>  /verify  /memory  /memory approve <id>\n"
            "/status  /pillars  /help  /exit\n"
        )

    def _write(self, text: str) -> None:
        self.stdout.write(paint(text, "32", enabled=self._color))


def _verdict(verdict: Classification) -> str:
    state = "reversible" if verdict.reversible else "not reversible"
    return f"tier {verdict.tier} · {verdict.blast} blast · {state}: {verdict.why}"


def _provider_message(exc: ProviderError) -> str:
    if isinstance(exc, ProviderNotConfiguredError):
        return (
            "no Anthropic credentials - source ~/TARS/.env or set ANTHROPIC_AUTH_TOKEN"
        )
    text = re.sub(r"<[^>]*>", " ", str(exc))
    text = " ".join(text.split())
    match = re.search(r"HTTP\s+\d{3}\s+from\s+provider", text, re.IGNORECASE)
    prefix = match.group(0) if match else "provider error"
    detail = text[match.end() :] if match else text
    return f"{prefix}{detail[:200]}"


def write_report(report: RunReport, stream: TextIO) -> None:
    """Print REPORT without dropping its honest-detail fields."""
    verified = (
        "not run"
        if report.verification_passed is None
        else ("pass" if report.verification_passed else "fail")
    )
    stream.write("REPORT\n")
    stream.write(f"task: {report.task}\n")
    stream.write(f"verdict: {report.verdict or 'unknown'}\n")
    stream.write(f"verification: {verified}\n")
    stream.write(f"touched: {', '.join(report.touched) or 'none'}\n")
    stream.write(
        f"memory: proposed={report.proposed} committed={report.committed} "
        f"queued={report.queued}\n"
    )
    stream.writelines(f"declined: {item}\n" for item in report.declined)
    stream.writelines(f"uncertainty: {item}\n" for item in report.uncertainties)
