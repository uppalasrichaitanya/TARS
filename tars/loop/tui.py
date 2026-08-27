"""Optional prompt-toolkit composer for the inline TARS workspace."""

from __future__ import annotations

import subprocess
from html import escape
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

from .types import InteractionMode

if TYPE_CHECKING:
    from .session import Session


def available() -> bool:
    try:
        import prompt_toolkit  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


class Composer:
    """One persistent composer, retaining history across turns."""

    def __init__(self, session: Session, commands: tuple[str, ...]) -> None:
        PromptSession = import_module("prompt_toolkit").PromptSession
        WordCompleter = import_module("prompt_toolkit.completion").WordCompleter
        self._HTML = import_module("prompt_toolkit.formatted_text").HTML
        InMemoryHistory = import_module("prompt_toolkit.history").InMemoryHistory
        KeyBindings = import_module("prompt_toolkit.key_binding").KeyBindings
        bindings = KeyBindings()

        @bindings.add("enter")
        def submit(event) -> None:
            event.current_buffer.validate_and_handle()

        @bindings.add("escape", "enter")
        @bindings.add("c-j")
        def newline(event) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("tab")
        def cycle_mode(event) -> None:
            self._session.set_mode(next_mode(self._session.mode))
            event.app.invalidate()

        self._session = session
        self._composer = PromptSession(
            history=InMemoryHistory(),
            completer=WordCompleter(list(commands), sentence=True),
            complete_while_typing=True,
            multiline=True,
            key_bindings=bindings,
        )

    def set_session(self, session: Session) -> None:
        """Follow a REPL session restored by `/resume`."""
        self._session = session

    def read(self) -> str:
        return self._composer.prompt(
            self._HTML(
                f"<ansigreen><b>◐ {self._session.mode.value} › </b></ansigreen>"
            ),
            bottom_toolbar=lambda: self._HTML(_status(self._session)),
        )


def prompt(session: Session, commands: tuple[str, ...]) -> str:
    """Compatibility helper for callers that need only one turn."""
    return Composer(session, commands).read()


def next_mode(mode: InteractionMode) -> InteractionMode:
    order = (InteractionMode.CHAT, InteractionMode.PLAN, InteractionMode.AGENT)
    return order[(order.index(mode) + 1) % len(order)]


def _status(session: Session) -> str:
    model = escape(session.model.rsplit("/", 1)[-1])
    repo = escape(session.root.resolve().name or str(session.root.resolve()))
    dirty = _dirty(session.root)
    verdict = escape(session.last_verdict or "not verified")
    return (
        f"<b>{session.mode.value.upper()}</b> · {model} · {repo} · {dirty} · "
        f"{verdict}    Tab mode · Enter send · Alt+Enter newline"
    )


def _dirty(root: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "no git"
    return "dirty" if result.stdout.strip() else "clean"
