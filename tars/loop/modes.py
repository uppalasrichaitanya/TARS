"""Chat and read-only Plan interactions, separate from the Agent engine."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from .. import providers
from .session import Session

READ_ONLY_TOOLS = (
    providers.ToolDef(
        "read_file",
        "Read a UTF-8 file inside the repository.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    ),
    providers.ToolDef(
        "list_dir",
        "List a repository directory.",
        {"type": "object", "properties": {"path": {"type": "string"}}},
    ),
    providers.ToolDef(
        "git_status", "Show repository status.", {"type": "object", "properties": {}}
    ),
    providers.ToolDef(
        "git_log",
        "Show recent repository commits.",
        {"type": "object", "properties": {}},
    ),
    providers.ToolDef(
        "verification_surface",
        "List the checks TARS detects for this repository.",
        {"type": "object", "properties": {}},
    ),
)


class ModeService:
    """Provider calls for non-mutating interactive modes."""

    def __init__(
        self, provider: Callable[..., providers.LLMResponse] = providers.generate
    ) -> None:
        self.provider = provider

    def chat(self, session: Session, text: str, *, output: TextIO | None = None) -> str:
        session.record(providers.Message.user(text))
        response = self.provider(session.messages, (), session.model)
        answer = response.text or ""
        if answer:
            self._write(output, answer + "\n")
        session.record(providers.Message.assistant(answer))
        return answer

    def plan(self, session: Session, task: str, *, output: TextIO | None = None) -> str:
        session.record(providers.Message.user(task))
        system = providers.Message.system(
            "You are in TARS Plan mode. Produce an actionable implementation plan. "
            "You may inspect the repository only with read-only tools; never edit, "
            "delete, browse, or run arbitrary commands."
        )
        response = self.provider(
            (system, *session.messages), READ_ONLY_TOOLS, session.model
        )
        turns = 0
        while response.tool_calls and turns < 6:
            session.record(
                providers.Message.assistant(response.text or None, response.tool_calls)
            )
            for call in response.tool_calls:
                result = read_only_tool(session.root, call.name, call.arguments)
                session.record(providers.Message.tool_result(call.id, result))
            response = self.provider(
                (system, *session.messages), READ_ONLY_TOOLS, session.model
            )
            turns += 1
        answer = response.text or ""
        if answer:
            self._write(output, answer + "\n")
        session.record(providers.Message.assistant(answer, response.tool_calls))
        session.save_plan(task, answer)
        return answer

    @staticmethod
    def _write(output: TextIO | None, text: str) -> None:
        if output is not None:
            output.write(text)
            output.flush()


def read_only_tool(root: Path, name: str, arguments: dict[str, Any]) -> str:
    """Execute one structural read-only inspection operation."""
    if name == "git_status":
        return _git(root, "status", "--short")
    if name == "git_log":
        return _git(root, "log", "--oneline", "-10")
    if name == "verification_surface":
        from ..verification.detect import detect_checks

        checks = detect_checks(root)
        return "\n".join(f"{check.kind}: {' '.join(check.command)}" for check in checks)
    path = str(arguments.get("path", "."))
    target = (root / path).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        return f"path outside repository: {path}"
    if name == "read_file":
        return (
            target.read_text(encoding="utf-8")
            if target.is_file()
            else f"file not found: {path}"
        )
    if name == "list_dir":
        return (
            "\n".join(
                sorted(
                    item.name + ("/" if item.is_dir() else "")
                    for item in target.iterdir()
                )
            )
            if target.is_dir()
            else f"directory not found: {path}"
        )
    return f"unknown read-only tool: {name}"


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else result.stderr
