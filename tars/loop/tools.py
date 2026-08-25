"""Model-facing ACT tools.

The model may request only these four operations. This adapter validates the
untrusted argument dictionaries, keeps paths inside the session root, and
turns executor outcomes into ``ToolResult`` content instead of leaking
exceptions into the loop.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..act import (
    DEFAULT_COMMAND_TIMEOUT_S,
    Command,
    Edit,
    Executor,
    Hunk,
    LocalExecutor,
    resolve_inside,
)
from .types import ToolCall, ToolResult

Confirm = Callable[[str], bool]
TOOL_NAMES = ("read_file", "list_dir", "apply_edit", "run_command")


class Toolbox:
    """Dispatch the loop's four tools against one repository root."""

    def __init__(
        self,
        root: Path | str = ".",
        *,
        executor: Executor | None = None,
        confirm: Confirm | None = None,
    ) -> None:
        self.root = Path(root)
        self.executor = executor or LocalExecutor(self.root)
        self._preview = LocalExecutor(self.root, dry_run=True)
        self.confirm = confirm or (lambda _diff: False)

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Run one model request, converting expected failures to content."""
        handlers = {
            "read_file": self._read_file,
            "list_dir": self._list_dir,
            "apply_edit": self._apply_edit,
            "run_command": self._run_command,
        }
        handler = handlers.get(call.name)
        if handler is None:
            return ToolResult.error_result(call.name, f"unknown tool {call.name!r}")
        try:
            return handler(call.arguments)
        except (KeyError, TypeError, ValueError) as exc:
            return ToolResult.error_result(call.name, f"invalid arguments: {exc}")

    def _read_file(self, args: Mapping[str, Any]) -> ToolResult:
        path = _required_str(args, "path")
        target = self._inside(path)
        if target is None:
            return ToolResult.error_result(
                "read_file", f"path {path!r} is outside the repo root"
            )
        if not target.is_file():
            return ToolResult.error_result("read_file", f"file not found: {path}")
        try:
            return ToolResult.ok_result("read_file", target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            return ToolResult.error_result("read_file", f"could not read {path}: {exc}")

    def _list_dir(self, args: Mapping[str, Any]) -> ToolResult:
        path = args.get("path", ".")
        if not isinstance(path, str):
            raise TypeError("path must be a string")
        target = self._inside(path)
        if target is None:
            return ToolResult.error_result(
                "list_dir", f"path {path!r} is outside the repo root"
            )
        if not target.is_dir():
            return ToolResult.error_result("list_dir", f"directory not found: {path}")
        try:
            entries = sorted(
                (
                    entry.name + ("/" if entry.is_dir() else "")
                    for entry in target.iterdir()
                ),
                key=str.casefold,
            )
        except OSError as exc:
            return ToolResult.error_result("list_dir", f"could not list {path}: {exc}")
        return ToolResult.ok_result("list_dir", "\n".join(entries))

    def _apply_edit(self, args: Mapping[str, Any]) -> ToolResult:
        edit = _build_edit(args)
        preview = self._preview.apply(edit)
        if not preview.ok:
            return _action_error("apply_edit", preview)
        if not self.confirm(preview.diff):
            return ToolResult.error_result("apply_edit", "edit declined")
        result = self.executor.apply(edit)
        if not result.ok:
            return _action_error("apply_edit", result)
        return ToolResult.ok_result("apply_edit", result.diff or result.detail)

    def _run_command(self, args: Mapping[str, Any]) -> ToolResult:
        argv = args.get("argv")
        if not isinstance(argv, list) or not all(
            isinstance(item, str) for item in argv
        ):
            raise TypeError("argv must be a list of strings")
        command = Command(
            argv,
            cwd=args.get("cwd"),
            timeout=args.get("timeout", DEFAULT_COMMAND_TIMEOUT_S),
        )
        result = self.executor.run(command)
        if not result.ok:
            return _action_error("run_command", result)
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        return ToolResult.ok_result("run_command", output)

    def _inside(self, path: str) -> Path | None:
        return resolve_inside(self.root, path)


def _required_str(args: Mapping[str, Any], name: str) -> str:
    value = args.get(name)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _build_edit(args: Mapping[str, Any]) -> Edit:
    path = _required_str(args, "path")
    kind = args.get("kind", "replace")
    if kind == "replace":
        raw_hunks = args.get("hunks")
        if not isinstance(raw_hunks, list):
            raise TypeError("hunks must be a list")
        hunks = tuple(
            Hunk(_required_str(hunk, "old"), _required_str(hunk, "new"))
            for hunk in raw_hunks
            if isinstance(hunk, Mapping)
        )
        if len(hunks) != len(raw_hunks):
            raise TypeError("each hunk must be an object")
        return Edit.replace(path, *hunks)
    if kind in ("create", "overwrite"):
        return Edit(path, kind=kind, content=_required_str(args, "content"))
    if kind == "delete":
        return Edit.delete(path)
    raise ValueError(f"unknown edit kind {kind!r}")


def _action_error(name: str, result: Any) -> ToolResult:
    reason = result.reason or "failed"
    return ToolResult.error_result(name, f"{reason}: {result.detail}")
