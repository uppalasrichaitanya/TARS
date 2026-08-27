"""Model-facing ACT tools.

The model may request only these four operations. This adapter validates the
untrusted argument dictionaries, keeps paths inside the session root, and
turns executor outcomes into ``ToolResult`` content instead of leaking
exceptions into the loop.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from difflib import get_close_matches
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
Mutation = Callable[[tuple[str, ...]], None]
TOOL_NAMES = (
    "read_file",
    "list_dir",
    "apply_edit",
    "run_command",
    "browser_goto",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_check",
    "browser_uncheck",
    "browser_back",
    "browser_forward",
    "browser_reload",
)


class Toolbox:
    """Dispatch the loop's four tools against one repository root."""

    def __init__(
        self,
        root: Path | str = ".",
        *,
        executor: Executor | None = None,
        confirm: Confirm | None = None,
        browser_factory: Callable[[], Any] | None = None,
        on_mutation: Mutation | None = None,
    ) -> None:
        self.root = Path(root)
        self.executor = executor or LocalExecutor(self.root)
        self._preview = LocalExecutor(self.root, dry_run=True)
        self.confirm = confirm or (lambda _diff: False)
        self._browser_session: Any | None = None
        self._browser_factory = browser_factory
        self._browser_error: str | None = None
        self._last_browser_snapshot: str | None = None
        self.on_mutation = on_mutation or (lambda _paths: None)

    def _get_browser(self) -> Any:
        if self._browser_session is not None:
            return self._browser_session
        if self._browser_error is not None:
            raise RuntimeError(self._browser_error)
        try:
            from tars.browser import BrowserSession
        except ImportError as exc:
            self._browser_error = "browser extra not installed; install tars[browser]"
            raise RuntimeError(self._browser_error) from exc
        factory = self._browser_factory or BrowserSession  # type: ignore[assignment]
        try:
            browser = factory()
            # Support both context-manager and plain instance factories
            if hasattr(browser, "__enter__"):
                browser = browser.__enter__()
            self._browser_session = browser
            return self._browser_session
        except RuntimeError:
            raise
        except ImportError as exc:
            self._browser_error = "browser extra not installed; install tars[browser]"
            raise RuntimeError(self._browser_error) from exc
        except Exception as exc:
            self._browser_error = str(exc)
            raise RuntimeError(self._browser_error) from exc

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Run one model request, converting expected failures to content."""
        handlers = {
            "read_file": self._read_file,
            "list_dir": self._list_dir,
            "apply_edit": self._apply_edit,
            "run_command": self._run_command,
            "browser_goto": self._browser_goto,
            "browser_snapshot": self._browser_snapshot,
            "browser_click": self._browser_click,
            "browser_type": self._browser_type,
            "browser_check": self._browser_check,
            "browser_uncheck": self._browser_uncheck,
            "browser_back": self._browser_back,
            "browser_forward": self._browser_forward,
            "browser_reload": self._browser_reload,
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
            return ToolResult.error_result("read_file", _missing_file(self.root, path))
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
        self.on_mutation((edit.path,))
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
        if not self.confirm(command.display):
            return ToolResult.error_result("run_command", "command declined")
        before = _git_status(self.root)
        result = self.executor.run(command)
        if not result.ok:
            return _action_error("run_command", result)
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        touched: tuple[str, ...] = (command.display,)
        # Enrich with actual file paths changed in the repo (so edits via
        # run_command, e.g. printf > file, are reported as file touches).
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if proc.returncode == 0:
                files: list[str] = []
                for line in proc.stdout.splitlines():
                    # porcelains: " M path", "?? path", "A  path"
                    path = line[3:].strip().split(" -> ")[-1].strip()
                    if path:
                        # strip quotes if git quotes the path
                        if path.startswith('"') and path.endswith('"'):
                            path = path[1:-1]
                        files.append(path)
                if files:
                    # Keep command display first, then file paths not already present
                    seen = set(touched)
                    extra = tuple(p for p in files if p not in seen)
                    touched = (*touched, *extra)
        except Exception:  # noqa: BLE001, S110
            pass
        after = _git_status(self.root)
        changed = tuple(
            sorted(path for path, state in after.items() if before.get(path) != state)
        )
        if changed:
            self.on_mutation(changed)
        return ToolResult.ok_result("run_command", output, touched=touched)

    def _browser_goto(self, args: Mapping[str, Any]) -> ToolResult:
        url = _required_str(args, "url")
        try:
            self._get_browser().goto(url)
        except RuntimeError as exc:
            return ToolResult.error_result("browser_goto", str(exc))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.error_result("browser_goto", str(exc))
        return ToolResult.ok_result("browser_goto", f"navigated to {url}")

    def _browser_snapshot(self, args: Mapping[str, Any]) -> ToolResult:
        incremental = args.get("incremental", False)
        if not isinstance(incremental, bool):
            raise TypeError("incremental must be a boolean")
        try:
            snapshot = self._get_browser().snapshot()
        except RuntimeError as exc:
            return ToolResult.error_result("browser_snapshot", str(exc))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.error_result("browser_snapshot", str(exc))
        output = snapshot
        if incremental and self._last_browser_snapshot is not None:
            before = set(self._last_browser_snapshot.splitlines())
            changed = [line for line in snapshot.splitlines() if line not in before]
            output = "\n".join(changed) or "no accessibility changes"
        self._last_browser_snapshot = snapshot
        return ToolResult.ok_result("browser_snapshot", output)

    def _browser_click(self, args: Mapping[str, Any]) -> ToolResult:
        ref = args.get("ref")
        if not isinstance(ref, int):
            raise TypeError("ref must be an integer")
        try:
            self._get_browser().click(ref)
        except (RuntimeError, KeyError) as exc:
            return ToolResult.error_result("browser_click", str(exc))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.error_result("browser_click", str(exc))
        return ToolResult.ok_result("browser_click", f"clicked ref {ref}")

    def _browser_type(self, args: Mapping[str, Any]) -> ToolResult:
        ref = args.get("ref")
        if not isinstance(ref, int):
            raise TypeError("ref must be an integer")
        text = args.get("text")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        try:
            self._get_browser().type(ref, text)
        except (RuntimeError, KeyError) as exc:
            return ToolResult.error_result("browser_type", str(exc))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.error_result("browser_type", str(exc))
        return ToolResult.ok_result("browser_type", f"typed into ref {ref}")

    def _browser_check(self, args: Mapping[str, Any]) -> ToolResult:
        return self._browser_ref_action("browser_check", "check", args)

    def _browser_uncheck(self, args: Mapping[str, Any]) -> ToolResult:
        return self._browser_ref_action("browser_uncheck", "uncheck", args)

    def _browser_ref_action(
        self, name: str, method: str, args: Mapping[str, Any]
    ) -> ToolResult:
        ref = args.get("ref")
        if not isinstance(ref, int):
            raise TypeError("ref must be an integer")
        try:
            getattr(self._get_browser(), method)(ref)
        except (RuntimeError, KeyError) as exc:
            return ToolResult.error_result(name, str(exc))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.error_result(name, str(exc))
        return ToolResult.ok_result(name, f"{method}ed ref {ref}")

    def _browser_back(self, _args: Mapping[str, Any]) -> ToolResult:
        return self._browser_navigation("browser_back", "back")

    def _browser_forward(self, _args: Mapping[str, Any]) -> ToolResult:
        return self._browser_navigation("browser_forward", "forward")

    def _browser_reload(self, _args: Mapping[str, Any]) -> ToolResult:
        return self._browser_navigation("browser_reload", "reload")

    def _browser_navigation(self, name: str, method: str) -> ToolResult:
        try:
            getattr(self._get_browser(), method)()
        except RuntimeError as exc:
            return ToolResult.error_result(name, str(exc))
        except Exception as exc:  # noqa: BLE001
            return ToolResult.error_result(name, str(exc))
        return ToolResult.ok_result(name, method)

    def close(self) -> None:
        if self._browser_session is not None:
            try:
                self._browser_session.close()
            except Exception:  # noqa: BLE001, S110
                pass
            self._browser_session = None

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
    if kind in ("replace", "anchored"):
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


def _git_status(root: Path) -> dict[str, str]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return {}
    if result.returncode != 0:
        return {}
    return {line[3:].strip(): line[:2] for line in result.stdout.splitlines()}


def _missing_file(root: Path, path: str) -> str:
    candidates = [
        str(item.relative_to(root))
        for item in root.rglob(Path(path).name)
        if item.is_file() and ".git" not in item.parts
    ]
    match = get_close_matches(path, candidates, n=1, cutoff=0.3)
    suffix = f" - did you mean {match[0]}?" if match else ""
    return f"file not found: {path}{suffix}"
