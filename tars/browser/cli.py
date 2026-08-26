"""`tars-browser`: scriptable accessibility-tree web tests.

Exit codes: 0 match/success, 1 expectation mismatch, 2 usage or input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, TextIO

from . import BrowserSession

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_USAGE = 2


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError(message)


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="tars-browser",
        description="Efficient accessibility-tree-first web testing.",
        epilog=("exit codes: 0 success, 1 expectation mismatch, 2 usage/input error"),
    )
    s = p.add_subparsers(dest="command", required=True)
    s.add_parser("snapshot", help="print the accessibility tree").add_argument("url")
    for name in ("run", "check"):
        q = s.add_parser(name, help="execute scripted steps")
        q.add_argument("url")
        q.add_argument("steps")
        if name == "check":
            q.add_argument("expect")
    return p


def _match(snapshot: str, expectations: list[dict[str, Any]]) -> bool:
    for exp in expectations:
        if "text" in exp and str(exp["text"]) not in snapshot:
            return False
        if "role" in exp:
            needle = str(exp["role"])
            if exp.get("name"):
                needle += f' "{exp["name"]}"'
            if needle not in snapshot:
                return False
    return True


def _load_list(path: str, label: str) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must be a JSON list of objects")
    return value


def run_browser_task(
    url: str,
    steps: list[dict[str, Any]] | None,
    expect: list[dict[str, Any]] | None,
    *,
    session_factory=BrowserSession,
) -> tuple[int, str]:
    """Execute one browser task and return the exit code plus a snapshot.

    This is the shared core of the CLI and the verification web target.
    ``steps`` may be ``None`` for a bare snapshot; ``expect`` may be ``None``
    when no expectation check is desired.
    """
    with session_factory() as browser:
        browser.goto(url)
        if steps is None:
            snap = browser.snapshot()
            if expect is None:
                return EXIT_OK, snap
            return (EXIT_OK if _match(snap, expect) else EXIT_MISMATCH), snap
        for step in steps:
            action = step.get("action")
            raw_ref = step.get("ref")
            if not isinstance(raw_ref, (str, int)):
                raise TypeError("step ref must be an integer")
            ref = int(raw_ref)
            if action == "click":
                browser.click(ref)
            elif action == "type":
                browser.type(ref, str(step.get("text", "")))
            elif action == "check":
                browser.check(ref)
            elif action == "uncheck":
                browser.uncheck(ref)
            else:
                raise ValueError(f"unknown action: {action}")
        snap = browser.snapshot()
        if expect is None:
            return EXIT_OK, snap
        return (EXIT_OK if _match(snap, expect) else EXIT_MISMATCH), snap


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    session_factory=BrowserSession,
) -> int:
    out, err = stdout or sys.stdout, stderr or sys.stderr
    try:
        args = build_parser().parse_args(list(argv) if argv is not None else None)
    except (ValueError, SystemExit) as exc:
        if isinstance(exc, SystemExit):
            return exc.code if isinstance(exc.code, int) else EXIT_USAGE
        err.write(f"usage: tars-browser: {exc}\n")
        return EXIT_USAGE
    try:
        if args.command == "snapshot":
            code, snap = run_browser_task(
                args.url, None, None, session_factory=session_factory
            )
            out.write(snap + "\n")
            return code
        steps = _load_list(args.steps, "steps")
        expect: list[dict[str, Any]] | None = None
        if args.command == "check":
            expect = _load_list(args.expect, "expectations")
        code, snap = run_browser_task(
            args.url, steps, expect, session_factory=session_factory
        )
        out.write(snap + "\n")
        return code
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        err.write(f"tars-browser: {exc}\n")
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
