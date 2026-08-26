"""`tars-verify` — the verification gate as a standalone command.

Usable outside the agent loop entirely: point it at any repo and it reports that
repo's check surface and how it did.

The exit-code contract is the interesting part:

    0  pass          something ran and nothing failed
    1  fail          a check failed, timed out, or couldn't be launched
    2  usage         bad flags, or a path that isn't a directory
    3  no-checks     nothing could be verified here

3 is deliberately not 0. A CI step that treats 0 as "verified" would otherwise
read a repo with no check surface as a green build. `--allow-no-checks` maps it
back to 0 for callers that genuinely don't care.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TextIO

from .._cli_helpers import is_tty as _is_tty
from .._cli_helpers import use_color as _use_color
from .detect import detect_candidates
from .gate import select_checks, verify
from .runner import DEFAULT_TIMEOUT_S, build_child_env, resolve_executable
from .types import ALL_KINDS, DEFAULT_KINDS, VALID_KINDS, Check, CheckResult, GateReport

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_NO_CHECKS = 3

PROG = "tars-verify"

_STATUS_LABELS = {
    "passed": "PASS",
    "failed": "FAIL",
    "timeout": "TIMEOUT",
    "error": "ERROR",
    "unavailable": "UNAVAILABLE",
    "skipped": "SKIPPED",
}

_STATUS_COLORS = {
    "passed": "32",  # green
    "failed": "31",  # red
    "timeout": "31",
    "error": "31",
    "unavailable": "33",  # yellow
    "skipped": "90",  # grey
}

_VERDICT_COLORS = {"pass": "32", "fail": "31", "no-checks": "33"}

#: Commands longer than this get elided in the row; the full text is always in
#: --json, so nothing is actually lost.
_MAX_COMMAND_WIDTH = 46


class _UsageError(Exception):
    """A bad invocation. Carried out to main() so nothing writes to sys.stderr
    behind the caller's back."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise _UsageError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog=PROG,
        description=(
            "Detect a repo's test runner, linter, type-checker and formatter, "
            "then run them uniformly."
        ),
        epilog=(
            "exit codes: 0 pass, 1 a check failed, 2 usage error, "
            "3 nothing could be verified (see --allow-no-checks)"
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="repo to verify (default: the current directory)",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="KIND[,KIND]",
        help=f"run only these kinds, replacing the default set ({', '.join(DEFAULT_KINDS)})",
    )
    parser.add_argument(
        "--include",
        action="append",
        metavar="KIND[,KIND]",
        help=f"add a kind to the default set (the opt-in kinds: {', '.join(k for k in ALL_KINDS if k not in DEFAULT_KINDS)})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="show what would run, and why, without running anything",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit one JSON object instead of text"
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first failing check (a missing tool never stops the run)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        metavar="SECONDS",
        help=f"per-check timeout; 0 means no limit (default: {DEFAULT_TIMEOUT_S:g})",
    )
    parser.add_argument(
        "--output-lines",
        type=int,
        default=40,
        metavar="N",
        help="lines of output to show for a failing check; 0 to suppress (default: 40)",
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=4000,
        metavar="N",
        help="per-stream cap on captured output in --json (default: 4000)",
    )
    parser.add_argument(
        "--allow-no-checks",
        action="store_true",
        help="exit 0 instead of 3 when nothing could be verified",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="print only the verdict line"
    )
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    parser = build_parser()

    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        kinds = _resolve_kinds(args.only, args.include)
    except _UsageError as exc:
        err.write(f"{PROG}: {exc}\n")
        return EXIT_USAGE
    except SystemExit as exc:  # --help / --version
        return int(exc.code or 0)

    printer = _Printer(out, color=_use_color(out, args.no_color), live=_is_tty(out))
    timeout = args.timeout if args.timeout and args.timeout > 0 else None

    try:
        if args.list:
            return _run_list(args, kinds, printer)
        return _run_gate(args, kinds, timeout, printer)
    except FileNotFoundError as exc:
        err.write(f"{PROG}: {exc}\n")
        return EXIT_USAGE
    except PermissionError as exc:
        err.write(f"{PROG}: {exc}\n")
        return EXIT_USAGE


def _run_gate(
    args: argparse.Namespace,
    kinds: tuple[str, ...],
    timeout: float | None,
    printer: _Printer,
) -> int:
    live = not args.json and not args.quiet
    root = Path(args.path)
    # Validate before any output, so a typo'd path yields a clean usage error
    # rather than a header followed by an error.
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    if live:
        printer.header(str(root.resolve()))
    report = verify(
        args.path,
        kinds=kinds,
        timeout=timeout,
        fail_fast=args.fail_fast,
        on_start=printer.check_started if live else None,
        on_result=printer.check_finished if live else None,
    )
    code = _exit_code(report, allow_no_checks=args.allow_no_checks)

    if args.json:
        payload = report.to_dict(max_output_chars=args.max_output_chars)
        payload["exit_code"] = code
        printer.raw(json.dumps(payload, indent=2))
        return code

    if args.quiet:
        printer.verdict(report)
        return code

    printer.details(report, output_lines=args.output_lines, kinds=kinds)
    printer.verdict(report)
    return code


def _run_list(
    args: argparse.Namespace, kinds: tuple[str, ...], printer: _Printer
) -> int:
    root = Path(args.path)
    candidates = detect_candidates(root, kinds=kinds)
    selected, _ = select_checks(root, kinds=kinds)
    child_env = build_child_env()
    installed = {
        check: resolve_executable(check.command[0], root, env=child_env) is not None
        for check in (c.check for c in candidates)
    }

    code = (
        EXIT_PASS
        if candidates
        else (EXIT_PASS if args.allow_no_checks else EXIT_NO_CHECKS)
    )

    if args.json:
        printer.raw(
            json.dumps(
                {
                    "root": str(root.resolve()),
                    "checks": [
                        {
                            "kind": c.check.kind,
                            "tool": c.check.tool,
                            "ecosystem": c.check.ecosystem,
                            "command": list(c.check.command),
                            "command_line": c.check.command_line,
                            "evidence": c.check.evidence,
                            "selected": c.check in selected,
                            "installed": installed[c.check],
                        }
                        for c in candidates
                    ],
                    "exit_code": code,
                },
                indent=2,
            )
        )
        return code

    printer.header(str(root.resolve()), verb="would run")
    if not candidates:
        printer.line("  no checks detected — nothing here to verify")
        return code
    for candidate in candidates:
        printer.candidate_row(
            candidate.check,
            selected=candidate.check in selected,
            installed=installed[candidate.check],
        )
    return code


def _exit_code(report: GateReport, *, allow_no_checks: bool) -> int:
    if report.verdict == "fail":
        return EXIT_FAIL
    if report.verdict == "no-checks":
        return EXIT_PASS if allow_no_checks else EXIT_NO_CHECKS
    return EXIT_PASS


def _resolve_kinds(
    only: list[str] | None, include: list[str] | None
) -> tuple[str, ...]:
    base = _split_kinds(only) if only else set(DEFAULT_KINDS)
    if include:
        base |= _split_kinds(include)
    # Canonical order, so downstream output is stable regardless of flag order.
    return tuple(k for k in ALL_KINDS if k in base)


def _split_kinds(values: Iterable[str]) -> set[str]:
    kinds: set[str] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if part not in VALID_KINDS:
                raise _UsageError(
                    f"unknown kind {part!r}; expected one of {', '.join(ALL_KINDS)}"
                )
            kinds.add(part)
    return kinds


def _elide(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


class _Printer:
    """Text output. Live rows on a tty, plain rows everywhere else.

    On a terminal the row is written before the check runs and rewritten with
    its status after, so a slow type-check shows what it's waiting on instead of
    a silent pause. Piped or captured, only the finished row is written — no
    control characters in a log an agent has to read back.
    """

    def __init__(self, stream: TextIO, *, color: bool, live: bool) -> None:
        self.stream = stream
        self.color = color
        self.live = live
        self._header_written = False

    # ── primitives ──
    def raw(self, text: str) -> None:
        self.stream.write(text + "\n")

    def line(self, text: str = "") -> None:
        self.stream.write(text + "\n")

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    # ── rows ──
    def header(self, root: str, verb: str = "verifying") -> None:
        if self._header_written:
            return
        self._header_written = True
        self.line(f"{self._paint('verification gate', '1')} · {verb} {root}")

    def _row(self, check: Check, status: str = "", duration: str = "") -> str:
        return (
            f"  {check.kind:<10} {check.tool:<14} "
            f"{_elide(check.command_line, _MAX_COMMAND_WIDTH):<{_MAX_COMMAND_WIDTH}} "
            f"{status:<12}{duration}"
        )

    def check_started(self, check: Check) -> None:
        if self.live:
            self.stream.write(self._row(check, self._paint("running…", "90")))
            self.stream.flush()

    def check_finished(self, result: CheckResult) -> None:
        label = _STATUS_LABELS[result.status]
        painted = self._paint(label, _STATUS_COLORS[result.status])
        # Pad before painting so ANSI codes don't count toward the column width.
        padded = painted + " " * max(0, 12 - len(label))
        row = (
            f"  {result.check.kind:<10} {result.check.tool:<14} "
            f"{_elide(result.check.command_line, _MAX_COMMAND_WIDTH):<{_MAX_COMMAND_WIDTH}} "
            f"{padded}{result.duration_s:.1f}s"
        )
        if self.live:
            self.stream.write("\r\033[2K" if self.color else "\r")
        self.line(row)

    def candidate_row(self, check: Check, *, selected: bool, installed: bool) -> None:
        marker = "*" if selected else " "
        note = "" if installed else self._paint("  (not installed)", "33")
        self.line(
            f"  {marker} {check.kind:<10} {check.tool:<14} "
            f"{_elide(check.command_line, _MAX_COMMAND_WIDTH):<{_MAX_COMMAND_WIDTH}} "
            f"← {check.evidence}{note}"
        )

    # ── report sections ──
    def details(
        self, report: GateReport, *, output_lines: int, kinds: Sequence[str]
    ) -> None:
        self.header(report.root)
        for result in report.results:
            # Rows for checks that ran were already streamed by the callbacks;
            # anything not run (skipped) still needs a line.
            if not result.executed and result.status != "unavailable":
                self.check_finished(result)

        if output_lines > 0:
            for result in report.failures:
                body = result.tail(output_lines)
                if not body:
                    continue
                self.line()
                self.line(
                    f"  {self._paint(result.check.label, '1')} — "
                    f"{result.detail or result.status}, last {output_lines} lines:"
                )
                for line in body.splitlines():
                    self.line(f"    {line}")

        gaps = [r for r in report.results if r.status == "unavailable"]
        if gaps or report.missing_kinds(kinds) or report.not_selected or report.notes:
            self.line()
        for result in gaps:
            self.line(
                f"  {self._paint('not installed', '33')}: {result.check.tool} "
                f"(wanted because: {result.check.evidence})"
            )
        for kind in report.missing_kinds(kinds):
            self.line(f"  {self._paint('no check found', '33')}: {kind}")
        for check in report.not_selected:
            self.line(f"  not used: {check.label} ({check.evidence})")
        for note in report.notes:
            self.line(f"  note: {note}")

    def verdict(self, report: GateReport) -> None:
        label = report.verdict.upper()
        painted = self._paint(label, _VERDICT_COLORS[report.verdict])
        counts = report.summary_line().split("—", 1)[1].strip()
        self.line(f"{painted} — {counts}")
