"""`tars` — persistent REPL and one-shot loop commands.

Exit codes:

    0  command completed
    1  a one-shot run did not verify
    2  invalid command-line usage
    3  provider unavailable or unconfigured
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NoReturn, TextIO

from .. import router
from ..providers.errors import ProviderError, ProviderNotConfiguredError
from ..router import Classification
from .engine import Engine
from .repl import EngineRunner, Repl, _verdict, write_report
from .session import Session
from .types import RunPromptSpec

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_PROVIDER = 3
PROG = "tars"


class _UsageError(Exception):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog=PROG,
        description="Run TARS's disciplined agent loop.",
        epilog="exit codes: 0 completed, 1 verification failed, 2 usage error, 3 provider unavailable",
    )
    subs = parser.add_subparsers(dest="command")
    run = subs.add_parser("run", help="run one task and exit")
    run.add_argument("task")
    run.add_argument("--yes", action="store_true", help="approve ACT previews")
    run.add_argument("--root", default=".", metavar="PATH")
    run.add_argument("--model", metavar="MODEL")
    route = subs.add_parser("route", help="classify one task without acting")
    route.add_argument("task")
    route.add_argument("--root", default=".", metavar="PATH")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    engine: EngineRunner | None = None,
    classify: Callable[[str], Classification] = router.classify,
) -> int:
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    try:
        args = build_parser().parse_args(list(argv) if argv is not None else None)
    except _UsageError as exc:
        err.write(f"{PROG}: {exc}\n")
        return EXIT_USAGE
    except SystemExit as exc:
        return int(exc.code or 0)

    if args.command is None:
        return Repl(stdin=stdin, stdout=out, engine=engine).run()
    if args.command == "route":
        out.write(_verdict(classify(args.task)) + "\n")
        return EXIT_OK

    session = Session(
        root=Path(args.root), **({"model": args.model} if args.model else {})
    )
    runner = engine or Engine()
    try:
        report = runner.run(
            RunPromptSpec(args.task, auto_confirm=args.yes), session, output=out
        )
    except ProviderError as exc:
        err.write(_provider_message(exc) + "\n")
        return EXIT_PROVIDER
    write_report(report, out)
    return EXIT_FAILED if report.verification_passed is False else EXIT_OK


def _provider_message(exc: ProviderError) -> str:
    if isinstance(exc, ProviderNotConfiguredError):
        return (
            "no Anthropic credentials - source ~/TARS/.env or set ANTHROPIC_AUTH_TOKEN"
        )
    text = str(exc)
    text = re.sub(r"<[^>]*>", " ", text)
    text = " ".join(text.split())
    match = re.search(r"HTTP\s+\d{3}\s+from\s+provider", text, re.IGNORECASE)
    prefix = match.group(0) if match else "provider error"
    detail = text[match.end() :] if match else text
    return f"{prefix}{detail[:200]}"
