"""`tars` — persistent REPL and one-shot loop commands.

Exit codes:

    0  command completed
    1  a one-shot run did not verify
    2  invalid command-line usage
    3  provider unavailable or unconfigured
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NoReturn, TextIO

from .. import router
from .. import __version__
from ..providers.errors import ProviderError
from ..router import Classification
from ._io import provider_message
from .engine import Engine
from .repl import EngineRunner, Repl, _verdict, write_report
from .session import Session
from .session_store import SessionStore
from .types import RunPromptSpec, WebTarget

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
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subs = parser.add_subparsers(dest="command")
    run = subs.add_parser("run", help="run one task and exit")
    run.add_argument("task")
    run.add_argument("--yes", action="store_true", help="approve ACT previews")
    run.add_argument("--root", default=".", metavar="PATH")
    run.add_argument("--model", metavar="MODEL")
    run.add_argument("--web-url", metavar="URL", help="also verify a web surface")
    run.add_argument("--web-steps", metavar="PATH", help="JSON file of browser steps")
    run.add_argument("--web-expect", metavar="PATH", help="JSON file of expectations")
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
        home = Path(os.environ.get("TARS_HOME", Path.home() / ".tars"))
        session_store = SessionStore(home / "sessions")
        session = Session(store=session_store)
        return Repl(
            session, stdin=stdin, stdout=out, engine=engine, session_store=session_store
        ).run()
    if args.command == "route":
        out.write(_verdict(classify(args.task)) + "\n")
        return EXIT_OK

    session = Session(
        root=Path(args.root), **({"model": args.model} if args.model else {})
    )
    runner = engine or Engine()
    try:
        web: WebTarget | None = None
        if getattr(args, "web_url", None):
            import json

            steps: tuple[dict, ...] = ()
            expect: tuple[dict, ...] = ()
            if args.web_steps:
                try:
                    raw = json.loads(Path(args.web_steps).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    err.write(f"{PROG}: cannot read --web-steps: {exc}\n")
                    return EXIT_USAGE
                if not isinstance(raw, list):
                    err.write(f"{PROG}: --web-steps must be a JSON list\n")
                    return EXIT_USAGE
                steps = tuple(raw)
            if args.web_expect:
                try:
                    raw = json.loads(Path(args.web_expect).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    err.write(f"{PROG}: cannot read --web-expect: {exc}\n")
                    return EXIT_USAGE
                if not isinstance(raw, list):
                    err.write(f"{PROG}: --web-expect must be a JSON list\n")
                    return EXIT_USAGE
                expect = tuple(raw)
            web = WebTarget(url=args.web_url, steps=steps, expect=expect)
        report = runner.run(
            RunPromptSpec(args.task, auto_confirm=args.yes, web=web),
            session,
            output=out,
        )
    except ProviderError as exc:
        err.write(provider_message(exc) + "\n")
        return EXIT_PROVIDER
    write_report(report, out)
    return EXIT_FAILED if report.verification_passed is False else EXIT_OK
