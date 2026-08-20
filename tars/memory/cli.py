"""`tars-memory` — the REFLECT phase as a command.

Point `reflect` at a session transcript and it prints the memory entries that
session earned, each with a confidence, the domains it was tagged with, why it
matters and the turn it came from. `--write` is what actually puts them into
`.tars/memory/`; without it nothing is touched.

That default is the safety property of this whole phase. A memory entry is a
claim a future session will act on *without* re-deriving it, so a missed entry
costs one re-derivation while a wrong one gets believed. Printing by default and
writing only when asked is the same asymmetry the scorer encodes, at the level a
person operates.

The exit-code contract mirrors the verification gate's and the readme writer's:

    0  ok        something was kept, and nothing needs a human
    1  review    something is waiting for a human — a queued candidate, or a
                 queue that still has entries in it
    2  usage     bad flags, an unreadable transcript, an id that isn't queued
    3  nothing   nothing memory-worthy this cycle, or nothing stored to recall

3 is deliberately not 0, for the same reason it isn't in the readme writer: a
loop treating 0 as "learned something" would otherwise read a cycle that learned
nothing as a productive one. And 1 is not a failure — "a human should look at
this" is the correct outcome for a low-confidence entry, not an error. It is
distinct from 0 only so an unattended run can tell that it happened.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .propose import MAX_CANDIDATES, propose
from .store import MemoryStore
from .transcript import load_transcript
from .types import VALID_KINDS, Candidate, MemoryEntry, ReflectionReport

EXIT_OK = 0
EXIT_REVIEW = 1
EXIT_USAGE = 2
EXIT_NOTHING = 3

PROG = "tars-memory"

#: Recorded as the session that wrote an approved entry. A person, not the tool:
#: the only thing that distinguishes an approved entry from a queued one is that
#: somebody looked at it, so that is what the provenance should say.
DEFAULT_APPROVER = "human"

#: Colours per disposition, so a long report can be skimmed for the entries that
#: are about to become fact.
_DISPOSITION_COLOR = {"commit": "32", "review": "33", "drop": "90"}


class _UsageError(Exception):
    """A bad invocation, carried out to main() so nothing writes to stderr
    behind the caller's back."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise _UsageError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog=PROG,
        description=(
            "Read a session transcript and propose the memory entries it "
            "earned: domain-tagged, scored, and quoted from the turn they came "
            "from. Also reads back what earlier sessions kept."
        ),
        epilog=(
            "exit codes: 0 ok, 1 something is waiting for a human, "
            "2 usage error, 3 nothing memory-worthy"
        ),
    )
    subs = parser.add_subparsers(dest="command", metavar="COMMAND")

    reflect = subs.add_parser(
        "reflect",
        parents=[_common()],
        help="propose memory entries from a transcript",
        description=(
            "Score every statement in a transcript and report what is worth "
            "remembering. Prints only, unless --write is given."
        ),
    )
    reflect.add_argument("transcript", metavar="TRANSCRIPT", help="session to read")
    reflect.add_argument(
        "--write",
        action="store_true",
        help="apply the report: commit what may commit, queue the rest",
    )
    reflect.add_argument(
        "--session",
        metavar="NAME",
        help="label entries with NAME (default: the transcript's file name)",
    )
    reflect.add_argument(
        "--max",
        dest="max_candidates",
        type=int,
        default=MAX_CANDIDATES,
        metavar="N",
        help=f"propose at most N entries (default: {MAX_CANDIDATES})",
    )
    reflect.add_argument(
        "--now",
        metavar="STAMP",
        help="timestamp to write entries with (default: now, UTC)",
    )

    recall = subs.add_parser(
        "recall",
        parents=[_common()],
        help="read back what earlier sessions kept",
        description=(
            "Live entries only, strongest first. Queued candidates are "
            "deliberately unreachable from here — see `pending`."
        ),
    )
    recall.add_argument(
        "--domain",
        action="append",
        default=[],
        metavar="D",
        help="only this domain; repeatable (default: every partition)",
    )
    recall.add_argument(
        "--kind",
        action="append",
        default=[],
        choices=sorted(VALID_KINDS),
        metavar="K",
        help=f"only this kind; repeatable (one of {', '.join(sorted(VALID_KINDS))})",
    )
    recall.add_argument("--limit", type=int, metavar="N", help="stop after N entries")

    subs.add_parser(
        "pending",
        parents=[_common()],
        help="show the human review queue",
        description="Candidates the writer would not commit on its own.",
    )

    approve = subs.add_parser(
        "approve",
        parents=[_common()],
        help="move a queued candidate into memory",
        description="Drain one entry from the review queue into its partitions.",
    )
    approve.add_argument("entry_id", metavar="ID", help="id shown by `pending`")
    approve.add_argument(
        "--session",
        metavar="NAME",
        default=DEFAULT_APPROVER,
        help=f"who approved it (default: {DEFAULT_APPROVER})",
    )
    approve.add_argument("--now", metavar="STAMP", help="timestamp to write with")

    reject = subs.add_parser(
        "reject",
        parents=[_common()],
        help="drop a queued candidate",
        description="Remove one entry from the review queue without storing it.",
    )
    reject.add_argument("entry_id", metavar="ID", help="id shown by `pending`")

    return parser


def _common() -> _Parser:
    """Flags every subcommand takes, added by `parents=`.

    A `_Parser` rather than a plain one so that a bad shared flag raises
    `_UsageError` like every other bad flag; a parent parser never parses
    anything itself, but the subparsers inherit its class.
    """
    shared = _Parser(add_help=False)
    shared.add_argument(
        "--root",
        default=".",
        metavar="DIR",
        help="repo whose .tars/memory to use (default: the current directory)",
    )
    shared.add_argument(
        "--json", action="store_true", help="emit JSON instead of a report"
    )
    shared.add_argument("--no-color", action="store_true", help="disable ANSI color")
    return shared


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
        if args.command is None:
            raise _UsageError(
                "no command given; expected one of reflect, recall, pending, "
                "approve, reject"
            )
        _reject_impossible_arguments(args)
    except _UsageError as exc:
        err.write(f"{PROG}: {exc}\n")
        return EXIT_USAGE
    except SystemExit as exc:  # --help
        return int(exc.code or 0)

    handler = {
        "reflect": _reflect,
        "recall": _recall,
        "pending": _pending,
        "approve": _approve,
        "reject": _reject,
    }[args.command]
    try:
        return handler(args, out, err)
    except (OSError, ValueError, TypeError) as exc:
        # The store and the transcript parser both raise on input a person can
        # fix — a missing file, malformed JSON, a domain that isn't a name. None
        # of those is this tool crashing, so none of them should look like it.
        err.write(f"{PROG}: {exc}\n")
        return EXIT_USAGE


def _reject_impossible_arguments(args: argparse.Namespace) -> None:
    if getattr(args, "max_candidates", 1) < 1:
        raise _UsageError(
            f"--max must be at least 1: {args.max_candidates} would report "
            "'nothing memory-worthy' over a transcript that had something"
        )
    if getattr(args, "limit", None) is not None and args.limit < 0:
        raise _UsageError(f"--limit cannot be negative: {args.limit}")


# ── reflect ───────────────────────────────────────────────────────────────


def _reflect(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    script = load_transcript(args.transcript, session=args.session)
    store = MemoryStore(root=Path(args.root))
    report = propose(
        script,
        existing=store.entries(),
        max_candidates=args.max_candidates,
        root=str(args.root),
    )
    code = _verdict(report)

    result = None
    if args.write and not report.nothing_worth_remembering:
        result = store.write(report, created_at=args.now or _stamp())

    if args.json:
        payload = report.to_dict()
        payload["exit_code"] = code
        if result is not None:
            payload["write"] = result.to_dict()
        out.write(json.dumps(payload, indent=2) + "\n")
    else:
        out.write(_render_report(report, color=_color(out, args)))

    if result is not None:
        err.write(f"{PROG}: {result.summary_line()} in {store.directory}\n")
        err.writelines(f"{PROG}: {note}\n" for note in result.notes)
    elif not report.nothing_worth_remembering:
        err.write(f"{PROG}: nothing written; pass --write to apply this report\n")
    return code


def _verdict(report: ReflectionReport) -> int:
    """What one reflection pass means to a caller reading only the code."""
    if report.nothing_worth_remembering:
        return EXIT_NOTHING
    if report.to_review:
        return EXIT_REVIEW
    return EXIT_OK


def _stamp() -> str:
    """Now, in UTC. The only place in this phase that reads the clock.

    Everything downstream takes time as a parameter so its output stays
    comparable between runs; the boundary has to get it from somewhere, and here
    is where a `--now` override can still displace it.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _render_report(report: ReflectionReport, *, color: bool) -> str:
    lines = [report.summary_line(), ""]
    for cand in report.candidates:
        lines.extend(_render_candidate(cand, color=color))
    for note in report.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines).rstrip() + "\n"


def _render_candidate(cand: Candidate, *, color: bool) -> list[str]:
    label = _paint(
        f"{cand.disposition:<6}", _DISPOSITION_COLOR[cand.disposition], color
    )
    domains = ", ".join(cand.domains) or "no domain"
    lines = [
        f"  {label}  {cand.confidence:.2f}  {cand.kind:<10}  {domains}",
        f"      {cand.text}",
        f"      why: {cand.why}",
    ]
    if cand.evidence:
        lines.append(f"      from: {', '.join(e.location for e in cand.evidence)}")
    if cand.supersedes:
        lines.append(f"      supersedes: {', '.join(cand.supersedes)}")
    if cand.signals:
        lines.append(f"      signals: {', '.join(cand.signals)}")
    lines.append("")
    return lines


# ── recall ────────────────────────────────────────────────────────────────


def _recall(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    store = MemoryStore(root=Path(args.root))
    found = store.recall(
        domains=tuple(args.domain), kinds=tuple(args.kind), limit=args.limit
    )
    code = EXIT_OK if found else EXIT_NOTHING

    if args.json:
        out.write(
            json.dumps(
                {
                    "root": str(args.root),
                    "domains": list(args.domain),
                    "kinds": list(args.kind),
                    "entries": [e.to_dict() for e in found],
                    "exit_code": code,
                },
                indent=2,
            )
            + "\n"
        )
        return code

    if not found:
        out.write(f"nothing in memory for {_asked_for(args)}\n")
        return code
    color = _color(out, args)
    out.write(f"{len(found)} entr{'y' if len(found) == 1 else 'ies'}\n\n")
    out.writelines("\n".join(_render_entry(e, color=color)) + "\n" for e in found)
    return code


def _asked_for(args: argparse.Namespace) -> str:
    parts = []
    if args.domain:
        parts.append(
            f"domain{'s' if len(args.domain) > 1 else ''} {', '.join(args.domain)}"
        )
    if args.kind:
        parts.append(f"kind{'s' if len(args.kind) > 1 else ''} {', '.join(args.kind)}")
    return " and ".join(parts) or f"{args.root}"


def _render_entry(entry: MemoryEntry, *, color: bool) -> list[str]:
    kind = _paint(f"{entry.kind:<10}", "36", color)
    domains = ", ".join(entry.domains) or "no domain"
    lines = [
        f"  {entry.id}  {entry.confidence:.2f}  {kind}  {domains}",
        f"      {entry.text}",
        f"      why: {entry.why}",
    ]
    if entry.session or entry.created_at:
        origin = " ".join(p for p in (entry.session, entry.created_at) if p)
        lines.append(f"      from: {origin}")
    lines.append("")
    return lines


# ── the review queue ──────────────────────────────────────────────────────


def _pending(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    store = MemoryStore(root=Path(args.root))
    queue = store.pending()
    code = EXIT_REVIEW if queue else EXIT_NOTHING

    if args.json:
        out.write(
            json.dumps(
                {
                    "pending": [
                        {"id": MemoryEntry.from_candidate(c).id, **c.to_dict()}
                        for c in queue
                    ],
                    "exit_code": code,
                },
                indent=2,
            )
            + "\n"
        )
        return code

    if not queue:
        out.write(f"nothing waiting for review in {store.directory}\n")
        return code
    out.write(f"{len(queue)} waiting for review\n\n")
    color = _color(out, args)
    for cand in queue:
        entry_id = MemoryEntry.from_candidate(cand).id
        out.write(f"  {_paint(entry_id, '33', color)}\n")
        out.write("\n".join(_render_candidate(cand, color=color)[1:]) + "\n")
    out.write(f"approve or reject with: {PROG} approve <ID>\n")
    return code


def _approve(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    store = MemoryStore(root=Path(args.root))
    entry = store.approve(
        args.entry_id, session=args.session, created_at=args.now or _stamp()
    )
    if entry is None:
        err.write(
            f"{PROG}: {args.entry_id} is not in the review queue; "
            f"`{PROG} pending` lists what is\n"
        )
        return EXIT_USAGE
    if args.json:
        out.write(json.dumps(entry.to_dict(), indent=2) + "\n")
    else:
        out.write(f"committed {entry.id} to {', '.join(entry.domains)}\n")
    return EXIT_OK


def _reject(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    store = MemoryStore(root=Path(args.root))
    if not store.reject(args.entry_id):
        err.write(
            f"{PROG}: {args.entry_id} is not in the review queue; "
            f"`{PROG} pending` lists what is\n"
        )
        return EXIT_USAGE
    out.write(f"dropped {args.entry_id}\n")
    return EXIT_OK


# ── color ─────────────────────────────────────────────────────────────────


def _color(stream: TextIO, args: argparse.Namespace) -> bool:
    # NO_COLOR is the cross-tool convention; honour it before anything else.
    if args.no_color or os.environ.get("NO_COLOR"):
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):  # pragma: no cover - closed/odd stream
        return False


def _paint(text: str, code: str, color: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if color else text
