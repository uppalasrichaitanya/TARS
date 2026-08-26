"""`tars-readme` — write a README out of what a repo actually contains.

Point it at any checkout and it prints a README to stdout. It writes into the
tree only when asked, and it refuses to overwrite an existing file without
`--force`: this runs unattended, and a hand-written README replaced by a
generated one is work nobody can get back.

The exit-code contract mirrors the verification gate's, for the same reason:

    0  ok            a README was produced (or the one on disk is current)
    1  out-of-date   `--check` found the file on disk differs from the README
                     this repo's source implies
    2  usage         bad flags, a path that isn't a directory, or an output
                     file that already exists without `--force`
    3  nothing       there was nothing here to document

3 is deliberately not 0. A pipeline treating 0 as "documented" would otherwise
read an empty directory as a documented one — the docs equivalent of the gate
reading a repo with no checks as a green build.

`--check` is the CI mode and never writes anything, including the file it is
checking. It prints a diff, because "your README is stale" is unactionable and
"these four lines are stale" is a patch.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from .._cli_helpers import use_color
from .inspection import DEFAULT_MAX_EXAMPLES, inspect_project
from .render import render_readme
from .types import ProjectFacts

EXIT_OK = 0
EXIT_STALE = 1
EXIT_USAGE = 2
EXIT_NOTHING = 3

PROG = "tars-readme"

#: What `--check` reads when no `-o` was given. Not configurable beyond `-o`,
#: because a checker that hunts for candidate filenames can pass by finding the
#: wrong file.
DEFAULT_OUTPUT = "README.md"

#: Diff lines printed by `--check` before the rest is elided. Enough to see what
#: moved; short enough that a stale README doesn't bury the CI log.
MAX_DIFF_LINES = 60


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
            "Write a README from what a repo actually contains: usage examples "
            "lifted from its own docstrings and tests, the commands it declares, "
            "and the checks it can run."
        ),
        epilog=(
            "exit codes: 0 ok, 1 --check found the file out of date, "
            "2 usage error, 3 nothing here to document"
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="repo to document (default: the current directory)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help=f"write to FILE instead of stdout (--check reads {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing output file",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the file on disk against this repo's source; write nothing",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the extracted facts as JSON instead of markdown",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="print examples that no longer resolve, labelled as unverified",
    )
    parser.add_argument(
        "--notes",
        action="store_true",
        help="append what was dropped during generation, and why",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="do not mine examples out of the test suite",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        metavar="N",
        help=f"examples to show at most (default: {DEFAULT_MAX_EXAMPLES})",
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

    try:
        args = build_parser().parse_args(list(argv) if argv is not None else None)
        _reject_impossible_combinations(args)
    except _UsageError as exc:
        err.write(f"{PROG}: {exc}\n")
        return EXIT_USAGE
    except SystemExit as exc:  # --help
        return int(exc.code or 0)

    try:
        facts = inspect_project(
            args.path,
            include_tests=not args.no_tests,
            max_examples=args.max_examples,
        )
    except OSError as exc:
        err.write(f"{PROG}: {exc}\n")
        return EXIT_USAGE

    if not facts.documentable and not args.json:
        err.write(
            f"{PROG}: nothing to document in {facts.root}: no modules, commands "
            "or examples were found\n"
        )
        return EXIT_NOTHING

    code = EXIT_OK if facts.documentable else EXIT_NOTHING
    body = _body(facts, args, code)

    if args.check:
        return _report_check(body, _target(args), out, err, color=_color(out, args))
    if args.output:
        return _write(body, Path(args.output), facts, args, err) or code
    out.write(body)
    return code


def _body(facts: ProjectFacts, args: argparse.Namespace, code: int) -> str:
    """The bytes this invocation produces — markdown, or the facts behind it."""
    if args.json:
        payload = facts.to_dict()
        # Carried in the payload too: a caller reading only stdout would
        # otherwise have to infer the verdict from the shape of the facts.
        payload["exit_code"] = code
        return json.dumps(payload, indent=2) + "\n"
    return render_readme(
        facts,
        include_unverified=args.include_unverified,
        include_notes=args.notes,
    )


def _reject_impossible_combinations(args: argparse.Namespace) -> None:
    if args.check and args.json:
        raise _UsageError(
            "--check compares markdown against a file on disk; --json emits "
            "facts instead, so the two cannot be combined"
        )
    if args.max_examples < 0:
        raise _UsageError(f"--max-examples cannot be negative: {args.max_examples}")


# ── writing ───────────────────────────────────────────────────────────────


def _write(
    body: str,
    target: Path,
    facts: ProjectFacts,
    args: argparse.Namespace,
    err: TextIO,
) -> int:
    """Write `body` to `target`. Returns 0, or a usage code without writing.

    Refusing an existing file by default is the whole safety property here. A
    missing parent directory is refused rather than created: a tool that runs
    unattended and mkdirs into a typo'd path scatters directories through a tree
    nobody asked it to touch.
    """
    if target.exists() and not args.force:
        err.write(
            f"{PROG}: {target} already exists; pass --force to overwrite it "
            "(or -o to write elsewhere)\n"
        )
        return EXIT_USAGE
    try:
        target.write_text(body, encoding="utf-8")
    except OSError as exc:
        err.write(f"{PROG}: cannot write {target}: {exc}\n")
        return EXIT_USAGE
    err.write(f"{PROG}: wrote {target} ({_written_summary(body, facts)})\n")
    return EXIT_OK


def _written_summary(body: str, facts: ProjectFacts) -> str:
    shown = len(facts.verified_examples)
    lines = body.count("\n")
    return f"{lines} lines, {shown} verified example{'' if shown == 1 else 's'}"


# ── --check ───────────────────────────────────────────────────────────────


def _target(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    return Path(args.path) / DEFAULT_OUTPUT


def _report_check(
    expected: str, target: Path, out: TextIO, err: TextIO, *, color: bool
) -> int:
    try:
        actual = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        out.write(f"{_paint('out of date', '31', color)}: {target} does not exist\n")
        return EXIT_STALE
    except OSError as exc:
        err.write(f"{PROG}: cannot read {target}: {exc}\n")
        return EXIT_USAGE

    if actual == expected:
        out.write(f"{_paint('up to date', '32', color)}: {target}\n")
        return EXIT_OK

    out.write(f"{_paint('out of date', '31', color)}: {target}\n")
    out.writelines(line + "\n" for line in _diff(actual, expected, target))
    return EXIT_STALE


def _diff(actual: str, expected: str, target: Path) -> list[str]:
    lines = list(
        difflib.unified_diff(
            actual.splitlines(),
            expected.splitlines(),
            fromfile=f"{target} (on disk)",
            tofile=f"{PROG} (from source)",
            lineterm="",
        )
    )
    if len(lines) <= MAX_DIFF_LINES:
        return lines
    held = len(lines) - MAX_DIFF_LINES
    return [*lines[:MAX_DIFF_LINES], f"… {held} more diff line(s) not shown"]


# ── color ─────────────────────────────────────────────────────────────────


def _color(stream: TextIO, args: argparse.Namespace) -> bool:
    return use_color(stream, no_color=args.no_color)


def _paint(text: str, code: str, color: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if color else text
