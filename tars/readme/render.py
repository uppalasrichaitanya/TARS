"""Turning facts into markdown, in the order the spec fixes.

*What this is, why it exists, how to run it, real examples, how it's
structured. In that order. Skip filler sections.*

The last clause does most of the work here. Every section is conditional on
having something sourced to put in it: no commands means no command list, no
supplied prose means no "why it exists", no verified examples means no examples
section at all. A generated README earns its place by being shorter than the
hand-written one it replaces — a wall of headings with a sentence under each is
the failure mode, and it is worse than no README because it looks maintained.

Two conventions in the output are deliberate:

* **Every example names its source line.** `widget/__init__.py:3` under a snippet
  turns "trust me" into "go look". It is the whole reason the extractor carries
  provenance around.
* **Unverified examples are withheld, not fixed.** If a snippet no longer
  resolves, printing it with a caveat teaches the reader something false and
  printing a corrected version invents an example. It is left out, and the
  inspection notes say how many were left out and why.
"""

from __future__ import annotations

from collections.abc import Sequence

from .types import CheckFact, CommandFact, Example, ModuleFact, ProjectFacts

#: Section headings, in the spec's order. "What this is" has no heading of its
#: own — it is the title and the sentence under it, which is where a reader
#: already looks.
SECTION_ORDER = (
    "Why it exists",
    "Install",
    "How to run it",
    "Examples",
    "How it's structured",
    "Verifying it",
)

#: Named in the footer so a reader knows the file is generated, and by what.
TOOL = "tars-readme"

#: Modules listed under "how it's structured" before the list stops being a map
#: and becomes a directory dump.
MAX_MODULES = 24


def render_readme(
    facts: ProjectFacts,
    *,
    include_unverified: bool = False,
    include_notes: bool = False,
) -> str:
    """Render `facts` as markdown.

    Raises `ValueError` when there is nothing to document: emitting a title with
    no body would produce a file that looks like documentation and isn't.
    """
    if not facts.documentable:
        raise ValueError(
            f"nothing to document in {facts.root}: no name, modules, commands or "
            "examples were found"
        )

    blocks = [_title(facts)]
    for section in (
        _why(facts),
        _install(facts),
        _how_to_run(facts),
        _examples(facts, include_unverified=include_unverified),
        _structure(facts),
        _verifying(facts),
        _notes(facts) if include_notes else "",
    ):
        if section.strip():
            blocks.append(section.strip())
    blocks.append(_footer(facts))
    return "\n\n".join(blocks) + "\n"


# ── what this is ──────────────────────────────────────────────────────────


def _title(facts: ProjectFacts) -> str:
    lines = [f"# {facts.name}"]
    described = facts.description or _root_summary(facts)
    if described:
        lines += ["", described]
    supplied = facts.prose.get("what")
    if supplied:
        lines += ["", supplied.text]
    return "\n".join(lines)


def _root_summary(facts: ProjectFacts) -> str:
    """The top-level package's own one-liner, when the manifest had none."""
    root = facts.root_module
    return root.summary if root else ""


# ── why it exists ─────────────────────────────────────────────────────────


def _why(facts: ProjectFacts) -> str:
    supplied = facts.prose.get("why")
    if not supplied:
        return ""
    return f"## Why it exists\n\n{supplied.text}"


# ── install ───────────────────────────────────────────────────────────────


def _install(facts: ProjectFacts) -> str:
    if not facts.installable:
        return ""
    extra = _dev_extra(facts)
    target = f'".[{extra}]"' if extra else "."
    lines = ["## Install", "", "```console", f"pip install -e {target}", "```"]
    facts_line = _requirements_line(facts)
    if facts_line:
        lines += ["", facts_line]
    if facts.dependencies:
        lines += [
            "",
            "Dependencies: " + ", ".join(f"`{d}`" for d in facts.dependencies),
        ]
    else:
        lines += ["", "This project has no runtime dependencies."]
    return "\n".join(lines)


def _dev_extra(facts: ProjectFacts) -> str:
    for candidate in ("dev", "test", "tests", "develop"):
        if candidate in facts.extras:
            return candidate
    return ""


def _requirements_line(facts: ProjectFacts) -> str:
    if not facts.requires_python:
        return ""
    return f"Requires Python `{facts.requires_python}`."


# ── how to run it ─────────────────────────────────────────────────────────


def _how_to_run(facts: ProjectFacts) -> str:
    if not facts.commands:
        return ""
    lines = ["## How to run it"]
    for command in facts.commands:
        lines += ["", *_command_block(command)]
    return "\n".join(lines)


def _command_block(command: CommandFact) -> list[str]:
    lines = [f"### `{command.name}`"]
    if command.description:
        lines += ["", command.description]
    lines += ["", "```console", *command.invocations, "```"]
    if command.flags:
        lines += ["", "Flags: " + ", ".join(f"`{f}`" for f in command.flags)]
    lines += ["", f"<sub>declared in `{command.source}`</sub>"]
    return lines


# ── examples ──────────────────────────────────────────────────────────────


def _examples(facts: ProjectFacts, *, include_unverified: bool) -> str:
    shown: Sequence[Example] = (
        facts.examples if include_unverified else facts.verified_examples
    )
    if not shown:
        return ""
    preamble = (
        "Every snippet below was taken from the file named under it, and checked "
        "against the current source."
    )
    lines = ["## Examples", "", preamble]
    for example in shown:
        lines += ["", *_example_block(example)]
    return "\n".join(lines)


def _example_block(example: Example) -> list[str]:
    lines: list[str] = []
    if example.title:
        lines += [f"**{example.title}**", ""]
    lines.append(example.fenced())
    origin = f"<sub>`{example.location}`"
    if not example.verified:
        origin += f" — unverified: {example.verification}"
    lines += ["", origin + "</sub>"]
    return lines


# ── how it's structured ───────────────────────────────────────────────────


def _structure(facts: ProjectFacts) -> str:
    if not facts.modules:
        return ""
    lines = ["## How it's structured", ""]
    # Ordered by path, not by depth: depth order lists every submodule after
    # every package, so the indentation ends up nesting `a/one.py` under package
    # `b` and claims a containment that doesn't exist.
    ordered = sorted(facts.modules, key=lambda m: m.path)
    shown = ordered[:MAX_MODULES]
    for module in shown:
        lines.append(_module_line(module))
    if len(ordered) > len(shown):
        lines.append(f"- …and {len(ordered) - len(shown)} more modules")
    if facts.test_paths:
        joined = ", ".join(f"`{p}/`" for p in facts.test_paths)
        lines += ["", f"Tests live in {joined}."]
    return "\n".join(lines)


def _module_line(module: ModuleFact) -> str:
    indent = "  " * module.depth
    parts = [f"{indent}- `{module.path}`"]
    if module.summary:
        parts.append(f" — {module.summary}")
    if module.runnable:
        parts.append(f" (`python -m {module.name}`)")
    return "".join(parts)


# ── verifying it ──────────────────────────────────────────────────────────


def _verifying(facts: ProjectFacts) -> str:
    """The checks the project itself defines, not a generic CI snippet."""
    if not facts.checks:
        return ""
    lines = ["## Verifying it", "", "```console"]
    lines += [check.command_line for check in facts.checks]
    lines += ["```", "", _evidence_line(facts.checks)]
    return "\n".join(lines)


def _evidence_line(checks: Sequence[CheckFact]) -> str:
    kinds = ", ".join(sorted({check.kind for check in checks}))
    return f"<sub>detected from this repo: {kinds}</sub>"


# ── notes ─────────────────────────────────────────────────────────────────


def _notes(facts: ProjectFacts) -> str:
    if not facts.notes:
        return ""
    lines = ["## Notes from generation", ""]
    lines += [f"- {note}" for note in facts.notes]
    return "\n".join(lines)


# ── footer ────────────────────────────────────────────────────────────────


def _footer(facts: ProjectFacts) -> str:
    shown = len(facts.verified_examples)
    counted = (
        f"{shown} example{'s' if shown != 1 else ''} extracted from the paths above"
        if shown
        else "no examples could be extracted and checked"
    )
    return (
        f"---\n\n<sub>Generated by `{TOOL}` from the source of this repository: "
        f"{counted}. Sections with nothing to source were left out.</sub>"
    )
