"""TARS readme writer — a README built out of what a repo actually contains.

    from tars.readme import readme_for

    print(readme_for("/path/to/repo"))

Usage examples are never composed. They are lifted verbatim out of the repo's
own docstrings, doctests and short tests, each one carrying the file and line it
came from, and checked against the project's real API before anything prints it:

    from tars.readme import inspect_project

    facts = inspect_project(".")
    facts.verified_examples    # snippets whose imports still resolve
    facts.unverified_examples  # snippets that rotted since they were written
    facts.notes                # what was dropped on the way, and why

An example that no longer resolves is withheld rather than caveated or
corrected: printing it teaches the reader something false, and rewriting it
would be inventing one. Sections work the same way — no commands means no
command list, and no supplied prose means no "why it exists".

Prose the source cannot supply is not written either. *Why does this exist* has
no answer anywhere in an AST, so that section is read from `.tars/readme.json`
or left out:

    {"why": "Because the hand-written one drifted from the code every release."}

Nothing here imports or executes the project being documented. Every fact comes
out of a parsed file, which is what makes it safe to point at a checkout you
have not read yet.

Also a standalone command, with no TARS loop involved:

    tars-readme /path/to/repo --output README.md
    tars-readme . --check
"""

from __future__ import annotations

from pathlib import Path

from .examples import ProjectApi, extract_examples, extract_test_examples
from .inspection import (
    DEFAULT_MAX_EXAMPLES,
    PROSE_PATH,
    inspect_project,
    read_prose,
    select_examples,
)
from .render import render_readme
from .types import CheckFact, CommandFact, Example, ModuleFact, ProjectFacts, Prose


def readme_for(
    root: Path | str,
    *,
    include_tests: bool = True,
    include_unverified: bool = False,
    include_notes: bool = False,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> str:
    """Inspect `root` and render its README, in one call.

    Raises `ValueError` when the checkout has nothing to document and
    `FileNotFoundError` when `root` isn't a directory. Neither is downgraded to
    an empty string: a caller writing the result to a file would otherwise
    truncate a README on a typo.
    """
    facts = inspect_project(
        root, include_tests=include_tests, max_examples=max_examples
    )
    return render_readme(
        facts,
        include_unverified=include_unverified,
        include_notes=include_notes,
    )


__all__ = [
    "DEFAULT_MAX_EXAMPLES",
    "PROSE_PATH",
    "CheckFact",
    "CommandFact",
    "Example",
    "ModuleFact",
    "ProjectApi",
    "ProjectFacts",
    "Prose",
    "extract_examples",
    "extract_test_examples",
    "inspect_project",
    "read_prose",
    "readme_for",
    "render_readme",
    "select_examples",
]
