"""Types the readme writer speaks: what was found, and where it was found.

One shape per concern:
  Example      — a usage snippet lifted verbatim out of the repo, with its origin.
  ModuleFact   — one importable unit and what it says about itself.
  CommandFact  — an entry point a user can actually type.
  ProjectFacts — everything one inspection pass learned.

Two fields do the real work and are worth reading carefully:

``Example.path``/``line`` exist because a generated README is only as
trustworthy as its worst example. When the tool prints a snippet, the next
question is always "did you find that or invent it?" — carrying the origin makes
the answer checkable instead of a matter of faith.

``Example.verified`` defaults to *False*. Extraction alone proves a snippet was
in the tree, not that it still works; a docstring example that rotted two
refactors ago is as misleading as a fabricated one. Something has to prove an
example before it is trusted, and the default has to be distrust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

#: Where a snippet came from, in descending order of how much it means.
#:   doctest   — ``>>>`` lines; the author wrote them to be executed.
#:   docstring — a code block in prose the author wrote as usage.
#:   test      — a real, passing call, but written to test rather than to teach.
ExampleKind = Literal["doctest", "docstring", "test"]

#: Fence languages emitted. `console` is a shell transcript; `text` is a snippet
#: that is real but that nothing can parse (a tree diagram, a log line).
ExampleLanguage = Literal["python", "console", "json", "text"]

#: Ordering is trust order, used to pick which examples survive a length budget.
KIND_ORDER: tuple[ExampleKind, ...] = ("doctest", "docstring", "test")

#: Also trust order, within one kind. A call shows how the project is used; a
#: config fragment is supporting material for someone already using it.
LANGUAGE_ORDER: tuple[ExampleLanguage, ...] = ("python", "console", "json", "text")


@dataclass(frozen=True)
class Example:
    """A usage snippet copied verbatim out of the repo.

    `code` is never rewritten beyond dedenting: the moment the tool starts
    "improving" a snippet, the origin recorded in `path`/`line` stops being a
    claim anyone can check.
    """

    code: str
    language: ExampleLanguage
    kind: ExampleKind
    path: str
    line: int
    #: The module the snippet was found in, when it was found in one.
    module: str | None = None
    #: What it demonstrates, if the surrounding prose or test name said.
    title: str | None = None
    #: Dotted project names the snippet uses, resolved during validation.
    symbols: tuple[str, ...] = ()
    verified: bool = False
    #: Why it is trusted — or, when it isn't, exactly what failed to resolve.
    verification: str = "unverified"

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError(f"example at {self.path}:{self.line} has no code")
        if not isinstance(self.symbols, tuple):
            object.__setattr__(self, "symbols", tuple(self.symbols))

    @property
    def location(self) -> str:
        """The origin as an editor-clickable string."""
        return f"{self.path}:{self.line}"

    def fenced(self) -> str:
        """The snippet as a markdown code block."""
        return f"```{self.language}\n{self.code.strip()}\n```"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "language": self.language,
            "kind": self.kind,
            "path": self.path,
            "line": self.line,
            "location": self.location,
            "module": self.module,
            "title": self.title,
            "symbols": list(self.symbols),
            "verified": self.verified,
            "verification": self.verification,
        }


def _first_line(text: str) -> str:
    for line in text.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


@dataclass(frozen=True)
class ModuleFact:
    """One module or package, as the source file describes itself."""

    name: str
    path: str
    doc: str = ""
    is_package: bool = False
    #: Names the module exports: `__all__` if it declares one, else its public
    #: top-level definitions. This is what example validation resolves against.
    public: tuple[str, ...] = ()
    #: True when `__all__` is declared, i.e. the export list is deliberate.
    declares_all: bool = False
    #: True when the package can be run as `python -m <name>`.
    runnable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.public, tuple):
            object.__setattr__(self, "public", tuple(self.public))

    @property
    def summary(self) -> str:
        """First line of the docstring — the module's own one-liner."""
        return _first_line(self.doc)

    @property
    def depth(self) -> int:
        return self.name.count(".")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "summary": self.summary,
            "is_package": self.is_package,
            "public": list(self.public),
            "runnable": self.runnable,
        }


@dataclass(frozen=True)
class CommandFact:
    """An entry point a reader can type, and where it is declared."""

    name: str
    target: str
    source: str
    #: One-line description, taken from the command's own argparse `description`.
    description: str = ""
    #: Flags the command really accepts, read out of its parser definition. One
    #: entry per option, so `-q/--quiet` is one flag and not two.
    flags: tuple[str, ...] = ()
    #: Other ways to type this same entry point — `python -m widget` for a
    #: console script whose target lives inside `widget`. Same code, same
    #: `--help`, so it belongs in this command's section rather than its own.
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.flags, tuple):
            object.__setattr__(self, "flags", tuple(self.flags))
        if not isinstance(self.aliases, tuple):
            object.__setattr__(self, "aliases", tuple(self.aliases))

    @property
    def invocations(self) -> tuple[str, ...]:
        """Every spelling of this command, the canonical one first."""
        return (self.name, *self.aliases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "source": self.source,
            "description": self.description,
            "flags": list(self.flags),
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True)
class CheckFact:
    """How the project is verified, as reported by the verification gate."""

    kind: str
    tool: str
    command_line: str
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "tool": self.tool,
            "command_line": self.command_line,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class Prose:
    """Human-written text the tool is allowed to reuse, and its origin.

    The tool never writes prose. When a README needs a sentence that no code
    can supply, the sentence has to come from somewhere a reader could look up,
    and `source` is that somewhere.
    """

    text: str
    source: str


@dataclass(frozen=True)
class ProjectFacts:
    """Everything one inspection pass learned about a checkout."""

    root: str
    name: str = ""
    version: str = ""
    description: str = ""
    requires_python: str = ""
    #: Runtime dependencies, as declared.
    dependencies: tuple[str, ...] = ()
    #: extra name -> what installing it pulls in.
    extras: dict[str, tuple[str, ...]] = field(default_factory=dict)
    commands: tuple[CommandFact, ...] = ()
    modules: tuple[ModuleFact, ...] = ()
    examples: tuple[Example, ...] = ()
    checks: tuple[CheckFact, ...] = ()
    #: Directories holding tests, as found on disk.
    test_paths: tuple[str, ...] = ()
    ecosystems: tuple[str, ...] = ()
    #: Prose the project supplied for a named section, keyed by section id.
    prose: dict[str, Prose] = field(default_factory=dict)
    #: Anything the caller should know about how this inspection went:
    #: degraded parsing, unreadable files, examples that were dropped.
    notes: tuple[str, ...] = ()
    #: True when the project declares a build backend (so it is installable).
    installable: bool = False

    def __post_init__(self) -> None:
        for name in (
            "dependencies",
            "commands",
            "modules",
            "examples",
            "checks",
            "test_paths",
            "ecosystems",
            "notes",
        ):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))

    @property
    def documentable(self) -> bool:
        """Whether there is enough here to write a README about.

        A directory with no name, no modules, no commands and no examples has
        nothing a README could truthfully say. Reporting that plainly is the
        point: it is the docs equivalent of the gate's `no-checks` verdict.
        """
        has_content = bool(self.modules or self.commands or self.examples)
        return bool(self.name) and has_content

    @property
    def verified_examples(self) -> tuple[Example, ...]:
        return tuple(e for e in self.examples if e.verified)

    @property
    def unverified_examples(self) -> tuple[Example, ...]:
        return tuple(e for e in self.examples if not e.verified)

    @property
    def root_module(self) -> ModuleFact | None:
        """The top-level package, if there is exactly one obvious candidate."""
        tops = [m for m in self.modules if m.depth == 0]
        if len(tops) == 1:
            return tops[0]
        for module in tops:
            if module.name == self.name.replace("-", "_"):
                return module
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "requires_python": self.requires_python,
            "dependencies": list(self.dependencies),
            "extras": {k: list(v) for k, v in self.extras.items()},
            "commands": [c.to_dict() for c in self.commands],
            "modules": [m.to_dict() for m in self.modules],
            "examples": [e.to_dict() for e in self.examples],
            "checks": [c.to_dict() for c in self.checks],
            "test_paths": list(self.test_paths),
            "ecosystems": list(self.ecosystems),
            "prose": {
                k: {"text": v.text, "source": v.source} for k, v in self.prose.items()
            },
            "documentable": self.documentable,
            "installable": self.installable,
            "notes": list(self.notes),
        }
