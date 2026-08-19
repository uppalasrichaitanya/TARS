"""Mining usage examples out of a repo, and distrusting them until they resolve.

The spec rule this implements: *never invent usage examples, pull them from real
working code paths*. Two halves:

**Extraction** lifts snippets verbatim out of docstrings and tests, recording the
file and line each came from. Nothing is rewritten beyond removing indentation,
so the recorded origin stays a claim anyone can check with an editor.

**Validation** then tries to break every snippet against the project's real API.
An example that imports a symbol the project stopped exporting, runs
`python -m` on a package with no `__main__`, or types a console script that no
longer exists, is not printed — it is reported as dropped. Extraction proves a
snippet was *in the tree*; only validation suggests it still *works*, and a
docstring example that rotted two refactors ago misleads a reader exactly as
badly as a fabricated one.

Three classification details worth knowing, because each one is a trap:

* **Shell before Python.** `python -m pkg --list` is syntactically valid Python
  (a chain of subtractions and unary minuses), so a Python-first classifier tags
  shell transcripts as Python.
* **JSON before Python.** `{"lint": null}` is a valid Python dict display with a
  `Name` in it, so the same trap applies to config snippets.
* **Indentation of four, measured from the quote column.** Code blocks in these
  docstrings are indented four spaces from the docstring's own column; two-space
  "term — description" prose lists are not. `col_offset` is the baseline rather
  than the minimum indent of the block, because a function docstring whose only
  indented lines *are* the code block would otherwise dedent to zero.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .source import SourceFile
from .types import Example, ExampleKind, ExampleLanguage

#: A code block in a docstring is indented this far past the docstring's own
#: column. Anything shallower is prose.
CODE_INDENT = 4

#: Blocks longer than this are documentation of something else — a whole
#: implementation pasted into a docstring, a fixture file. Not README material.
MAX_EXAMPLE_LINES = 40

#: A test longer than this stops reading as "here is how you call it".
MAX_TEST_STATEMENTS = 8

#: First tokens that mean "this line is a shell command". Deliberately not
#: exhaustive: an unrecognised first token makes the block fail classification,
#: which drops it. Dropping beats mislabelling.
SHELL_COMMANDS = frozenset(
    {
        "python",
        "python3",
        "py",
        "pip",
        "pip3",
        "pipx",
        "uv",
        "uvx",
        "poetry",
        "hatch",
        "pdm",
        "tox",
        "nox",
        "pytest",
        "ruff",
        "mypy",
        "black",
        "flake8",
        "pylint",
        "pyright",
        "coverage",
        "twine",
        "npm",
        "npx",
        "pnpm",
        "yarn",
        "bun",
        "node",
        "deno",
        "tsc",
        "eslint",
        "prettier",
        "vitest",
        "jest",
        "make",
        "cargo",
        "go",
        "rustup",
        "git",
        "docker",
        "kubectl",
        "curl",
        "wget",
        "sh",
        "bash",
        "zsh",
        "fish",
        "export",
        "cd",
        "ls",
        "mkdir",
        "rm",
        "cp",
        "mv",
        "cat",
        "echo",
        "source",
        "sudo",
        "env",
        "which",
        "open",
        "code",
    }
)

#: Nodes that mean a snippet does something rather than merely naming things.
#: Without this, a two-line list of class names parses fine and reads as code.
_ACTIVE_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Call,
    ast.Attribute,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Subscript,
    ast.Assert,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.With,
    ast.For,
    ast.While,
    ast.Return,
    ast.Await,
)

_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

#: Fence info strings we trust to name the language, so classification is
#: skipped. An unrecognised fence (```diff, ```yaml) falls through to the
#: classifier and is dropped if it isn't code we can check.
FENCE_LANGUAGES: dict[str, ExampleLanguage] = {
    "python": "python",
    "py": "python",
    "pycon": "python",
    "console": "console",
    "shell": "console",
    "bash": "console",
    "sh": "console",
    "zsh": "console",
    "json": "json",
    "text": "text",
}

_FENCE = re.compile(r"^\s*(?:```+|~~~+)\s*([A-Za-z0-9_+-]*)\s*$")
_PROMPT = re.compile(r"^\s*[$%>]\s+")


@dataclass(frozen=True)
class ProjectApi:
    """What the project really exposes, for examples to be checked against.

    Everything here is derived from the source tree and the manifest, never from
    importing the project.
    """

    #: dotted module name -> the names it exports.
    modules: Mapping[str, Sequence[str]] = field(default_factory=dict)
    #: Packages with a `__main__.py`, so `python -m <name>` is real.
    runnable: Sequence[str] = ()
    #: Console scripts the manifest declares.
    scripts: Sequence[str] = ()
    #: Extras that `pip install .[x]` can name.
    extras: Sequence[str] = ()
    installable: bool = False
    #: Dotted symbols the test suite actually calls — evidence beyond existence.
    exercised: Sequence[str] = ()

    def __post_init__(self) -> None:
        # A string is a sequence, so `{"pkg": "run"}` silently means "pkg
        # exports r, u, n". Rejecting it here beats a README that claims the
        # project exports single letters.
        flat = sorted(m for m, names in self.modules.items() if isinstance(names, str))
        if flat:
            raise ValueError(
                "ProjectApi.modules values must be sequences of names, not "
                f"strings: {', '.join(flat)}"
            )

    @property
    def packages(self) -> frozenset[str]:
        """Top-level names that belong to this project."""
        return frozenset(name.split(".", 1)[0] for name in self.modules)

    def exports(self, module: str) -> frozenset[str]:
        return frozenset(self.modules.get(module, ()))

    def has_module(self, module: str) -> bool:
        return module in self.modules


# ── extraction ────────────────────────────────────────────────────────────


def extract_examples(
    source: SourceFile, *, commands: Sequence[str] = ()
) -> tuple[Example, ...]:
    """Every usable snippet in `source`'s docstrings, in file order."""
    found: list[Example] = []
    for node in ast.walk(source.tree):
        if not isinstance(node, _DOCSTRING_OWNERS):
            continue
        literal = _docstring_literal(node)
        if literal is None:
            continue
        title = _qualname(source, node)
        for code, line, kind, language in _blocks(literal, source, commands):
            found.append(
                Example(
                    code=code,
                    language=language,
                    kind=kind,
                    path=source.path,
                    line=line,
                    module=source.module,
                    title=title,
                )
            )
    found.sort(key=lambda e: e.line)
    return tuple(found)


def _docstring_literal(node: ast.AST) -> ast.Constant | None:
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.value
    return None


def _qualname(source: SourceFile, node: ast.AST) -> str | None:
    if isinstance(node, ast.Module):
        return None
    name = getattr(node, "name", None)
    return f"{source.module}.{name}" if name else None


def _blocks(
    literal: ast.Constant, source: SourceFile, commands: Sequence[str]
) -> list[tuple[str, int, ExampleKind, ExampleLanguage]]:
    """Candidate snippets in one docstring, as (code, file line, kind, language).

    Line mapping: the literal's first value-line shares the line of the opening
    quote, so value line *i* is file line `literal.lineno + i`. That holds for
    ordinary docstrings and is checked against the file text afterwards, because
    an implicit concatenation or an escaped-newline docstring would break it and
    a wrong line number is worse than none.
    """
    raw = literal.value
    if not isinstance(raw, str):  # pragma: no cover - guarded by caller
        return []
    lines = raw.expandtabs().split("\n")
    base = literal.col_offset
    out: list[tuple[str, int, ExampleKind, ExampleLanguage]] = []

    index = 0
    while index < len(lines):
        consumed, block = _block_at(lines, index, base)
        if block is None:
            index += max(consumed, 1)
            continue
        code, kind, forced = block
        language = forced or _classify(code, commands)
        if language is not None and 0 < len(code.splitlines()) <= MAX_EXAMPLE_LINES:
            line = _locate(source, code, literal.lineno + index)
            out.append((code, line, kind, language))
        index += max(consumed, 1)
    return out


def _block_at(
    lines: list[str], index: int, base: int
) -> tuple[int, tuple[str, ExampleKind, ExampleLanguage | None] | None]:
    """Read whatever block starts at `index`. Returns (lines consumed, block)."""
    line = lines[index]
    stripped = line.strip()

    fence = _FENCE.match(line)
    if fence:
        body: list[str] = []
        cursor = index + 1
        while cursor < len(lines) and not _FENCE.match(lines[cursor]):
            body.append(lines[cursor])
            cursor += 1
        info = fence.group(1).lower()
        forced = FENCE_LANGUAGES.get(info)
        code = _dedent(body)
        # +1 for the closing fence; the opening fence is not part of the code, so
        # the reported line is the first body line.
        return cursor - index + 1, (
            (code, "docstring", forced) if code.strip() else None
        )

    if stripped.startswith(">>>"):
        body = []
        cursor = index
        while cursor < len(lines) and lines[cursor].strip():
            body.append(lines[cursor])
            cursor += 1
        return cursor - index, (_dedent(body), "doctest", "python")

    # An indented literal block: runs until the indentation drops back. Blank
    # lines inside it do not end it — "import, blank line, call" is the single
    # most common shape a usage example has.
    if index >= 1 and stripped and _indent(line) - base >= CODE_INDENT:
        body = []
        cursor = index
        while cursor < len(lines):
            current = lines[cursor]
            if not current.strip():
                body.append(current)
                cursor += 1
                continue
            if _indent(current) - base < CODE_INDENT:
                break
            body.append(current)
            cursor += 1
        while body and not body[-1].strip():
            body.pop()
        return cursor - index, (_dedent(body), "docstring", None)

    return 1, None


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _dedent(lines: Sequence[str]) -> str:
    """Strip the block's common indentation without touching anything else."""
    indents = [_indent(line) for line in lines if line.strip()]
    margin = min(indents) if indents else 0
    return "\n".join(
        line[margin:] if len(line) >= margin else line.lstrip() for line in lines
    ).strip("\n")


def _locate(source: SourceFile, code: str, guess: int) -> int:
    """Confirm the computed line, searching the file if it doesn't match."""
    first = code.splitlines()[0].strip()
    file_lines = source.text.expandtabs().splitlines()
    if 1 <= guess <= len(file_lines) and file_lines[guess - 1].strip() == first:
        return guess
    for number, text in enumerate(file_lines, start=1):
        if text.strip() == first:
            return number
    return guess


def _classify(code: str, commands: Sequence[str]) -> ExampleLanguage | None:
    """Which language a snippet is, or None when it isn't code at all."""
    text = code.strip()
    if not text:
        return None

    if text[0] in "{[":
        try:
            json.loads(text)
        except ValueError:
            pass
        else:
            return "json"

    if _is_shell(text, commands):
        return "console"

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    if any(isinstance(node, _ACTIVE_NODES) for node in ast.walk(tree)):
        return "python"
    return None


def _is_shell(text: str, commands: Sequence[str]) -> bool:
    """True when every line reads as a command someone could type."""
    known = set(SHELL_COMMANDS) | set(commands)
    saw_command = False
    for raw in text.splitlines():
        line = _PROMPT.sub("", raw).strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        if token.startswith("./") or token in known:
            saw_command = True
            continue
        return False
    return saw_command


# ── extraction from tests ─────────────────────────────────────────────────


def extract_test_examples(
    source: SourceFile, *, packages: Sequence[str] = ()
) -> tuple[Example, ...]:
    """Short, fixtureless tests, as examples, with the imports they need.

    Fixtures are the cutoff. A test taking `tmp_path` or a custom fixture is not
    a snippet a reader can run; a test taking nothing is a complete call.
    """
    imports = _module_imports(source)
    found: list[Example] = []
    for node in source.tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test"):
            continue
        args = node.args
        if args.args or args.posonlyargs or args.kwonlyargs or args.vararg:
            continue
        body = [stmt for stmt in node.body if _docstring_literal_stmt(stmt) is None]
        if not body or len(body) > MAX_TEST_STATEMENTS:
            continue

        segments = [ast.get_source_segment(source.text, stmt) for stmt in body]
        if any(segment is None for segment in segments):  # pragma: no cover
            continue
        code_body = "\n".join(s for s in segments if s)
        used = _names_used(node)
        needed = [line for name, line in imports if name in used]
        if not needed:
            continue
        if packages and not any(
            _root_of_import(line) in set(packages) for line in needed
        ):
            continue

        code = "\n".join(needed) + "\n\n" + code_body
        found.append(
            Example(
                code=code,
                language="python",
                kind="test",
                path=source.path,
                line=node.lineno,
                module=source.module,
                title=node.name,
            )
        )
    return tuple(found)


def _docstring_literal_stmt(stmt: ast.stmt) -> ast.Constant | None:
    if (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    ):
        return stmt.value
    return None


def _module_imports(source: SourceFile) -> list[tuple[str, str]]:
    """(bound name, verbatim import line) for every top-level import."""
    out: list[tuple[str, str]] = []
    for node in source.tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        segment = ast.get_source_segment(source.text, node)
        if segment is None:  # pragma: no cover - needs a synthesised tree
            continue
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            out.append((bound, segment))
    return out


def _names_used(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _root_of_import(line: str) -> str:
    parts = line.split()
    if not parts:  # pragma: no cover - segments always have tokens
        return ""
    if parts[0] == "from" and len(parts) > 1:
        return parts[1].split(".")[0]
    if parts[0] == "import" and len(parts) > 1:
        return parts[1].split(".")[0]
    return ""


def exercised_symbols(
    tests: Iterable[SourceFile], *, packages: Sequence[str] = ()
) -> frozenset[str]:
    """Dotted project symbols the test suite imports and uses.

    This is the difference between "this name exists" and "something ran it",
    and it is the strongest evidence available without executing anything.
    """
    roots = set(packages)
    found: set[str] = set()
    for source in tests:
        aliases: dict[str, str] = {}
        for node in ast.walk(source.tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".")[0] not in roots:
                    continue
                for alias in node.names:
                    found.add(f"{module}.{alias.name}")
                    aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in roots:
                        found.add(alias.name)
                        aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        for node in ast.walk(source.tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                target = aliases.get(node.value.id)
                if target:
                    found.add(f"{target}.{node.attr}")
    return frozenset(found)


# ── validation ────────────────────────────────────────────────────────────


def validate_examples(
    examples: Iterable[Example], api: ProjectApi
) -> tuple[Example, ...]:
    """Resolve every example against `api`.

    Returns the examples that say something about *this* project, each carrying
    a verdict and the reason for it. Snippets with no project reference at all
    are dropped: they may be perfectly good code, but they are not examples of
    using this project, and a README full of `import os` is noise.
    """
    out: list[Example] = []
    for example in examples:
        checked = _validate(example, api)
        if checked is not None:
            out.append(checked)
    return tuple(out)


def _validate(example: Example, api: ProjectApi) -> Example | None:
    if example.language == "json":
        # Nothing here can be resolved against code. Extraction already proved
        # it parses, and the note says exactly how far that goes.
        return _with(
            example,
            verified=True,
            verification="valid JSON (keys are not resolvable against the API)",
        )
    if example.language == "console":
        return _validate_console(example, api)
    if example.language == "python":
        return _validate_python(example, api)
    return None


def _with(example: Example, **changes: object) -> Example:
    from dataclasses import replace

    return replace(example, **changes)  # type: ignore[arg-type]


def _validate_python(example: Example, api: ProjectApi) -> Example | None:
    code = _doctest_source(example.code) if example.kind == "doctest" else example.code
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        return _with(
            example, verified=False, verification=f"does not parse as Python ({exc})"
        )

    symbols: list[str] = []
    problems: list[str] = []
    touches_project = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                # A relative import in an example can't be resolved without
                # knowing where the reader is standing.
                problems.append(
                    f"relative import {'.' * node.level}{node.module or ''}"
                )
                touches_project = True
                continue
            module = node.module or ""
            if module.split(".")[0] not in api.packages:
                continue
            touches_project = True
            if not api.has_module(module):
                problems.append(f"no module {module}")
                continue
            exports = api.exports(module)
            for alias in node.names:
                dotted = f"{module}.{alias.name}"
                if alias.name in exports or api.has_module(dotted):
                    symbols.append(dotted)
                else:
                    problems.append(f"{module} does not export {alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in api.packages:
                    continue
                touches_project = True
                if api.has_module(alias.name):
                    symbols.append(alias.name)
                else:
                    problems.append(f"no module {alias.name}")

    if not touches_project:
        return None
    if problems:
        return _with(
            example,
            symbols=tuple(dict.fromkeys(symbols)),
            verified=False,
            verification="stale: " + "; ".join(dict.fromkeys(problems)),
        )

    resolved = tuple(dict.fromkeys(symbols))
    exercised = [s for s in resolved if s in set(api.exercised)]
    note = f"imports resolve ({', '.join(resolved)})" if resolved else "imports resolve"
    if exercised:
        note += f"; exercised by tests: {', '.join(exercised)}"
    return _with(example, symbols=resolved, verified=True, verification=note)


def _doctest_source(code: str) -> str:
    """The executable half of a doctest block, without prompts or output."""
    out: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith((">>> ", "... ")):
            out.append(stripped[4:])
        elif stripped in (">>>", "..."):
            out.append("")
    return "\n".join(out)


_EXTRA = re.compile(r"\[([^\]]+)\]")


def _validate_console(example: Example, api: ProjectApi) -> Example | None:
    scripts = set(api.scripts)
    runnable = set(api.runnable)
    extras = set(api.extras)

    symbols: list[str] = []
    problems: list[str] = []
    touches_project = False

    for raw in example.code.splitlines():
        line = _PROMPT.sub("", raw).strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        token = tokens[0]

        if token not in SHELL_COMMANDS and not token.startswith("./"):
            touches_project = True
            if token in scripts:
                symbols.append(token)
            else:
                problems.append(f"{token} is not a command this project declares")
            continue

        if "-m" in tokens:
            target = _after(tokens, "-m")
            if target and target.split(".")[0] in api.packages:
                touches_project = True
                if target in runnable:
                    symbols.append(f"python -m {target}")
                else:
                    problems.append(f"{target} has no __main__ to run")

        if token in ("pip", "pip3", "uv", "poetry") and _installs_this_project(tokens):
            touches_project = True
            named = _EXTRA.search(line)
            if named:
                for extra in (e.strip() for e in named.group(1).split(",")):
                    if extra in extras:
                        symbols.append(f"extra {extra}")
                    else:
                        problems.append(f"no extra named {extra}")
            elif api.installable:
                symbols.append("installable")
            else:
                problems.append("no build backend is declared, so this cannot install")

    if not touches_project:
        return None
    if problems:
        return _with(
            example,
            symbols=tuple(dict.fromkeys(symbols)),
            verified=False,
            verification="stale: " + "; ".join(dict.fromkeys(problems)),
        )
    resolved = tuple(dict.fromkeys(symbols))
    return _with(
        example,
        symbols=resolved,
        verified=True,
        verification=f"resolves ({', '.join(resolved)})" if resolved else "resolves",
    )


def _after(tokens: Sequence[str], flag: str) -> str | None:
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def _installs_this_project(tokens: Sequence[str]) -> bool:
    """True for `pip install .`, `-e .`, `.[extra]` — this repo, not PyPI."""
    for token in tokens[1:]:
        cleaned = token.strip("\"'")
        if cleaned == "." or cleaned.startswith(".["):
            return True
    return False
