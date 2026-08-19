"""One inspection pass: everything the tool knows, and where it learned it.

The pieces below are deliberately separate — manifest, sources, examples, checks
— and this module is the only place they meet. It is also the only place allowed
to decide *what goes in*, which makes it the natural home for the two rules that
keep a generated README honest:

**Nothing is invented.** Every field traces to a file. The project name comes
from the manifest, or failing that from the one top-level package in the tree; a
command's description is the string its own argparse prints under `--help`, not
a sentence composed from its name. Where a README needs a claim no code can
supply — *why does this exist* — the tool does not write it. It reads it from
`.tars/readme.json` or leaves the section out.

**Selection is part of honesty.** A repo with 168 tests can yield 168 "examples",
and printing them all is a different kind of lie: it buries the two snippets a
reader needed. Examples are ranked by how much they mean (a doctest was written
to be run; a test was written to catch regressions) and capped, and whatever the
cap discarded is recorded in `notes` rather than quietly dropped.

Nothing here imports or executes the project being inspected.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Sequence
from pathlib import Path

from ..verification.detect import detect_checks, detect_ecosystems
from ..verification.types import DEFAULT_KINDS
from .examples import (
    ProjectApi,
    exercised_symbols,
    extract_examples,
    extract_test_examples,
    validate_examples,
)
from .manifest import Manifest, read_manifest
from .source import SourceFile, SourceTree, collect_sources
from .types import (
    KIND_ORDER,
    LANGUAGE_ORDER,
    CheckFact,
    CommandFact,
    Example,
    ModuleFact,
    ProjectFacts,
    Prose,
)

#: Where a project supplies the prose the tool refuses to write for it. JSON for
#: the same reason the gate's override file is JSON: it parses everywhere without
#: a dependency.
#:
#:     {"why": "Because the old converter was unmaintainable."}
PROSE_PATH = ".tars/readme.json"

#: Sections prose may be supplied for. An unrecognised key is almost always a
#: typo in a hand-edited file, and a silently ignored typo means a section the
#: author thinks they wrote and a reader never sees.
PROSE_SECTIONS = ("what", "why", "install", "run", "examples", "structure", "notes")

#: How many examples a README gets before it stops being a README.
DEFAULT_MAX_EXAMPLES = 6

#: At most this many test-derived examples, however many tests exist. Tests are
#: the weakest kind of example and by far the most numerous.
MAX_TEST_EXAMPLES = 2


def inspect_project(
    root: Path | str,
    *,
    include_tests: bool = True,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
) -> ProjectFacts:
    """Learn everything one pass can learn about the checkout at `root`.

    Raises `FileNotFoundError` for a path that isn't a directory — a typo must
    not read as "this project has nothing to document".
    """
    root = Path(root)
    tree = collect_sources(root)  # raises for a non-directory
    manifest = read_manifest(root)
    prose, prose_notes = read_prose(root)
    notes = list(tree.notes) + list(manifest.notes) + list(prose_notes)

    modules = _module_facts(tree)
    name = manifest.name or _inferred_name(root, modules)
    api = _build_api(manifest, tree, modules)

    found, dropped = _mine(tree, manifest, api, include_tests=include_tests)
    examples = select_examples(found, limit=max_examples)
    notes.extend(dropped)
    notes.extend(_example_notes(found, examples))

    checks, check_notes = _check_facts(root)
    notes.extend(check_notes)

    return ProjectFacts(
        root=str(root),
        name=name,
        version=manifest.version,
        description=manifest.description,
        requires_python=manifest.requires_python,
        dependencies=manifest.dependencies,
        extras=dict(manifest.extras),
        commands=_command_facts(manifest, tree),
        modules=modules,
        examples=examples,
        checks=checks,
        test_paths=tree.test_paths,
        ecosystems=detect_ecosystems(root),
        prose=prose,
        notes=tuple(notes),
        installable=manifest.installable,
    )


# ── modules ───────────────────────────────────────────────────────────────


def _module_facts(tree: SourceTree) -> tuple[ModuleFact, ...]:
    runnable = set(tree.runnable)
    facts: list[ModuleFact] = []
    for source in tree.modules:
        public, declares_all = public_names(source)
        facts.append(
            ModuleFact(
                name=source.module,
                path=source.path,
                doc=source.doc,
                is_package=source.is_package,
                public=public,
                declares_all=declares_all,
                runnable=source.module in runnable,
            )
        )
    return tuple(facts)


def public_names(source: SourceFile) -> tuple[tuple[str, ...], bool]:
    """`(names, declares_all)` for one module.

    A declared `__all__` is an author's statement of intent and wins. When it
    can't be read statically — `__all__ = [*_BASE, "extra"]` — the fallback is
    the module's public definitions, because "this module exports nothing" is a
    worse answer than an approximate list.
    """
    declared = _declared_all(source.tree)
    if declared is not None:
        return declared, True
    return _defined_names(source.tree), False


def _declared_all(tree: ast.Module) -> tuple[str, ...] | None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
            value: ast.expr | None = node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            continue
        if value is None:
            continue
        try:
            loaded = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError):
            return None
        if isinstance(loaded, (list, tuple, set)):
            return tuple(str(item) for item in loaded)
        return None
    return None


def _defined_names(tree: ast.Module) -> tuple[str, ...]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and not node.name.startswith("_"):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names.extend(
                t.id
                for t in node.targets
                if isinstance(t, ast.Name) and not t.id.startswith("_")
            )
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and not node.target.id.startswith("_")
        ):
            names.append(node.target.id)
    return tuple(dict.fromkeys(names))


def _inferred_name(root: Path, modules: Sequence[ModuleFact]) -> str:
    """A name for a repo whose manifest didn't give one.

    The single top-level package first, then the directory. Both are things a
    reader can see; neither is invented.
    """
    tops = [m.name for m in modules if m.depth == 0]
    if len(tops) == 1:
        return tops[0]
    return root.resolve().name


# ── the API examples are checked against ──────────────────────────────────


def _build_api(
    manifest: Manifest, tree: SourceTree, modules: Sequence[ModuleFact]
) -> ProjectApi:
    packages = tuple(m.name.split(".")[0] for m in modules if m.depth == 0)
    return ProjectApi(
        modules={m.name: m.public for m in modules},
        runnable=tree.runnable,
        scripts=tuple(manifest.scripts),
        extras=tuple(manifest.extras),
        installable=manifest.installable,
        exercised=tuple(sorted(exercised_symbols(tree.tests, packages=packages))),
    )


# ── examples ──────────────────────────────────────────────────────────────


def _mine(
    tree: SourceTree,
    manifest: Manifest,
    api: ProjectApi,
    *,
    include_tests: bool,
) -> tuple[tuple[Example, ...], list[str]]:
    """Every example in the tree, validated. Returns (examples, notes)."""
    commands = tuple(manifest.scripts)
    packages = tuple(sorted(api.packages))
    found: list[Example] = []
    for source in tree.modules:
        found.extend(extract_examples(source, commands=commands))
    if include_tests:
        for source in tree.tests:
            found.extend(extract_test_examples(source, packages=packages))

    validated = validate_examples(found, api)
    notes: list[str] = []
    lost = len(found) - len(validated)
    if lost:
        notes.append(
            f"{lost} extracted snippet(s) referenced nothing in this project and "
            "were dropped"
        )
    return validated, notes


def select_examples(
    examples: Sequence[Example], *, limit: int = DEFAULT_MAX_EXAMPLES
) -> tuple[Example, ...]:
    """The best `limit` examples, in the order a README should show them.

    Ranking, in order: verified before unverified, then by kind (a doctest was
    written to be executed, a docstring block to be read, a test to catch
    regressions), then by language (a call shows how the project is used; a
    config fragment supports someone already using it), then shallower modules
    before deeper ones, then file order.
    Near-duplicates are dropped, since two spellings of the same call teach a
    reader nothing the first one didn't.
    """
    ranked = sorted(examples, key=_rank)
    chosen: list[Example] = []
    seen: set[str] = set()
    tests = 0
    for example in ranked:
        if len(chosen) >= limit:
            break
        fingerprint = _fingerprint(example.code)
        if fingerprint in seen:
            continue
        if example.kind == "test":
            if tests >= MAX_TEST_EXAMPLES:
                continue
            tests += 1
        seen.add(fingerprint)
        chosen.append(example)
    return tuple(chosen)


def _rank(example: Example) -> tuple[int, int, int, int, str, int]:
    kind = _index(KIND_ORDER, example.kind)
    language = _index(LANGUAGE_ORDER, example.language)
    depth = example.module.count(".") if example.module else 0
    return (
        0 if example.verified else 1,
        kind,
        language,
        depth,
        example.path,
        example.line,
    )


def _index(order: Sequence[str], value: str) -> int:
    return order.index(value) if value in order else len(order)


def _fingerprint(code: str) -> str:
    """Whitespace-insensitive identity, to catch the same snippet twice."""
    return re.sub(r"\s+", " ", code).strip()


def _example_notes(found: Sequence[Example], chosen: Sequence[Example]) -> list[str]:
    notes: list[str] = []
    stale = [e for e in chosen if not e.verified]
    if stale:
        notes.append(
            f"{len(stale)} example(s) are unverified: "
            + "; ".join(f"{e.location} ({e.verification})" for e in stale)
        )
    held = len(found) - len(chosen)
    if held > 0:
        notes.append(
            f"{held} further example(s) were found but not shown (length budget)"
        )
    return notes


# ── commands ──────────────────────────────────────────────────────────────


def _command_facts(manifest: Manifest, tree: SourceTree) -> tuple[CommandFact, ...]:
    by_module = tree.by_module()
    facts: list[CommandFact] = []
    claimed: set[str] = set()
    for name, target in sorted(manifest.scripts.items()):
        module = target.split(":")[0].strip()
        description, flags = _describe_entry_point(by_module.get(module))
        package = _runnable_package_behind(module, tree)
        if package:
            claimed.add(package)
        facts.append(
            CommandFact(
                name=name,
                target=target,
                source=manifest.source,
                description=description,
                flags=flags,
                aliases=(f"python -m {package}",) if package else (),
            )
        )
    for package in tree.runnable:
        if package in claimed:
            continue
        description, flags = _describe_entry_point(_parser_module(package, tree))
        facts.append(
            CommandFact(
                name=f"python -m {package}",
                target=f"{package}.__main__",
                source=_main_path(package, by_module),
                description=description,
                flags=flags,
            )
        )
    return tuple(facts)


def _runnable_package_behind(module: str, tree: SourceTree) -> str:
    """The runnable package a console script's target lives in, if any.

    `tars-readme = "tars.readme.cli:main"` and `python -m tars.readme` run the
    same code, print the same `--help` and take the same flags, so listing them
    as two commands asks a reader to find a difference that isn't there. The
    evidence for merging them is containment: the script's target module has to
    live inside a package that has a `__main__.py`. A script pointing at
    `other.cli:main` says nothing about `python -m widget`.
    """
    for package in sorted(tree.runnable, key=len, reverse=True):
        if module == package or module.startswith(f"{package}."):
            return package
    return ""


def _main_path(package: str, by_module: dict[str, SourceFile]) -> str:
    """Where `python -m <package>` really lives.

    Derived from the package's own `__init__.py` path rather than composed from
    the dotted name, which would turn `widget.sub` into the non-path
    `widget.sub/__main__.py` and miss a `src/` prefix entirely.
    """
    init = by_module.get(package)
    if init is None:  # pragma: no cover - a runnable package always has one
        return f"{package.replace('.', '/')}/__main__.py"
    directory = init.path.rsplit("/", 1)[0] if "/" in init.path else ""
    return f"{directory}/__main__.py" if directory else "__main__.py"


def _parser_module(package: str, tree: SourceTree) -> SourceFile | None:
    """The module in `package` that builds the command-line parser.

    Found by looking for the parser construction rather than by assuming the
    module is called `cli`: the convention is common but it is still a guess, and
    a guess that misses leaves a command with no description for no reason.
    """
    candidates = [
        source
        for source in tree.modules
        if (source.module == package or source.module.startswith(f"{package}."))
        and _builds_a_parser(source)
    ]
    candidates.sort(key=lambda s: (s.depth, s.module))
    return candidates[0] if candidates else None


def _builds_a_parser(source: SourceFile) -> bool:
    return any(
        isinstance(node, ast.Call) and _callee_name(node.func).endswith("Parser")
        for node in ast.walk(source.tree)
    )


def _describe_entry_point(source: SourceFile | None) -> tuple[str, tuple[str, ...]]:
    """A command's own `--help` text and flags, read from its parser.

    Matched on the callee name ending in `Parser` rather than on
    `argparse.ArgumentParser` exactly, because a project that overrides
    `error()` to raise — as this one does — constructs a subclass.
    """
    if source is None:
        return "", ()
    description = ""
    flags: list[str] = []
    for node in ast.walk(source.tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _callee_name(node.func)
        if callee.endswith("Parser"):
            if not description:
                description = _keyword_string(node, "description")
        elif callee == "add_argument":
            option = _option_spellings(node)
            if option:
                flags.append(option)
    return _one_line(description), tuple(dict.fromkeys(flags))


def _option_spellings(call: ast.Call) -> str:
    """`-q/--quiet` for one `add_argument("-q", "--quiet", ...)` call.

    Grouped per call because that is what the file says: they are two spellings
    of one option. Flattened, a command with five aliased options looks like a
    command with ten switches.
    """
    spellings = [
        arg.value
        for arg in call.args
        if isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and arg.value.startswith("-")
    ]
    return "/".join(spellings)


def _callee_name(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _keyword_string(call: ast.Call, name: str) -> str:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            if isinstance(value, str):
                return value
    return ""


def _one_line(text: str) -> str:
    return " ".join(text.split())


# ── checks ────────────────────────────────────────────────────────────────


def _check_facts(root: Path) -> tuple[tuple[CheckFact, ...], list[str]]:
    """What the verification gate says this repo can run.

    Reusing the gate means the README's "how to check it" section describes the
    same commands CI would run, rather than a second guess at them. Restricted
    to the gate's default kinds for the same reason: `build` is opt-in there
    (`--include build`), so printing `python -m build` under "verifying it"
    would document a check `tars-verify` doesn't run.
    """
    try:
        checks = detect_checks(root, kinds=DEFAULT_KINDS)
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        return (), [f"check detection failed ({exc})"]
    return (
        tuple(
            CheckFact(
                kind=check.kind,
                tool=check.tool,
                command_line=check.command_line,
                evidence=check.evidence,
            )
            for check in checks
        ),
        [],
    )


# ── prose ─────────────────────────────────────────────────────────────────


def read_prose(root: Path | str) -> tuple[dict[str, Prose], tuple[str, ...]]:
    """Read `.tars/readme.json`. Returns (prose by section, notes).

    Never raises. A malformed override file degrades to "no supplied prose" plus
    a note, because one bad edit should cost a section, not the README.
    """
    path = Path(root) / PROSE_PATH
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}, ()

    try:
        loaded = json.loads(text)
    except ValueError as exc:
        return {}, (f"{PROSE_PATH}: not valid JSON ({exc})",)
    if not isinstance(loaded, dict):
        return {}, (f"{PROSE_PATH}: expected a JSON object of section -> text",)

    prose: dict[str, Prose] = {}
    notes: list[str] = []
    for key, value in loaded.items():
        if key not in PROSE_SECTIONS:
            notes.append(
                f"{PROSE_PATH}: ignoring unknown section {key!r} "
                f"(known: {', '.join(PROSE_SECTIONS)})"
            )
            continue
        if not isinstance(value, str) or not value.strip():
            notes.append(f"{PROSE_PATH}: section {key!r} is not a non-empty string")
            continue
        prose[key] = Prose(text=value.strip(), source=PROSE_PATH)
    return prose, tuple(notes)
