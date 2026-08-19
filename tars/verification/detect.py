"""Mapping a repo to the checks it can actually run.

Three rules shape everything here:

1. **Evidence, or nothing.** A check is only proposed when something in the
   repo points at it — a config section, a declared dependency, a tool in the
   repo's own venv. The gate never invents a linter a repo gave no sign of
   using, because "lint failed" on a tool the repo doesn't use is worse than
   "no linter found".

2. **Filesystem only.** Detection never executes anything, so it's fast, safe
   on an untrusted checkout, and trivially testable against a tmp dir. Whether
   a detected tool is *installed* is the runner's question, not this module's.

3. **Candidates, not a single answer.** A kind can have several candidates in
   preference order (ruff, then flake8, then pylint). The gate runs the first
   one that's installed. Ordering is by evidence strength first, then by tool
   preference — a repo with a .flake8 file and an incidental ruff in its venv
   means flake8.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import ALL_KINDS, VALID_KINDS, Check, CheckKind

PYTHON = "python"
NODE = "node"
GO = "go"
RUST = "rust"
MAKE = "make"
OVERRIDE = "override"

#: A repo can declare its own check surface here when detection can't know
#: better — a bespoke test harness, a linter behind a wrapper script, or a
#: check that must simply be turned off. JSON rather than TOML so it parses on
#: every supported Python without a dependency.
#:
#:     {"checks": {"test": "make integration", "lint": null}}
OVERRIDES_PATH = ".tars/verification.json"

#: Evidence strength. Lower is stronger; it's the primary sort key, so a
#: configured tool always outranks one that merely happens to be installed.
_CONFIG = 0  # a config file or config section names the tool
_DECLARED = 1  # a dependency file declares it
_INSTALLED = 2  # it's in the repo's own venv/node_modules but declared nowhere
_INFERRED = 3  # no mention of the tool at all; inferred from layout alone

#: Never treated as source, never walked for packages.
_IGNORED_DIRS = frozenset(
    {
        "venv",
        "env",
        "build",
        "dist",
        "node_modules",
        "__pycache__",
        "site-packages",
        "htmlcov",
        "target",
    }
)

#: Files that can declare a Python tool as a dependency.
_PY_DEPENDENCY_FILES = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "tox.ini",
    "Pipfile",
    "environment.yml",
)

_PY_DEPENDENCY_GLOBS = ("requirements*.txt", "requirements/*.txt")

#: (file, section-prefix or None) pairs that count as config evidence.
#: A None section means the file's mere existence is the evidence.
_PY_CONFIG_MARKERS: dict[str, tuple[tuple[str, str | None], ...]] = {
    "ruff": (
        ("pyproject.toml", "[tool.ruff"),
        ("ruff.toml", None),
        (".ruff.toml", None),
    ),
    "flake8": (
        (".flake8", None),
        ("setup.cfg", "[flake8]"),
        ("tox.ini", "[flake8]"),
    ),
    "pylint": (
        (".pylintrc", None),
        ("pylintrc", None),
        ("pyproject.toml", "[tool.pylint"),
    ),
    "black": (("pyproject.toml", "[tool.black]"),),
    "mypy": (
        ("pyproject.toml", "[tool.mypy"),
        ("mypy.ini", None),
        (".mypy.ini", None),
        ("setup.cfg", "[mypy]"),
        ("tox.ini", "[mypy]"),
    ),
    "pyright": (
        ("pyrightconfig.json", None),
        ("pyproject.toml", "[tool.pyright"),
    ),
    "pytest": (
        ("pyproject.toml", "[tool.pytest"),
        ("pytest.ini", None),
        ("tox.ini", "[pytest]"),
        ("setup.cfg", "[tool:pytest]"),
        ("conftest.py", None),
    ),
}

_PY_MARKER_FILES = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "tox.ini",
    "Pipfile",
    "poetry.lock",
    "uv.lock",
    "conftest.py",
    "mypy.ini",
    "pytest.ini",
    "ruff.toml",
    ".ruff.toml",
    ".flake8",
    "pyrightconfig.json",
)


@dataclass(frozen=True)
class _Candidate:
    """A check plus the sort keys that decide preference within its kind."""

    rank: int
    preference: int
    check: Check


class Repo:
    """Cached read-only view of a checkout.

    Detectors ask a lot of overlapping questions ("does pyproject mention X?"),
    so file reads are cached — detection stays one pass over the tree even with
    a dozen detectors.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._text_cache: dict[str, str | None] = {}

    def exists(self, rel: str) -> bool:
        return (self.root / rel).exists()

    def is_dir(self, rel: str) -> bool:
        return (self.root / rel).is_dir()

    def read(self, rel: str) -> str | None:
        """File contents, or None if unreadable. Never raises."""
        if rel not in self._text_cache:
            path = self.root / rel
            try:
                self._text_cache[rel] = path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                self._text_cache[rel] = None
        return self._text_cache[rel]

    def glob(self, pattern: str) -> list[Path]:
        try:
            return sorted(p for p in self.root.glob(pattern) if p.is_file())
        except OSError:  # pragma: no cover - unreadable tree
            return []

    def has_local_tool(self, name: str) -> str | None:
        """The repo-local bin dir holding `name`, if any.

        Deliberately does not consult PATH: a tool on the system PATH says
        nothing about this repo, while `.venv/bin/ruff` is a real statement
        about how this repo is checked.
        """
        for rel in (
            ".venv/bin",
            ".venv/Scripts",
            "venv/bin",
            "env/bin",
            "node_modules/.bin",
        ):
            candidate = self.root / rel / name
            if candidate.exists():
                return rel
        return None

    def top_level_dirs(self) -> list[Path]:
        try:
            entries = sorted(p for p in self.root.iterdir() if p.is_dir())
        except OSError:  # pragma: no cover - unreadable tree
            return []
        return [
            p
            for p in entries
            if not p.name.startswith(".") and p.name not in _IGNORED_DIRS
        ]


# ── evidence helpers ──────────────────────────────────────────────────────


def _config_evidence(repo: Repo, tool: str) -> str | None:
    for filename, section in _PY_CONFIG_MARKERS.get(tool, ()):
        if section is None:
            if repo.exists(filename):
                return filename
            continue
        text = repo.read(filename)
        if text and section in text:
            return (
                f"{filename} {section}]"
                if not section.endswith("]")
                else f"{filename} {section}"
            )
    return None


def _declared_evidence(
    repo: Repo, tool: str, files: Sequence[str], globs: Sequence[str]
) -> str | None:
    pattern = re.compile(rf"(?<![\w-]){re.escape(tool)}(?![\w-])", re.IGNORECASE)
    for filename in files:
        text = repo.read(filename)
        if text and pattern.search(text):
            return f"declared in {filename}"
    for glob in globs:
        for path in repo.glob(glob):
            rel = path.relative_to(repo.root).as_posix()
            text = repo.read(rel)
            if text and pattern.search(text):
                return f"declared in {rel}"
    return None


def _tool_evidence(
    repo: Repo,
    tool: str,
    *,
    dependency_files: Sequence[str] = _PY_DEPENDENCY_FILES,
    dependency_globs: Sequence[str] = _PY_DEPENDENCY_GLOBS,
    binary: str | None = None,
) -> tuple[int, str] | None:
    """Strongest evidence that `tool` is part of this repo's check surface."""
    config = _config_evidence(repo, tool)
    if config:
        return _CONFIG, config
    declared = _declared_evidence(repo, tool, dependency_files, dependency_globs)
    if declared:
        return _DECLARED, declared
    bin_dir = repo.has_local_tool(binary or tool)
    if bin_dir:
        return _INSTALLED, f"{binary or tool} installed in {bin_dir}"
    return None


# ── python ────────────────────────────────────────────────────────────────


def _is_python_repo(repo: Repo) -> bool:
    if any(repo.exists(name) for name in _PY_MARKER_FILES):
        return True
    if repo.glob("requirements*.txt"):
        return True
    return bool(repo.glob("*.py"))


def _python_targets(repo: Repo) -> tuple[str, ...]:
    """Source paths to hand a type-checker.

    Explicit targets, not `.`: mypy walks whatever you give it, and `.` in a
    repo with a virtualenv means type-checking the entire dependency tree.
    """
    targets: list[str] = []
    if repo.is_dir("src") and (repo.glob("src/**/*.py") or repo.glob("src/*.py")):
        targets.append("src")
    else:
        targets.extend(
            d.name for d in repo.top_level_dirs() if (d / "__init__.py").exists()
        )
    if repo.is_dir("tests") and "tests" not in targets:
        targets.append("tests")
    return tuple(targets) if targets else (".",)


def _has_test_layout(repo: Repo) -> str | None:
    """The directory holding tests, if the layout makes one obvious."""
    for name in ("tests", "test"):
        if repo.is_dir(name):
            return name
    if repo.glob("test_*.py") or repo.glob("*_test.py"):
        return "."
    return None


def _detect_python(repo: Repo) -> list[_Candidate]:
    if not _is_python_repo(repo):
        return []

    found: list[_Candidate] = []

    def add(
        kind: CheckKind,
        tool: str,
        command: tuple[str, ...],
        rank: int,
        evidence: str,
        preference: int,
    ) -> None:
        found.append(
            _Candidate(
                rank=rank,
                preference=preference,
                check=Check(
                    kind=kind,
                    tool=tool,
                    command=command,
                    ecosystem=PYTHON,
                    evidence=evidence,
                ),
            )
        )

    # lint / format / typecheck: straight tool lookups, preference by list order.
    linters: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("ruff", ("ruff", "check", ".")),
        ("flake8", ("flake8",)),
        ("pylint", ("pylint", *_python_targets(repo))),
    )
    for pref, (tool, command) in enumerate(linters):
        evidence = _tool_evidence(repo, tool)
        if evidence:
            add("lint", tool, command, evidence[0], evidence[1], pref)

    # Check-only, always. A gate that reformats the repo it is inspecting has
    # changed the thing it was asked to measure.
    formatters: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("ruff-format", "ruff", ("ruff", "format", "--check", ".")),
        ("black", "black", ("black", "--check", ".")),
    )
    for pref, (tool, binary, command) in enumerate(formatters):
        evidence = _tool_evidence(repo, binary, binary=binary)
        if evidence:
            add("format", tool, command, evidence[0], evidence[1], pref)

    typecheckers: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("mypy", ("mypy", *_python_targets(repo))),
        ("pyright", ("pyright",)),
    )
    for pref, (tool, command) in enumerate(typecheckers):
        evidence = _tool_evidence(repo, tool)
        if evidence:
            add("typecheck", tool, command, evidence[0], evidence[1], pref)

    # tests: pytest if anything points at it, else inferred from layout.
    test_dir = _has_test_layout(repo)
    pytest_evidence = _tool_evidence(repo, "pytest")
    if pytest_evidence is None and test_dir:
        pytest_evidence = (_INFERRED, f"{test_dir} contains tests but names no runner")
    if pytest_evidence:
        add("test", "pytest", ("pytest",), pytest_evidence[0], pytest_evidence[1], 0)

    # unittest is the stdlib safety net, offered only when the pytest evidence
    # is weak. If a repo genuinely configures pytest, running its suite under a
    # different runner would produce misleading failures — better to report
    # pytest as unavailable.
    if test_dir and (pytest_evidence is None or pytest_evidence[0] >= _INSTALLED):
        add(
            "test",
            "unittest",
            ("python", "-m", "unittest", "discover", "-s", test_dir, "-t", "."),
            _INFERRED,
            f"{test_dir} contains tests; unittest is stdlib",
            9,
        )

    if (text := repo.read("pyproject.toml")) and "[build-system]" in text:
        add(
            "build",
            "build",
            ("python", "-m", "build"),
            _CONFIG,
            "pyproject.toml [build-system]",
            0,
        )

    return found


class _Collector:
    """Accumulates candidates for one ecosystem."""

    def __init__(self, ecosystem: str) -> None:
        self.ecosystem = ecosystem
        self.found: list[_Candidate] = []

    def add(
        self,
        kind: CheckKind,
        tool: str,
        command: Sequence[str],
        *,
        evidence: str,
        rank: int = _CONFIG,
        preference: int = 0,
    ) -> None:
        self.found.append(
            _Candidate(
                rank=rank,
                preference=preference,
                check=Check(
                    kind=kind,
                    tool=tool,
                    command=tuple(command),
                    ecosystem=self.ecosystem,
                    evidence=evidence,
                ),
            )
        )


# ── node / typescript ─────────────────────────────────────────────────────

#: Script names accepted per kind. For `format`, only check-shaped names: a bare
#: `npm run format` rewrites the working tree, and the gate must not mutate the
#: repo it is measuring.
_NODE_SCRIPTS: dict[str, tuple[str, ...]] = {
    "lint": ("lint",),
    "format": (
        "format:check",
        "fmt:check",
        "format-check",
        "check-format",
        "prettier:check",
    ),
    "typecheck": ("typecheck", "type-check", "types", "tsc"),
    "test": ("test",),
    "build": ("build",),
}

_PACKAGE_MANAGERS = ("npm", "pnpm", "yarn", "bun")

#: Lockfile -> package manager, most specific first.
_LOCKFILES = (
    ("pnpm-lock.yaml", "pnpm"),
    ("yarn.lock", "yarn"),
    ("bun.lockb", "bun"),
    ("bun.lock", "bun"),
    ("package-lock.json", "npm"),
    ("npm-shrinkwrap.json", "npm"),
)

_ESLINT_CONFIGS = (
    "eslint.config.js",
    "eslint.config.mjs",
    "eslint.config.cjs",
    "eslint.config.ts",
    ".eslintrc",
    ".eslintrc.js",
    ".eslintrc.cjs",
    ".eslintrc.json",
    ".eslintrc.yml",
    ".eslintrc.yaml",
)

_PRETTIER_CONFIGS = (
    ".prettierrc",
    ".prettierrc.json",
    ".prettierrc.js",
    ".prettierrc.cjs",
    ".prettierrc.yml",
    ".prettierrc.yaml",
    "prettier.config.js",
    "prettier.config.mjs",
    "prettier.config.cjs",
)


def _is_node_repo(repo: Repo) -> bool:
    return repo.exists("package.json")


def _package_json(repo: Repo) -> dict[str, Any]:
    """package.json as a dict, or empty if it's unparseable.

    A malformed manifest must degrade to "no script signals", not crash the
    gate — plenty of real repos have a broken package.json on some branch.
    """
    text = repo.read("package.json")
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _package_manager(repo: Repo, pkg: dict[str, Any]) -> tuple[str, str]:
    """Which package manager to invoke, and why."""
    declared = pkg.get("packageManager")
    if isinstance(declared, str) and declared:
        name = declared.split("@", 1)[0].strip()
        if name in _PACKAGE_MANAGERS:
            return name, "package.json packageManager"
    for lockfile, name in _LOCKFILES:
        if repo.exists(lockfile):
            return name, lockfile
    return "npm", "package.json (no lockfile, defaulting to npm)"


def _node_declares(pkg: dict[str, Any], name: str) -> str | None:
    for section in ("devDependencies", "dependencies"):
        deps = pkg.get(section)
        if isinstance(deps, dict) and name in deps:
            return f"package.json {section}.{name}"
    return None


def _first_existing(repo: Repo, names: Sequence[str]) -> str | None:
    return next((n for n in names if repo.exists(n)), None)


def _detect_node(repo: Repo) -> list[_Candidate]:
    out = _Collector(NODE)
    pkg = _package_json(repo)
    raw_scripts = pkg.get("scripts")
    scripts: dict[str, Any] = raw_scripts if isinstance(raw_scripts, dict) else {}
    manager, manager_evidence = _package_manager(repo, pkg)

    # The repo's own scripts win: they encode flags (--max-warnings 0, project
    # globs, reporters) that the gate has no business guessing at.
    for kind, names in _NODE_SCRIPTS.items():
        for script in names:
            if script in scripts:
                out.add(
                    kind,  # type: ignore[arg-type]
                    f"{manager}:{script}",
                    (manager, "run", script),
                    evidence=f"package.json scripts.{script} (via {manager_evidence})",
                    preference=0,
                )
                break

    def direct(
        kind: CheckKind,
        tool: str,
        command: Sequence[str],
        configs: Sequence[str] = (),
        manifest_key: str | None = None,
        dep_name: str | None = None,
    ) -> None:
        """A tool invoked straight, for repos without a matching script."""
        dep = dep_name or tool
        config = _first_existing(repo, configs)
        if config:
            rank, evidence = _CONFIG, config
        elif manifest_key and manifest_key in pkg:
            rank, evidence = _CONFIG, f"package.json {manifest_key}"
        elif (declared := _node_declares(pkg, dep)) is not None:
            rank, evidence = _DECLARED, declared
        elif (bin_dir := repo.has_local_tool(command[0])) is not None:
            rank, evidence = _INSTALLED, f"{command[0]} installed in {bin_dir}"
        else:
            return
        out.add(kind, tool, command, evidence=evidence, rank=rank, preference=1)

    direct("lint", "eslint", ("eslint", "."), _ESLINT_CONFIGS, "eslintConfig")
    direct(
        "format",
        "prettier",
        ("prettier", "--check", "."),
        _PRETTIER_CONFIGS,
        "prettier",
    )
    # --noEmit is not optional: plain `tsc` writes .js files next to the sources,
    # which would leave the repo dirty after a verification run.
    direct(
        "typecheck",
        "tsc",
        ("tsc", "--noEmit"),
        ("tsconfig.json",),
        dep_name="typescript",
    )
    # `vitest` alone watches; `vitest run` exits.
    direct(
        "test", "vitest", ("vitest", "run"), ("vitest.config.ts", "vitest.config.js")
    )
    direct(
        "test",
        "jest",
        ("jest",),
        ("jest.config.js", "jest.config.ts", "jest.config.cjs"),
        "jest",
    )
    direct(
        "test", "mocha", ("mocha",), (".mocharc.json", ".mocharc.yml", ".mocharc.js")
    )
    return out.found


# ── go ────────────────────────────────────────────────────────────────────

_GOLANGCI_CONFIGS = (
    ".golangci.yml",
    ".golangci.yaml",
    ".golangci.toml",
    ".golangci.json",
)


def _is_go_repo(repo: Repo) -> bool:
    return repo.exists("go.mod") or repo.exists("go.sum")


def _detect_go(repo: Repo) -> list[_Candidate]:
    out = _Collector(GO)
    marker = "go.mod" if repo.exists("go.mod") else "go.sum"
    out.add("test", "go-test", ("go", "test", "./..."), evidence=marker)
    # `go vet` is the closest thing to a type/soundness check that ships with
    # the toolchain; `go build` is the build kind, not this one.
    out.add("typecheck", "go-vet", ("go", "vet", "./..."), evidence=marker)
    out.add("build", "go-build", ("go", "build", "./..."), evidence=marker)
    if (config := _first_existing(repo, _GOLANGCI_CONFIGS)) is not None:
        out.add("lint", "golangci-lint", ("golangci-lint", "run"), evidence=config)
    # No format check on purpose: `gofmt -l .` prints unformatted files and still
    # exits 0, so an exit-code gate would report a permanent pass. A check that
    # cannot fail is worse than no check at all.
    return out.found


# ── rust ──────────────────────────────────────────────────────────────────


def _is_rust_repo(repo: Repo) -> bool:
    return repo.exists("Cargo.toml")


def _detect_rust(repo: Repo) -> list[_Candidate]:
    out = _Collector(RUST)
    out.add("test", "cargo-test", ("cargo", "test"), evidence="Cargo.toml")
    out.add(
        "typecheck",
        "cargo-check",
        ("cargo", "check", "--all-targets"),
        evidence="Cargo.toml",
    )
    # rustfmt ships with the default toolchain and `--check` is a real exit code.
    out.add("format", "cargo-fmt", ("cargo", "fmt", "--check"), evidence="Cargo.toml")
    out.add("build", "cargo-build", ("cargo", "build"), evidence="Cargo.toml")

    # Clippy needs evidence: on a toolchain without the component, `cargo clippy`
    # exits non-zero, which the gate would report as a lint failure rather than
    # a missing tool. `-D warnings` is required for the check to mean anything —
    # clippy exits 0 on warnings otherwise.
    cargo_toml = repo.read("Cargo.toml") or ""
    evidence = _first_existing(repo, ("clippy.toml", ".clippy.toml"))
    if evidence is None and "[lints.clippy]" in cargo_toml:
        evidence = "Cargo.toml [lints.clippy]"
    if evidence:
        out.add(
            "lint",
            "clippy",
            ("cargo", "clippy", "--all-targets", "--", "-D", "warnings"),
            evidence=evidence,
        )
    return out.found


# ── makefile fallback ─────────────────────────────────────────────────────

_MAKEFILES = ("Makefile", "makefile", "GNUmakefile")

#: Target names accepted per kind. `format` is check-shaped only, same rule as
#: the node scripts.
_MAKE_TARGETS: dict[str, tuple[str, ...]] = {
    "lint": ("lint",),
    "format": ("fmt-check", "format-check", "check-format", "fmt_check"),
    "typecheck": ("typecheck", "type-check", "types", "mypy"),
    "test": ("test", "tests"),
    "build": ("build",),
}

#: A target is `name:` but not `name :=` / `name ::=` — assignments are not
#: targets, and treating one as a target produces a check that can't run.
_MAKE_TARGET_RE = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_.\-/]*)\s*:(?!=)", re.MULTILINE)


def _is_make_repo(repo: Repo) -> bool:
    return any(repo.exists(name) for name in _MAKEFILES)


def _detect_make(repo: Repo) -> list[_Candidate]:
    out = _Collector(MAKE)
    filename = _first_existing(repo, _MAKEFILES)
    if filename is None:  # pragma: no cover - guarded by _is_make_repo
        return []
    text = repo.read(filename) or ""
    targets = {m.group(1) for m in _MAKE_TARGET_RE.finditer(text)}
    for kind, names in _MAKE_TARGETS.items():
        for target in names:
            if target in targets:
                out.add(
                    kind,  # type: ignore[arg-type]
                    f"make:{target}",
                    ("make", target),
                    evidence=f"{filename} target {target!r}",
                )
                break
    return out.found


#: (ecosystem name, applies?, detector). Order sets the order ecosystems appear
DETECTORS: tuple[
    tuple[str, Callable[[Repo], bool], Callable[[Repo], list[_Candidate]]], ...
] = (
    (PYTHON, _is_python_repo, _detect_python),
    (NODE, _is_node_repo, _detect_node),
    (GO, _is_go_repo, _detect_go),
    (RUST, _is_rust_repo, _detect_rust),
    (MAKE, _is_make_repo, _detect_make),
)


def detect_ecosystems(root: Path | str) -> tuple[str, ...]:
    """Which ecosystems this checkout looks like, in report order."""
    repo = Repo(root)
    _require_dir(repo.root)
    return tuple(name for name, applies, _ in DETECTORS if applies(repo))


def detect_candidates(
    root: Path | str,
    *,
    kinds: Iterable[str] | None = None,
    overrides: dict[str, tuple[str, ...] | None] | None = None,
) -> tuple[_Candidate, ...]:
    """Every check candidate, sorted by kind, then evidence, then preference."""
    repo = Repo(root)
    _require_dir(repo.root)
    wanted = set(kinds) if kinds is not None else set(ALL_KINDS)
    if overrides is None:
        overrides, _ = load_overrides(repo.root)

    candidates: list[_Candidate] = []
    for _, applies, detector in DETECTORS:
        if applies(repo):
            candidates.extend(detector(repo))
    candidates = _drop_redundant_make(candidates)
    candidates = _apply_overrides(candidates, overrides)

    kind_order = {kind: i for i, kind in enumerate(ALL_KINDS)}
    selected = [c for c in candidates if c.check.kind in wanted]
    selected.sort(
        key=lambda c: (
            kind_order.get(c.check.kind, len(kind_order)),
            c.rank,
            c.preference,
            c.check.tool,
        )
    )
    return tuple(selected)


def detect_checks(
    root: Path | str,
    *,
    kinds: Iterable[str] | None = None,
    overrides: dict[str, tuple[str, ...] | None] | None = None,
) -> tuple[Check, ...]:
    """Checks this repo could run, best candidate per kind first."""
    return tuple(
        c.check for c in detect_candidates(root, kinds=kinds, overrides=overrides)
    )


def load_overrides(
    root: Path | str,
) -> tuple[dict[str, tuple[str, ...] | None], tuple[str, ...]]:
    """Read `.tars/verification.json`, returning (overrides, notes).

    Never raises: a broken override file degrades to detection plus a note, so
    one bad edit can't take the gate offline. A value of `null` means "this kind
    is deliberately disabled here", which is different from "not detected" and
    is worth being able to say.
    """
    repo = Repo(root)
    text = repo.read(OVERRIDES_PATH)
    if text is None:
        return {}, ()

    try:
        data = json.loads(text)
    except ValueError as exc:
        return {}, (f"ignored {OVERRIDES_PATH}: not valid JSON ({exc})",)
    if not isinstance(data, dict):
        return {}, (
            f"ignored {OVERRIDES_PATH}: expected a JSON object at the top level",
        )

    raw = data.get("checks")
    if raw is None:
        return {}, (f"ignored {OVERRIDES_PATH}: no 'checks' object",)
    if not isinstance(raw, dict):
        return {}, (f"ignored {OVERRIDES_PATH}: 'checks' must be an object",)

    overrides: dict[str, tuple[str, ...] | None] = {}
    notes: list[str] = []
    for kind, value in raw.items():
        if kind not in VALID_KINDS:
            notes.append(
                f"ignored unknown kind {kind!r} in {OVERRIDES_PATH} "
                f"(expected one of {sorted(VALID_KINDS)})"
            )
            continue
        if value is None or value is False:
            overrides[kind] = None
            continue
        if isinstance(value, str):
            argv = tuple(shlex.split(value))
        elif isinstance(value, list) and all(isinstance(part, str) for part in value):
            argv = tuple(value)
        else:
            notes.append(
                f"ignored {OVERRIDES_PATH} checks.{kind}: expected a command string, "
                "an argv list, or null"
            )
            continue
        if not argv:
            notes.append(f"ignored {OVERRIDES_PATH} checks.{kind}: empty command")
            continue
        overrides[kind] = argv
    return overrides, tuple(notes)


def _apply_overrides(
    candidates: Sequence[_Candidate],
    overrides: dict[str, tuple[str, ...] | None],
) -> list[_Candidate]:
    """Replace detection for any kind the repo declared itself."""
    if not overrides:
        return list(candidates)

    kept = [c for c in candidates if c.check.kind not in overrides]
    for kind, argv in overrides.items():
        if argv is None:  # deliberately disabled
            continue
        kept.append(
            _Candidate(
                rank=_CONFIG,
                preference=0,
                check=Check(
                    kind=kind,  # type: ignore[arg-type]
                    tool=Path(argv[0]).name,
                    command=argv,
                    ecosystem=OVERRIDE,
                    evidence=f"{OVERRIDES_PATH} checks.{kind}",
                ),
            )
        )
    return kept


def _require_dir(root: Path) -> None:
    # A typo'd path must not look like "this repo has nothing to verify".
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")


def _drop_redundant_make(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    """Keep Makefile candidates only for kinds no real ecosystem covers.

    `make test` in a Python repo almost always just calls pytest; running both
    doubles the wall clock for one signal. Where make covers a kind nothing else
    does (a shellcheck lint target, say), it's the only thing that will find it.
    """
    covered = {c.check.kind for c in candidates if c.check.ecosystem != MAKE}
    return [
        c
        for c in candidates
        if c.check.ecosystem != MAKE or c.check.kind not in covered
    ]
