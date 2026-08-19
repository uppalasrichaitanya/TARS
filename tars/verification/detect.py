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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .types import ALL_KINDS, Check, CheckKind

PYTHON = "python"

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
                self._text_cache[rel] = path.read_text(encoding="utf-8", errors="replace")
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
        for rel in (".venv/bin", ".venv/Scripts", "venv/bin", "env/bin", "node_modules/.bin"):
            candidate = self.root / rel / name
            if candidate.exists():
                return rel
        return None

    def top_level_dirs(self) -> list[Path]:
        try:
            entries = sorted(p for p in self.root.iterdir() if p.is_dir())
        except OSError:  # pragma: no cover - unreadable tree
            return []
        return [p for p in entries if not p.name.startswith(".") and p.name not in _IGNORED_DIRS]


# ── evidence helpers ──────────────────────────────────────────────────────


def _config_evidence(repo: Repo, tool: str) -> str | None:
    for filename, section in _PY_CONFIG_MARKERS.get(tool, ()):
        if section is None:
            if repo.exists(filename):
                return filename
            continue
        text = repo.read(filename)
        if text and section in text:
            return f"{filename} {section}]" if not section.endswith("]") else f"{filename} {section}"
    return None


def _declared_evidence(repo: Repo, tool: str, files: Sequence[str], globs: Sequence[str]) -> str | None:
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
        add("build", "build", ("python", "-m", "build"), _CONFIG, "pyproject.toml [build-system]", 0)

    return found


#: (ecosystem name, applies?, detector). Order sets the order ecosystems appear
#: in a report; within a kind each ecosystem is selected independently, because
#: a repo with both a pyproject and a package.json really does need both suites.
DETECTORS: tuple[tuple[str, Callable[[Repo], bool], Callable[[Repo], list[_Candidate]]], ...] = (
    (PYTHON, _is_python_repo, _detect_python),
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
) -> tuple[_Candidate, ...]:
    """Every check candidate, sorted by kind, then evidence, then preference."""
    repo = Repo(root)
    _require_dir(repo.root)
    wanted = set(kinds) if kinds is not None else set(ALL_KINDS)

    candidates: list[_Candidate] = []
    for _, applies, detector in DETECTORS:
        if applies(repo):
            candidates.extend(detector(repo))

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
) -> tuple[Check, ...]:
    """Checks this repo could run, best candidate per kind first."""
    return tuple(c.check for c in detect_candidates(root, kinds=kinds))


def _require_dir(root: Path) -> None:
    # A typo'd path must not look like "this repo has nothing to verify".
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
