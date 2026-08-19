"""The corpus: which files to read, parsed but never imported.

`ast` does all the work. A docs tool pointed at an unfamiliar checkout has no
business importing that checkout — an import runs module-level code, needs the
dependencies installed, and fails on the half-finished branch you most want to
read. Parsing gets docstrings, `__all__`, signatures and argparse definitions
without any of that.

Tests are collected as a separate corpus on purpose. They are the strongest
evidence that a call actually works (something ran them), and simultaneously the
worst thing to present as public API. Keeping them apart lets the example miner
use them as evidence while the structure section ignores them.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

#: Directories never walked: environments, caches, build output, VCS metadata.
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        ".env",
        ".tox",
        ".nox",
        "build",
        "dist",
        "node_modules",
        "__pycache__",
        "site-packages",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "htmlcov",
        "target",
        ".eggs",
    }
)

#: Directory names that mean "these files are tests" wherever they appear.
TEST_DIRS = frozenset({"tests", "test", "testing"})

#: Build and test plumbing. Real files, but nothing a README should introduce as
#: part of the project's structure.
SCAFFOLDING = frozenset(
    {"setup.py", "conftest.py", "noxfile.py", "tasks.py", "manage.py"}
)


def _is_test_path(rel: Path) -> bool:
    name = rel.name
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    return any(part in TEST_DIRS for part in rel.parts[:-1])


@dataclass(frozen=True)
class SourceFile:
    """One parsed Python file.

    `module` is the dotted name it would be imported as, which is what example
    validation resolves imports against.
    """

    path: str
    module: str
    text: str
    tree: ast.Module
    is_package: bool = False

    @property
    def doc(self) -> str:
        return ast.get_docstring(self.tree) or ""

    @property
    def depth(self) -> int:
        return self.module.count(".")


@dataclass(frozen=True)
class SourceTree:
    """Everything worth reading in one checkout."""

    root: str
    modules: tuple[SourceFile, ...] = ()
    tests: tuple[SourceFile, ...] = ()
    #: Packages with a `__main__.py`, i.e. the ones `python -m` can run.
    runnable: tuple[str, ...] = ()
    #: Directories where tests were found, relative to the root.
    test_paths: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_files(self) -> tuple[SourceFile, ...]:
        return self.modules + self.tests

    def by_module(self) -> dict[str, SourceFile]:
        return {s.module: s for s in self.modules}


def collect_sources(root: Path | str) -> SourceTree:
    """Parse every Python file in `root` that is worth documenting.

    Raises `FileNotFoundError` for a path that isn't a directory: a typo must
    not read as "this project has no code in it".
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")

    modules: list[SourceFile] = []
    tests: list[SourceFile] = []
    runnable: list[str] = []
    test_dirs: list[str] = []
    notes: list[str] = []

    package_roots = _package_roots(root)
    python_files = _walk_python(root)
    # A repo with real packages is documented through them; a repo with none may
    # still be a single-file tool, and that file is then the whole project.
    has_packages = any(p.name == "__init__.py" for p in python_files)

    for path in python_files:
        rel = path.relative_to(root)
        posix = rel.as_posix()
        text = _read(path)
        if text is None:
            notes.append(f"{posix}: could not be read")
            continue
        try:
            tree = ast.parse(text, filename=posix)
        except SyntaxError as exc:
            notes.append(
                f"{posix}: skipped, does not parse ({exc.msg} at line {exc.lineno})"
            )
            continue

        is_test = _is_test_path(rel)
        module = _module_name(rel, package_roots)
        source = SourceFile(
            path=posix,
            module=module,
            text=text,
            tree=tree,
            is_package=path.name == "__init__.py",
        )

        if is_test:
            tests.append(source)
            top = rel.parts[0] if len(rel.parts) > 1 else "."
            if top not in test_dirs:
                test_dirs.append(top)
            continue

        if path.name == "__main__.py":
            # Documented through its package, not as a module of its own.
            parent = module.rsplit(".", 1)[0] if "." in module else module
            runnable.append(parent)
            continue
        if rel.name in SCAFFOLDING:
            continue
        # A loose script next to real packages is almost always scratch work.
        if has_packages and len(rel.parts) == 1:
            continue
        modules.append(source)

    modules.sort(key=lambda s: (s.depth, s.module))
    tests.sort(key=lambda s: s.path)
    return SourceTree(
        root=str(root),
        modules=tuple(modules),
        tests=tuple(tests),
        runnable=tuple(sorted(set(runnable))),
        test_paths=tuple(test_dirs),
        notes=tuple(notes),
    )


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _walk_python(root: Path) -> list[Path]:
    """Every .py file under `root`, skipping ignored directories entirely."""
    found: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError:  # pragma: no cover - unreadable dir
            continue
        for entry in entries:
            if entry.is_symlink():
                continue  # a symlinked tree can loop, or leave the checkout
            if entry.is_dir():
                if entry.name in IGNORED_DIRS or entry.name.startswith("."):
                    continue
                stack.append(entry)
            elif entry.suffix == ".py":
                found.append(entry)
    return sorted(found)


def _package_roots(root: Path) -> tuple[Path, ...]:
    """Directories that dotted module names are relative to.

    `src/` layouts import as `pkg.core`, not `src.pkg.core`, so `src` has to be
    stripped from the name while staying in the path.
    """
    roots: list[Path] = []
    for name in ("src", "lib"):
        candidate = root / name
        if candidate.is_dir() and any(
            (child / "__init__.py").exists()
            for child in candidate.iterdir()
            if child.is_dir()
        ):
            roots.append(candidate)
    roots.append(root)
    return tuple(roots)


def _module_name(rel: Path, package_roots: tuple[Path, ...]) -> str:
    parts = list(rel.parts)
    for candidate in package_roots[:-1]:
        if parts and parts[0] == candidate.name:
            parts = parts[1:]
            break
    if parts and parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)
