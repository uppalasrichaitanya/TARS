"""What the project says about itself, read out of its manifest.

Two things make this more than a TOML load:

1. **`tomllib` is 3.11+.** This package supports 3.10, so there is a fallback
   subset parser for the flat `key = "value"` / `key = [list]` shapes a
   pyproject actually uses for metadata. It is not a TOML implementation and
   does not pretend to be — it raises rather than guess, and a degraded parse is
   recorded as a note so the caller can say so out loud.

2. **A broken manifest is data, not an exception.** Same rule the verification
   gate applies to `package.json`: a repo with an unparseable manifest still has
   modules and docstrings worth documenting, so parsing failure degrades to
   empty metadata plus a note.

Only metadata is read here. Nothing in this module imports or executes the
project being inspected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PYPROJECT = "pyproject.toml"
PACKAGE_JSON = "package.json"


@dataclass(frozen=True)
class Manifest:
    """Declared project metadata, normalised across manifest formats."""

    name: str = ""
    version: str = ""
    description: str = ""
    requires_python: str = ""
    dependencies: tuple[str, ...] = ()
    #: extra/feature name -> the requirements installing it pulls in.
    extras: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: A command a reader can type -> what it runs.
    scripts: dict[str, str] = field(default_factory=dict)
    #: True only when the manifest declares a backend that could build it.
    installable: bool = False
    #: Which file this came from, or "" when there was no manifest.
    source: str = ""
    notes: tuple[str, ...] = ()


def read_manifest(root: Path | str) -> Manifest:
    """Read `root`'s manifest. Never raises on malformed input."""
    root = Path(root)
    notes: list[str] = []

    text = _read(root / PYPROJECT)
    if text is not None:
        table, note = load_toml(text)
        if note:
            notes.append(f"{PYPROJECT}: {note}")
        if table is not None:
            return _from_pyproject(table, tuple(notes))

    text = _read(root / PACKAGE_JSON)
    if text is not None:
        try:
            loaded = json.loads(text)
        except ValueError as exc:
            notes.append(f"{PACKAGE_JSON}: not valid JSON ({exc})")
        else:
            if isinstance(loaded, dict):
                return _from_package_json(loaded, tuple(notes))
            notes.append(f"{PACKAGE_JSON}: expected a JSON object at the top level")

    return Manifest(notes=tuple(notes))


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


# ── pyproject ─────────────────────────────────────────────────────────────


def _from_pyproject(table: dict[str, Any], notes: tuple[str, ...]) -> Manifest:
    project = _sub(table, "project")
    poetry = _sub(_sub(table, "tool"), "poetry")
    # PEP 621 first; poetry's own table is the common non-standard case.
    source = project or poetry

    extras_raw = _sub(project, "optional-dependencies") or _sub(poetry, "extras")
    scripts_raw = _sub(project, "scripts") or _sub(poetry, "scripts")

    return Manifest(
        name=_string(source.get("name")),
        version=_string(source.get("version")),
        description=_string(source.get("description")),
        requires_python=_string(
            project.get("requires-python") or _sub(poetry, "dependencies").get("python")
        ),
        dependencies=_strings(project.get("dependencies")),
        extras={k: _strings(v) for k, v in extras_raw.items()},
        scripts={
            str(k): _string(v) for k, v in scripts_raw.items() if isinstance(k, str)
        },
        installable="build-system" in table,
        source=PYPROJECT,
        notes=notes,
    )


def _from_package_json(pkg: dict[str, Any], notes: tuple[str, ...]) -> Manifest:
    deps = _sub(pkg, "dependencies")
    dev = _sub(pkg, "devDependencies")
    # npm scripts are only runnable as `npm run <name>`, so record what a reader
    # would actually type rather than the bare script name.
    scripts = {
        f"npm run {k}": _string(v)
        for k, v in _sub(pkg, "scripts").items()
        if isinstance(k, str)
    }
    return Manifest(
        name=_string(pkg.get("name")),
        version=_string(pkg.get("version")),
        description=_string(pkg.get("description")),
        dependencies=tuple(
            f"{k}{v}" if str(v).startswith("^") else k for k, v in deps.items()
        ),
        extras={"dev": tuple(dev)} if dev else {},
        scripts=scripts,
        source=PACKAGE_JSON,
        notes=notes,
    )


def _sub(table: dict[str, Any], key: str) -> dict[str, Any]:
    value = table.get(key)
    return value if isinstance(value, dict) else {}


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(v.strip() for v in value if isinstance(v, str))
    return ()


# ── toml ──────────────────────────────────────────────────────────────────


def load_toml(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse `text` as TOML. Returns (table, note); table is None on failure.

    Uses `tomllib` where it exists and the subset parser where it doesn't. The
    note names which path was taken when it wasn't the real parser, so a caller
    can pass that uncertainty on instead of swallowing it.
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            return parse_toml_subset(text), (
                "parsed with the built-in subset parser (no tomllib before "
                "Python 3.11); tables and string/list values only"
            )
        except ValueError as exc:
            return None, f"could not be parsed ({exc})"

    try:
        return tomllib.loads(text), None
    except ValueError as exc:  # TOMLDecodeError subclasses ValueError
        return None, f"not valid TOML ({exc})"


def parse_toml_subset(text: str) -> dict[str, Any]:
    """Parse the slice of TOML that project metadata actually uses.

    Handles `[table]` and `[dotted.table]` headers, `key = "string"`,
    `key = ["a", "b"]` inline lists, multi-line lists, bare numbers and
    booleans, and comments. Anything else raises `ValueError`: for a docs tool,
    "I couldn't read this" is a usable answer, while a silently wrong parse
    produces a README full of confident nonsense.
    """
    root: dict[str, Any] = {}
    current = root
    pending_key: str | None = None
    pending_items: list[Any] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw).strip()
        if not line:
            continue

        if pending_key is not None:
            chunk, closed = _split_list_chunk(line)
            pending_items.extend(_parse_list_items(chunk, lineno))
            if closed:
                current[pending_key] = pending_items
                pending_key, pending_items = None, []
            continue

        if line.startswith("["):
            if not line.endswith("]"):
                raise ValueError(f"line {lineno}: unterminated table header {line!r}")
            header = line[1:-1].strip()
            if header.startswith("["):  # [[array.of.tables]]
                raise ValueError(f"line {lineno}: arrays of tables are not supported")
            if not header:
                raise ValueError(f"line {lineno}: empty table header")
            current = _descend(root, _split_key(header, lineno))
            continue

        if "=" not in line:
            raise ValueError(f"line {lineno}: not a key/value pair: {line!r}")
        key, _, value = line.partition("=")
        key, value = _unquote(key.strip()), value.strip()
        if not key:
            raise ValueError(f"line {lineno}: empty key")

        if value.startswith("[") and not value.endswith("]"):
            pending_key = key
            pending_items = list(_parse_list_items(value[1:], lineno))
            continue
        current[key] = _parse_value(value, lineno)

    if pending_key is not None:
        raise ValueError(f"unterminated list for key {pending_key!r}")
    return root


def _strip_comment(line: str) -> str:
    """Drop a trailing comment, respecting quotes."""
    out: list[str] = []
    quote: str | None = None
    for char in line:
        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#":
            break
        out.append(char)
    return "".join(out)


def _split_key(header: str, lineno: int) -> list[str]:
    parts = [_unquote(p.strip()) for p in header.split(".")]
    if not all(parts):
        raise ValueError(f"line {lineno}: malformed table header {header!r}")
    return parts


def _descend(root: dict[str, Any], parts: list[str]) -> dict[str, Any]:
    node = root
    for part in parts:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    return node


def _split_list_chunk(line: str) -> tuple[str, bool]:
    """Split a continuation line into (items, list-is-now-closed)."""
    if line.endswith("]"):
        return line[:-1], True
    return line, False


def _parse_list_items(chunk: str, lineno: int) -> list[Any]:
    return [_parse_value(part, lineno) for part in _split_items(chunk) if part]


def _split_items(chunk: str) -> list[str]:
    """Split on commas that aren't inside quotes."""
    items: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    for char in chunk:
        if quote:
            buf.append(char)
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            buf.append(char)
            continue
        if char == ",":
            items.append("".join(buf).strip())
            buf = []
            continue
        buf.append(char)
    items.append("".join(buf).strip())
    return [i for i in items if i]


def _parse_value(value: str, lineno: int) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        return _parse_list_items(value[1:-1], lineno)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    if value in ("true", "false"):
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    raise ValueError(f"line {lineno}: unsupported value {value!r}")


def _unquote(key: str) -> str:
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        return key[1:-1]
    return key
