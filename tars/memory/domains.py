"""Which partition does a fact belong to, and what said so.

The whole file is one rule from the spec, made mechanical: *tag cross-cutting
entries with every domain they touch — don't infer hidden connections between
domains; tag explicitly, or not at all.*

So a tag is only ever produced by a marker that literally appears in the text: a
path the domain owns, or one of its keywords, matched on word boundaries. There
is no similarity, no co-occurrence, and no relationship graph between domains.
Matching `frontend` never implies `backend`, however obviously the two talk to
each other in any real system — because a future ORIENT retrieves *by domain*,
and an entry that arrives in a partition it has nothing to do with is worse than
an entry that arrives nowhere.

Zero tags is therefore a normal outcome, not a bug, and the schema makes it a
safe one: an untagged candidate cannot auto-commit.

The default map is a starting point aimed at this project plus the four
partitions the spec names as examples. Any project can replace or extend it in
``.tars/memory/domains.json``, matching the convention the verification gate
already uses for ``.tars/verification.json``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Where a project states its own partitions, relative to the repo root.
DOMAINS_PATH = Path(".tars") / "memory" / "domains.json"

#: Domain names become filenames in the store, so the character set is fixed
#: rather than sanitised: quietly rewriting `../etc` into `etc` would hide a
#: mistake that is much better raised.
_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

_MARKER_KINDS = ("path", "keyword")


@dataclass(frozen=True)
class DomainMatch:
    """One domain, and the marker that put it there."""

    domain: str
    marker: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in _MARKER_KINDS:
            raise ValueError(f"marker kind {self.kind!r} is not one of {_MARKER_KINDS}")

    def to_dict(self) -> dict[str, Any]:
        return {"domain": self.domain, "marker": self.marker, "kind": self.kind}


@dataclass(frozen=True)
class Domain:
    """One memory partition, and the markers that name it.

    `paths` are matched as substrings, because a path fragment in prose ("the
    retry lives in tars/providers/base.py") is unambiguous. `keywords` are
    matched on word boundaries, because substring matching on words tags half a
    corpus off "test" appearing inside "protests".
    """

    name: str
    description: str = ""
    paths: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("paths", "keywords"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                object.__setattr__(self, field_name, tuple(value))
        name = self.name.strip().lower()
        if not _SAFE_NAME.match(name):
            raise ValueError(
                f"domain name {self.name!r} is not usable as a partition file name; "
                "expected lowercase letters, digits, dot, dash or underscore"
            )
        object.__setattr__(self, "name", name)

    def match(self, text: str) -> DomainMatch | None:
        """The first marker of this domain that appears in `text`, if any."""
        lowered = text.lower()
        for path in self.paths:
            if path.lower() in lowered:
                return DomainMatch(domain=self.name, marker=path, kind="path")
        for keyword in self.keywords:
            pattern = r"\b" + re.escape(keyword.lower()).replace(r"\ ", r"\s+") + r"\b"
            if re.search(pattern, lowered):
                return DomainMatch(domain=self.name, marker=keyword, kind="keyword")
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "paths": list(self.paths),
            "keywords": list(self.keywords),
        }


@dataclass(frozen=True)
class DomainMap:
    """The partitions of one project."""

    domains: tuple[Domain, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.domains, tuple):
            object.__setattr__(self, "domains", tuple(self.domains))
        seen: set[str] = set()
        for domain in self.domains:
            if domain.name in seen:
                raise ValueError(f"domain {domain.name!r} is declared twice")
            seen.add(domain.name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.domains)

    def match(self, text: str) -> tuple[DomainMatch, ...]:
        """Every domain named in `text`, with the marker that named it.

        Sorted by domain name so two runs over the same text agree.
        """
        matches = [m for m in (d.match(text) for d in self.domains) if m is not None]
        return tuple(sorted(matches, key=lambda m: m.domain))

    def domains_for(self, text: str) -> tuple[str, ...]:
        return tuple(m.domain for m in self.match(text))

    def get(self, name: str) -> Domain | None:
        for domain in self.domains:
            if domain.name == name.lower():
                return domain
        return None

    def extended_with(self, others: tuple[Domain, ...]) -> DomainMap:
        """This map plus `others`, with later definitions winning by name."""
        kept = [d for d in self.domains if d.name not in {o.name for o in others}]
        return DomainMap(domains=tuple(kept) + others)

    def to_dict(self) -> dict[str, Any]:
        return {"domains": {d.name: d.to_dict() for d in self.domains}}


#: The built-in map: the four partitions the spec names as examples, plus the
#: ones this project actually works in. Keywords are deliberately narrow —
#: "model" and "api" would tag everything, so neither is here. A project whose
#: vocabulary differs should say so in `.tars/memory/domains.json` rather than
#: hope these generalise.
DEFAULT_DOMAINS = DomainMap(
    domains=(
        Domain(
            name="backend",
            description="server-side request handling and business logic",
            paths=("src/api", "app/api", "server/"),
            keywords=(
                "backend",
                "endpoint",
                "handler",
                "route",
                "middleware",
                "fastapi",
                "django",
                "flask",
                "express",
            ),
        ),
        Domain(
            name="frontend",
            description="user interface and anything rendered",
            paths=("src/components", "src/pages", "web/"),
            keywords=(
                "frontend",
                "react",
                "vue",
                "svelte",
                "component",
                "css",
                "tailwind",
                "jsx",
                "tsx",
                "viewport",
            ),
        ),
        Domain(
            name="infra",
            description="deployment, containers, CI, and the machines",
            paths=(".github/workflows", "deploy/", "infra/", "terraform/"),
            keywords=(
                "infra",
                "docker",
                "dockerfile",
                "kubernetes",
                "k8s",
                "terraform",
                "nginx",
                "systemd",
                "cron",
                "ci pipeline",
                "github actions",
                "deployment",
            ),
        ),
        Domain(
            name="data-model",
            description="persisted shapes: schemas, migrations, stores",
            paths=("migrations/", "schema/"),
            keywords=(
                "data model",
                "schema",
                "migration",
                "database",
                "sqlite",
                "postgres",
                "mysql",
                "orm",
                "foreign key",
                "sql",
            ),
        ),
        Domain(
            name="providers",
            description="LLM provider adapters and the generate() interface",
            paths=("tars/providers",),
            keywords=(
                "provider",
                "providers",
                "anthropic",
                "openai",
                "openrouter",
                "claude",
                "gpt",
                "adapter",
                "auth token",
                "api key",
            ),
        ),
        Domain(
            name="verification",
            description="the check surface: tests, lint, types, build",
            paths=("tars/verification",),
            keywords=(
                "verification gate",
                "check surface",
                "ruff",
                "mypy",
                "linter",
                "type-checker",
                "typecheck",
                "tars-verify",
            ),
        ),
        Domain(
            name="testing",
            description="the test suite itself",
            paths=("tests/",),
            keywords=(
                "pytest",
                "test suite",
                "unit test",
                "integration test",
                "fixture",
                "coverage",
                "tests",
            ),
        ),
        Domain(
            name="memory",
            description="memory partitions, retrieval, and write-back",
            paths=("tars/memory",),
            keywords=(
                "memory entry",
                "memory partition",
                "memory writer",
                "write-back",
                "recall",
                "retrieval",
                "embedding",
                "vector db",
                "reflect phase",
            ),
        ),
        Domain(
            name="router",
            description="task classification and model routing",
            paths=("tars/router",),
            keywords=(
                "router",
                "routing",
                "classifier",
                "tier 0",
                "tier 1",
                "tier 2",
                "blast radius",
            ),
        ),
        Domain(
            name="browser",
            description="browser automation and accessibility-tree reads",
            paths=("tars/browser",),
            keywords=(
                "playwright",
                "selenium",
                "accessibility tree",
                "a11y",
                "selector",
                "dom",
                "screenshot",
            ),
        ),
        Domain(
            name="cli",
            description="the command surface a person types",
            paths=("cli.py",),
            keywords=(
                "cli",
                "argparse",
                "command line",
                "console script",
                "typer",
                "subcommand",
                "exit code",
            ),
        ),
        Domain(
            name="docs",
            description="prose meant for humans: README, docstrings, guides",
            paths=("docs/", "readme.md"),
            keywords=("readme", "docstring", "documentation", "changelog", "docs"),
        ),
        Domain(
            name="build",
            description="packaging, dependencies, and the build backend",
            paths=("pyproject.toml", "package.json", "requirements.txt"),
            keywords=(
                "dependency",
                "dependencies",
                "packaging",
                "wheel",
                "sdist",
                "lockfile",
                "setuptools",
                "virtualenv",
                "venv",
            ),
        ),
        Domain(
            name="security",
            description="secrets, credentials, and access",
            paths=(".env",),
            keywords=(
                "secret",
                "secrets",
                "credential",
                "credentials",
                "password",
                "permission",
                "sandbox",
                "leak",
            ),
        ),
    )
)


def _domains_from_dict(data: dict[str, Any], source: str) -> tuple[Domain, ...]:
    raw = data.get("domains", {})
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: 'domains' must be an object of name -> markers")
    domains: list[Domain] = []
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{source}: domain {name!r} must be an object")
        domains.append(
            Domain(
                name=str(name),
                description=str(spec.get("description", "")),
                paths=tuple(spec.get("paths", ())),
                keywords=tuple(spec.get("keywords", ())),
            )
        )
    return tuple(domains)


def load_domain_map(root: str | Path = ".") -> DomainMap:
    """The project's partitions, or the default map when it hasn't said.

    An override either replaces the defaults or, with ``"extends": "default"``,
    adds to them. A malformed override raises rather than silently falling back:
    a project that tried to declare its partitions and got them wrong should
    hear about it, not get somebody else's taxonomy.
    """
    path = Path(root) / DOMAINS_PATH
    if not path.is_file():
        return DEFAULT_DOMAINS
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object")
    declared = _domains_from_dict(data, str(path))
    if str(data.get("extends", "")).lower() == "default":
        return DEFAULT_DOMAINS.extended_with(declared)
    return DomainMap(domains=declared)
