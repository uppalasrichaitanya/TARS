"""Where kept entries live: domain-partitioned JSON, plus one review queue.

Plain files, one per partition, under `.tars/memory/`. No index, no database, no
embedding — retrieval in this design is *by domain*, which a directory listing
already does, and a memory a person can open in an editor and correct is worth
more here than a fast one.

Four properties are load-bearing, and each is a rule from the spec rather than a
storage preference:

**Pending is not memory.** `recall()` reads partitions only. Anything awaiting a
human sits in `pending.json` and is invisible to retrieval, because a future
session that can recall an unreviewed entry as fact has skipped the review step
the spec asks for.

**Superseding marks, never deletes.** The replaced entry keeps its row and gains
a pointer to what replaced it. "Contradicting entries aren't both live" is
satisfied by `live`, and "we used to think X" stays readable — which is what
makes a bad session's writes auditable after the fact.

**Ids are content-derived, so writing twice is safe.** An overnight run that
reflects on the same transcript twice must not double every fact. A second write
of an entry that already exists keeps the original, timestamp included, and says
so in the result.

**Time comes from the caller.** The store never asks the clock. Two runs over the
same content produce byte-identical files, which is the only way a store's diffs
stay reviewable.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import Candidate, MemoryEntry, ReflectionReport

#: Where entries live, relative to the repo root.
STORE_DIR = Path(".tars") / "memory"

#: The human review queue. One file rather than per-domain, because a reviewer
#: works through a queue, not through a taxonomy — and because an untagged
#: candidate has no partition to sit in.
PENDING_FILE = "pending.json"

#: Files under the store directory that are not partitions.
_RESERVED = frozenset({PENDING_FILE, "domains.json"})

#: The same rule :class:`tars.memory.domains.Domain` applies at construction,
#: repeated here because a domain can reach the store as a bare string from a CLI
#: flag or a hand-edited file, and this is the point where it becomes a path.
_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class WriteResult:
    """What one write actually did, in the store's own terms."""

    committed: tuple[MemoryEntry, ...] = ()
    queued: tuple[Candidate, ...] = ()
    #: Ids of entries a commit marked as replaced.
    superseded: tuple[str, ...] = ()
    #: Ids already present, left exactly as they were.
    unchanged: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary_line(self) -> str:
        parts = [f"{len(self.committed)} committed", f"{len(self.queued)} queued"]
        if self.superseded:
            parts.append(f"{len(self.superseded)} superseded")
        if self.unchanged:
            parts.append(f"{len(self.unchanged)} already known")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary_line(),
            "committed": [e.to_dict() for e in self.committed],
            "queued": [c.to_dict() for c in self.queued],
            "superseded": list(self.superseded),
            "unchanged": list(self.unchanged),
            "notes": list(self.notes),
        }


def _check_name(domain: str) -> str:
    name = domain.strip().lower()
    if not _SAFE_NAME.match(name):
        raise ValueError(
            f"domain {domain!r} is not usable as a partition file name; "
            "expected lowercase letters, digits, dot, dash or underscore"
        )
    return name


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain an object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write `data` to `path` without ever leaving a half-written file.

    An interrupted overnight run is the expected case, not a rare one, and a
    truncated partition would take the whole domain's memory with it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(body, encoding="utf-8")
    os.replace(temp, path)


@dataclass(frozen=True)
class MemoryStore:
    """The memory of one project, on disk."""

    root: Path = Path()

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    # ── locations ────────────────────────────────────────────────────────

    @property
    def directory(self) -> Path:
        return self.root / STORE_DIR

    def partition_path(self, domain: str) -> Path:
        return self.directory / f"{_check_name(domain)}.json"

    @property
    def pending_path(self) -> Path:
        return self.directory / PENDING_FILE

    def partitions(self) -> tuple[str, ...]:
        """Domain names this store has entries for, sorted."""
        if not self.directory.is_dir():
            return ()
        return tuple(
            sorted(
                p.stem for p in self.directory.glob("*.json") if p.name not in _RESERVED
            )
        )

    # ── reading ──────────────────────────────────────────────────────────

    def _partition_entries(self, domain: str) -> tuple[MemoryEntry, ...]:
        path = self.partition_path(domain)
        if not path.is_file():
            return ()
        data = _read_json(path)
        raw = data.get("entries", [])
        if not isinstance(raw, list):
            raise TypeError(f"{path}: 'entries' must be a list")
        return tuple(MemoryEntry.from_dict(e) for e in raw)

    def entries(self, domains: Sequence[str] = ()) -> tuple[MemoryEntry, ...]:
        """Every stored entry, superseded ones included, deduped by id.

        This is the audit view. Retrieval uses :meth:`recall`.
        """
        wanted = tuple(domains) if domains else self.partitions()
        seen: dict[str, MemoryEntry] = {}
        for domain in wanted:
            for entry in self._partition_entries(domain):
                seen.setdefault(entry.id, entry)
        return tuple(seen.values())

    def recall(
        self,
        domains: Sequence[str] = (),
        kinds: Sequence[str] = (),
        limit: int | None = None,
    ) -> tuple[MemoryEntry, ...]:
        """What a future ORIENT gets back: live entries only, strongest first.

        Pending candidates are deliberately unreachable from here. So are
        superseded entries — both are readable through other methods, and neither
        is something a session should act on without being told.
        """
        found = [e for e in self.entries(domains) if e.live]
        if kinds:
            wanted = {k.lower() for k in kinds}
            found = [e for e in found if e.kind in wanted]
        found.sort(key=lambda e: (-e.confidence, e.id))
        return tuple(found[:limit] if limit is not None else found)

    def pending(self) -> tuple[Candidate, ...]:
        """The review queue, in the order it was written."""
        if not self.pending_path.is_file():
            return ()
        data = _read_json(self.pending_path)
        raw = data.get("pending", [])
        if not isinstance(raw, list):
            raise TypeError(f"{self.pending_path}: 'pending' must be a list")
        return tuple(Candidate.from_dict(c) for c in raw)

    # ── writing ──────────────────────────────────────────────────────────

    def _save_partition(self, domain: str, entries: Iterable[MemoryEntry]) -> None:
        ordered = sorted(entries, key=lambda e: (-e.confidence, e.id))
        _write_json(
            self.partition_path(domain),
            {"domain": _check_name(domain), "entries": [e.to_dict() for e in ordered]},
        )

    def _save_pending(self, queue: Iterable[Candidate]) -> None:
        _write_json(self.pending_path, {"pending": [c.to_dict() for c in queue]})

    def _mark_superseded(self, entry_id: str, replaced_by: str) -> bool:
        """Point `entry_id` at its replacement in every partition holding it."""
        touched = False
        for domain in self.partitions():
            entries = self._partition_entries(domain)
            if not any(e.id == entry_id for e in entries):
                continue
            self._save_partition(
                domain,
                [
                    e.superseded_by_entry(replaced_by) if e.id == entry_id else e
                    for e in entries
                ],
            )
            touched = True
        return touched

    def _commit(
        self, candidate: Candidate, session: str, created_at: str | None
    ) -> tuple[MemoryEntry | None, tuple[str, ...], tuple[str, ...]]:
        """Store one candidate. Returns (entry, superseded ids, notes)."""
        entry = MemoryEntry.from_candidate(
            candidate, session=session, created_at=created_at
        )
        known = {e.id for e in self.entries()}
        if entry.id in known:
            return None, (), ()

        for domain in entry.domains:
            existing = list(self._partition_entries(domain))
            existing.append(entry)
            self._save_partition(domain, existing)

        superseded: list[str] = []
        notes: list[str] = []
        for old_id in entry.supersedes:
            if self._mark_superseded(old_id, entry.id):
                superseded.append(old_id)
            else:
                notes.append(
                    f"{entry.id} claims to supersede {old_id}, which is not in the "
                    "store; nothing was marked"
                )
        return entry, tuple(superseded), tuple(notes)

    def write(
        self,
        report: ReflectionReport,
        created_at: str | None = None,
    ) -> WriteResult:
        """Apply one reflection report: commit what may commit, queue the rest.

        The split is :attr:`Candidate.disposition`'s call, not this method's, so
        the rule cannot drift between the store and the CLI. Dropped candidates
        are neither stored nor queued — below the floor a candidate is not a
        proposal, and the report is where it stays readable.
        """
        committed: list[MemoryEntry] = []
        superseded: list[str] = []
        unchanged: list[str] = []
        notes: list[str] = []

        for candidate in report.to_commit:
            entry, marked, entry_notes = self._commit(
                candidate, report.session, created_at
            )
            if entry is None:
                unchanged.append(MemoryEntry.from_candidate(candidate).id)
                continue
            committed.append(entry)
            superseded.extend(marked)
            notes.extend(entry_notes)

        queued: list[Candidate] = []
        if report.to_review:
            queue = list(self.pending())
            known = {MemoryEntry.from_candidate(c).id for c in queue}
            known |= {e.id for e in self.entries()}
            for candidate in report.to_review:
                entry_id = MemoryEntry.from_candidate(candidate).id
                if entry_id in known:
                    unchanged.append(entry_id)
                    continue
                queue.append(candidate)
                known.add(entry_id)
                queued.append(candidate)
            if queued:
                self._save_pending(queue)

        return WriteResult(
            committed=tuple(committed),
            queued=tuple(queued),
            superseded=tuple(superseded),
            unchanged=tuple(unchanged),
            notes=tuple(notes),
        )

    # ── draining the queue ───────────────────────────────────────────────

    def approve(
        self,
        entry_id: str,
        session: str = "",
        created_at: str | None = None,
    ) -> MemoryEntry | None:
        """Move a queued candidate into its partitions.

        Returns None when nothing in the queue has that id — approving something
        that isn't there is a mistake worth reporting, not an exception.
        """
        queue = list(self.pending())
        for index, candidate in enumerate(queue):
            if MemoryEntry.from_candidate(candidate).id != entry_id:
                continue
            if not candidate.domains:
                raise ValueError(
                    f"{entry_id} has no domain, so there is no partition to "
                    "approve it into; give it one first"
                )
            entry, _, _ = self._commit(candidate, session, created_at)
            del queue[index]
            self._save_pending(queue)
            return entry or MemoryEntry.from_candidate(
                candidate, session=session, created_at=created_at
            )
        return None

    def reject(self, entry_id: str) -> bool:
        """Drop a queued candidate. True when there was one to drop."""
        queue = list(self.pending())
        kept = [c for c in queue if MemoryEntry.from_candidate(c).id != entry_id]
        if len(kept) == len(queue):
            return False
        self._save_pending(kept)
        return True
