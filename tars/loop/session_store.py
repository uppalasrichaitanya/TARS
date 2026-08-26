"""Durable event log for REPL sessions.

The log is append-only so a crash cannot rewrite earlier conversation, while
the loader remains strict enough that corruption is never mistaken for history.
"""

from __future__ import annotations

import json
import os
import uuid
import warnings
from pathlib import Path
from typing import Any

KNOWN_TYPES = frozenset(
    {
        "session_started",
        "message",
        "classification",
        "verification",
        "config",
        "run_started",
        "run_finished",
        "run_interrupted",
    }
)


class SessionStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.jsonl"

    def append(self, session_id: str, event_type: str, data: dict[str, Any]) -> str:
        if event_type not in KNOWN_TYPES:
            raise ValueError(f"unknown session event type: {event_type}")
        event_id = uuid.uuid4().hex
        event = {"v": 1, "id": event_id, "type": event_type, **data}
        file = self.path(session_id)
        with file.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        return event_id

    def append_raw(self, path: Path, event: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def load(self, session_id: str) -> list[dict[str, Any]]:
        file = self.path(session_id)
        if not file.is_file():
            raise FileNotFoundError(file)
        lines = file.read_text(encoding="utf-8").splitlines()
        events: list[dict[str, Any]] = []
        seen: set[str] = set()
        unknown_warned = False
        for index, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                if index == len(lines):
                    warnings.warn(
                        f"ignored truncated final event in {file}", UserWarning
                    )
                    continue
                raise ValueError(f"{file} line {index} is not valid JSON") from exc
            if not isinstance(event, dict) or event.get("v") != 1:
                raise ValueError(f"{file} line {index} has unsupported event schema")
            event_id = str(event.get("id", ""))
            if event_id in seen:
                continue
            seen.add(event_id)
            if event.get("type") not in KNOWN_TYPES:
                if not unknown_warned:
                    warnings.warn(
                        f"skipped unknown session event type: {event.get('type')}",
                        UserWarning,
                    )
                    unknown_warned = True
                continue
            events.append(event)
        return events

    def list_sessions(self, *, root: Path) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for file in self.directory.glob("*.jsonl"):
            try:
                events = self.load(file.stem)
            except (OSError, ValueError):
                continue
            started = next((e for e in events if e["type"] == "session_started"), None)
            if (
                started
                and Path(str(started.get("root", ""))).resolve() == root.resolve()
            ):
                configs = [e for e in events if e["type"] == "config"]
                verdicts = [e for e in events if e["type"] == "verification"]
                runs = [e for e in events if e["type"] == "run_started"]
                found.append(
                    {
                        "id": file.stem,
                        "events": events,
                        "updated": file.stat().st_mtime,
                        "label": next(
                            (
                                str(e.get("label", ""))
                                for e in configs
                                if e.get("label")
                            ),
                            str(runs[0].get("task", ""))[:80] if runs else "",
                        ),
                        "verdict": str(verdicts[-1].get("verdict", ""))
                        if verdicts
                        else "",
                    }
                )
        return sorted(found, key=lambda item: item["updated"], reverse=True)
