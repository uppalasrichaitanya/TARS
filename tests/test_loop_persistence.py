from __future__ import annotations

from pathlib import Path

import pytest

from tars.loop.session_store import SessionStore


def test_store_appends_versioned_events_and_loads_idempotently(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.append("s1", "session_started", {"root": str(tmp_path)})
    path = store.path("s1")
    event_id = store.append("s1", "config", {"model": "fake"})
    store.append_raw(path, {"v": 1, "id": event_id, "type": "config", "model": "fake"})
    events = store.load("s1")
    assert [event["type"] for event in events] == ["session_started", "config"]
    assert all(event["v"] == 1 for event in events)


def test_store_tolerates_truncated_final_line_but_rejects_middle(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path)
    store.append("s1", "session_started", {})
    path = store.path("s1")
    store.append("s1", "message", {"role": "user", "content": "hi"})
    path.write_text(path.read_text() + '{"v":1,"type":"message"', encoding="utf-8")
    assert len(store.load("s1")) == 2
    lines = path.read_text().splitlines()
    path.write_text(
        lines[0] + "\n{" + "bad" + "}\n" + lines[1] + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="line 2"):
        store.load("s1")


def test_unknown_future_events_are_skipped_with_one_warning(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.append("s1", "session_started", {"root": str(tmp_path)})
    path = store.path("s1")
    store.append_raw(path, {"v": 1, "id": "future-1", "type": "future_a"})
    store.append_raw(path, {"v": 1, "id": "future-2", "type": "future_b"})
    with pytest.warns(UserWarning, match="unknown session event") as caught:
        events = store.load("s1")
    assert len(caught) == 1
    assert [event["type"] for event in events] == ["session_started"]


def test_session_ids_sort_chronologically() -> None:
    from datetime import UTC, datetime

    from tars.loop.session import new_session_id

    first = new_session_id(datetime(2026, 8, 26, 10, 0, tzinfo=UTC), suffix="aaaa")
    second = new_session_id(datetime(2026, 8, 26, 10, 1, tzinfo=UTC), suffix="bbbb")
    assert first < second
