"""What persists between commands.

A `Session` is the REPL's only mutable state, and it is deliberately small: the
repo it works on, the model in use, the transcript the loop grows, and the last
verdicts from CLASSIFY and VERIFY. That set is chosen by what the commands need
from each other — `/memory` reflects on the transcript `/run` produced, `/status`
reports the gate result `/verify` left behind — and nothing more, because state
the loop does not read is state that can go stale unnoticed.

The transcript is kept as `tars.providers.Message` objects rather than as text.
Provider calls use those objects directly; REFLECT gets an explicit mapping
conversion at this loop boundary because `tars.memory.parse_messages` accepts
mappings, not provider types.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..providers import Message
from ..router import Classification
from .session_store import SessionStore

#: The flagship. The loop's router is what steps down to something cheaper, so
#: the default here is deliberately the capable model, not the frugal one.
FLAGSHIP = "claude-opus-5"


def new_session_id(now: datetime | None = None, *, suffix: str | None = None) -> str:
    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{suffix or uuid.uuid4().hex[:4]}"


def default_model(env: Mapping[str, str] | None = None) -> str:
    """Which model string to use when the caller names none.

    Machine-dependent on purpose. A bare ``claude-opus-5`` routes to
    api.anthropic.com, so on a host whose only credential is an AgentRouter
    bearer token that default would fail at the first call with a credentials
    error that says nothing about the actual cause. When the environment looks
    like AgentRouter — an auth token *and* a base URL — the default routes there
    instead.

    An auth token on its own is not taken as evidence: it may be a direct
    Anthropic bearer token, and guessing wrong would silently send a real
    Anthropic credential to a third-party host.
    """
    env = os.environ if env is None else env

    explicit = env.get("TARS_MODEL")
    if explicit:
        return explicit

    base_url = env.get("ANTHROPIC_BASE_URL", "")
    has_token = bool(env.get("ANTHROPIC_AUTH_TOKEN"))
    if has_token and base_url and "agentrouter" in base_url.lower():
        return "agentrouter:" + FLAGSHIP
    return FLAGSHIP


def default_tier0_model(env: Mapping[str, str] | None = None) -> str | None:
    """The cheap route is explicit because free-model ids are not stable facts."""
    env = os.environ if env is None else env
    return env.get("TARS_TIER0_MODEL") or None


@dataclass
class Session:
    """One REPL session's state. Mutable by design; nothing else here is."""

    root: Path = field(default_factory=Path)
    model: str = field(default_factory=default_model)
    #: Skip the per-edit confirmation for every run in this session.
    auto_confirm: bool = False
    tier_models: dict[str, str | None] = field(default_factory=dict)
    session_id: str = field(default_factory=new_session_id)
    store: SessionStore | None = field(default=None, repr=False, compare=False)

    _messages: list[Message] = field(default_factory=list, repr=False)
    #: What CLASSIFY last decided, so /run can report what /route already showed.
    last_classification: Classification | None = None
    #: What VERIFY last concluded, as the gate's own verdict string.
    last_verdict: str | None = None
    interrupted: bool = False
    checkpoint_sha: str | None = None
    checkpoint_paths: set[str] = field(default_factory=set, repr=False)
    tier0_source: str = field(default="unset", init=False)

    def __post_init__(self) -> None:
        explicit_tier0 = "0" in self.tier_models
        tier0 = self.tier_models.get("0") if explicit_tier0 else default_tier0_model()
        self.tier0_source = (
            "config" if explicit_tier0 else ("env" if tier0 else "unset")
        )
        self.tier_models = {
            "0": tier0,
            "1": self.model,
            "2": self.model,
        }
        if self.store is not None and not self.store.path(self.session_id).exists():
            self.store.append(
                self.session_id,
                "session_started",
                {
                    "root": str(self.root.resolve()),
                    "model": self.model,
                    "auto_confirm": self.auto_confirm,
                },
            )
            self.record_event("config", {"tier_models": self.tier_models})

    @property
    def messages(self) -> tuple[Message, ...]:
        """The transcript so far. A tuple, so a caller cannot append behind us."""
        return tuple(self._messages)

    def record(self, *messages: Message) -> None:
        """Append turns to the transcript."""
        self._messages.extend(messages)
        if self.store is not None:
            for message in messages:
                self.store.append(self.session_id, "message", _message_data(message))

    def record_event(self, event_type: str, data: dict[str, Any]) -> None:
        if self.store is not None:
            self.store.append(self.session_id, event_type, data)

    @classmethod
    def load(cls, store: SessionStore, session_id: str) -> Session:
        events = store.load(session_id)
        started = next(
            (event for event in events if event["type"] == "session_started"), {}
        )
        session = cls(
            root=Path(str(started.get("root", "."))),
            model=str(started.get("model", default_model())),
            auto_confirm=bool(started.get("auto_confirm", False)),
            tier_models={"0": None},
            session_id=session_id,
            store=store,
        )
        for event in events:
            if event["type"] == "message":
                session._messages.append(_message_from_data(event))
            elif event["type"] == "classification":
                session.last_classification = Classification(
                    int(event["tier"]),
                    str(event["blast"]),
                    bool(event["reversible"]),
                    tuple(event.get("signals", ())),
                    str(event.get("why", "")),
                )
            elif event["type"] == "verification":
                session.last_verdict = str(event.get("verdict", ""))
            elif event["type"] == "config":
                if event.get("clear_transcript"):
                    session._messages.clear()
                    session.last_classification = None
                    session.last_verdict = None
                    session.interrupted = False
                if "model" in event:
                    session.model = str(event["model"])
                if "auto_confirm" in event:
                    session.auto_confirm = bool(event["auto_confirm"])
                if "tier_models" in event and isinstance(event["tier_models"], dict):
                    saved = event["tier_models"]
                    session.tier_models = {
                        "0": saved.get("0"),
                        "1": session.model,
                        "2": session.model,
                    }
                    session.tier0_source = "config"
            elif event["type"] == "run_interrupted":
                session.interrupted = True
            elif event["type"] == "run_finished":
                session.interrupted = False
        return session

    def as_messages(self) -> list[dict[str, str | None]]:
        """Return the transcript in the mapping shape expected by memory."""
        return [
            {"role": message.role, "content": message.content}
            for message in self._messages
        ]

    def ensure_checkpoint(self) -> str | None:
        """Snapshot the pre-ACT worktree without changing its index or files."""
        if self.checkpoint_sha is not None:
            return self.checkpoint_sha
        if not (self.root / ".git").exists():
            return None
        with tempfile.NamedTemporaryFile() as index:
            env = {**os.environ, "GIT_INDEX_FILE": index.name}
            subprocess.run(
                ["git", "read-tree", "HEAD"],
                cwd=self.root,
                env=env,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.root,
                env=env,
                check=True,
                capture_output=True,
            )
            tree = subprocess.run(
                ["git", "write-tree"],
                cwd=self.root,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        parent = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "commit-tree", tree, "-p", parent, "-m", "TARS pre-ACT checkpoint"],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "TARS",
                "GIT_AUTHOR_EMAIL": "tars@localhost",
                "GIT_COMMITTER_NAME": "TARS",
                "GIT_COMMITTER_EMAIL": "tars@localhost",
            },
        ).stdout.strip()
        ref = f"refs/tars/checkpoints/{self.session_id}"
        subprocess.run(["git", "update-ref", ref, commit], cwd=self.root, check=True)
        self.checkpoint_sha = commit
        return commit

    def note_mutation(self, paths: tuple[str, ...]) -> None:
        self.checkpoint_paths.update(paths)

    def undo(self) -> tuple[str, ...]:
        """Restore only paths TARS mutated, preserving unrelated user work."""
        if self.checkpoint_sha is None:
            return ()
        restored: list[str] = []
        for relative in sorted(self.checkpoint_paths):
            target = self.root / relative
            blob = subprocess.run(
                ["git", "show", f"{self.checkpoint_sha}:{relative}"],
                cwd=self.root,
                capture_output=True,
            )
            if blob.returncode == 0:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(blob.stdout)
            elif target.exists():
                target.unlink()
            restored.append(relative)
        self.checkpoint_paths.clear()
        return tuple(restored)

    def clear(self) -> None:
        """Drop the transcript and the verdicts, keeping the configuration.

        `/clear` is for starting a new task in the same repo with the same model,
        so root, model and auto_confirm deliberately survive.
        """
        self._messages.clear()
        self.last_classification = None
        self.last_verdict = None
        self.interrupted = False
        self.record_event("config", {"clear_transcript": True})

    def snapshot(self) -> dict[str, Any]:
        """What `/status` reports, as data rather than as printed text."""
        return {
            "id": self.session_id,
            "root": str(self.root.resolve()),
            "model": self.model,
            "turns": len(self._messages),
            "auto_confirm": self.auto_confirm,
            "tier_0_model": self.tier_models["0"],
            "last_tier": (
                None
                if self.last_classification is None
                else self.last_classification.tier
            ),
            "last_verdict": self.last_verdict,
            "interrupted": self.interrupted,
        }


def _message_data(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_call_id": message.tool_call_id,
        "is_error": message.is_error,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ],
    }


def _message_from_data(data: dict[str, Any]) -> Message:
    from ..providers import ToolCall

    return Message(
        role=data["role"],
        content=data.get("content"),
        tool_call_id=data.get("tool_call_id"),
        is_error=bool(data.get("is_error", False)),
        tool_calls=tuple(ToolCall(**call) for call in data.get("tool_calls", ())),
    )
