"""What persists between commands.

A `Session` is the REPL's only mutable state, and it is deliberately small: the
repo it works on, the model in use, the transcript the loop grows, and the last
verdicts from CLASSIFY and VERIFY. That set is chosen by what the commands need
from each other — `/memory` reflects on the transcript `/run` produced, `/status`
reports the gate result `/verify` left behind — and nothing more, because state
the loop does not read is state that can go stale unnoticed.

The transcript is kept as `tars.providers.Message` objects rather than as text.
They are what the provider layer already speaks, and `tars.memory.parse_messages`
already reads that shape, so REFLECT needs no third format in between.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..providers import Message
from ..router import Classification

#: The flagship. The loop's router is what steps down to something cheaper, so
#: the default here is deliberately the capable model, not the frugal one.
FLAGSHIP = "claude-opus-5"


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


@dataclass
class Session:
    """One REPL session's state. Mutable by design; nothing else here is."""

    root: Path = field(default_factory=Path)
    model: str = field(default_factory=default_model)
    #: Skip the per-edit confirmation for every run in this session.
    auto_confirm: bool = False

    _messages: list[Message] = field(default_factory=list, repr=False)
    #: What CLASSIFY last decided, so /run can report what /route already showed.
    last_classification: Classification | None = None
    #: What VERIFY last concluded, as the gate's own verdict string.
    last_verdict: str | None = None

    @property
    def messages(self) -> tuple[Message, ...]:
        """The transcript so far. A tuple, so a caller cannot append behind us."""
        return tuple(self._messages)

    def record(self, *messages: Message) -> None:
        """Append turns to the transcript."""
        self._messages.extend(messages)

    def as_messages(self) -> list[Message]:
        """A fresh list for a provider call or for `memory.parse_messages`."""
        return list(self._messages)

    def clear(self) -> None:
        """Drop the transcript and the verdicts, keeping the configuration.

        `/clear` is for starting a new task in the same repo with the same model,
        so root, model and auto_confirm deliberately survive.
        """
        self._messages.clear()
        self.last_classification = None
        self.last_verdict = None

    def snapshot(self) -> dict[str, Any]:
        """What `/status` reports, as data rather than as printed text."""
        return {
            "root": str(self.root.resolve()),
            "model": self.model,
            "turns": len(self._messages),
            "auto_confirm": self.auto_confirm,
            "last_tier": (
                None if self.last_classification is None
                else self.last_classification.tier
            ),
            "last_verdict": self.last_verdict,
        }
