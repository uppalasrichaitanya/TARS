"""Turning whatever the loop kept into turns the writer can quote.

The input shapes this accepts on purpose:

* provider-shaped message lists (``[{"role": ..., "content": ...}]``), with
  content as a string or as a list of blocks — that is what both SDK families
  hand back, and accepting it costs nothing and avoids importing either one;
* ``{"session": ..., "turns": [...]}`` when the caller has a label to give;
* ``.json`` / ``.jsonl`` files of the above;
* plain text — a captured phase log, split on the loop's own phase headings.

One rule drives the design: **a turn's role has to say whose voice it is, not
which envelope it arrived in.** Tool results are delivered inside `user`
messages by the Anthropic message format; the scorer trusts human voice more
than machine output, so a parser that preserves the envelope's label would
hand a grep result the authority of an instruction. Tool-only messages become
`tool` turns here, and that is the single most important thing this file does.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The loop's phases, in loop order. Used for phase-heading detection and for
#: reporting which phases a transcript actually contains.
PHASES: tuple[str, ...] = (
    "ORIENT",
    "CLASSIFY",
    "PLAN",
    "ACT",
    "VERIFY",
    "REFLECT",
    "REPORT",
)

_PHASE_INDEX = {name: i for i, name in enumerate(PHASES)}

#: Roles whose voice the scorer knows how to weigh. Anything else is kept
#: verbatim and simply earns no role bonus.
KNOWN_ROLES: frozenset[str] = frozenset({"user", "assistant", "tool", "system"})

#: Roles that are machine output rather than someone's statement.
MACHINE_ROLES: frozenset[str] = frozenset({"tool"})

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

#: A phase heading, once markdown decoration is stripped: `## PHASE 4 — ACT`,
#: `### VERIFY`, `**REFLECT**`. Anchored end-to-end so that a sentence merely
#: containing "act" or "report" is left alone.
_PHASE_HEADING = re.compile(
    r"^(?:phase\s*\d*\s*[-—:.]*\s*)?(" + "|".join(p.lower() for p in PHASES) + r")$",
    re.IGNORECASE,
)

_DECORATION = "#*_-─=•> \t"

_ROLE_PREFIX = re.compile(
    r"^(" + "|".join(sorted(KNOWN_ROLES)) + r")\s*:\s*(.*)$", re.IGNORECASE
)


@dataclass(frozen=True)
class Turn:
    """One turn of a session, as something the writer may quote."""

    index: int
    role: str
    text: str
    #: Loop phase this turn belongs to, when the transcript said so. `None` is
    #: common and fine: most transcripts are not phase-annotated.
    phase: str | None = None
    #: Names of tools called in this turn. Kept out of `text` so the sentence
    #: segmenter never sees JSON, but kept at all because "this turn ran the
    #: test suite" is a real signal about what the turn was doing.
    tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tools, tuple):
            object.__setattr__(self, "tools", tuple(self.tools))
        if self.phase is not None:
            object.__setattr__(self, "phase", self.phase.upper())

    @property
    def is_machine(self) -> bool:
        """True when this turn is output rather than someone's statement."""
        return self.role in MACHINE_ROLES

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "role": self.role,
            "text": self.text,
            "phase": self.phase,
            "tools": list(self.tools),
        }


@dataclass(frozen=True)
class Transcript:
    """A session, as turns."""

    session: str = ""
    turns: tuple[Turn, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.turns, tuple):
            object.__setattr__(self, "turns", tuple(self.turns))

    def __bool__(self) -> bool:
        return bool(self.turns)

    def __len__(self) -> int:
        return len(self.turns)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.turns)

    def phases_seen(self) -> tuple[str, ...]:
        """Phases present in this transcript, in loop order."""
        seen = {t.phase for t in self.turns if t.phase}
        return tuple(p for p in PHASES if p in seen)

    def by_phase(self, phase: str) -> tuple[Turn, ...]:
        wanted = phase.upper()
        return tuple(t for t in self.turns if t.phase == wanted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session,
            "turns": [t.to_dict() for t in self.turns],
            "phases_seen": list(self.phases_seen()),
        }


def _clean(text: str) -> str:
    """Strip terminal escapes and trailing whitespace.

    Captured logs (`claude --print > phase.log`) are full of ANSI. Left in, the
    escape bytes end up inside quoted evidence.
    """
    return _ANSI.sub("", text).replace("\r\n", "\n").replace("\r", "\n").strip()


def _blocks_to_text(blocks: Sequence[Any]) -> tuple[str, tuple[str, ...], bool, bool]:
    """Flatten content blocks.

    Returns the prose, the tool names called, and two flags: whether any
    tool-result block was seen, and whether any block was prose. The flags are
    what decide a message's real voice.
    """
    parts: list[str] = []
    tools: list[str] = []
    saw_result = False
    saw_prose = False
    for block in blocks:
        if isinstance(block, str):
            if block.strip():
                parts.append(block)
                saw_prose = True
            continue
        if not isinstance(block, Mapping):
            continue
        kind = str(block.get("type", "")).lower()
        if kind in {"tool_use", "tool_call", "function_call"}:
            name = block.get("name") or block.get("id") or "tool"
            tools.append(str(name))
            continue
        if kind in {"tool_result", "function_call_output"}:
            saw_result = True
            content = block.get("content", block.get("output", ""))
            if isinstance(content, str):
                if content.strip():
                    parts.append(content)
            elif isinstance(content, Sequence):
                nested, _, _, _ = _blocks_to_text(content)
                if nested:
                    parts.append(nested)
            continue
        text = block.get("text", block.get("content", ""))
        if isinstance(text, str) and text.strip():
            parts.append(text)
            saw_prose = True
    return "\n".join(p.strip() for p in parts), tuple(tools), saw_result, saw_prose


def _turn_from_message(message: Mapping[str, Any], index: int) -> Turn | None:
    role = str(message.get("role", "assistant")).lower()
    content = message.get("content", message.get("text", ""))
    tools: tuple[str, ...] = ()
    if isinstance(content, str):
        text = content
    elif isinstance(content, Sequence):
        text, tools, saw_result, saw_prose = _blocks_to_text(content)
        if saw_result and not saw_prose:
            # Machine output wearing a `user` envelope. Relabel it: the scorer
            # weighs human voice, and this is not one.
            role = "tool"
    else:
        text = ""
    text = _clean(text)
    if not text and not tools:
        return None
    phase = message.get("phase")
    return Turn(
        index=index,
        role=role,
        text=text,
        phase=str(phase) if phase else None,
        tools=tools,
    )


def parse_messages(messages: Iterable[Any], session: str = "") -> Transcript:
    """Turns from a provider-shaped message list."""
    turns: list[Turn] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError(
                f"transcript messages must be mappings with a role and content, "
                f"got {type(message).__name__}"
            )
        turn = _turn_from_message(message, len(turns) + 1)
        if turn is not None:
            turns.append(turn)
    return Transcript(session=session, turns=tuple(turns))


def _phase_of_heading(line: str) -> str | None:
    stripped = line.strip().strip(_DECORATION).strip()
    if not stripped or len(stripped) > 40:
        return None
    match = _PHASE_HEADING.match(stripped)
    return match.group(1).upper() if match else None


def parse_text(text: str, session: str = "") -> Transcript:
    """Turns from plain text — a captured log, or a pasted session.

    Two markers are honoured, both conservative: a line that is *only* a phase
    heading opens a phase, and a line beginning with a known role and a colon
    attributes what follows. Ordinary prose containing a colon or the word
    "act" is left alone, because a parser that guesses here produces phase tags
    and attributions that are worse than none.
    """
    turns: list[Turn] = []
    phase: str | None = None
    role = "assistant"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, role
        body = _clean("\n".join(buffer))
        buffer = []
        if body:
            turns.append(Turn(index=len(turns) + 1, role=role, text=body, phase=phase))
        role = "assistant"

    for raw in _clean(text).splitlines():
        heading = _phase_of_heading(raw)
        if heading:
            flush()
            phase = heading
            continue
        match = _ROLE_PREFIX.match(raw.strip())
        if match:
            flush()
            role = match.group(1).lower()
            buffer = [match.group(2)]
            flush()
            continue
        if not raw.strip():
            flush()
            continue
        buffer.append(raw)
    flush()
    return Transcript(session=session, turns=tuple(turns))


def parse_transcript(source: Any, session: str = "") -> Transcript:
    """Turns from a message list, or from a `{"session", "turns"}` object."""
    if isinstance(source, Transcript):
        return source
    if isinstance(source, str):
        return parse_text(source, session=session)
    if isinstance(source, Mapping):
        label = str(source.get("session", session) or session)
        messages = source.get("turns", source.get("messages", ()))
        if isinstance(messages, str):
            return parse_text(messages, session=label)
        return parse_messages(messages, session=label)
    if isinstance(source, Iterable):
        return parse_messages(source, session=session)
    raise TypeError(f"cannot read a transcript from {type(source).__name__}")


def _parse_jsonl(text: str, path: Path) -> list[Any]:
    messages: list[Any] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
    return messages


def load_transcript(path: str | Path, session: str | None = None) -> Transcript:
    """Read a transcript from disk.

    The file's stem is the default session label — an honest one, since it is
    the only name the file actually has. `.json` and `.jsonl` are parsed as
    data and a parse error is raised rather than swallowed: falling back to
    prose on a malformed `.json` would turn a typo into a transcript that reads
    fine and means nothing. Anything else is treated as text.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"no transcript at {file}")
    label = session if session is not None else file.stem
    raw = file.read_text(encoding="utf-8", errors="replace")
    suffix = file.suffix.lower()
    if suffix == ".jsonl":
        return parse_transcript(_parse_jsonl(raw, file), session=label)
    if suffix == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file} is not valid JSON: {exc}") from exc
        return parse_transcript(data, session=label)
    return parse_text(raw, session=label)


def phase_order(phase: str | None) -> int:
    """Position of a phase in the loop, or -1 for untagged."""
    if not phase:
        return -1
    return _PHASE_INDEX.get(phase.upper(), -1)
