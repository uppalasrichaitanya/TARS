"""Small output helpers for the REPL.

The loop writes to terminals and captured streams alike. Keeping TTY and
``NO_COLOR`` handling here prevents ANSI control bytes from leaking into logs,
while leaving the rest of the REPL concerned only with its messages.
"""

from __future__ import annotations

import os
from typing import TextIO


def is_tty(stream: TextIO) -> bool:
    """Return whether *stream* is an interactive terminal."""
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def color_enabled(stream: TextIO, *, no_color: bool = False) -> bool:
    """Return whether ANSI color should be emitted to *stream*."""
    if no_color or os.environ.get("NO_COLOR"):
        return False
    return is_tty(stream)


def paint(text: str, code: str, *, enabled: bool) -> str:
    """Wrap *text* in an ANSI color sequence when enabled."""
    return f"\033[{code}m{text}\033[0m" if enabled else text
