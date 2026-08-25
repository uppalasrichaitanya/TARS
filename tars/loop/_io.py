"""Small output helpers for the REPL.

The loop writes to terminals and captured streams alike. Keeping TTY and
``NO_COLOR`` handling here prevents ANSI control bytes from leaking into logs,
while leaving the rest of the REPL concerned only with its messages.
"""

from __future__ import annotations

import os
import re
from typing import TextIO

from ..providers.errors import ProviderError, ProviderNotConfiguredError


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


def provider_message(exc: ProviderError) -> str:
    if isinstance(exc, ProviderNotConfiguredError):
        return (
            "no Anthropic credentials - source ~/TARS/.env or set ANTHROPIC_AUTH_TOKEN"
        )
    text = " ".join(re.sub(r"<[^>]*>", " ", str(exc)).split())
    match = re.search(r"HTTP\s+\d{3}\s+from\s+provider", text, re.IGNORECASE)
    prefix = match.group(0) if match else "provider error"
    detail = text[match.end() :] if match else text
    return f"{prefix}{detail[:200]}"
