"""Shared CLI helpers: color and tty detection."""

from __future__ import annotations

import os
from typing import TextIO


def is_tty(stream: TextIO) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def use_color(stream: TextIO, no_color: bool = False) -> bool:
    if no_color or os.environ.get("NO_COLOR"):
        return False
    return is_tty(stream)
