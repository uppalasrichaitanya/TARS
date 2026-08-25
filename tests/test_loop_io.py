"""The REPL's output stays readable in terminals and captured logs."""

from __future__ import annotations

import io

from tars.loop._io import color_enabled, is_tty, paint


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_is_tty_reads_stream_capability() -> None:
    assert is_tty(_TTY())
    assert not is_tty(io.StringIO())


def test_is_tty_treats_stream_errors_as_not_a_terminal() -> None:
    class Broken(io.StringIO):
        def isatty(self) -> bool:
            raise ValueError("closed")

    assert not is_tty(Broken())


def test_color_enabled_requires_tty_and_respects_no_color(monkeypatch) -> None:
    stream = _TTY()
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert color_enabled(stream)
    monkeypatch.setenv("NO_COLOR", "1")
    assert not color_enabled(stream)
    assert not color_enabled(stream, no_color=True)


def test_paint_wraps_text_only_when_enabled() -> None:
    assert paint("PASS", "32", enabled=True) == "\033[32mPASS\033[0m"
    assert paint("PASS", "32", enabled=False) == "PASS"
