"""Toolbox browser tools are lazy, optional, and ref-addressed."""

from __future__ import annotations

from pathlib import Path

from tars.loop.tools import Toolbox
from tars.loop.types import ToolCall


class FakeBrowser:
    def __init__(self) -> None:
        self.goto_url: str | None = None
        self.snapshot_text = '[ref=1] button "Save"'
        self.clicked: list[int] = []
        self.typed: list[tuple[int, str]] = []

    def goto(self, url: str) -> None:
        self.goto_url = url

    def snapshot(self) -> str:
        return self.snapshot_text

    def click(self, ref: int) -> None:
        self.clicked.append(ref)

    def type(self, ref: int, text: str) -> None:
        self.typed.append((ref, text))

    def check(self, ref: int) -> None:
        self.clicked.append(ref)

    def uncheck(self, ref: int) -> None:
        self.clicked.append(-ref)

    def back(self) -> None:
        self.goto_url = "back"

    def forward(self) -> None:
        self.goto_url = "forward"

    def reload(self) -> None:
        self.goto_url = "reload"

    def close(self) -> None:
        pass


def test_browser_tools_drive_fake_session(tmp_path: Path) -> None:
    browser = FakeBrowser()
    toolbox = Toolbox(root=tmp_path, browser_factory=lambda: browser)

    result = toolbox.dispatch(
        ToolCall("1", "browser_goto", {"url": "http://example.com"})
    )
    assert result.ok
    assert browser.goto_url == "http://example.com"

    result = toolbox.dispatch(ToolCall("2", "browser_snapshot", {}))
    assert result.ok
    assert "button" in result.output

    result = toolbox.dispatch(ToolCall("3", "browser_click", {"ref": 1}))
    assert result.ok
    assert browser.clicked == [1]

    result = toolbox.dispatch(
        ToolCall("4", "browser_type", {"ref": 1, "text": "hello"})
    )
    assert result.ok
    assert browser.typed == [(1, "hello")]

    assert toolbox.dispatch(ToolCall("5", "browser_check", {"ref": 2})).ok
    assert toolbox.dispatch(ToolCall("6", "browser_uncheck", {"ref": 3})).ok
    assert toolbox.dispatch(ToolCall("7", "browser_back", {})).ok
    assert toolbox.dispatch(ToolCall("8", "browser_forward", {})).ok
    assert toolbox.dispatch(ToolCall("9", "browser_reload", {})).ok
    assert browser.clicked[-2:] == [2, -3]
    assert browser.goto_url == "reload"


def test_browser_snapshot_can_return_only_changes(tmp_path: Path) -> None:
    browser = FakeBrowser()
    toolbox = Toolbox(root=tmp_path, browser_factory=lambda: browser)
    assert toolbox.dispatch(ToolCall("1", "browser_snapshot", {})).output
    assert (
        toolbox.dispatch(
            ToolCall("2", "browser_snapshot", {"incremental": True})
        ).output
        == "no accessibility changes"
    )
    browser.snapshot_text += '\n[ref=2] status "Saved"'
    changed = toolbox.dispatch(ToolCall("3", "browser_snapshot", {"incremental": True}))
    assert 'status "Saved"' in changed.output


def test_browser_tools_report_missing_extra(tmp_path: Path) -> None:
    toolbox = Toolbox(
        root=tmp_path,
        browser_factory=lambda: (_ for _ in ()).throw(
            RuntimeError("Playwright is required")
        ),
    )

    result = toolbox.dispatch(ToolCall("1", "browser_snapshot", {}))
    assert not result.ok
    assert "Playwright" in (result.error or "")


def test_browser_tools_reuse_session(tmp_path: Path) -> None:
    created: list[FakeBrowser] = []

    def factory():
        b = FakeBrowser()
        created.append(b)
        return b

    toolbox = Toolbox(root=tmp_path, browser_factory=factory)
    toolbox.dispatch(ToolCall("1", "browser_goto", {"url": "http://a.com"}))
    toolbox.dispatch(ToolCall("2", "browser_snapshot", {}))
    assert len(created) == 1


def test_browser_snapshot_without_playwright_is_error(tmp_path: Path) -> None:
    def bad_factory():
        raise ImportError("No module named 'playwright'")

    toolbox2 = Toolbox(root=tmp_path, browser_factory=bad_factory)
    result = toolbox2.dispatch(ToolCall("1", "browser_snapshot", {}))
    assert not result.ok
    assert "browser extra not installed" in (result.error or "")
