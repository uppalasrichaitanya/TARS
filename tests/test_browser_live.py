"""Guarded proof that an AX-tree ref can drive a real Chromium element."""

import os
import re

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("TARS_BROWSER_LIVE") != "1",
    reason="set TARS_BROWSER_LIVE=1 to run Playwright browser tests",
)


def test_accessibility_snapshot_click_round_trip() -> None:
    pytest.importorskip("playwright")
    from playwright.sync_api import Error as PlaywrightError

    from tars.browser import BrowserSession

    page = """
    <button aria-label="Run check" id="runbtn">Run</button>
    <p id="status" role="status">Ready</p>
    <script>
      document.getElementById('runbtn').addEventListener('click', function () {
        document.getElementById('status').textContent = 'Complete';
      });
    </script>
    """
    try:
        with BrowserSession() as browser:
            browser.goto("data:text/html," + page)
            before = browser.snapshot()
            assert 'button "Run check"' in before
            match = re.search(r'\[ref=(\d+)\] button "Run check"', before)
            assert match is not None
            ref = int(match.group(1))
            browser.click(ref)
            after = browser.snapshot()
            assert 'StaticText "Complete"' in after
            assert 'StaticText "Ready"' not in after
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc):
            pytest.skip("Playwright is installed but Chromium is not")
        raise
