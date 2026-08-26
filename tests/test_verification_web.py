"""Web verification is a lazy, optional check via the a11y tree."""

from __future__ import annotations

from tars.verification import verify_web


class FakeBrowser:
    def __init__(self, snapshot: str = "") -> None:
        self._snapshot = snapshot
        self.url: str | None = None
        self.steps: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def goto(self, url: str) -> None:
        self.url = url

    def snapshot(self) -> str:
        return self._snapshot

    def click(self, ref: int) -> None:
        self.steps.append({"action": "click", "ref": ref})

    def type(self, ref: int, text: str) -> None:
        self.steps.append({"action": "type", "ref": ref, "text": text})

    def check(self, ref: int) -> None:
        self.steps.append({"action": "check", "ref": ref})

    def uncheck(self, ref: int) -> None:
        self.steps.append({"action": "uncheck", "ref": ref})


class FakeFactory:
    def __init__(self, snapshot: str) -> None:
        self.snapshot = snapshot
        self.last: FakeBrowser | None = None

    def __call__(self):
        self.last = FakeBrowser(self.snapshot)
        return self.last


def test_verify_web_pass(tmp_path) -> None:
    factory = FakeFactory('[ref=1] button "Save"')
    report = verify_web(
        "http://example.com",
        None,
        [{"role": "button", "name": "Save"}],
        session_factory=factory,
        root=tmp_path,
    )
    assert report.verdict == "pass"
    assert report.results[0].status == "passed"


def test_verify_web_fail(tmp_path) -> None:
    factory = FakeFactory('[ref=1] button "Save"')
    report = verify_web(
        "http://example.com",
        None,
        [{"role": "button", "name": "Missing"}],
        session_factory=factory,
        root=tmp_path,
    )
    assert report.verdict == "fail"
    assert report.results[0].status == "failed"


def test_verify_web_with_steps(tmp_path) -> None:
    factory = FakeFactory('[ref=1] status "Complete"')
    report = verify_web(
        "http://example.com",
        [{"action": "click", "ref": 1}],
        [{"text": "Complete"}],
        session_factory=factory,
        root=tmp_path,
    )
    assert report.verdict == "pass"
    assert factory.last is not None
    assert factory.last.steps == [{"action": "click", "ref": 1}]


def test_verify_web_unavailable_when_playwright_missing(monkeypatch, tmp_path) -> None:
    class Boom:
        def __call__(self):
            raise RuntimeError(
                "Playwright is required for browser sessions; install tars[browser]"
            )

        def __enter__(self):  # pragma: no cover
            return self

        def __exit__(self, *_a: object) -> None:
            return None

    # verify_web catches the RuntimeError and maps to unavailable
    def raise_factory():
        raise RuntimeError(
            "Playwright is required for browser sessions; install tars[browser]"
        )

    # Use a factory that raises on construction via run_browser_task
    class RaiseFactory:
        def __call__(self):
            raise RuntimeError(
                "Playwright is required for browser sessions; install tars[browser]"
            )

    report = verify_web(
        "http://example.com",
        None,
        None,
        session_factory=RaiseFactory(),
        root=tmp_path,
    )
    assert report.verdict == "no-checks"
    assert report.results[0].status == "unavailable"
