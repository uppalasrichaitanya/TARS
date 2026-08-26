import json
from io import StringIO

import pytest

from tars.browser.cli import EXIT_MISMATCH, EXIT_OK, EXIT_USAGE, build_parser, main


@pytest.mark.parametrize(
    ("argv", "command"),
    [
        (["snapshot", "https://example.test"], "snapshot"),
        (["run", "https://example.test", "steps.json"], "run"),
        (
            ["check", "https://example.test", "steps.json", "expect.json"],
            "check",
        ),
    ],
)
def test_parser_accepts_each_subcommand(argv, command) -> None:
    assert build_parser().parse_args(argv).command == command


def test_bad_arguments_exit_two() -> None:
    out, err = StringIO(), StringIO()
    assert main(["run", "only-a-url"], stdout=out, stderr=err) == EXIT_USAGE
    assert "usage" in err.getvalue().lower()


def test_check_matches_a_faked_snapshot_without_opening_a_browser(tmp_path) -> None:
    steps = tmp_path / "steps.json"
    expect = tmp_path / "expect.json"
    steps.write_text("[]")
    expect.write_text(
        json.dumps([{"role": "button", "name": "Save"}, {"text": "Ready"}])
    )
    snapshot = '[ref=1] document "Settings"\n  [ref=2] button "Save"\n  [ref=3] status "Ready"\n'

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def goto(self, _url):
            return None

        def snapshot(self):
            return snapshot

    out, err = StringIO(), StringIO()
    code = main(
        ["check", "https://example.test", str(steps), str(expect)],
        stdout=out,
        stderr=err,
        session_factory=FakeSession,
    )
    assert code == EXIT_OK
    assert snapshot.strip() in out.getvalue()
    assert err.getvalue() == ""


def test_check_mismatch_exits_one(tmp_path) -> None:
    steps = tmp_path / "steps.json"
    expect = tmp_path / "expect.json"
    steps.write_text("[]")
    expect.write_text('[{"role": "alert", "name": "Failed"}]')

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def goto(self, _url):
            return None

        def snapshot(self):
            return '[ref=1] document "Settings"'

    code = main(
        ["check", "https://example.test", str(steps), str(expect)],
        stdout=StringIO(),
        stderr=StringIO(),
        session_factory=FakeSession,
    )
    assert code == EXIT_MISMATCH
