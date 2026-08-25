"""One-shot loop CLI commands use injected streams and dependencies."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from tars.loop.types import RunReport
from tars.router import Classification


def test_run_subcommand_prints_report_and_honours_yes(tmp_path: Path) -> None:
    from tars.loop.cli import main

    seen: list[bool] = []

    class FakeEngine:
        def run(self, spec, session, *, output):
            seen.append(spec.auto_confirm)
            return RunReport(task=spec.task, verification_passed=True)

    out, err = StringIO(), StringIO()
    code = main(
        ["run", "fix typo", "--yes", "--root", str(tmp_path)],
        stdout=out,
        stderr=err,
        engine=FakeEngine(),
    )
    assert code == 0
    assert seen == [True]
    assert "task: fix typo" in out.getvalue()
    assert not err.getvalue()


def test_route_subcommand_only_classifies(tmp_path: Path) -> None:
    from tars.loop.cli import main

    out = StringIO()
    code = main(
        ["route", "inspect it", "--root", str(tmp_path)],
        stdout=out,
        stderr=StringIO(),
        classify=lambda _task: Classification(0, "low", True, (), "small"),
    )
    assert code == 0
    assert "tier 0" in out.getvalue()


def test_cli_reports_unconfigured_provider_without_traceback(tmp_path: Path) -> None:
    from tars.loop.cli import EXIT_PROVIDER, main
    from tars.providers.errors import ProviderNotConfiguredError

    class BrokenEngine:
        def run(self, spec, session, *, output):
            raise ProviderNotConfiguredError("missing")

    out, err = StringIO(), StringIO()
    code = main(
        ["run", "fix it", "--root", str(tmp_path)],
        stdout=out,
        stderr=err,
        engine=BrokenEngine(),
    )
    assert code == EXIT_PROVIDER
    assert "no Anthropic credentials" in err.getvalue()
    assert "Traceback" not in err.getvalue()
