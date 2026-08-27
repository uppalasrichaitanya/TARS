"""Provider onboarding keeps credentials hidden, validated, and mode 600."""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path


def test_setup_writes_user_config_validates_and_never_prints_secret(
    tmp_path: Path, monkeypatch
) -> None:
    from tars.loop.setup import run_setup

    secret = "ar-secret with spaces"
    answers = iter(["1", "", "", "1"])
    seen: list[str] = []
    out = StringIO()
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    code = run_setup(
        input_fn=lambda _prompt: next(answers),
        secret_fn=lambda _prompt: secret,
        output=out,
        home=tmp_path,
        cwd=tmp_path / "project",
        validate=lambda model: seen.append(model),
    )

    target = tmp_path / ".config" / "tars" / "env"
    assert code == 0
    assert seen == ["agentrouter:claude-opus-5"]
    assert secret not in out.getvalue()
    assert "ANTHROPIC_AUTH_TOKEN" in target.read_text(encoding="utf-8")
    assert target.stat().st_mode & 0o777 == 0o600


def test_second_run_can_load_the_user_config(tmp_path: Path, monkeypatch) -> None:
    from tars.loop.setup import load_env, run_setup

    secret = "saved-secret"
    answers = iter(["2", "", "1"])
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert (
        run_setup(
            input_fn=lambda _prompt: next(answers),
            secret_fn=lambda _prompt: secret,
            output=StringIO(),
            home=tmp_path,
            cwd=tmp_path,
            validate=lambda _model: None,
        )
        == 0
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    load_env(home=tmp_path, cwd=tmp_path / "elsewhere")
    assert os.environ["OPENROUTER_API_KEY"] == secret


def test_setup_preserves_existing_values_in_target(tmp_path: Path, monkeypatch) -> None:
    from tars.loop.setup import run_setup

    target = tmp_path / ".config" / "tars" / "env"
    target.parent.mkdir(parents=True)
    target.write_text("export KEEP_ME=yes\n", encoding="utf-8")
    answers = iter(["4", "1"])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert (
        run_setup(
            input_fn=lambda _prompt: next(answers),
            secret_fn=lambda _prompt: "direct-secret",
            output=StringIO(),
            home=tmp_path,
            cwd=tmp_path,
            validate=lambda _model: None,
        )
        == 0
    )
    text = target.read_text(encoding="utf-8")
    assert "KEEP_ME=yes" in text
    assert "ANTHROPIC_API_KEY=direct-secret" in text


def test_validation_failure_writes_nothing_and_hides_secret(
    tmp_path: Path, monkeypatch
) -> None:
    from tars.loop.setup import run_setup
    from tars.providers.errors import ProviderAPIError

    secret = "do-not-leak"
    answers = iter(["3"])
    monkeypatch.delenv("TOKENROUTER_API_KEY", raising=False)
    out = StringIO()

    def fail(_model: str) -> None:
        raise ProviderAPIError("upstream rejected do-not-leak", provider="fake")

    assert (
        run_setup(
            input_fn=lambda _prompt: next(answers),
            secret_fn=lambda _prompt: secret,
            output=out,
            home=tmp_path,
            cwd=tmp_path,
            validate=fail,
        )
        == 3
    )
    assert secret not in out.getvalue()
    assert not (tmp_path / ".config" / "tars" / "env").exists()


def test_setup_parser_and_version_route_smokes() -> None:
    from tars.loop.cli import build_parser, main

    assert build_parser().parse_args(["setup"]).command == "setup"
    assert main(["--version"], stdout=StringIO(), stderr=StringIO()) == 0
    assert main(["route", "hello"], stdout=StringIO(), stderr=StringIO()) == 0


def test_setup_subcommand_dispatches_to_wizard() -> None:
    from tars.loop.cli import main

    out = StringIO()
    called: list[object] = []

    def setup_runner(*, output) -> int:
        called.append(output)
        output.write("wizard ran\n")
        return 0

    assert main(["setup"], stdout=out, setup_runner=setup_runner) == 0
    assert called == [out]
    assert out.getvalue() == "wizard ran\n"
