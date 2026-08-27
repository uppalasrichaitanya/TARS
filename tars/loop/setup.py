"""Interactive, deliberately boring credential setup for TARS.

Secrets live in user-owned files, not session logs or the repository.  This
module keeps prompting, validation, and persistence injectable so the wizard is
fully testable without a terminal or a network call.
"""

from __future__ import annotations

import getpass
import os
import re
import shlex
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TextIO

from .. import __version__, providers, router

BANNER = (
    "◐ TARS — a disciplined, multi-session coding agent loop\n"
    "This wizard configures a provider so TARS can plan and act safely.\n"
)
DEFAULT_AGENTROUTER_URL = "https://agentrouter.org"
DEFAULT_TIER0 = "openrouter:minimax/minimax-m3:free"


def _redact_values(text: str, values: Mapping[str, str]) -> str:
    """Remove credentials even when an upstream error unexpectedly echoes one."""
    for value in values.values():
        if value:
            text = text.replace(value, "***")
    return text


def parse_env_file(path: Path) -> dict[str, str]:
    """Read simple dotenv exports; malformed or commented lines are ignored."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
        if not match:
            continue
        value = match.group(2).strip()
        try:
            parsed = shlex.split(value, comments=False, posix=True)
        except ValueError:
            continue
        if len(parsed) != 1:
            continue
        value = parsed[0]
        values[match.group(1)] = value
    return values


def load_env(*, home: Path | None = None, cwd: Path | None = None) -> dict[str, str]:
    """Load persisted config into the process without overriding real env."""
    home = home or Path.home()
    cwd = cwd or Path.cwd()
    merged = parse_env_file(home / ".config" / "tars" / "env")
    merged.update(parse_env_file(cwd / ".env"))
    for key, value in merged.items():
        os.environ.setdefault(key, value)
    return merged


def _write_env(path: Path, values: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = parse_env_file(path)
    merged.update(values)
    content = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in merged.items())
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def run_setup(
    *,
    input_fn: Callable[[str], str] = input,
    secret_fn: Callable[[str], str] = getpass.getpass,
    output: TextIO,
    home: Path | None = None,
    cwd: Path | None = None,
    validate: Callable[[str], object] | None = None,
) -> int:
    """Run the provider picker and persist only the selected credentials."""
    home, cwd = home or Path.home(), cwd or Path.cwd()
    output.write(BANNER)
    output.write(
        "1) AgentRouter  2) OpenRouter  3) TokenRouter  4) Direct Anthropic  5) Skip\n"
    )
    _install_choice_completion(("1", "2", "3", "4", "5"))
    choice = input_fn("Choose provider [5]: ").strip()
    if choice == "5" or not choice:
        output.write("Setup skipped.\n")
        return 0
    values: dict[str, str] = {}
    if choice == "1":
        token = secret_fn("AgentRouter key (hidden): ")
        if token:
            values["ANTHROPIC_AUTH_TOKEN"] = token
        values["ANTHROPIC_BASE_URL"] = (
            input_fn(f"Base URL [{DEFAULT_AGENTROUTER_URL}]: ").strip()
            or DEFAULT_AGENTROUTER_URL
        )
        model = input_fn(f"Tier-0 model [{DEFAULT_TIER0}]: ").strip() or DEFAULT_TIER0
        values["TARS_TIER0_MODEL"] = model
        model_to_validate = "agentrouter:claude-opus-5"
    elif choice == "2":
        token = secret_fn("OpenRouter API key (hidden): ")
        if token:
            values["OPENROUTER_API_KEY"] = token
        model_to_validate = values.setdefault(
            "TARS_TIER0_MODEL",
            input_fn(f"Tier-0 model [{DEFAULT_TIER0}]: ").strip() or DEFAULT_TIER0,
        )
    elif choice == "3":
        token = secret_fn("TokenRouter API key (hidden): ")
        if token:
            values["TOKENROUTER_API_KEY"] = token
        model_to_validate = "tokenrouter:qwen/qwen3.8-max-free"
    elif choice == "4":
        token = secret_fn("Anthropic API key (hidden): ")
        if token:
            values["ANTHROPIC_API_KEY"] = token
        model_to_validate = "claude-opus-5"
    else:
        output.write("Unknown provider; setup skipped.\n")
        return 2
    old_values = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    providers.reset_providers()
    try:
        if validate:
            validate(model_to_validate)
        else:
            providers.generate(
                [providers.Message.system("ping")], [], model_to_validate
            )
    except providers.ProviderError as exc:
        from ._io import provider_message

        output.write(f"Validation failed: {type(exc).__name__}.\n")
        output.write(_redact_values(provider_message(exc), values) + "\n")
        if isinstance(exc, providers.ProviderNotInstalledError) and choice == "1":
            output.write("→ run 'pipx inject tars-agent anthropic' in this env\n")
            output.write('→ or, in a venv: pip install "tars-agent[anthropic]"\n')
        elif isinstance(exc, providers.ProviderAPIError) and choice == "2":
            hints = {
                401: "→ check your OpenRouter key at openrouter.ai/keys",
                404: "→ model not found, try openrouter:anthropic/claude-3.5-sonnet",
                429: "→ transient, retry in 30s",
                405: "→ transient, retry in 30s",
            }
            if exc.status_code in hints:
                output.write(hints[exc.status_code] + "\n")
        output.write("No files changed.\n")
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        providers.reset_providers()
        return 3
    _install_choice_completion(("1", "2", "3"))
    target = (
        input_fn("Write to (1) ~/.config/tars/env  (2) ./.env  (3) both [1]: ").strip()
        or "1"
    )
    paths = []
    if target in {"1", "3"}:
        paths.append(home / ".config" / "tars" / "env")
    if target in {"2", "3"}:
        paths.append(cwd / ".env")
    for path in paths:
        _write_env(path, values)
    load_env(home=home, cwd=cwd)
    output.write(f"verified: tars {__version__}\n")
    output.write(f"verified: tars route hello → {router.classify('hello').tier}\n")
    output.write("Setup complete. Run `tars` to begin.\n")
    return 0


def _install_choice_completion(choices: tuple[str, ...]) -> None:
    """Offer numeric choices when readline is available."""
    try:
        import readline
    except ImportError:  # pragma: no cover - platform dependent
        return

    def complete(text: str, state: int) -> str | None:
        matches = [choice for choice in choices if choice.startswith(text)]
        return matches[state] if state < len(matches) else None

    readline.set_completer(complete)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")
