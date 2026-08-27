# tars-agent

TARS — a disciplined, multi-session coding agent loop

## Why it exists

Named for TARS - the robot from Interstellar - whose worth lies not in cleverness but in discipline: it does what it said, reports honestly what it did, and asks before it touches anything. This tool tries to earn that name.

## Install

```console
pip install -e ".[dev]"
```

Or install the published package:

```console
pipx install tars-agent
pip install tars-agent
```

Requires Python `>=3.10`.

This project has no runtime dependencies.

## Quick Start

Get TARS running in 60 seconds:

```console
pipx install tars-agent
# or: pip install tars-agent
export ANTHROPIC_AUTH_TOKEN=sk-...      # or OPENROUTER_API_KEY / TOKENROUTER_API_KEY
export ANTHROPIC_BASE_URL=https://agentrouter.org   # if using AgentRouter
tars
```

Inside the REPL, just type what you want — bare prompts run the full
`ORIENT → CLASSIFY → PLAN → ACT → VERIFY → REFLECT → REPORT` loop:

```console
tars> Fix the failing test in tests/test_login.py
tars> Add dark mode to src/App.tsx and verify with tests
tars> Explain how tars/memory scores candidates
```

Slash commands force a single phase:

| Command | What it does |
|---|---|
| `/run <task>` | Full loop for one task |
| `/route <task>` | Show tier/blast/reversibility without acting |
| `/verify` | Run the repo's gate locally (`tars-verify`) |
| `/memory [approve <id> \| correct <id> "reason"]` | Review or correct remembered decisions |
| `/status` | Model, tier, session, last verdict |
| `/undo` | Restore the pre-ACT Git checkpoint |
| `/resume [ID]` | Resume a prior session (survives Ctrl-C) |
| `/pillars` | Show the seven-phase loop |

One-shot (no REPL):

```console
tars run "Fix bug in app.py --yes"
tars route "Add a new CLI flag"
```

**Configure the cheapest model for simple tasks:**

```console
export TARS_TIER0_MODEL=openrouter:minimax/minimax-m3:free
# tier 0 (scaffold/create-artifact) uses this; tiers 1-2 stay on the flagship
```

**Browser testing (optional):**

```console
pip install "tars-agent[browser]"
tars run "Check https://example.com loads and has a login button" --web-url https://example.com --web-expect "login"
```

## How to run it

### `tars`

Run TARS's disciplined agent loop.

```console
tars
python -m tars.loop
```

Flags: `--version`, `--yes`, `--root`, `--model`, `--web-url`, `--web-steps`, `--web-expect`

<sub>declared in `pyproject.toml`</sub>

### `tars-browser`

Efficient accessibility-tree-first web testing.

```console
tars-browser
python -m tars.browser
```

<sub>declared in `pyproject.toml`</sub>

### `tars-memory`

Read a session transcript and propose the memory entries it earned: domain-tagged, scored, and quoted from the turn they came from. Also reads back what earlier sessions kept.

```console
tars-memory
python -m tars.memory
```

Flags: `--write`, `--session`, `--max`, `--now`, `--domain`, `--kind`, `--limit`, `--root`, `--json`, `--no-color`

<sub>declared in `pyproject.toml`</sub>

### `tars-readme`

Write a README from what a repo actually contains: usage examples lifted from its own docstrings and tests, the commands it declares, and the checks it can run.

```console
tars-readme
python -m tars.readme
```

Flags: `-o/--output`, `--force`, `--check`, `--json`, `--include-unverified`, `--notes`, `--no-tests`, `--max-examples`, `--no-color`

<sub>declared in `pyproject.toml`</sub>

### `tars-verify`

Detect a repo's test runner, linter, type-checker and formatter, then run them uniformly.

```console
tars-verify
python -m tars.verification
```

Flags: `--only`, `--include`, `--list`, `--json`, `--fail-fast`, `--timeout`, `--output-lines`, `--max-output-chars`, `--allow-no-checks`, `--quiet/-q`, `--no-color`

<sub>declared in `pyproject.toml`</sub>

### `python -m tars`

Efficient accessibility-tree-first web testing.

```console
python -m tars
```

<sub>declared in `tars/__main__.py`</sub>

## Examples

Every snippet below was taken from the file named under it, and checked against the current source.

```python
from tars.act import Edit, Hunk, LocalExecutor

executor = LocalExecutor(".")
result = executor.apply(Edit.replace("a.py", Hunk("x = 1", "x = 2")))
result.ok, result.diff, result.touched
```

<sub>`tars/act/__init__.py:5`</sub>

```python
from tars.act import Command, LocalExecutor

LocalExecutor(".").run(Command(["pytest", "-q"])).reason
```

<sub>`tars/act/__init__.py:24`</sub>

```python
from tars.act import Edit, Hunk, LocalExecutor

preview = LocalExecutor(".", dry_run=True)
preview.apply(Edit.replace("a.py", Hunk("x = 1", "x = 2"))).diff
```

<sub>`tars/act/__init__.py:40`</sub>

```python
from tars.act import get_executor, known_executors

get_executor("local")
known_executors()
```

<sub>`tars/act/__init__.py:48`</sub>

```python
from tars.memory import reflect_on

report = reflect_on("run/session.jsonl")
report.summary_line()
report.to_commit    # high-confidence, decision-shaped, domain-tagged
report.to_review    # what a human should see before anything believes it
report.dropped      # considered, scored too low, kept so you can see it
```

<sub>`tars/memory/__init__.py:7`</sub>

```python
from tars.memory import reflect_on

reflect_on("run/session.jsonl").nothing_worth_remembering
```

<sub>`tars/memory/__init__.py:22`</sub>

## How it's structured

- `tars/__init__.py` — TARS — a disciplined, multi-session coding agent loop. (`python -m tars`)
  - `tars/_cli_helpers.py` — Shared CLI helpers: color and tty detection.
  - `tars/act/__init__.py` — TARS ACT — PHASE 4, isolated: the phase that actually touches the repo.
    - `tars/act/base.py` — The one interface every executor implements.
    - `tars/act/local.py` — The built-in executor: this machine, this repo, stdlib only.
    - `tars/act/registry.py` — Which executor does ACT's work, chosen by name rather than by import.
    - `tars/act/types.py` — What ACT was asked to do, and what came of it.
  - `tars/browser/__init__.py` — Efficient, accessibility-tree-first browser testing. (`python -m tars.browser`)
    - `tars/browser/cli.py` — `tars-browser`: scriptable accessibility-tree web tests.
  - `tars/loop/__init__.py` — TARS loop orchestrator — the seven phases, driven. (`python -m tars.loop`)
    - `tars/loop/_io.py` — Small output helpers for the REPL.
    - `tars/loop/cli.py` — `tars` — persistent REPL and one-shot loop commands.
    - `tars/loop/engine.py` — The seven-phase loop, composed from TARS's tested phase modules.
    - `tars/loop/repl.py` — The persistent slash-command interface over one live :class:`Session`.
    - `tars/loop/session.py` — What persists between commands.
    - `tars/loop/session_store.py` — Durable event log for REPL sessions.
    - `tars/loop/tools.py` — Model-facing ACT tools.
    - `tars/loop/types.py` — The loop's own vocabulary.
  - `tars/memory/__init__.py` — TARS memory writer — PHASE 6 (REFLECT), isolated. (`python -m tars.memory`)
    - `tars/memory/cli.py` — `tars-memory` — the REFLECT phase as a command.
    - `tars/memory/domains.py` — Which partition does a fact belong to, and what said so.
    - `tars/memory/propose.py` — The REFLECT pass: read a transcript, propose what is worth remembering.
    - `tars/memory/signals.py` — Detectors for the sentence in core directive 4.
    - `tars/memory/similarity.py` — Small lexical similarity for memory retrieval without an embedding service.
- …and 32 more modules

Tests live in `tests/`.

## Verifying it

```console
ruff check .
ruff format --check .
mypy tars tests
pytest
```

<sub>detected from this repo: format, lint, test, typecheck</sub>

---

<sub>Generated by `tars-readme` from the source of this repository: 6 examples extracted from the paths above. Sections with nothing to source were left out.</sub>
