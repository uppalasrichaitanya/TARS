# tars

TARS — a disciplined, multi-session coding agent loop

## Why it exists

Named for TARS - the robot from Interstellar - whose worth lies not in cleverness but in discipline: it does what it said, reports honestly what it did, and asks before it touches anything. This tool tries to earn that name.

## Install

```console
pip install -e ".[dev]"
```

Requires Python `>=3.10`.

This project has no runtime dependencies.

## How to run it

### `tars`

Run TARS's disciplined agent loop.

```console
tars
```

Flags: `--yes`, `--root`, `--model`

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

- `tars/__init__.py` — TARS — a disciplined, multi-session coding agent loop.
  - `tars/act/__init__.py` — TARS ACT — PHASE 4, isolated: the phase that actually touches the repo.
    - `tars/act/base.py` — The one interface every executor implements.
    - `tars/act/local.py` — The built-in executor: this machine, this repo, stdlib only.
    - `tars/act/registry.py` — Which executor does ACT's work, chosen by name rather than by import.
    - `tars/act/types.py` — What ACT was asked to do, and what came of it.
  - `tars/loop/__init__.py` — TARS loop orchestrator — the seven phases, driven.
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
    - `tars/memory/store.py` — Where kept entries live: domain-partitioned JSON, plus one review queue.
    - `tars/memory/transcript.py` — Turning whatever the loop kept into turns the writer can quote.
    - `tars/memory/types.py` — Types the memory writer speaks: what a candidate is, and what makes one
  - `tars/providers/__init__.py` — TARS provider adapter layer.
- …and 26 more modules

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
