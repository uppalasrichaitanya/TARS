# tars

TARS — a disciplined, multi-session coding agent loop

## Install

```console
pip install -e ".[dev]"
```

Requires Python `>=3.10`.

This project has no runtime dependencies.

## How to run it

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

```python
from tars.memory import propose, parse_text

candidate = propose(parse_text("We decided to use stdlib http.")).candidates[0]
candidate.signals      # ("cue:decided-to", "role:assistant", ...)
candidate.evidence     # the turns it was read out of, quoted verbatim
candidate.disposition  # "commit", "review" or "drop"
```

<sub>`tars/memory/__init__.py:29`</sub>

```python
from tars.memory import reflect_on, remember

remember(reflect_on("run/session.jsonl"))
```

<sub>`tars/memory/__init__.py:42`</sub>

```python
from tars.memory import recall

recall(domains=("providers",))
recall(kinds=("constraint",), limit=5)
```

<sub>`tars/memory/__init__.py:51`</sub>

```python
from tars.providers import Message, ToolDef, generate

response = generate(
    [Message.user("What files are in this repo?")],
    tools=[ToolDef("ls", "List files", {"type": "object", "properties": {}})],
    model="claude-opus-5",          # or "gpt-5", or "qwen/qwen-3-235b"
)
response.text, response.tool_calls, response.usage.total_tokens
```

<sub>`tars/providers/__init__.py:5`</sub>

## How it's structured

- `tars/__init__.py` — TARS — a disciplined, multi-session coding agent loop.
  - `tars/memory/__init__.py` — TARS memory writer — PHASE 6 (REFLECT), isolated. (`python -m tars.memory`)
    - `tars/memory/cli.py` — `tars-memory` — the REFLECT phase as a command.
    - `tars/memory/domains.py` — Which partition does a fact belong to, and what said so.
    - `tars/memory/propose.py` — The REFLECT pass: read a transcript, propose what is worth remembering.
    - `tars/memory/signals.py` — Detectors for the sentence in core directive 4.
    - `tars/memory/store.py` — Where kept entries live: domain-partitioned JSON, plus one review queue.
    - `tars/memory/transcript.py` — Turning whatever the loop kept into turns the writer can quote.
    - `tars/memory/types.py` — Types the memory writer speaks: what a candidate is, and what makes one
  - `tars/providers/__init__.py` — TARS provider adapter layer.
    - `tars/providers/_openai_compat.py` — Translation shared by every OpenAI-compatible endpoint.
    - `tars/providers/agentrouter_provider.py` — AgentRouter adapter — the Anthropic Messages API at a third-party host.
    - `tars/providers/anthropic_provider.py` — Anthropic adapter — direct SDK, first-class per the spec.
    - `tars/providers/base.py` — The one interface every provider implements.
    - `tars/providers/errors.py` — Errors raised by the provider adapter layer.
    - `tars/providers/openai_provider.py` — OpenAI adapter — direct SDK, first-class per the spec.
    - `tars/providers/openrouter_provider.py` — OpenRouter adapter — plain HTTP, no SDK.
    - `tars/providers/registry.py` — Model-string routing and provider construction.
    - `tars/providers/types.py` — Provider-neutral types spoken by the TARS agent loop.
  - `tars/readme/__init__.py` — TARS readme writer — a README built out of what a repo actually contains. (`python -m tars.readme`)
    - `tars/readme/cli.py` — `tars-readme` — write a README out of what a repo actually contains.
    - `tars/readme/examples.py` — Mining usage examples out of a repo, and distrusting them until they resolve.
    - `tars/readme/inspection.py` — One inspection pass: everything the tool knows, and where it learned it.
    - `tars/readme/manifest.py` — What the project says about itself, read out of its manifest.
- …and 12 more modules

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
