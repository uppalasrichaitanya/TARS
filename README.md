# tars

TARS — a disciplined, multi-session coding agent loop

## Install

```console
pip install -e ".[dev]"
```

Requires Python `>=3.10`.

This project has no runtime dependencies.

## How to run it

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
from tars.providers import Message, ToolDef, generate

response = generate(
    [Message.user("What files are in this repo?")],
    tools=[ToolDef("ls", "List files", {"type": "object", "properties": {}})],
    model="claude-opus-5",          # or "gpt-5", or "qwen/qwen-3-235b"
)
response.text, response.tool_calls, response.usage.total_tokens
```

<sub>`tars/providers/__init__.py:5`</sub>

```python
from tars.readme import readme_for

print(readme_for("/path/to/repo"))
```

<sub>`tars/readme/__init__.py:3`</sub>

```python
from tars.readme import inspect_project

facts = inspect_project(".")
facts.verified_examples    # snippets whose imports still resolve
facts.unverified_examples  # snippets that rotted since they were written
facts.notes                # what was dropped on the way, and why
```

<sub>`tars/readme/__init__.py:11`</sub>

```python
from tars.verification import verify

report = verify("/path/to/repo")
report.verdict        # "pass" | "fail" | "no-checks"
report.failures       # the checks that failed, with output
report.missing_kinds()  # what this repo can't verify at all
```

<sub>`tars/verification/__init__.py:3`</sub>

```console
tars-readme /path/to/repo --output README.md
tars-readme . --check
```

<sub>`tars/readme/__init__.py:35`</sub>

```console
python -m tars.verification --list
tars-verify /path/to/repo --json
```

<sub>`tars/verification/__init__.py:31`</sub>

## How it's structured

- `tars/__init__.py` — TARS — a disciplined, multi-session coding agent loop.
  - `tars/providers/__init__.py` — TARS provider adapter layer.
    - `tars/providers/_openai_compat.py` — Translation shared by every OpenAI-compatible endpoint.
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
    - `tars/readme/render.py` — Turning facts into markdown, in the order the spec fixes.
    - `tars/readme/source.py` — The corpus: which files to read, parsed but never imported.
    - `tars/readme/types.py` — Types the readme writer speaks: what was found, and where it was found.
  - `tars/verification/__init__.py` — TARS verification gate — map a repo to its check surface, then run it. (`python -m tars.verification`)
    - `tars/verification/cli.py` — `tars-verify` — the verification gate as a standalone command.
    - `tars/verification/detect.py` — Mapping a repo to the checks it can actually run.
    - `tars/verification/gate.py` — The gate itself: detect → select → run → report.
    - `tars/verification/runner.py` — Running one check, uniformly.
    - `tars/verification/types.py` — Types the verification gate speaks: what to run, what happened, what it means.

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
