# AGENTS.md — read this before touching TARS

You are working on **TARS**, a disciplined multi-session coding agent. This file
is the handover: the conventions, the traps, and the decisions already made.
Everything here was verified on this machine, not assumed.

Two companion documents:

- **`map.md`** — full status: what's built, measured numbers, what's left, and the
  reasoning behind every settled decision. Read it after this file.
- **`master_agent_loop.md`** — the spec. It is the authority on intent.

---

## The prime rule

TARS is **not a chatbot, it is a loop**: `ORIENT → CLASSIFY → PLAN → ACT →
VERIFY → REFLECT → REPORT`, in that order, no phases skipped. The loop's
discipline *is* the product — this is a portfolio piece whose whole thesis is that
the control flow is trustworthy. Do not replace it with a framework's loop; see
"Settled decisions" below.

The spec's five directives govern your own work too, not just the agent's:

1. **Memory before opinion** — check what this project already decided.
2. **Verification before completion** — written ≠ done.
3. **Cheapest capable path** — don't use more model than the task needs.
4. **Write memory like it costs something.**
5. **Narrate honestly, not impressively** — never present a guess as verified.

---

## Run these first

```bash
cd /home/chaitanya/TARS
.venv/bin/python -m pytest              # the suite
.venv/bin/tars-verify .                 # the gate: ruff, ruff-format, mypy, pytest
```

Use `.venv/bin/…` — there is no active virtualenv by default and the repo's own
tools live there.

**State as of 2026-08-25:** at the last commit (`aab5b4f`) the gate was **PASS
4/4** with 934 tests. The working tree then picked up in-progress `tars/loop/`
work which is **currently red** — one failing test and unformatted files. That is
known, described in `map.md` §5, and is the first thing to fix.

---

## Seven traps that will cost you hours

### 1. AgentRouter whitelists client software

**With a valid token**, plain `curl` and the `openai` Python SDK both get
`401 unauthorized_client_error`. The official `anthropic` SDK succeeds. So:

> **A 401 from AgentRouter means "client not whitelisted", NOT "bad key."**

Probed 2026-08-26: the OpenAI-compatible `/v1` surface rejects the official
OpenAI SDK on **both** `/chat/completions` and `/responses` — the whitelist is
client-fingerprint-based, not endpoint-shaped. The codex key works only inside
the Codex CLI itself. Consequence: **a GPT-5.6 provider adapter for TARS is
blocked on AgentRouter's whitelist**, not on code — do not attempt to build one
until a client is whitelisted or different credentials exist.

AgentRouter is opus-only here; set `TARS_TIER0_MODEL` to an explicitly chosen
OpenRouter model to enable tier-0 routing.

Do not go hunting for a new token, and do not try to forge a whitelisted client's
fingerprint. Route through the official SDK — that is why
`tars/providers/agentrouter_provider.py` subclasses the Anthropic adapter instead
of using the stdlib HTTP path.

### 2. The two AgentRouter surfaces differ in path

| Surface | Base URL | Auth |
|---|---|---|
| Anthropic-compatible | `https://agentrouter.org` — **no `/v1`** | `ANTHROPIC_AUTH_TOKEN`, bearer |
| OpenAI-compatible | `https://agentrouter.org/v1` — **`/v1` required** | bearer |

The Anthropic SDK appends `/v1/messages` itself, so a `/v1` base URL requests
`/v1/v1/messages` and 404s with a message that explains nothing.
`agentrouter_provider.py` refuses such a base URL up front, with the fix in the
error text.

### 3. Credentials on this machine

Set (in `~/TARS/.env`, gitignored): `ANTHROPIC_AUTH_TOKEN` +
`ANTHROPIC_BASE_URL=https://agentrouter.org`, `OPENROUTER_API_KEY`,
`TOKENROUTER_API_KEY`, and `TARS_TIER0_MODEL=tokenrouter:qwen/qwen3.8-max-free`.
**Unset:** `ANTHROPIC_API_KEY` (do not set it — see below), `OPENAI_API_KEY`.

Consequences: a bare `claude-opus-5` routes to api.anthropic.com and **fails
here** — use `agentrouter:claude-opus-5` for anything live. The Anthropic
(AgentRouter) path and the TokenRouter path were both live-verified this
session (TokenRouter's `qwen/qwen3.8-max-free` completed a real `tool_use`
round-trip). The OpenAI and OpenRouter adapters are unit-tested only — do not
describe them as live-verified.

`ANTHROPIC_API_KEY` must never be used as an AgentRouter credential — that would
send a real Anthropic key to a third party. The adapter deliberately does not
consult it. By the same rule, `OPENAI_API_KEY` is never consulted by the
TokenRouter adapter (`tars/providers/tokenrouter_provider.py`): it reads
`TOKENROUTER_API_KEY` only, so an OpenAI key cannot leak to a third-party host.
TokenRouter's OpenAI-compatible base URL **requires** the `/v1` suffix — the
opposite of AgentRouter's Anthropic host, where a `/v1` base URL would
double-version the path.

**Your own Codex credentials** (a second AgentRouter key, `gpt-5.6-sol`, the `/v1`
surface) are in `~/.codex/config.toml`. That file holds a live bearer token in
plaintext: don't paste its contents into code, commits, or logs.

### 4. Stale bytecode makes a working fix look broken

CPython validates a `.pyc` against its source's mtime **in whole seconds** plus
its **size**. An equal-length edit (`a - b` → `a + b`) written inside one second
leaves the stale cache valid, so the next process imports **pre-edit code**.

This bit hard: VERIFY graded code ACT had already replaced and reported a correct
fix as failed. `tars/act/base.py:invalidate_bytecode` handles the Python case.
Treat it as one instance of a class — any mtime+size cache is exposed the same way
(`.mypy_cache`, `.ruff_cache`, `tsc` incremental). The one-step diagnostic:
**delete the suspected cache, re-run, see if the result changes.**

### 5. `ruff format` rewrites Python inside markdown fences

`README.md` is generated by `tars-readme` from snippets copied **verbatim** out of
this repo, and a test pins that verbatim property. So the formatter excludes it
(`[tool.ruff.format] exclude`, plus `force-exclude = true` because an explicit
path otherwise bypasses an exclude). Don't "fix" the formatting of the generated
README — regenerate it instead.

### 6. AgentRouter's WAF intermittently blocks valid requests

Some hours, ~half of live calls fail with `HTTP 405` carrying an Aliyun WAF
block page — while other calls in the same run pass. Retry once; if it fails
again, report interrupted-with-reason, never as verified or failed-by-content.
Never paste the block-page HTML into user-facing output. This is their
infrastructure, not the whitelist rule (that one is a 401).

### 7. Session logs contain raw tool output

Once session persistence lands, `~/.tars/sessions/` persists raw tool output
unencrypted. Treat those files as sensitive; never paste their contents into
commits, logs, or output.

---

## Conventions this codebase actually follows

Match these; they are consistent across all six modules.

- **Zero runtime dependencies.** `dependencies = []`, and `tars/router`,
  `tars/memory` and `tars/act` each ship a test asserting no third-party import.
  Adding a hard dependency **turns the gate red on purpose**. SDKs are optional
  extras, imported lazily. If you think you need a library, raise it explicitly.
- **TDD, and watch it fail first.** Every module here was built test-first. The
  RED must fail for the right reason — `ModuleNotFoundError` counts, a typo
  doesn't.
- **A failure must say *why*.** `ActionResult` refuses to hold `ok=False` without
  a reason; `CheckResult` carries a status, never a bare bool. This is the spec's
  "never blind-retry" made structural.
- **Docstrings explain *why*, not what.** Look at any module header: they document
  the trap avoided or the choice made. Match that register — the code is the
  portfolio artifact.
- **Injected dependencies over reaching for globals.** Adapters take `env=`,
  executors take a confirm callable, the gate takes `on_start`/`on_result`. That
  is what keeps the suite hermetic (959 tests in ~6s, no network, no spend).
- **argparse, not click/typer.** All three CLIs. Documented exit codes.
- **Commit messages state the reasoning**, including what was deliberately not
  done. `git log` is part of the handover; read a few.

---

## Settled decisions — do not silently reopen these

Each has its full reasoning in `map.md` §7.

| Decision | One-line reason |
|---|---|
| **Don't build on `deepseek-harness`** | Its finest unit is a *prompt*, not a tool call, so adopting it dissolves the loop that is the product. `tars/act`'s `Executor` seam keeps it adoptable later. |
| **Provider layer = master's stdlib version** | The rival httpx branch was deleted. httpx broke the zero-dependency invariant. Its tip `fd24eb9` is now unreachable — the object survives and `git checkout fd24eb9` still works, but only until `git gc` prunes it. |
| **No rich/textual** | Cosmetics don't get to turn a green gate red. Optional `try: import rich` with fallback remains available as a later opt-in. |
| **ACT previews and confirms by default** | `--yes` / `TARS_NO_CONFIRM` skips. `LocalExecutor(dry_run=True)` exists for this. |
| **Router verdict always shown before ACT acts** | Silent routing is how trust erodes. |
| **Anchored hunks, never line numbers** | A stale line number edits the wrong place *silently*; a stale anchor fails loudly. An anchor matching twice is refused, never applied to the first match. |
| **Only `timeout` is retryable** | `not-found`/`ambiguous` mean the plan was wrong; `failed` means the program ran and said no. |

If you believe one of these is wrong, say so explicitly and argue it — the spec's
directive 1 asks exactly that. Do not just quietly do the other thing.

---

## The map

```
tars/providers/     one generate(); Anthropic + OpenAI SDKs, OpenRouter (stdlib
                    HTTP), AgentRouter, TokenRouter. Anthropic (AgentRouter) and
                    TokenRouter paths live-verified incl. tool round-trip;
                    OpenAI + OpenRouter adapters unit-tested only.
tars/verification/  VERIFY — detect a repo's check surface, run it uniformly.
tars/readme/        generate a README from real code; --check for CI.
tars/memory/        REFLECT — scored domain-tagged candidates, review queue,
                    supersede-marks never deletes.
tars/router/        CLASSIFY — tier 0/1/2 + blast + reversibility, named signals.
tars/act/           ACT — anchored edits, no partial writes, argv-only commands.
tars/loop/          THE REPL, IN PROGRESS. types.py done; the rest unbuilt.
```

Commands today: `tars-verify`, `tars-readme`, `tars-memory`. The `tars` REPL is
what's being built — see `map.md` §5 for its file-by-file plan and §9 for the
next five steps.

---

## Where the loop is going

A persistent slash-command REPL: `/run`, `/route`, `/verify`, `/memory`,
`/memory approve`, `/status`, `/pillars`.

**Every phase is a real call into an already-tested module** — the loop
contributes only composition, model→tool dispatch, and the confirm gate. Nothing
stubbed:

| Phase | Calls |
|---|---|
| ORIENT | `memory.recall(domains=…)` |
| CLASSIFY | `router.classify(task)` |
| PLAN | `providers.generate(…)` |
| ACT | model tool calls → `act.LocalExecutor` (dry-run → diff → confirm → apply) |
| VERIFY | `verification.verify(root)` |
| REFLECT | `memory.propose(...)` over the session transcript |
| REPORT | a `RunReport` — including an `uncertainties` field that is printed verbatim |

**The current failing test is a genuine design question, not a bug to paper
over.** `Session` holds `providers.Message` objects; `memory.parse_messages`
wants `{"role", "content"}` mappings. The fix is an explicit conversion in
`Session` — the provider layer shouldn't learn memory's format and memory
shouldn't learn the provider's, and the loop is the thing that knows both.

---

## Before you claim done

1. `.venv/bin/tars-verify .` → **PASS 4/4**. Not "the tests I wrote pass".
2. Then a **real live run**, not only a mocked one. A passing mock and a working
   product are different claims: `act`'s bytecode bug was invisible to 86 unit
   tests and obvious the first time it ran for real, and AgentRouter's whitelist
   was invisible until a live call was made.
3. `.venv/bin/tars-readme . -o README.md --force` then `--check` → exit 0. Adding
   a module changes the README.
4. If something is unverified, **say which part and why** rather than rounding up.
   That is directive 5, and it is the whole point of this project.
