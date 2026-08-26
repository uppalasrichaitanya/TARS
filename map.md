# map.md — where TARS stands, and where it is going

*Written 2026-08-25. Every number below was measured against the tree, not
recalled. Where something is unbuilt or broken, it says so.*

---

## 1. What TARS is

A disciplined, multi-session coding agent. Not a chatbot — a **loop**. Every task
runs seven phases in a fixed order, and the discipline is the product:

```
ORIENT → CLASSIFY → PLAN → ACT → VERIFY → REFLECT → REPORT
```

The spec lives in `master_agent_loop.md`. Its five non-negotiable directives:

1. **Memory before opinion** — check what the project already decided.
2. **Verification before completion** — written ≠ done.
3. **Cheapest capable path** — never use more model than the task needs.
4. **Write memory like it costs something** — not everything is worth keeping.
5. **Narrate honestly, not impressively** — never present a guess as verified.

---

## 2. Status, measured right now

| | |
|---|---|
| Tests | **987 passed, 0 failed, 2 skipped** |
| `tars-verify .` | **PASS 4/4** |
| Live `/run` | **VERIFIED end-to-end through AgentRouter** (2026-08-26): classify → edit → report on a temp repo; `add(2,3) = 5` after the fix |
| Live graded run | **VERIFIED** (2026-08-26): a repo with a real failing pytest went FAIL → PASS across one live `/run`; VERIFY graded an actual fix |
| Runtime dependencies | **zero** (`dependencies = []`), enforced by tests |

Known caveats, stated plainly:
- AgentRouter's WAF intermittently 405s valid requests (AGENTS.md trap #6);
  live runs need a retry and honest interruption reports.
- **REPORT under-reports `touched` when the model edits via `run_command`**
  instead of `apply_edit` (seen in the graded run: repo changed,
  `touched: none`). Unfixed; next slice.

---

## 3. Built and green — six modules

| Module | Phase | What it does | How it was proven |
|---|---|---|---|
| `tars/providers` | — | one `generate(messages, tools, model)`; Anthropic + OpenAI direct SDKs, OpenRouter over stdlib HTTP, **AgentRouter** | live on `claude-opus-5`, including a full tool round-trip |
| `tars/verification` | VERIFY | detects a repo's check surface (test/lint/typecheck/format) and runs it uniformly | runs on TARS itself as `tars-verify` |
| `tars/readme` | — | generates a README from real code, with `--check` for CI | this repo's README is its own output, committed verbatim |
| `tars/memory` | REFLECT | scored, domain-tagged candidates; human review queue; supersede-marks, never deletes | 76-statement hand-labelled corpus, every floor met |
| `tars/router` | CLASSIFY | tier 0/1/2 + blast radius + reversibility, with named signals | tier **1.00 dev / 1.00 holdout** vs a 0.33 majority baseline |
| `tars/act` | ACT | anchored edits, no partial writes, argv-only commands | end-to-end in a temp repo against the gate |

**Commands that work today:** `tars-verify`, `tars-readme`, `tars-memory`
(`reflect` / `recall` / `pending` / `approve` / `reject`). All argparse, stdlib
only, documented exit codes.

### Numbers worth keeping

- **Memory benchmark** (76 statements: 45 memory-worthy, 31 not): detector
  precision 0.93, recall 0.87, kind accuracy 0.97, store precision 1.00. The
  scorecard names its 6 misses and 3 false positives rather than hiding them.
- **Router benchmark** (48 labelled cases + the rubric that produced them): tier
  1.00/1.00, blast 0.88/0.92, reversibility 1.00/1.00. **The holdout is
  structural, not unseen** — corpus and cues were written in one sitting by one
  author, and the benchmark's docstring says so rather than letting 1.00 imply
  more than it should.

---

## 4. What this session did

| Commit | What |
|---|---|
| `e882893` | **agentrouter** — AgentRouter's Claude surface as a first-class provider, live-verified |
| `6504600` | **router-classifier** — the module its own phase never wrote; the suite could not even collect before this |
| `4bb8ff1` | **memory-writer close-out** — reconstructed phase report, README regenerated |
| `3c0b342` | **act** — the executor seam, so a harness stays optional |
| `aab5b4f` | **providers** — cleared all lint/type debt, deleted the rival branch, first green gate in the project's history |

Three findings from this work that outlived the tasks that produced them:

- **AgentRouter whitelists clients.** With a *valid* token, plain `curl` and the
  `openai` SDK both return `401 unauthorized_client_error`; only the official
  `anthropic` SDK succeeds. So that 401 means "client not whitelisted", not "bad
  key" — and AgentRouter must be reached through the SDK, never raw HTTP.
- **The router phase had been checkpointed as finished with no module at all** —
  just a test importing `tars.router`. That single broken import failed
  collection for the *entire* suite, so `pytest` exited 2 with zero tests run and
  the gate reported FAIL for reasons unrelated to any code.
- **Stale bytecode breaks edit-then-verify.** CPython validates a `.pyc` against
  its source's mtime *in whole seconds* plus size, so an equal-length edit
  (`a - b` → `a + b`) leaves the stale cache valid and the next process imports
  pre-edit code. VERIFY then grades code ACT already replaced and reports a
  working fix as failed — which under the escalation rule burns a tier-2 attempt
  on nothing. **86 unit tests missed this; running it for real caught it in one
  step.**

---

## 5. The `tars` REPL (`tars/loop/`) — COMPLETE and live-verified

Six good libraries, now one product. Built file-by-file TDD-first across
commits `d99a961`…`fdc849a`, shipped as the `tars` command in `487455a`.

| File | Contents | State |
|---|---|---|
| `types.py` | `Phase`, `RunPromptSpec`, `RunReport`, `ToolCall`, `ToolResult` | ✅ |
| `session.py` | `Session` — root, model, transcript, last verdicts; env-aware default model | ✅ |
| `_io.py` | raw-ANSI output: `NO_COLOR`, `isatty`, `\033[…m` | ✅ |
| `tools.py` | `read_file` / `list_dir` / `apply_edit` / `run_command`; both mutating tools behind the confirm gate | ✅ |
| `engine.py` | the seven phases + model→tool dispatch, retry via ACT's `RETRYABLE_REASONS` | ✅ |
| `repl.py` | slash dispatch over one live `Session`, injected streams | ✅ |
| `cli.py` | bare `tars` opens the REPL; `tars run`/`tars route` one-shots; exit codes 0/1/2 | ✅ |

**Live proof (2026-08-26):** a temp repo whose `add()` subtracted was fixed
by a real `/run` through AgentRouter — CLASSIFY printed its verdict, the
model requested an anchored edit which applied correctly, VERIFY honestly
reported "no checks" rather than passing, REFLECT proposed two candidates
without auto-committing them, REPORT rendered everything including
uncertainties verbatim.

### Every phase is a real call, nothing stubbed

| Phase | Calls | Model? |
|---|---|---|
| ORIENT | `memory.recall(domains=…)` | no |
| CLASSIFY | `router.classify(task)` — **verdict always printed** | no |
| PLAN | `providers.generate(…)`, shown as the plan | yes |
| ACT | model tool calls → `act.LocalExecutor`; each edit dry-run → diff → confirm → apply | yes |
| VERIFY | `verification.verify(root)` | no |
| REFLECT | `memory.propose(...)` over the session transcript | no |
| REPORT | a `RunReport`: done / verified / written to memory / **uncertain** | no |

The loop contributes only composition, model→tool dispatch, and the confirm gate.

**Commands:** `/run <task>`, `/route <task>`, `/verify`, `/memory`,
`/memory approve`, `/status`, `/pillars`, plus `/help` and `/exit`.

### The one red test, and why it is a good failure

`Session` stores `providers.Message` objects — that is what the provider layer
speaks. `memory.parse_messages` wants `{"role", "content"}` mappings. The test
that asserts the REFLECT seam actually connects is precisely what caught it.

**Fix:** an explicit conversion on `Session` (Message → mapping) rather than
making either module tolerate the other's type. The provider layer should not
learn memory's format, and memory should not learn the provider's; the loop is
the thing that knows both, so the translation belongs here. Then format the new
files and the gate returns to PASS 4/4.

### One machine-specific choice, made explicit

A bare `claude-opus-5` routes to api.anthropic.com, which has **no credentials on
this machine**. So the default model is resolved from the environment: an
AgentRouter-looking base URL *plus* an auth token → `agentrouter:claude-opus-5`;
otherwise `claude-opus-5`. An auth token *alone* is deliberately not taken as
evidence — it may be a direct Anthropic bearer token, and guessing wrong would
send a real Anthropic credential to a third-party host. `/status` prints which
model is live so it is never a mystery.

---

## 6. Remaining after the REPL

1. **`browser-test-efficient`** — a11y-tree-first navigation, action batching,
   selector caching. The spec calls it the sharpest technical differentiator and
   says to save it until the loop is solid. Needs `playwright`. Note it is in the
   spec's skills list but was never in `overnight_build.sh`'s phase list, so the
   overnight run was never going to build it.
2. **Semantic recall for memory** — the spec wants your existing vector DB on the
   embedded side, with JSON/SQLite for the structured side. Only the structured
   side exists, so ORIENT cannot yet ask "what do we know *related* to this?" The
   spec calls this "the real synergy point".
 3. **GPT-5.6 path — BLOCKED on AgentRouter's client whitelist.** Probed
    2026-08-26 with the codex key: the `/v1` surface 401s the official OpenAI
    SDK on both `/chat/completions` and `/responses` (`unauthorized_client_error`),
    so the whitelist is client-fingerprint-based, not endpoint-shaped. The key
    works only inside Codex CLI itself. No adapter helps; needs a whitelisted
    client or different credentials.
 4. **Smaller, optional:** consolidate the (soon four) duplicated
    `_color`/`isatty` helpers across the CLIs; `overnight_build.sh` still has no
    lockfile and no resume point, so relaunching restarts from phase one.
 5. **Directive 3 is classified but not enacted** — the router emits a tier,
    yet PLAN/ACT always use `session.model`. Wiring tier→model selection
    (e.g. tier 0 → a cheap model) would make "cheapest capable path" real.
 6. **Transient provider errors are not auto-retried** — trap #6's rule says
    retry once on WAF-style failures; today that retry is manual.

---

## 7. Decisions settled, with their reasons

| Decision | Why |
|---|---|
| **Keep master's provider layer; delete the httpx branch** | httpx is a *runtime* dependency, and `router`/`memory`/`act` each assert zero third-party imports — a regression, not a style preference. The branch was also 21,213 deletions behind master. |
| **Don't build on `deepseek-harness`** | Real and good (MIT, 193k stars, active), but its finest unit is a **prompt**, not a tool call. Adopting it would dissolve the loop that *is* the product. `tars/act`'s `Executor` seam keeps it adoptable later as a prompt-level delegate. |
| **AgentRouter through the official SDK, never raw HTTP** | It whitelists clients; a valid key still 401s from `curl` or the `openai` SDK. |
| **Plain streaming output, no rich** | Cosmetics don't get to turn a green gate red. Optional `try: import rich` with a fallback stays available as a later opt-in. |
| **ACT previews and confirms by default** | `--yes` / `TARS_NO_CONFIRM` skips. `LocalExecutor(dry_run=True)` exists for exactly this. |
| **Router verdict always shown before ACT** | Silent routing is how trust erodes. |
| **Anchored hunks, never line numbers** | A stale line number edits the wrong place *silently*; a stale anchor fails loudly. An anchor matching twice is refused, never applied to the first hit. |
| **Only `timeout` is retryable** | `not-found`/`ambiguous` mean the plan was wrong; `failed` means the program ran and said no. This is what makes "never blind-retry" enforceable. |
| **Nothing written until every hunk resolves** | Stronger than "apply incrementally": there is no partial state to localize. |

---

## 8. The verification standard

Applied to everything above, and to everything still to come:

1. **TDD** — watch the test fail first, for the right reason. Not "it errored".
2. **`tars-verify .` stays PASS 4/4.** New files clean under ruff, ruff-format
   and mypy.
3. **Hermetic tests** — no network, injected fakes, never blocking on stdin. The
   suite runs in seconds and spends nothing.
4. **Then a real live run.** A passing mock and a working product are different
   claims. `act`'s stale-bytecode bug was invisible to 86 unit tests and obvious
   the moment it ran for real; the AgentRouter whitelist was invisible until a
   live call was made.
5. **Regenerate the README** (`tars-readme . --check` → exit 0) — adding a module
   changes it.

---

## 9. Next steps

1. ~~Fix the seam, build the loop, ship `tars`~~ — done, live-verified
   (see §5).
2. ~~A live `/run` on a repo with a real check surface~~ — done, verified
   2026-08-26 (FAIL → PASS graded by real pytest).
3. ~~GPT-5.6 provider path~~ — **blocked on AgentRouter's client whitelist**
   (probed; see §6 item 3). Parked until a whitelisted client or new
   credentials exist.
4. **Session persistence** — plan first, build after review (the hinge for
   semantic recall).
5. Then §6: tier→model routing, transient-retry, browser-test-efficient,
   semantic recall, smaller items.
