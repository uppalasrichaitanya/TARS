#!/usr/bin/env bash
# overnight_build.sh — unattended Claude Code build + self-directed refine loop
#
# Phase 1: fixed foundation phases (provider adapter, verification gate, etc.)
# Phase 2: open refinement loop — Opus reviews the project itself, picks the
#          most valuable next improvement, builds + verifies it, self-rates
#          overall quality, and either continues or declares itself done.
#
# Usage:
#   chmod +x overnight_build.sh
#   nohup ./overnight_build.sh > /dev/null 2>&1 &
#   disown
#
# Morning check-in:
#   cat run/summary.log
#   git log --oneline

set -uo pipefail

# ── Lock — a second launch exits, not restarts from phase one ──
LOCK_FILE="$REPO_DIR/.overnight_build.lock"
exec 9>"$LOCK_FILE" 2>/dev/null || true
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    echo "overnight_build.sh is already running (lock $LOCK_FILE)" >&2
    exit 1
  fi
fi
trap 'rm -f "$LOCK_FILE" 2>/dev/null; exec 9>&- 2>/dev/null || true' EXIT

# ── Resume point ──────────────────────────────────────────
PROGRESS_FILE="$RUN_DIR/progress"
mkdir -p "$RUN_DIR"
touch "$PROGRESS_FILE"
mark_done() { echo "$1" >> "$PROGRESS_FILE"; }
already_done() { grep -qxF "$1" "$PROGRESS_FILE" 2>/dev/null; }

# ── Config ────────────────────────────────────────────────
REPO_DIR="$(pwd)"
RUN_DIR="$REPO_DIR/run"
SPEC_FILE="$REPO_DIR/master_agent_loop.md"
MODEL="opus"
BUDGET_PER_CALL_USD="9999"       # no real cap, per your call
MAX_CONSECUTIVE_FAILURES=3       # only counts REAL failures (after retries exhausted) — "needs a human" guard
MAX_REFINE_ITERATIONS=60         # sane infinite-loop backstop, not a quality cap — shouldn't bind if it's working
SATISFIED_TOKEN="TARS_SATISFIED"
RETRY_LIMIT=8                    # per-call retries before it counts as a real failure
RETRY_BACKOFF_BASE=30            # seconds; doubles each retry (30,60,120,240...) to ride out rate limits/outages

FOUNDATION_PHASES=(
  "provider-adapter:Build the provider adapter layer per the spec — unified generate(messages, tools, model) interface, Claude + OpenAI direct SDKs, OpenRouter via HTTP for everything else. Include basic tests."
  "verification-gate:Build the verification-gate module — detect the repo's test runner, linter, type-checker, and run them uniformly. Standalone CLI-callable."
  "readme-writer:Build the readme-writer skill — inspect the codebase, pull real usage examples from working code, generate a structured README. Test it on this repo itself."
  "router-classifier:Build the router-classifier module — given a task description, output tier (0/1/2) + blast radius. Include a small labeled test set and report accuracy."
  "memory-writer:Build the memory-writer module — takes a session transcript, proposes domain-tagged memory candidates with confidence scores, per the REFLECT phase rules in the spec."
  "loop-orchestrator:Wire provider-adapter, verification-gate, router-classifier, and memory-writer into the actual TARS loop from the spec (ORIENT→CLASSIFY→PLAN→ACT→VERIFY→REFLECT→REPORT). CLI entrypoint, rich/textual output."
)
# ─────────────────────────────────────────────────────────

mkdir -p "$RUN_DIR"
SUMMARY="$RUN_DIR/summary.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SUMMARY"; }

# Retries a claude call up to RETRY_LIMIT times with exponential backoff.
# Handles transient stuff (rate limits, network blips, momentary API outages)
# without burning the consecutive-failure budget. Only returns non-zero if
# every retry failed — that's a real failure, not a hiccup.
run_claude() {
  local log_file="$1" prompt="$2"
  local attempt=1
  local backoff=$RETRY_BACKOFF_BASE

  while (( attempt <= RETRY_LIMIT )); do
    claude \
      --print \
      --dangerously-skip-permissions \
      --model "$MODEL" \
      --max-budget-usd "$BUDGET_PER_CALL_USD" \
      "$prompt" \
      < /dev/null \
      > "$log_file" 2>&1
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
      if (( attempt > 1 )); then
        log "  (succeeded on retry attempt $attempt)"
      fi
      return 0
    fi

    log "  call failed (exit $exit_code), attempt $attempt/$RETRY_LIMIT — retrying in ${backoff}s. See $log_file"
    sleep "$backoff"
    backoff=$(( backoff * 2 ))
    attempt=$(( attempt + 1 ))
  done

  log "  all $RETRY_LIMIT retries exhausted for this call — treating as a real failure."
  return 1
}

foundation_prompt() {
  local name="$1" desc="$2"
  cat <<EOF
You are working inside the TARS project. Full spec: $SPEC_FILE (read it first).

This phase: $name
Task: $desc

Rules:
- Follow the loop discipline in the spec (classify before acting, verify
  before declaring done, propose memory entries only where genuinely
  warranted).
- Commit your work with git as you go, clear commit messages, so partial
  progress survives even if this phase doesn't finish.
- If stuck or ambiguous, make the most reasonable assumption, note it
  clearly in a commit message, and continue. Nobody is available tonight
  to answer questions — do not stall waiting for input.
- Stay in scope for this phase only.
- End by running whatever verification applies (tests/build) and report
  pass/fail honestly.
EOF
}

refine_prompt() {
  local iteration="$1"
  cat <<EOF
You are working inside the TARS project (refinement iteration #$iteration).
Spec: $SPEC_FILE. Nobody is watching tonight — you're deciding what to work
on next, not being assigned a task.

Do the following, in order:

1. REVIEW: Look at the current state of the codebase, tests, and git log.
   Identify what's incomplete, fragile, undertested, or missing relative to
   the spec's four pillars (provider layer, browser-test efficiency,
   verification-gated edits, router, partitioned memory) and the loop
   orchestrator.

2. PICK: Choose the SINGLE most valuable improvement to make right now —
   the thing that most increases whether this is a genuinely solid,
   working product. Prefer fixing something broken or half-done over
   adding something new. Prefer real test coverage over more features.
   State what you picked and why in one or two sentences.

3. BUILD IT: Implement it. Keep the change scoped to what you picked —
   don't sprawl into unrelated areas.

4. VERIFY: Run the actual checks (tests/build/lint) for what you changed.
   Do not claim success without running them. If it fails, fix it or
   revert — do not leave the repo in a broken state.

5. COMMIT: Clear message describing what changed and why.

6. SELF-ASSESS: Rate the overall project 0-100 on: does it actually run
   end-to-end, is each of the four pillars genuinely implemented (not
   stubbed), is there real test coverage, would a stranger reading the
   README understand and be able to run it. Give the number and a one-line
   justification.

7. DECIDE: If the project is genuinely solid — most pillars working,
   tested, documented, no known broken state — output the exact line:
   $SATISFIED_TOKEN
   on its own, at the very end of your response. Only do this if you mean
   it; do not output it just because you've done many iterations, and do
   not withhold it out of excessive caution once the thing genuinely works.
   Otherwise, end normally (no token) and the loop will call you again for
   the next iteration.
EOF
}

# ── Phase 1: foundation ─────────────────────────────────────
consecutive_failures=0
for phase in "${FOUNDATION_PHASES[@]}"; do
  name="${phase%%:*}"; desc="${phase#*:}"

  if already_done "foundation:$name"; then
    log "── Foundation phase: $name — already done, skipping ──"
    continue
  fi
  if (( consecutive_failures >= MAX_CONSECUTIVE_FAILURES )); then
    log "HALT (foundation): $MAX_CONSECUTIVE_FAILURES consecutive failures. Needs a human before '$name'."
    break
  fi

  log "── Foundation phase: $name ──"
  PHASE_LOG="$RUN_DIR/phase_${name}.log"
  run_claude "$PHASE_LOG" "$(foundation_prompt "$name" "$desc")"

  if [ $? -eq 0 ]; then
    log "Phase '$name' finished. Committing checkpoint."
    git add -A
    git commit -m "checkpoint: $name (foundation)" --quiet || log "  (nothing to commit)"
    mark_done "foundation:$name"
    consecutive_failures=0
  else
    log "Phase '$name' FAILED. See $PHASE_LOG."
    consecutive_failures=$((consecutive_failures + 1))
  fi
  sleep 5
done

# ── Phase 2: self-directed refinement loop ─────────────────
log "── Foundation complete. Entering refinement loop (cap: $MAX_REFINE_ITERATIONS iterations) ──"
consecutive_failures=0

for (( i=1; i<=MAX_REFINE_ITERATIONS; i++ )); do
  if already_done "refine:$i"; then
    log "── Refine iteration $i — already done, skipping ──"
    continue
  fi
  if (( consecutive_failures >= MAX_CONSECUTIVE_FAILURES )); then
    log "HALT (refine): $MAX_CONSECUTIVE_FAILURES consecutive failures. Stopping — needs a human."
    break
  fi

  log "── Refine iteration $i ──"
  ITER_LOG="$RUN_DIR/refine_${i}.log"
  run_claude "$ITER_LOG" "$(refine_prompt "$i")"
  exit_code=$?

  if [ $exit_code -eq 0 ]; then
    git add -A
    git commit -m "refine iteration $i" --quiet || log "  (nothing to commit iteration $i)"
    mark_done "refine:$i"
    consecutive_failures=0

    if grep -q "$SATISFIED_TOKEN" "$ITER_LOG"; then
      log "Model declared satisfaction at iteration $i. See $ITER_LOG for its self-assessment. Stopping."
      break
    fi
  else
    log "Refine iteration $i FAILED (exit $exit_code). See $ITER_LOG."
    consecutive_failures=$((consecutive_failures + 1))
  fi
  sleep 5
done

log "── Run complete ──"
log "Review: git log --oneline | head -30"
log "Last self-assessment: check the most recent run/refine_*.log"
