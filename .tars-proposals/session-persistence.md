# Session persistence proposal

Use append-only JSONL, one loop-owned event per line, with a versioned envelope:
`{"v":1,"type":"session_started",...}` followed by message, classification,
verification, and configuration events. JSONL fits a live REPL: flush each
completed transition without rewriting the file, and a crash loses at most the
final line. Reject malformed middle lines; explicitly report and ignore only a
truncated final line.

The file is not a `tars.memory` transcript. Persistence must preserve the full
provider-neutral `Message` shape, including tool calls, call ids, and error state.
`Session` reconstructs `Message` objects on resume and continues to convert via
`Session.as_messages()` for `memory.parse_messages`. Neither providers nor memory
learns the persistence schema.

A restart restores transcript, canonical root, model, `auto_confirm`, last
classification, and last verification verdict. Transcript alone would make
`/status` lie and could resume against a different model or repository. Toolbox,
provider client, streams, confirm callback, and pending command do not survive.
An incomplete `/run` remains marked interrupted; never continue midway through ACT.

Add `/resume [ID]`. With an ID, load after validating the root. Without one,
list current-repo sessions newest first and require an explicit ID. Never replace
a non-empty live session silently: require `/clear` or confirmation. On success
print ID, root, model, turn count, and last verdicts.

Store files under `${TARS_HOME:-~/.tars}/sessions/`. Use an opaque sortable ID
such as `20260826T143012Z-7f3a`; task-text filenames leak, collide, and stale.
Store a truncated first-task label inside the file for display, never identity.

`/status` keeps its snapshot, then lists same-canonical-root sessions with ID,
updated time, label, turns, model, and verdict. Discovery reads only relevant
first/last events and skips corrupt files with a named warning.

`Session` gains identity, event conversion, append hooks, and a load constructor;
mutations append only after succeeding in memory. `engine` records run-started
and run-finished/interrupted boundaries. `repl` owns `/resume`, switching,
warnings, and expanded `/status`. `cli` resolves/injects the session store.
One-shot `tars run` and `tars route` remain non-persistent.

The seven-phase order, provider adapters, router, ACT confirmation rules,
verification gate, memory store, `memory.parse_messages`, and `RunReport` do not
change. Persistence is composition state, not a new phase or a framework loop.

No streaming or partial-message recovery; fork/branch/merge; rename/export/
compaction/deletion UX; cross-repo resume or discovery; concurrent writers;
cloud sync; encryption; or semantic/vector indexing in this slice.

The largest risk is losing voice/provenance: tool results flattened into user
prose, dropped tool metadata, or recalled ORIENT memory replayed as human speech.
Preserve roles and exclude synthetic resume/status text. Give events IDs and load
idempotently so retries do not overweight claims. Make schema migration explicit.
Raw tool output can flood recall with code, logs, secrets, and stale observations;
semantic recall must index a deliberate phase-aware projection, not raw JSONL.
