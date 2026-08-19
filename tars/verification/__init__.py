"""TARS verification gate — map a repo to its check surface, then run it.

    from tars.verification import verify

    report = verify("/path/to/repo")
    report.verdict        # "pass" | "fail" | "no-checks"
    report.failures       # the checks that failed, with output
    report.missing_kinds()  # what this repo can't verify at all

Nothing here imposes a toolchain. Detection is evidence-based and read-only:
config files, dependency declarations, and what's actually installed in the
repo's own `.venv` / `node_modules` decide which commands get run, and every
`Check` carries the evidence that selected it. A repo that disagrees can say so
in `.tars/verification.json`:

    {"checks": {"test": "make integration", "lint": null}}

Three properties are load-bearing for a caller that trusts the result:

* **`no-checks` is not `pass`.** A repo with no detectable check surface gets
  its own verdict, because "nothing failed" and "nothing ran" are different
  facts and only one of them is evidence.
* **A missing tool is a coverage gap, not a failure.** It stays in the report as
  `unavailable`, with the evidence that made the gate look for it, rather than
  being dropped.
* **Checks never write to the tree.** Formatters run in `--check` mode, and the
  `build` kind is opt-in (`ALL_KINDS`, not `DEFAULT_KINDS`) for that reason.

Also runnable as a standalone command, with no TARS loop involved:

    python -m tars.verification --list
    tars-verify /path/to/repo --json
"""

from __future__ import annotations

from .detect import (
    OVERRIDES_PATH,
    detect_candidates,
    detect_checks,
    detect_ecosystems,
    load_overrides,
)
from .gate import select_checks, unavailable_tools, verify
from .runner import (
    DEFAULT_TIMEOUT_S,
    build_child_env,
    resolve_executable,
    run_check,
)
from .types import (
    ALL_KINDS,
    DEFAULT_KINDS,
    VALID_KINDS,
    Check,
    CheckKind,
    CheckResult,
    CheckStatus,
    GateReport,
    Verdict,
    sort_checks,
)

__all__ = [
    "ALL_KINDS",
    "Check",
    "CheckKind",
    "CheckResult",
    "CheckStatus",
    "DEFAULT_KINDS",
    "DEFAULT_TIMEOUT_S",
    "GateReport",
    "OVERRIDES_PATH",
    "VALID_KINDS",
    "Verdict",
    "build_child_env",
    "detect_candidates",
    "detect_checks",
    "detect_ecosystems",
    "load_overrides",
    "resolve_executable",
    "run_check",
    "select_checks",
    "sort_checks",
    "unavailable_tools",
    "verify",
]
