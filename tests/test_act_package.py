"""The ACT package's surface, and the two costs it must not have.

ACT is called on every task, including tier-0 ones the router says need no
reasoning at all. So importing it must not drag in the provider layer — deciding
*what* to do needs a model, doing it does not — and it must need nothing from
PyPI, matching the same guarantee `tars.router` and `tars.memory` already make.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tars import act


def _import_probe(statement: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_documented_entry_points_are_exported():
    for name in (
        "Edit",
        "Hunk",
        "Command",
        "ActionResult",
        "Executor",
        "LocalExecutor",
    ):
        assert name in act.__all__, name


def test_every_exported_name_actually_resolves():
    for name in act.__all__:
        assert hasattr(act, name), name


def test_dunder_all_has_no_duplicates():
    assert len(act.__all__) == len(set(act.__all__))


def test_the_failure_vocabulary_is_part_of_the_surface():
    """A caller branching on why an action failed needs the list, not a guess."""
    assert "not-found" in act.VALID_FAILURE_REASONS
    assert "ambiguous" in act.VALID_FAILURE_REASONS
    assert act.RETRYABLE_REASONS == frozenset({"timeout"})


def test_an_edit_and_a_command_can_be_driven_from_the_package_alone(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    executor = act.LocalExecutor(tmp_path)

    edit = executor.apply(act.Edit.replace("a.py", act.Hunk("x = 1", "x = 2")))
    ran = executor.run(act.Command([sys.executable, "-c", "print('ok')"]))

    assert edit.ok, edit.detail
    assert (tmp_path / "a.py").read_text() == "x = 2\n"
    assert ran.ok and "ok" in ran.stdout


def test_the_default_executor_is_reachable_by_name():
    assert isinstance(act.get_executor("local"), act.LocalExecutor)


# ── the cost of importing it ─────────────────────────────────────────────


def test_importing_act_does_not_import_the_provider_layer():
    """Doing the work needs no model, so it must not need the model layer."""
    probe = _import_probe(
        "import sys, tars.act;"
        "assert 'tars.providers' not in sys.modules, sorted("
        "m for m in sys.modules if m.startswith('tars.'))"
    )
    assert probe.returncode == 0, probe.stderr


def test_act_needs_no_third_party_package():
    probe = _import_probe(
        "import sys, tars.act;"
        "bad = [m for m in ('anthropic', 'openai', 'httpx', 'pydantic')"
        " if m in sys.modules];"
        "assert not bad, bad"
    )
    assert probe.returncode == 0, probe.stderr


def test_a_dry_run_executor_is_the_documented_way_to_preview(tmp_path: Path):
    """The CLI's approval model will need this, so it is part of the surface."""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    result = act.LocalExecutor(tmp_path, dry_run=True).apply(
        act.Edit.replace("a.py", act.Hunk("x = 1", "x = 2"))
    )
    assert result.ok
    assert "+x = 2" in result.diff
    assert (tmp_path / "a.py").read_text() == "x = 1\n"
