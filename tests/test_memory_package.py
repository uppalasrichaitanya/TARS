"""The package surface, and the one property this phase has to hold for itself.

`tars.memory` is the import the rest of the loop will use, so what it exports is
a promise. Three tests keep that promise checkable: every exported name resolves,
the three verbs work end to end in one call each, and this package's own
docstring examples are run through the readme writer's extractor — the same
self-check the readme phase applies to itself, on the file most likely to rot
quietly.

Also pinned: importing the memory writer must not drag in `tars.providers`. The
spec asks for this module to be testable in isolation, and on this machine the
provider layer is contested between two implementations and needs credentials
that a reflection pass has no business touching.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import tars.memory as pkg

ROOT = Path(__file__).resolve().parent.parent

DECISION = "We decided to use stdlib http for OpenRouter instead of adding an SDK.\n"

tomllib = pytest.importorskip(
    "tomllib", reason="tomllib is 3.11+; the check needs a TOML parser"
)


@pytest.fixture(scope="module")
def scripts():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle).get("project", {}).get("scripts", {})


def transcript(tmp_path, text=DECISION):
    path = tmp_path / "session.md"
    path.write_text(text)
    return path


# ── exports ───────────────────────────────────────────────────────────────


def test_the_documented_entry_points_are_exported():
    for name in (
        "reflect_on",
        "remember",
        "recall",
        "recall_related",
        "propose",
        "ReflectionReport",
    ):
        assert name in pkg.__all__, name


def test_every_exported_name_actually_resolves():
    missing = [name for name in pkg.__all__ if not hasattr(pkg, name)]
    assert missing == []


def test_dunder_all_has_no_duplicates():
    assert len(pkg.__all__) == len(set(pkg.__all__))


def test_the_thresholds_that_decide_an_entrys_fate_are_part_of_the_surface():
    # A caller that can read the score but not the line it is measured against
    # can't tell a review item from a committed one without guessing.
    assert pkg.REVIEW_FLOOR < pkg.COMMIT_THRESHOLD


# ── the three verbs ───────────────────────────────────────────────────────


def test_reflect_on_reads_a_transcript_and_scores_it_in_one_call(tmp_path):
    report = pkg.reflect_on(transcript(tmp_path), root=tmp_path)
    assert [c.text for c in report.to_commit] == [DECISION.strip()]


def test_reflect_on_labels_the_report_with_the_transcripts_name(tmp_path):
    assert pkg.reflect_on(transcript(tmp_path), root=tmp_path).session == "session"


def test_reflect_on_takes_an_explicit_session_label(tmp_path):
    report = pkg.reflect_on(
        transcript(tmp_path), root=tmp_path, session="phase/memory-writer"
    )
    assert report.session == "phase/memory-writer"


def test_reflect_on_refuses_a_transcript_that_is_not_there(tmp_path):
    # Not an empty report: a caller writing the result would otherwise read "this
    # session learned nothing" off a typo'd path.
    with pytest.raises(FileNotFoundError):
        pkg.reflect_on(tmp_path / "nope.jsonl", root=tmp_path)


def test_remember_writes_a_report_into_the_store(tmp_path):
    report = pkg.reflect_on(transcript(tmp_path), root=tmp_path)
    result = pkg.remember(report, root=tmp_path, created_at="2026-08-20T02:00:00+00:00")
    assert len(result.committed) == 1


def test_remember_stamps_entries_with_the_time_it_was_given(tmp_path):
    # The store never reads the clock so that two runs over one transcript
    # produce comparable files; a wrapper that swallowed the stamp would put the
    # clock back in the middle of the write path.
    report = pkg.reflect_on(transcript(tmp_path), root=tmp_path)
    pkg.remember(report, root=tmp_path, created_at="2026-08-20T02:00:00+00:00")
    assert pkg.recall(root=tmp_path)[0].created_at == "2026-08-20T02:00:00+00:00"


def test_recall_reads_back_what_remember_wrote(tmp_path):
    pkg.remember(pkg.reflect_on(transcript(tmp_path), root=tmp_path), root=tmp_path)
    assert [e.text for e in pkg.recall(root=tmp_path)] == [DECISION.strip()]


def test_recall_can_be_asked_for_one_domain(tmp_path):
    pkg.remember(pkg.reflect_on(transcript(tmp_path), root=tmp_path), root=tmp_path)
    assert pkg.recall(domains=("verification",), root=tmp_path) == ()


def test_recall_on_a_project_with_no_memory_yet_is_empty_not_an_error(tmp_path):
    # ORIENT calls this before anything else exists. Raising here would make
    # "memory before opinion" impossible to satisfy on a fresh checkout.
    assert pkg.recall(root=tmp_path) == ()


def test_reflect_on_sees_what_is_already_stored(tmp_path):
    # The convenience wrapper has to pass the store's entries to the proposer, or
    # a replacement can never claim the entry it retires.
    old = (
        "We decided to use ANTHROPIC_AUTH_TOKEN for the Claude path in "
        "`tars/providers/anthropic_provider.py`.\n"
    )
    first = transcript(tmp_path, old)
    pkg.remember(pkg.reflect_on(first, root=tmp_path), root=tmp_path)
    old_id = pkg.recall(root=tmp_path)[0].id

    second = tmp_path / "second.md"
    second.write_text(
        "The adapter in `tars/providers/anthropic_provider.py` no longer uses "
        "ANTHROPIC_AUTH_TOKEN.\n"
    )
    report = pkg.reflect_on(second, root=tmp_path)
    assert report.to_commit[0].supersedes == (old_id,)


# ── the command ───────────────────────────────────────────────────────────


def test_tars_memory_is_declared_as_a_console_script(scripts):
    assert "tars-memory" in scripts


def test_the_entry_point_resolves_to_a_callable(scripts):
    module_name, _, attr = scripts["tars-memory"].partition(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attr))


# ── structural ────────────────────────────────────────────────────────────


def test_importing_the_memory_writer_does_not_import_the_provider_layer():
    probe = (
        "import sys, tars.memory; "
        "print(any(m.startswith('tars.providers') for m in sys.modules))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_the_memory_writer_needs_no_third_party_package():
    # stdlib only, deliberately: this runs on a machine where the one working
    # provider path is behind a proxy and the SDKs are not all installed.
    probe = (
        "import sys, tars.memory; "
        "print(sorted({m.split('.')[0] for m in sys.modules} "
        "& {'httpx', 'anthropic', 'openai', 'pydantic', 'requests'}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"


# ── this package's own examples ───────────────────────────────────────────


def test_this_packages_own_docstring_examples_survive_the_readme_extractor():
    # The same self-check the readme phase runs on itself, pointed here. If
    # someone renames `reflect_on` and updates every caller but not this
    # docstring, the example stops resolving and this fails.
    readme = importlib.import_module("tars.readme")
    facts = readme.inspect_project(ROOT, max_examples=1000)
    own = [e for e in facts.examples if e.path == "tars/memory/__init__.py"]
    assert own, "the package docstring should carry usage examples"
    for example in own:
        assert example.verified, f"{example.location}: {example.verification}"
