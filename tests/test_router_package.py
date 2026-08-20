"""The router's public surface, and the promises that make it cheap to call.

PHASE 2 runs before every task, including the trivial ones, so two properties
matter as much as accuracy: importing it must not drag in the provider layer,
and it must need nothing from PyPI.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from tars import router


def test_the_documented_entry_points_are_exported():
    for name in (
        "classify",
        "Classification",
        "load_dataset",
        "dev_split",
        "holdout_split",
    ):
        assert name in router.__all__, name


def test_every_exported_name_actually_resolves():
    for name in router.__all__:
        assert hasattr(router, name), name


def test_dunder_all_has_no_duplicates():
    assert len(router.__all__) == len(set(router.__all__))


def test_the_taxonomy_is_part_of_the_surface():
    """A caller switching on a tier needs the vocabulary, not a magic number."""
    assert router.BLAST_ORDER == ("low", "medium", "high")
    assert router.DEFAULT_TIER == 1


def test_classify_answers_all_three_questions_in_one_call():
    result = router.classify("Migrate the memory store to SQLite.")
    assert result.tier == 2
    assert result.blast == "high"
    assert result.reversible is False
    assert result.signals
    assert result.why


def test_the_labeled_set_is_reachable_for_anyone_wanting_to_re_measure():
    cases = router.load_dataset()
    assert len(cases) >= 40
    assert router.rubric()["tier"].keys() == {"0", "1", "2"}


def test_needs_care_flags_the_tasks_worth_pausing_on():
    assert router.classify("Rotate the AgentRouter credential.").needs_care is True
    assert router.classify("Fix the typo in the README.").needs_care is False


# ── the cost of calling it ───────────────────────────────────────────────


def _import_probe(statement: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        check=False,
    )


def test_importing_the_router_does_not_import_the_provider_layer():
    """CLASSIFY decides which model to use; it must not need a model to decide."""
    probe = _import_probe(
        "import sys, tars.router;"
        "assert 'tars.providers' not in sys.modules, sorted("
        "m for m in sys.modules if m.startswith('tars.'))"
    )
    assert probe.returncode == 0, probe.stderr


def test_the_router_needs_no_third_party_package():
    probe = _import_probe(
        "import sys, tars.router;"
        "bad = [m for m in ('anthropic', 'openai', 'httpx', 'pydantic')"
        " if m in sys.modules];"
        "assert not bad, bad"
    )
    assert probe.returncode == 0, probe.stderr


def test_classifying_is_cheap_enough_to_run_before_every_task():
    """Not a benchmark — a guard against someone making this call a model."""
    import time

    tasks = [case.task for case in router.load_dataset()]
    start = time.perf_counter()
    for task in tasks:
        router.classify(task)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"{len(tasks)} classifications took {elapsed:.3f}s"


# ── the data file has to survive packaging ───────────────────────────────


def test_the_labeled_set_is_declared_as_package_data():
    """`load_dataset` reads a JSON file out of the installed package.

    Without an explicit package-data entry it is present in the source tree and
    absent from the wheel, so this passes in development and fails only once
    someone installs it.
    """
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - py<3.11
        pytest.skip("tomllib needs Python 3.11+")

    with open("pyproject.toml", "rb") as handle:
        config = tomllib.load(handle)

    package_data = config["tool"]["setuptools"]["package-data"]
    patterns = package_data["tars.router"]
    assert any("json" in pattern for pattern in patterns), patterns
