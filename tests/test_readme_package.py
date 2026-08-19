"""The package surface, and one property the tool has to hold for itself.

The interesting test here is the self-referential one: this package's own
docstring examples get run through this package's own extractor, and every one
of them has to come out verified. That is the rule the whole phase exists to
enforce — *never invent usage examples, pull them from real working code paths* —
applied to the file most likely to rot quietly.

Also pinned: importing the readme writer must not drag in `tars.providers`. It
reads source and never executes it, so it needs no credentials and no SDKs.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import tars.readme as pkg

ROOT = Path(__file__).resolve().parent.parent

tomllib = pytest.importorskip(
    "tomllib", reason="tomllib is 3.11+; the check needs a TOML parser"
)


@pytest.fixture(scope="module")
def scripts():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle).get("project", {}).get("scripts", {})


# ── exports ───────────────────────────────────────────────────────────────


def test_the_documented_entry_points_are_exported():
    for name in ("readme_for", "inspect_project", "render_readme", "ProjectFacts"):
        assert name in pkg.__all__, name


def test_every_exported_name_actually_resolves():
    missing = [name for name in pkg.__all__ if not hasattr(pkg, name)]
    assert missing == []


def test_dunder_all_has_no_duplicates():
    assert len(pkg.__all__) == len(set(pkg.__all__))


def test_readme_for_produces_markdown_in_one_call(tmp_path):
    (tmp_path / "widget").mkdir()
    (tmp_path / "widget" / "__init__.py").write_text(
        '"""Turns things into other things."""\n\n\ndef convert(thing):\n    return thing\n'
    )
    text = pkg.readme_for(tmp_path)
    assert text.startswith("# widget\n")


def test_readme_for_refuses_a_directory_with_nothing_to_document(tmp_path):
    with pytest.raises(ValueError):
        pkg.readme_for(tmp_path)


# ── the command ───────────────────────────────────────────────────────────


def test_tars_readme_is_declared_as_a_console_script(scripts):
    assert "tars-readme" in scripts


def test_the_entry_point_resolves_to_a_callable(scripts):
    module_name, _, attr = scripts["tars-readme"].partition(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attr))


# ── structural ────────────────────────────────────────────────────────────


def test_importing_the_readme_writer_does_not_import_the_provider_layer():
    probe = (
        "import sys, tars.readme; "
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


# ── the tool's own examples ───────────────────────────────────────────────


def test_this_packages_own_docstring_examples_survive_its_own_extractor():
    # The self-check. If someone renames `readme_for` and updates every caller
    # but not this docstring, the example stops resolving and this fails —
    # which is exactly what the tool promises to catch in other people's code.
    facts = pkg.inspect_project(ROOT, max_examples=1000)
    own = [e for e in facts.examples if e.path == "tars/readme/__init__.py"]
    assert own, "the package docstring should carry usage examples"
    for example in own:
        assert example.verified, f"{example.location}: {example.verification}"


def test_every_example_mined_from_this_repo_appears_verbatim_in_its_source():
    # `path:line` under a snippet is a promise a reader can check by opening the
    # file, and it is only worth anything if the snippet is a copy. Dedenting is
    # the one allowed edit. Anything else — normalising whitespace, aligning
    # comments, running a formatter over the block — makes the citation a
    # paraphrase, and there is no way for a reader to tell which.
    facts = pkg.inspect_project(ROOT, max_examples=1000)
    for example in facts.examples:
        if example.kind == "doctest":  # `>>> ` prefixes are stripped, by design
            continue
        source = (ROOT / example.path).read_text(encoding="utf-8")
        for original in example.code.strip().splitlines():
            if not original.strip():
                continue
            assert original.strip() in source, (
                f"{example.location}: {original.strip()!r} is not in {example.path}"
            )
