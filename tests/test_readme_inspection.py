"""Composing one inspection pass: manifest + sources + examples + checks.

This is where the tool decides what it *knows*, and the two rules it has to hold
are the same two the spec sets: nothing is invented, and everything printed can
be traced back to a file. So the tests here mostly police provenance — a name
that came from nowhere, a command that isn't declared, or prose the tool wrote
itself would all be defects.
"""

import json

import pytest

from tars.readme.inspection import (
    PROSE_PATH,
    inspect_project,
    public_names,
    read_prose,
    select_examples,
)
from tars.readme.types import Example


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


PYPROJECT = """\
[build-system]
requires = ["setuptools"]

[project]
name = "widget"
version = "2.1.0"
description = "Turns things into other things."
requires-python = ">=3.10"
dependencies = ["httpx"]

[project.optional-dependencies]
dev = ["pytest"]

[project.scripts]
widget = "widget.cli:main"
"""


def make_repo(tmp_path, *, manifest=PYPROJECT):
    if manifest:
        write(tmp_path / "pyproject.toml", manifest)
    write(
        tmp_path / "widget" / "__init__.py",
        '"""Turns things into other things.\n\n'
        "    from widget import convert\n\n"
        '    convert("thing")\n'
        '"""\n\n'
        '__all__ = ["convert"]\n\n\n'
        "def convert(thing):\n"
        "    return thing\n",
    )
    write(tmp_path / "widget" / "__main__.py", "from .cli import main\n\nmain()\n")
    write(
        tmp_path / "widget" / "cli.py",
        "import argparse\n\n\n"
        "def build_parser():\n"
        "    parser = argparse.ArgumentParser(\n"
        '        prog="widget", description="Convert a thing on the command line."\n'
        "    )\n"
        '    parser.add_argument("path")\n'
        '    parser.add_argument("--json", action="store_true", help="machine output")\n'
        '    parser.add_argument("-q", "--quiet", action="store_true")\n'
        "    return parser\n\n\n"
        "def main(argv=None):\n"
        "    return 0\n",
    )
    write(
        tmp_path / "tests" / "test_convert.py",
        "from widget import convert\n\n\n"
        "def test_convert_returns_its_input():\n"
        '    assert convert("x") == "x"\n',
    )
    return tmp_path


# ── identity ──────────────────────────────────────────────────────────────


def test_identity_comes_from_the_manifest(tmp_path):
    facts = inspect_project(make_repo(tmp_path))
    assert facts.name == "widget"
    assert facts.version == "2.1.0"
    assert facts.description == "Turns things into other things."
    assert facts.requires_python == ">=3.10"
    assert facts.installable is True


def test_a_repo_with_no_manifest_is_named_after_its_only_package(tmp_path):
    # Falling back to the package name keeps the name sourced from the tree. The
    # alternative — leaving it blank — makes a perfectly documentable repo look
    # like it has nothing to say.
    write(tmp_path / "widget" / "__init__.py", '"""Does things."""\n')
    facts = inspect_project(tmp_path)
    assert facts.name == "widget"
    assert facts.documentable is True


def test_an_empty_directory_has_nothing_to_document(tmp_path):
    facts = inspect_project(tmp_path)
    assert facts.documentable is False
    assert facts.modules == ()


def test_a_path_that_is_not_a_directory_is_an_error_not_an_empty_report(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_project(tmp_path / "nope")


# ── modules ───────────────────────────────────────────────────────────────


def test_modules_carry_their_own_docstring_summary(tmp_path):
    facts = inspect_project(make_repo(tmp_path))
    top = {m.name: m for m in facts.modules}["widget"]
    assert top.summary == "Turns things into other things."
    assert top.is_package is True


def test_a_declared_all_is_the_public_surface(tmp_path):
    facts = inspect_project(make_repo(tmp_path))
    top = {m.name: m for m in facts.modules}["widget"]
    assert top.public == ("convert",)
    assert top.declares_all is True


def test_without_all_the_public_surface_is_the_public_definitions(tmp_path):
    write(
        tmp_path / "widget" / "__init__.py",
        "def public():\n    pass\n\n\ndef _private():\n    pass\n\n\nclass Thing:\n    pass\n",
    )
    facts = inspect_project(tmp_path)
    top = {m.name: m for m in facts.modules}["widget"]
    assert top.public == ("public", "Thing")
    assert top.declares_all is False


def test_a_package_with_a_main_module_is_marked_runnable(tmp_path):
    facts = inspect_project(make_repo(tmp_path))
    top = {m.name: m for m in facts.modules}["widget"]
    assert top.runnable is True


def test_public_names_reads_all_even_when_it_is_a_tuple(tmp_path):
    from tars.readme.source import collect_sources

    write(tmp_path / "widget" / "__init__.py", '__all__ = ("a", "b")\n')
    source = collect_sources(tmp_path).modules[0]
    assert public_names(source) == (("a", "b"), True)


def test_public_names_survives_an_all_that_is_computed(tmp_path):
    # `__all__ = [*_BASE, "extra"]` cannot be read statically. Falling back to
    # the definitions beats crashing, and beats claiming an empty surface.
    from tars.readme.source import collect_sources

    write(
        tmp_path / "widget" / "__init__.py",
        '__all__ = [*_BASE, "extra"]\n\n\ndef visible():\n    pass\n',
    )
    source = collect_sources(tmp_path).modules[0]
    names, declared = public_names(source)
    assert names == ("visible",)
    assert declared is False


# ── commands ──────────────────────────────────────────────────────────────


def test_a_console_script_becomes_a_command_with_its_declaration_site(tmp_path):
    facts = inspect_project(make_repo(tmp_path))
    command = {c.name: c for c in facts.commands}["widget"]
    assert command.target == "widget.cli:main"
    assert command.source == "pyproject.toml"


def test_a_commands_description_and_flags_are_read_from_its_own_parser(tmp_path):
    # Not invented, and not guessed from the function name: this is the string
    # the command itself prints under --help.
    facts = inspect_project(make_repo(tmp_path))
    command = {c.name: c for c in facts.commands}["widget"]
    assert command.description == "Convert a thing on the command line."
    assert "--json" in command.flags


def test_a_runnable_package_is_offered_as_python_dash_m(tmp_path):
    write(tmp_path / "widget" / "__init__.py", "")
    write(tmp_path / "widget" / "__main__.py", "print(1)\n")
    names = [c.name for c in inspect_project(tmp_path).commands]
    assert "python -m widget" in names


def test_a_script_and_the_python_m_form_of_it_are_one_command_not_two(tmp_path):
    # `widget` and `python -m widget` run the same code, print the same --help
    # and take the same flags. Two identical sections is the filler the spec
    # forbids, and it makes a reader hunt for the difference between them.
    facts = inspect_project(make_repo(tmp_path))
    assert [c.name for c in facts.commands] == ["widget"]
    assert facts.commands[0].aliases == ("python -m widget",)


def test_a_runnable_package_no_script_points_at_stays_its_own_command(tmp_path):
    # Merging is evidence-based: the script's target has to live inside the
    # package. `other.cli:main` says nothing about `python -m widget`.
    write(
        tmp_path / "pyproject.toml",
        PYPROJECT.replace('widget = "widget.cli:main"', 'other = "other.cli:main"'),
    )
    write(tmp_path / "widget" / "__init__.py", "")
    write(tmp_path / "widget" / "__main__.py", "print(1)\n")
    write(tmp_path / "other" / "__init__.py", "")
    write(tmp_path / "other" / "cli.py", "def main():\n    return 0\n")
    names = [c.name for c in inspect_project(tmp_path).commands]
    assert "python -m widget" in names


def test_flag_aliases_are_reported_as_one_flag_with_two_spellings(tmp_path):
    # `-q, --quiet` is one option. Listing both doubles the flag line and
    # implies the command has twice as many switches as it does.
    facts = inspect_project(make_repo(tmp_path))
    command = {c.name: c for c in facts.commands}["widget"]
    assert "-q/--quiet" in command.flags
    assert "-q" not in command.flags


def test_a_python_dash_m_command_points_at_a_file_that_exists(tmp_path):
    # Composing "widget.sub/__main__.py" from the dotted name produces a string
    # that looks like a path and isn't one. The path has to be derived from the
    # package's real location, which is the only thing that survives a nested
    # package or a src/ layout.
    write(tmp_path / "widget" / "__init__.py", "")
    write(tmp_path / "widget" / "sub" / "__init__.py", "")
    write(tmp_path / "widget" / "sub" / "__main__.py", "print(1)\n")
    command = {c.name: c for c in inspect_project(tmp_path).commands}[
        "python -m widget.sub"
    ]
    assert command.source == "widget/sub/__main__.py"
    assert (tmp_path / command.source).is_file()


def test_a_python_dash_m_description_is_found_by_evidence_not_by_naming(tmp_path):
    # The parser lives in `console.py`, not `cli.py`. Looking for the module that
    # actually builds an ArgumentParser finds it; guessing the conventional name
    # does not.
    write(tmp_path / "widget" / "__init__.py", "")
    write(tmp_path / "widget" / "__main__.py", "from .console import main\n\nmain()\n")
    write(
        tmp_path / "widget" / "console.py",
        "import argparse\n\n\n"
        "def main():\n"
        '    parser = argparse.ArgumentParser(description="Runs the widget.")\n'
        '    parser.add_argument("--loud", action="store_true")\n'
        "    return 0\n",
    )
    command = {c.name: c for c in inspect_project(tmp_path).commands}[
        "python -m widget"
    ]
    assert command.description == "Runs the widget."
    assert command.flags == ("--loud",)


def test_a_command_pointing_at_a_module_that_does_not_exist_is_still_reported(tmp_path):
    # The manifest declares it, so a reader will find it. Silently dropping it
    # would hide a real problem; it just has no description to show.
    write(
        tmp_path / "pyproject.toml",
        PYPROJECT.replace('widget = "widget.cli:main"', 'widget = "widget.gone:main"'),
    )
    write(tmp_path / "widget" / "__init__.py", "")
    facts = inspect_project(tmp_path)
    command = {c.name: c for c in facts.commands}["widget"]
    assert command.description == ""


# ── examples ──────────────────────────────────────────────────────────────


def test_examples_are_mined_from_the_repo_and_verified_against_it(tmp_path):
    facts = inspect_project(make_repo(tmp_path))
    docstring = [e for e in facts.examples if e.kind == "docstring"]
    assert docstring, [e.to_dict() for e in facts.examples]
    assert docstring[0].verified is True
    assert docstring[0].path == "widget/__init__.py"


def test_a_stale_example_is_kept_but_never_marked_verified(tmp_path):
    write(
        tmp_path / "widget" / "__init__.py",
        '"""Old.\n\n    from widget import removed\n\n    removed()\n"""\n',
    )
    facts = inspect_project(tmp_path)
    assert facts.verified_examples == ()
    assert len(facts.unverified_examples) == 1
    assert any("unverified" in note or "stale" in note for note in facts.notes)


def test_tests_supply_examples_too_but_rank_below_docstrings(tmp_path):
    facts = inspect_project(make_repo(tmp_path))
    kinds = [e.kind for e in facts.examples]
    assert "test" in kinds
    assert kinds.index("docstring") < kinds.index("test")


def test_test_derived_examples_can_be_left_out_entirely(tmp_path):
    facts = inspect_project(make_repo(tmp_path), include_tests=False)
    assert all(e.kind != "test" for e in facts.examples)


def test_a_repo_full_of_examples_does_not_produce_a_readme_full_of_them(tmp_path):
    facts = inspect_project(make_repo(tmp_path), max_examples=1)
    assert len(facts.examples) == 1


def test_selection_prefers_verified_examples_over_unverified_ones(tmp_path):
    good = Example(
        code="from widget import convert",
        language="python",
        kind="docstring",
        path="a.py",
        line=1,
        verified=True,
    )
    bad = Example(
        code="from widget import gone",
        language="python",
        kind="docstring",
        path="b.py",
        line=1,
    )
    assert select_examples((bad, good), limit=1) == (good,)


def test_selection_drops_a_snippet_that_repeats_one_already_chosen(tmp_path):
    first = Example(
        code="from widget import convert\n\nconvert()",
        language="python",
        kind="docstring",
        path="a.py",
        line=1,
        verified=True,
    )
    same = Example(
        code="from widget import convert\nconvert()",
        language="python",
        kind="test",
        path="tests/test_a.py",
        line=9,
        verified=True,
    )
    assert select_examples((first, same), limit=5) == (first,)


def test_a_config_fragment_ranks_below_a_snippet_that_calls_something(tmp_path):
    # A JSON override file is supporting material; a call is how the project is
    # used. With room for one example, the call is the one worth printing.
    config = Example(
        code='{"why": "because"}',
        language="json",
        kind="docstring",
        path="a.py",
        line=1,
        verified=True,
    )
    usage = Example(
        code="from widget import convert\n\nconvert()",
        language="python",
        kind="docstring",
        path="z.py",
        line=9,
        verified=True,
    )
    assert select_examples((config, usage), limit=1) == (usage,)


# ── checks ────────────────────────────────────────────────────────────────


def test_checks_come_from_the_verification_gate_with_their_evidence(tmp_path):
    facts = inspect_project(make_repo(tmp_path))
    kinds = {c.kind for c in facts.checks}
    assert "test" in kinds
    test_check = next(c for c in facts.checks if c.kind == "test")
    assert test_check.command_line
    assert test_check.evidence


def test_the_checks_shown_are_the_ones_the_gate_would_actually_run(tmp_path):
    # `build` is opt-in in the gate (ALL_KINDS, not DEFAULT_KINDS). A README
    # telling a reader to run a command `tars-verify` doesn't run is describing
    # a verification surface this project doesn't have.
    facts = inspect_project(make_repo(tmp_path))
    assert "build" not in {c.kind for c in facts.checks}


def test_the_ecosystems_the_gate_recognised_are_recorded(tmp_path):
    facts = inspect_project(make_repo(tmp_path))
    assert "python" in facts.ecosystems


def test_test_directories_are_reported_as_found_on_disk(tmp_path):
    facts = inspect_project(make_repo(tmp_path))
    assert facts.test_paths == ("tests",)


# ── prose ─────────────────────────────────────────────────────────────────


def test_human_written_prose_is_read_from_the_override_file(tmp_path):
    # The tool never writes prose. "Why it exists" is a claim about intent that
    # no amount of AST reading can recover, so it has to be supplied.
    repo = make_repo(tmp_path)
    write(
        repo / PROSE_PATH,
        json.dumps({"why": "Because the old converter was unmaintainable."}),
    )
    facts = inspect_project(repo)
    assert facts.prose["why"].text.startswith("Because the old converter")
    assert facts.prose["why"].source == PROSE_PATH


def test_an_unknown_prose_section_is_noted_rather_than_silently_ignored(tmp_path):
    repo = make_repo(tmp_path)
    write(repo / PROSE_PATH, json.dumps({"wat": "hello"}))
    facts = inspect_project(repo)
    assert "wat" not in facts.prose
    assert any("wat" in note for note in facts.notes)


def test_a_broken_prose_file_is_a_note_not_a_crash(tmp_path):
    repo = make_repo(tmp_path)
    write(repo / PROSE_PATH, "{not json")
    facts = inspect_project(repo)
    assert facts.name == "widget"
    assert any(PROSE_PATH in note for note in facts.notes)


def test_prose_that_is_not_a_string_is_rejected(tmp_path):
    repo = make_repo(tmp_path)
    write(repo / PROSE_PATH, json.dumps({"why": ["a", "list"]}))
    facts = inspect_project(repo)
    assert "why" not in facts.prose


def test_read_prose_on_a_repo_with_no_override_file_is_empty_and_quiet(tmp_path):
    prose, notes = read_prose(tmp_path)
    assert prose == {}
    assert notes == ()


# ── notes ─────────────────────────────────────────────────────────────────


def test_a_file_that_could_not_be_parsed_is_reported_in_the_notes(tmp_path):
    repo = make_repo(tmp_path)
    write(repo / "widget" / "broken.py", "def oops(:\n")
    facts = inspect_project(repo)
    assert any("broken.py" in note for note in facts.notes)


def test_a_degraded_manifest_parse_is_reported_in_the_notes(tmp_path):
    write(tmp_path / "pyproject.toml", "[project\nname = 'x'\n")
    write(tmp_path / "widget" / "__init__.py", "")
    facts = inspect_project(tmp_path)
    assert any("pyproject.toml" in note for note in facts.notes)
