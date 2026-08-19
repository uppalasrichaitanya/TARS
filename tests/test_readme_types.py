"""The shapes the readme writer speaks.

The load-bearing property here is provenance: every example knows the file and
line it was lifted from, and nothing is trusted until something proved it. An
example that can't say where it came from is indistinguishable from one the tool
made up, which is the exact failure the README rules exist to prevent.
"""

import pytest

from tars.readme.types import Example, ModuleFact, ProjectFacts


def test_an_example_reports_the_file_and_line_it_came_from():
    example = Example(
        code="from tars.readme import inspect_project",
        language="python",
        kind="docstring",
        path="tars/readme/__init__.py",
        line=3,
    )
    assert example.location == "tars/readme/__init__.py:3"


def test_an_example_with_no_code_is_a_bug_not_an_empty_fence():
    # Rendering an empty code fence would look like a usage example that
    # somehow takes no code. Better to fail where the extractor is wrong.
    with pytest.raises(ValueError):
        Example(code="   \n", language="python", kind="docstring", path="x.py", line=1)


def test_an_example_is_unverified_until_something_proves_it():
    example = Example(
        code="import tars", language="python", kind="docstring", path="x.py", line=1
    )
    assert example.verified is False


def test_fenced_renders_a_markdown_block_tagged_with_the_language():
    example = Example(
        code="tars-verify --list",
        language="console",
        kind="docstring",
        path="x",
        line=1,
    )
    assert example.fenced() == "```console\ntars-verify --list\n```"


def test_a_module_summary_is_the_first_line_of_its_docstring():
    module = ModuleFact(
        name="tars.verification",
        path="tars/verification/__init__.py",
        doc="Map a repo to its check surface.\n\nMore detail follows.\n",
        is_package=True,
    )
    assert module.summary == "Map a repo to its check surface."


def test_a_module_with_no_docstring_has_an_empty_summary():
    module = ModuleFact(name="tars", path="tars/__init__.py", doc="")
    assert module.summary == ""


def test_an_empty_project_is_not_documentable():
    # Exit code 3: "nothing here to document" must be distinguishable from
    # "here is a README", the same way no-checks is not pass in the gate.
    assert ProjectFacts(root="/tmp/x").documentable is False


def test_a_project_with_a_name_and_a_module_is_documentable():
    facts = ProjectFacts(
        root="/tmp/x",
        name="tars",
        modules=(ModuleFact(name="tars", path="tars/__init__.py", doc="A thing."),),
    )
    assert facts.documentable is True


def test_facts_serialize_with_example_provenance_intact():
    example = Example(
        code="import tars",
        language="python",
        kind="docstring",
        path="tars/__init__.py",
        line=7,
        verified=True,
        verification="imports resolve",
    )
    payload = ProjectFacts(root="/tmp/x", name="tars", examples=(example,)).to_dict()
    assert payload["examples"][0]["location"] == "tars/__init__.py:7"
    assert payload["examples"][0]["verified"] is True
    assert payload["examples"][0]["verification"] == "imports resolve"


def test_verified_examples_are_reachable_without_filtering_by_hand():
    good = Example(
        code="import tars",
        language="python",
        kind="docstring",
        path="a",
        line=1,
        verified=True,
    )
    stale = Example(
        code="import gone", language="python", kind="docstring", path="b", line=1
    )
    facts = ProjectFacts(root="/tmp/x", examples=(good, stale))
    assert facts.verified_examples == (good,)
    assert facts.unverified_examples == (stale,)
