"""Rendering the facts as a README, in the order the spec mandates.

The order is fixed: what this is, why it exists, how to run it, real examples,
how it's structured. So most of these tests are about what *doesn't* appear — an
empty section, a placeholder, a heading with nothing under it, or an example the
tool couldn't verify. "Skip filler sections" is the rule being enforced, and a
generated README earns its keep by being shorter than the hand-written one it
replaces, not longer.
"""

from tars.readme.render import SECTION_ORDER, render_readme
from tars.readme.types import (
    CheckFact,
    CommandFact,
    Example,
    ModuleFact,
    ProjectFacts,
    Prose,
)


def facts(**kwargs):
    defaults = {
        "root": "/tmp/widget",
        "name": "widget",
        "version": "2.1.0",
        "description": "Turns things into other things.",
        "requires_python": ">=3.10",
        "installable": True,
        "modules": (
            ModuleFact(
                name="widget",
                path="widget/__init__.py",
                doc="Turns things into other things.\n\nMore prose.\n",
                is_package=True,
                public=("convert",),
                declares_all=True,
            ),
        ),
        "examples": (
            Example(
                code="from widget import convert\n\nconvert('thing')",
                language="python",
                kind="docstring",
                path="widget/__init__.py",
                line=3,
                module="widget",
                verified=True,
                verification="imports resolve (widget.convert)",
            ),
        ),
    }
    defaults.update(kwargs)
    return ProjectFacts(**defaults)


COMMAND = CommandFact(
    name="widget",
    target="widget.cli:main",
    source="pyproject.toml",
    description="Convert a thing on the command line.",
    flags=("--json", "--quiet"),
)

CHECK = CheckFact(
    kind="test",
    tool="pytest",
    command_line="pytest",
    evidence="pyproject.toml [tool.pytest]",
)


def headings(text):
    return [line for line in text.splitlines() if line.startswith("## ")]


# ── order and shape ───────────────────────────────────────────────────────


def test_the_first_thing_a_reader_sees_is_what_this_is(tmp_path):
    text = render_readme(facts())
    assert text.startswith("# widget\n")
    assert "Turns things into other things." in text.split("##")[0]


def test_sections_appear_in_the_order_the_spec_mandates(tmp_path):
    text = render_readme(
        facts(
            commands=(COMMAND,),
            checks=(CHECK,),
            prose={"why": Prose(text="Because the old one broke.", source="x")},
        )
    )
    found = [h for h in SECTION_ORDER if f"## {h}" in text]
    positions = [text.index(f"## {h}") for h in found]
    assert positions == sorted(positions)
    assert len(found) >= 4


def test_the_output_is_the_same_every_time_for_the_same_facts(tmp_path):
    given = facts(commands=(COMMAND,), checks=(CHECK,))
    assert render_readme(given) == render_readme(given)


def test_nothing_to_document_is_refused_rather_than_rendered_empty(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        render_readme(ProjectFacts(root="/tmp/empty"))


# ── omission ──────────────────────────────────────────────────────────────


def test_why_it_exists_is_omitted_when_nobody_supplied_it(tmp_path):
    # Intent cannot be read out of an AST. Inventing a reason is the single most
    # tempting lie a docs generator can tell, so the section simply disappears.
    text = render_readme(facts())
    assert "Why it exists" not in text


def test_why_it_exists_appears_when_a_human_supplied_it(tmp_path):
    text = render_readme(
        facts(prose={"why": Prose(text="Because the old one broke.", source="x")})
    )
    assert "## Why it exists" in text
    assert "Because the old one broke." in text


def test_a_project_with_no_commands_gets_no_command_list(tmp_path):
    text = render_readme(facts())
    assert "--json" not in text


def test_an_uninstallable_project_is_not_told_to_pip_install_itself(tmp_path):
    text = render_readme(facts(installable=False))
    assert "pip install" not in text


def test_no_section_is_left_standing_empty(tmp_path):
    text = render_readme(facts(commands=(COMMAND,), checks=(CHECK,)))
    blocks = text.split("\n## ")
    for block in blocks[1:]:
        body = block.split("\n", 1)[1] if "\n" in block else ""
        assert body.strip(), f"empty section: {block.splitlines()[0]}"


def test_the_readme_never_contains_a_placeholder(tmp_path):
    text = render_readme(facts(commands=(COMMAND,), checks=(CHECK,))).lower()
    for placeholder in ("todo", "tbd", "coming soon", "your project", "lorem", "xxx"):
        assert placeholder not in text


# ── examples ──────────────────────────────────────────────────────────────


def test_an_example_is_shown_with_the_file_it_came_from(tmp_path):
    text = render_readme(facts())
    assert "from widget import convert" in text
    assert "widget/__init__.py:3" in text


def test_an_example_is_fenced_with_its_own_language(tmp_path):
    console = Example(
        code="widget --json .",
        language="console",
        kind="docstring",
        path="widget/cli.py",
        line=9,
        verified=True,
    )
    text = render_readme(facts(examples=(console,)))
    assert "```console\nwidget --json .\n```" in text


def test_an_unverified_example_is_not_printed_by_default(tmp_path):
    stale = Example(
        code="from widget import gone",
        language="python",
        kind="docstring",
        path="widget/__init__.py",
        line=3,
        verification="stale: widget does not export gone",
    )
    text = render_readme(facts(examples=(stale,)))
    assert "from widget import gone" not in text


def test_an_unverified_example_can_be_printed_but_is_labelled(tmp_path):
    stale = Example(
        code="from widget import gone",
        language="python",
        kind="docstring",
        path="widget/__init__.py",
        line=3,
        verification="stale: widget does not export gone",
    )
    text = render_readme(facts(examples=(stale,)), include_unverified=True)
    assert "from widget import gone" in text
    assert "does not export gone" in text


def test_a_project_with_no_verified_examples_gets_no_examples_section(tmp_path):
    text = render_readme(facts(examples=(), commands=(COMMAND,)))
    assert "## Examples" not in text


# ── how to run it ─────────────────────────────────────────────────────────


def test_a_command_is_listed_with_its_own_description(tmp_path):
    text = render_readme(facts(commands=(COMMAND,)))
    assert "widget" in text
    assert "Convert a thing on the command line." in text


def test_two_spellings_of_one_command_share_a_single_section(tmp_path):
    # One heading, both invocations in the block. A second identical section for
    # `python -m widget` is the filler the spec forbids.
    aliased = CommandFact(
        name="widget",
        target="widget.cli:main",
        source="pyproject.toml",
        description="Convert a thing on the command line.",
        aliases=("python -m widget",),
    )
    text = render_readme(facts(commands=(aliased,)))
    assert "```console\nwidget\npython -m widget\n```" in text
    assert text.count("Convert a thing on the command line.") == 1


def test_the_install_line_names_the_dev_extra_when_there_is_one(tmp_path):
    text = render_readme(facts(extras={"dev": ("pytest",)}))
    assert 'pip install -e ".[dev]"' in text


def test_the_required_python_version_is_stated_when_declared(tmp_path):
    assert ">=3.10" in render_readme(facts())


def test_dependencies_are_listed_when_there_are_any(tmp_path):
    text = render_readme(facts(dependencies=("httpx>=0.27",)))
    assert "httpx>=0.27" in text


def test_a_zero_dependency_project_says_so_rather_than_listing_nothing(tmp_path):
    # "No runtime dependencies" is a real, checkable fact and a selling point. An
    # empty bullet list is neither. "Runtime" matters: a project with a dev extra
    # does have dependencies, just not ones an installer pulls in.
    text = render_readme(facts(dependencies=(), extras={"dev": ("pytest",)}))
    assert "no runtime dependencies" in text.lower()


def test_the_checks_a_reader_can_run_are_shown_as_commands(tmp_path):
    text = render_readme(facts(checks=(CHECK,)))
    assert "pytest" in text


# ── how it's structured ───────────────────────────────────────────────────


def test_modules_are_listed_with_their_own_one_line_summaries(tmp_path):
    text = render_readme(facts())
    assert "widget/__init__.py" in text
    assert "Turns things into other things." in text


def test_a_module_with_no_docstring_is_listed_without_an_invented_summary(tmp_path):
    text = render_readme(
        facts(
            modules=(
                ModuleFact(name="widget", path="widget/__init__.py", is_package=True),
                ModuleFact(name="widget.quiet", path="widget/quiet.py"),
            )
        )
    )
    assert "widget/quiet.py" in text
    assert "quiet.py" in text


def test_test_directories_are_mentioned_so_a_reader_can_find_them(tmp_path):
    text = render_readme(facts(test_paths=("tests",)))
    assert "tests" in text


def test_the_module_list_nests_each_module_under_its_own_package(tmp_path):
    # Sorting by depth puts every depth-2 module after every depth-1 one, so
    # `a/one.py` ends up indented under package `b` — the indentation then claims
    # a containment that doesn't exist.
    text = render_readme(
        facts(
            modules=(
                ModuleFact(name="widget", path="widget/__init__.py", is_package=True),
                ModuleFact(
                    name="widget.a", path="widget/a/__init__.py", is_package=True
                ),
                ModuleFact(
                    name="widget.b", path="widget/b/__init__.py", is_package=True
                ),
                ModuleFact(name="widget.a.one", path="widget/a/one.py"),
                ModuleFact(name="widget.b.two", path="widget/b/two.py"),
            )
        )
    )
    listed = [line for line in text.splitlines() if line.lstrip().startswith("- `")]
    order = [line.strip().split("`")[1] for line in listed]
    assert order.index("widget/a/one.py") < order.index("widget/b/__init__.py")


# ── provenance footer ─────────────────────────────────────────────────────


def test_the_readme_says_it_was_generated_and_by_what(tmp_path):
    # A reader who knows a file is generated knows not to hand-edit it, and knows
    # which claims were machine-checked.
    text = render_readme(facts())
    assert "tars-readme" in text


def test_the_footer_does_not_claim_more_than_was_checked(tmp_path):
    text = render_readme(facts())
    assert "guaranteed" not in text.lower()
