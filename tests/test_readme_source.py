"""Finding the source files worth reading, without importing any of them.

Everything downstream reads this corpus, so two properties matter more than
breadth:

* Nothing here imports or executes the project. A docs tool pointed at an
  unfamiliar checkout must not run its code, and `ast` gives us docstrings and
  signatures without doing so. This is the same "filesystem only" rule the
  verification gate's detection follows.
* Tests are collected as their own corpus rather than mixed into the modules.
  They are the best evidence that a call really works, and the worst thing to
  document as public API — so they have to be distinguishable.
"""

from tars.readme.source import collect_sources


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def names(sources):
    return sorted(s.module for s in sources)


def test_a_package_and_its_submodules_get_dotted_names(tmp_path):
    write(tmp_path / "pkg" / "__init__.py", '"""Top."""\n')
    write(tmp_path / "pkg" / "core.py", '"""Core."""\n')
    write(tmp_path / "pkg" / "sub" / "__init__.py", "")
    write(tmp_path / "pkg" / "sub" / "deep.py", "")

    found = collect_sources(tmp_path)
    assert names(found.modules) == ["pkg", "pkg.core", "pkg.sub", "pkg.sub.deep"]


def test_a_src_layout_does_not_leak_src_into_module_names(tmp_path):
    write(tmp_path / "src" / "pkg" / "__init__.py", "")
    write(tmp_path / "src" / "pkg" / "core.py", "")
    assert names(collect_sources(tmp_path).modules) == ["pkg", "pkg.core"]


def test_a_package_marks_itself_as_a_package(tmp_path):
    write(tmp_path / "pkg" / "__init__.py", "")
    write(tmp_path / "pkg" / "core.py", "")
    by_name = {s.module: s for s in collect_sources(tmp_path).modules}
    assert by_name["pkg"].is_package is True
    assert by_name["pkg.core"].is_package is False


def test_a_main_module_makes_its_package_runnable(tmp_path):
    # `python -m pkg` is a real invocation only when __main__.py exists.
    write(tmp_path / "pkg" / "__init__.py", "")
    write(tmp_path / "pkg" / "__main__.py", "")
    found = collect_sources(tmp_path)
    assert found.runnable == ("pkg",)


def test_a_package_without_a_main_module_is_not_runnable(tmp_path):
    write(tmp_path / "pkg" / "__init__.py", "")
    assert collect_sources(tmp_path).runnable == ()


def test_environments_and_build_output_are_never_walked(tmp_path):
    write(tmp_path / "pkg" / "__init__.py", "")
    for junk in (
        ".venv/lib/python3.12/site-packages/requests/__init__.py",
        "node_modules/thing/index.py",
        "build/lib/pkg/__init__.py",
        "dist/pkg/__init__.py",
        "pkg/__pycache__/core.py",
        ".git/hooks/thing.py",
    ):
        write(tmp_path / junk, "")
    assert names(collect_sources(tmp_path).modules) == ["pkg"]


def test_tests_are_collected_separately_from_modules(tmp_path):
    write(tmp_path / "pkg" / "__init__.py", "")
    write(tmp_path / "tests" / "test_core.py", "")
    found = collect_sources(tmp_path)
    assert names(found.modules) == ["pkg"]
    assert [s.path for s in found.tests] == ["tests/test_core.py"]
    assert found.test_paths == ("tests",)


def test_a_test_file_inside_a_package_is_still_a_test(tmp_path):
    write(tmp_path / "pkg" / "__init__.py", "")
    write(tmp_path / "pkg" / "test_core.py", "")
    found = collect_sources(tmp_path)
    assert names(found.modules) == ["pkg"]
    assert [s.path for s in found.tests] == ["pkg/test_core.py"]


def test_a_single_file_tool_is_a_module_when_the_repo_has_no_packages(tmp_path):
    write(tmp_path / "tool.py", '"""A tool."""\n')
    assert names(collect_sources(tmp_path).modules) == ["tool"]


def test_packaging_scaffolding_is_not_documented_as_a_module(tmp_path):
    # setup.py and conftest.py are build/test plumbing. Listing them under
    # "how it's structured" tells a reader nothing about the project.
    write(tmp_path / "setup.py", "")
    write(tmp_path / "conftest.py", "")
    write(tmp_path / "tool.py", "")
    assert names(collect_sources(tmp_path).modules) == ["tool"]


def test_loose_scripts_are_ignored_once_a_real_package_exists(tmp_path):
    write(tmp_path / "pkg" / "__init__.py", "")
    write(tmp_path / "scratch.py", "")
    assert names(collect_sources(tmp_path).modules) == ["pkg"]


def test_a_file_that_does_not_parse_is_noted_and_skipped(tmp_path):
    write(tmp_path / "pkg" / "__init__.py", "")
    write(tmp_path / "pkg" / "broken.py", "def oops(:\n")
    found = collect_sources(tmp_path)
    assert names(found.modules) == ["pkg"]
    assert any("pkg/broken.py" in note for note in found.notes)


def test_a_source_file_exposes_its_docstring_and_parsed_tree(tmp_path):
    write(tmp_path / "pkg" / "__init__.py", '"""Top line.\n\nMore.\n"""\n')
    source = collect_sources(tmp_path).modules[0]
    assert source.doc.startswith("Top line.")
    assert source.tree is not None
