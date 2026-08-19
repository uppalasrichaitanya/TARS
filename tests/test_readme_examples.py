"""Mining usage examples out of a repo, and refusing to trust them by default.

The rule this module exists to enforce: *never invent a usage example, pull them
from real working code paths*. So every snippet here is lifted verbatim, carries
the file and line it came from, and then has to survive validation against the
project's real API before anything prints it. A docstring example that rotted
two refactors ago is exactly as misleading as one that was made up.
"""

import ast

import pytest

from tars.readme.examples import (
    ProjectApi,
    extract_examples,
    extract_test_examples,
    validate_examples,
)
from tars.readme.source import SourceFile


def source(text, path="pkg/thing.py", module="pkg.thing"):
    return SourceFile(
        path=path, module=module, text=text, tree=ast.parse(text), is_package=False
    )


def api(**kwargs):
    defaults = {
        "modules": {"pkg": ("run",), "pkg.thing": ("run", "Thing")},
        "runnable": ("pkg",),
        "scripts": ("pkg-run",),
        "extras": ("dev",),
        "installable": True,
    }
    defaults.update(kwargs)
    return ProjectApi(**defaults)


def only(examples):
    assert len(examples) == 1, [e.code for e in examples]
    return examples[0]


# ── extraction: indented blocks ───────────────────────────────────────────


def test_an_indented_block_in_a_module_docstring_is_an_example():
    found = extract_examples(
        source('"""Does things.\n\n    from pkg import run\n\n    run()\n"""\n')
    )
    example = only(found)
    assert example.code == "from pkg import run\n\nrun()"
    assert example.language == "python"
    assert example.kind == "docstring"


def test_a_blank_line_does_not_split_one_block_into_two():
    # An import, a blank line, then the call is the single most common shape a
    # usage example takes. Splitting it produces a fragment with no import.
    found = extract_examples(
        source('"""T.\n\n    from pkg import run\n\n    run()\n    run()\n"""\n')
    )
    assert len(found) == 1


def test_the_reported_line_is_where_the_snippet_really_starts():
    text = '"""Title.\n\nProse here.\n\n    from pkg import run\n"""\n'
    example = only(extract_examples(source(text)))
    assert example.line == 5
    assert text.splitlines()[example.line - 1].strip() == "from pkg import run"


def test_a_docstring_indented_inside_a_function_still_reports_real_lines():
    text = 'def f():\n    """Doc.\n\n        from pkg import run\n    """\n'
    example = only(extract_examples(source(text)))
    assert example.line == 4


def test_prose_lists_are_not_mistaken_for_code():
    # Two-space "term — description" lists are prose. Requiring a real code
    # indent (4) is what keeps them out.
    found = extract_examples(
        source('"""T.\n\n  run   — does the thing\n  Thing — a thing\n"""\n')
    )
    assert found == ()


def test_a_bare_list_of_names_is_not_an_example():
    # `run` and `Thing` on their own lines parse as Python but demonstrate
    # nothing. An example has to import, call, assign, or access something.
    assert extract_examples(source('"""T.\n\n    run\n    Thing\n"""\n')) == ()


def test_an_indented_block_that_is_not_code_in_any_language_is_dropped():
    found = extract_examples(
        source('"""T.\n\n    Check       — a command we decided to run\n"""\n')
    )
    assert found == ()


# ── extraction: fences, doctests, shells, json ────────────────────────────


def test_a_fenced_block_is_an_example_and_the_fence_is_stripped():
    found = extract_examples(
        source('"""T.\n\n```python\nfrom pkg import run\n```\n"""\n')
    )
    assert only(found).code == "from pkg import run"


def test_a_doctest_block_keeps_its_prompts_and_is_marked_as_a_doctest():
    example = only(extract_examples(source('"""T.\n\n>>> run()\n42\n"""\n')))
    assert example.kind == "doctest"
    assert example.code == ">>> run()\n42"


def test_a_shell_transcript_is_tagged_console_not_python():
    # `python -m pkg --list` happens to be syntactically valid Python (a chain
    # of subtractions), so shell has to be recognised before Python is tried.
    example = only(extract_examples(source('"""T.\n\n    python -m pkg --list\n"""\n')))
    assert example.language == "console"


def test_a_projects_own_command_is_recognised_as_a_shell_line():
    example = only(
        extract_examples(
            source('"""T.\n\n    pkg-run /tmp/x --json\n"""\n'),
            commands=("pkg-run",),
        )
    )
    assert example.language == "console"


def test_a_json_snippet_is_tagged_json_even_though_it_parses_as_python():
    # {"a": null} is a valid Python dict display with a Name in it. JSON first.
    example = only(
        extract_examples(source('"""T.\n\n    {"checks": {"lint": null}}\n"""\n'))
    )
    assert example.language == "json"


def test_class_and_function_docstrings_are_mined_and_attributed():
    text = (
        '"""Module."""\n\n\n'
        "class Thing:\n"
        '    """A thing.\n\n        from pkg import Thing\n\n        Thing()\n    """\n'
    )
    example = only(extract_examples(source(text)))
    assert example.title == "pkg.thing.Thing"
    assert example.module == "pkg.thing"


def test_an_absurdly_long_block_is_not_pasted_into_a_readme():
    body = "\n".join(f"    run({i})" for i in range(80))
    found = extract_examples(source(f'"""T.\n\n{body}\n"""\n'))
    assert found == ()


# ── validation: python ────────────────────────────────────────────────────


def test_an_example_whose_imports_resolve_is_verified():
    found = extract_examples(
        source('"""T.\n\n    from pkg import run\n\n    run()\n"""\n')
    )
    example = only(validate_examples(found, api()))
    assert example.verified is True
    assert "pkg.run" in example.symbols


def test_an_example_importing_a_symbol_the_project_no_longer_exports_is_not_verified():
    found = extract_examples(
        source('"""T.\n\n    from pkg import gone\n\n    gone()\n"""\n')
    )
    example = only(validate_examples(found, api()))
    assert example.verified is False
    assert "gone" in example.verification


def test_an_example_importing_a_module_that_no_longer_exists_is_not_verified():
    found = extract_examples(
        source('"""T.\n\n    import pkg.removed\n\n    pkg.removed.f()\n"""\n')
    )
    assert validate_examples(found, api())[0].verified is False


def test_importing_a_submodule_by_name_resolves_against_the_module_list():
    found = extract_examples(
        source('"""T.\n\n    from pkg import thing\n\n    thing.run()\n"""\n')
    )
    assert validate_examples(found, api())[0].verified is True


def test_third_party_imports_are_left_alone(monkeypatch):
    # The tool cannot know whether `httpx` is installed without executing
    # something, and guessing would turn a fine example into a false negative.
    found = extract_examples(
        source(
            '"""T.\n\n    import httpx\n\n    from pkg import run\n\n    run()\n"""\n'
        )
    )
    assert validate_examples(found, api())[0].verified is True


def test_a_snippet_that_never_touches_the_project_is_dropped_as_irrelevant():
    found = extract_examples(source('"""T.\n\n    import os\n\n    os.getcwd()\n"""\n'))
    assert validate_examples(found, api()) == ()


def test_tests_that_exercise_a_symbol_strengthen_the_verification_note():
    found = extract_examples(
        source('"""T.\n\n    from pkg import run\n\n    run()\n"""\n')
    )
    verified = validate_examples(found, api(exercised=("pkg.run",)))
    assert "test" in only(verified).verification


# ── validation: console ───────────────────────────────────────────────────


def test_a_python_m_invocation_resolves_against_runnable_packages():
    found = extract_examples(source('"""T.\n\n    python -m pkg --list\n"""\n'))
    assert validate_examples(found, api())[0].verified is True


def test_a_python_m_invocation_of_a_package_with_no_main_is_not_verified():
    found = extract_examples(source('"""T.\n\n    python -m pkg.thing\n"""\n'))
    assert validate_examples(found, api())[0].verified is False


def test_a_console_script_the_project_declares_is_verified():
    found = extract_examples(
        source('"""T.\n\n    pkg-run --json\n"""\n'), commands=("pkg-run",)
    )
    assert validate_examples(found, api())[0].verified is True


def test_a_console_script_the_project_no_longer_declares_is_not_verified():
    found = extract_examples(
        source('"""T.\n\n    pkg-old --json\n"""\n'), commands=("pkg-old",)
    )
    assert validate_examples(found, api(scripts=("pkg-run",)))[0].verified is False


def test_an_install_line_naming_a_real_extra_is_verified():
    found = extract_examples(source('"""T.\n\n    pip install -e ".[dev]"\n"""\n'))
    assert validate_examples(found, api())[0].verified is True


def test_an_install_line_naming_an_extra_that_does_not_exist_is_not_verified():
    found = extract_examples(source('"""T.\n\n    pip install -e ".[gpu]"\n"""\n'))
    assert validate_examples(found, api())[0].verified is False


def test_a_generic_shell_line_with_nothing_project_specific_is_dropped():
    found = extract_examples(source('"""T.\n\n    pip install requests\n"""\n'))
    assert validate_examples(found, api()) == ()


# ── validation: syntax ────────────────────────────────────────────────────


def test_a_json_example_that_is_not_valid_json_never_gets_extracted():
    assert extract_examples(source('"""T.\n\n    {"broken": \n"""\n')) == ()


# ── test-derived examples ─────────────────────────────────────────────────

TEST_FILE = '''\
"""Tests."""

from pkg import run


def test_run_returns_the_answer():
    assert run() == 42


def test_run_with_a_fixture(tmp_path):
    assert run(tmp_path) == 42


def test_run_at_length():
    a = 1
    b = 2
    c = 3
    d = 4
    e = 5
    f = 6
    g = 7
    h = 8
    i = 9
    j = 10
    k = 11
    assert run() == a + b + c + d + e + f + g + h + i + j + k
'''


def test_a_short_fixtureless_test_becomes_an_example_with_its_imports():
    found = extract_test_examples(
        source(TEST_FILE, path="tests/test_run.py", module="tests.test_run")
    )
    example = only(found)
    assert example.kind == "test"
    assert example.code == "from pkg import run\n\nassert run() == 42"
    assert example.path == "tests/test_run.py"
    assert example.title == "test_run_returns_the_answer"


def test_a_test_that_needs_a_fixture_is_not_a_standalone_example():
    found = extract_test_examples(
        source(TEST_FILE, path="tests/test_run.py", module="tests.test_run")
    )
    assert all("tmp_path" not in e.code for e in found)


def test_a_long_test_is_not_a_standalone_example():
    found = extract_test_examples(
        source(TEST_FILE, path="tests/test_run.py", module="tests.test_run")
    )
    assert all("k = 11" not in e.code for e in found)


HELPER_TEST_FILE = """\
from pkg import run


def provider():
    return object(), object()


def test_run_against_a_helper():
    prov, client = provider()
    assert run(prov) == 42


def test_run_multiline():
    assert run() == [
        1,
        2,
    ]
"""


def test_a_test_that_leans_on_a_local_helper_is_not_a_standalone_example():
    # `prov, client = provider()` reads like usage and cannot be run: `provider`
    # is defined in the test file, not in the project. Copying it into a README
    # hands the reader a snippet that fails on line one.
    found = extract_test_examples(
        source(HELPER_TEST_FILE, path="tests/test_run.py", module="tests.test_run")
    )
    assert all("provider()" not in e.code for e in found)


def test_a_multiline_statement_keeps_its_shape_without_stray_indentation():
    # Joining per-statement source segments leaves continuation lines indented to
    # their original column inside the function, which renders as a broken block.
    found = extract_test_examples(
        source(HELPER_TEST_FILE, path="tests/test_run.py", module="tests.test_run")
    )
    example = only(found)
    assert example.code == "from pkg import run\n\nassert run() == [\n    1,\n    2,\n]"


def test_exercised_symbols_are_read_out_of_the_test_corpus():
    from tars.readme.examples import exercised_symbols

    found = exercised_symbols(
        [source(TEST_FILE, path="tests/test_run.py", module="tests.test_run")],
        packages=("pkg",),
    )
    assert "pkg.run" in found


def test_extraction_never_raises_on_a_docstring_full_of_junk():
    # The escapes are deliberately not resolved by this file: Python refuses to
    # compile source containing a real NUL, so the only way one reaches the
    # extractor is a docstring whose *value* decodes to one. `ast.parse` raises
    # ValueError rather than SyntaxError for that, which is easy to miss.
    weird = '"""T.\n\n    \\x00\\x01 not code\n\n\tmixed\ttabs\n"""\n'
    assert "\x00" in ast.parse(weird).body[0].value.value
    assert extract_examples(source(weird)) == ()


def test_an_api_can_be_built_from_nothing_without_blowing_up():
    assert validate_examples((), ProjectApi()) == ()


def test_validation_rejects_a_non_example_kind_loudly():
    with pytest.raises(ValueError):
        ProjectApi(modules={"pkg": "run"})  # public names must be a sequence
