"""Detecting a Python repo's check surface.

Detection is filesystem-only and evidence-based on purpose: it never runs a
tool to find out whether the tool applies, and it never invents a linter a repo
gave no sign of using. Every candidate carries the marker that produced it.

Detection also does not decide availability — it can propose several candidates
per kind (ruff *then* flake8) and let the gate pick the one that's installed.
"""

import pytest

from tars.verification.detect import detect_checks, detect_ecosystems


def kinds_of(checks, kind):
    return [c for c in checks if c.kind == kind]


def tools_for(checks, kind):
    return [c.tool for c in checks if c.kind == kind]


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def fake_tool(root, name):
    """A tool installed in the repo's venv but declared nowhere."""
    binary = root / ".venv" / "bin" / name
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    return binary


# ── ecosystem gating ──────────────────────────────────────────────────────


def test_a_bare_directory_has_no_ecosystem_and_no_checks(tmp_path):
    write(tmp_path / "notes.txt", "hello")
    assert detect_ecosystems(tmp_path) == ()
    assert detect_checks(tmp_path) == ()


def test_pyproject_marks_a_python_repo(tmp_path):
    write(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    assert "python" in detect_ecosystems(tmp_path)


def test_requirements_file_marks_a_python_repo(tmp_path):
    write(tmp_path / "requirements.txt", "requests\n")
    assert "python" in detect_ecosystems(tmp_path)


def test_a_missing_root_is_an_error_not_an_empty_result(tmp_path):
    # Silently returning "no checks" for a typo'd path would read as
    # "this repo has nothing to verify". That must be impossible to confuse.
    with pytest.raises(FileNotFoundError):
        detect_checks(tmp_path / "nope")


# ── tests ─────────────────────────────────────────────────────────────────


def test_pytest_section_in_pyproject_detects_pytest(tmp_path):
    write(
        tmp_path / "pyproject.toml", "[tool.pytest.ini_options]\ntestpaths=['tests']\n"
    )
    check = kinds_of(detect_checks(tmp_path), "test")[0]
    assert check.tool == "pytest"
    assert check.command == ("pytest",)
    assert check.ecosystem == "python"
    assert "pyproject.toml" in check.evidence


def test_pytest_ini_detects_pytest(tmp_path):
    write(tmp_path / "pytest.ini", "[pytest]\n")
    assert tools_for(detect_checks(tmp_path), "test")[0] == "pytest"


def test_conftest_detects_pytest(tmp_path):
    write(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    write(tmp_path / "conftest.py")
    assert "pytest" in tools_for(detect_checks(tmp_path), "test")


def test_a_tests_directory_alone_proposes_pytest_then_unittest(tmp_path):
    # No pytest config anywhere: pytest is still the likely runner, but
    # unittest is stdlib and always installable, so it's the safety net.
    write(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    write(tmp_path / "tests" / "test_thing.py", "def test_x(): pass\n")
    assert tools_for(detect_checks(tmp_path), "test") == ["pytest", "unittest"]


def test_unittest_fallback_targets_the_tests_directory(tmp_path):
    write(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    write(tmp_path / "tests" / "test_thing.py", "def test_x(): pass\n")
    unittest = next(c for c in detect_checks(tmp_path) if c.tool == "unittest")
    assert unittest.command == (
        "python",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-t",
        ".",
    )


def test_a_python_repo_with_no_tests_gets_no_test_check(tmp_path):
    # Reporting "no test runner" is the honest answer; inventing `pytest` here
    # would make an empty repo look verified.
    write(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    assert kinds_of(detect_checks(tmp_path), "test") == []


# ── lint ──────────────────────────────────────────────────────────────────


def test_ruff_section_detects_ruff_check(tmp_path):
    write(tmp_path / "pyproject.toml", "[tool.ruff]\nline-length=88\n")
    lint = kinds_of(detect_checks(tmp_path), "lint")[0]
    assert lint.tool == "ruff"
    assert lint.command == ("ruff", "check", ".")


def test_ruff_toml_detects_ruff(tmp_path):
    write(tmp_path / "ruff.toml", "line-length=88\n")
    assert tools_for(detect_checks(tmp_path), "lint") == ["ruff"]


def test_a_declared_dependency_is_evidence(tmp_path):
    write(tmp_path / "requirements-dev.txt", "flake8==7.0.0\n")
    lint = kinds_of(detect_checks(tmp_path), "lint")[0]
    assert lint.tool == "flake8"
    assert "requirements-dev.txt" in lint.evidence


def test_a_tool_installed_in_the_repo_venv_is_evidence(tmp_path):
    # Extremely common: the repo uses ruff, its CI installs ruff, but nothing
    # in pyproject mentions it. Refusing to lint here would be pedantic.
    write(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    fake_tool(tmp_path, "ruff")
    lint = kinds_of(detect_checks(tmp_path), "lint")[0]
    assert lint.tool == "ruff"
    assert ".venv" in lint.evidence


def test_config_evidence_outranks_venv_evidence(tmp_path):
    write(tmp_path / ".flake8", "[flake8]\n")
    fake_tool(tmp_path, "ruff")
    assert tools_for(detect_checks(tmp_path), "lint") == ["flake8", "ruff"]


def test_no_linter_evidence_means_no_lint_check(tmp_path):
    write(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    assert kinds_of(detect_checks(tmp_path), "lint") == []


# ── format ────────────────────────────────────────────────────────────────


def test_ruff_format_is_check_only(tmp_path):
    # The gate must never rewrite the repo it is inspecting.
    write(tmp_path / "pyproject.toml", "[tool.ruff]\n")
    fmt = kinds_of(detect_checks(tmp_path), "format")[0]
    assert fmt.command == ("ruff", "format", "--check", ".")


def test_black_is_check_only(tmp_path):
    write(tmp_path / "pyproject.toml", "[tool.black]\n")
    fmt = kinds_of(detect_checks(tmp_path), "format")[0]
    assert fmt.tool == "black"
    assert "--check" in fmt.command


# ── typecheck ─────────────────────────────────────────────────────────────


def test_mypy_section_detects_mypy_with_explicit_targets(tmp_path):
    # Explicit targets keep mypy out of .venv, which `mypy .` would walk.
    write(tmp_path / "pyproject.toml", "[tool.mypy]\nstrict=true\n")
    write(tmp_path / "mypkg" / "__init__.py")
    write(tmp_path / "tests" / "test_x.py")
    check = kinds_of(detect_checks(tmp_path), "typecheck")[0]
    assert check.tool == "mypy"
    assert check.command == ("mypy", "mypkg", "tests")


def test_mypy_uses_src_layout_when_present(tmp_path):
    write(tmp_path / "pyproject.toml", "[tool.mypy]\n")
    write(tmp_path / "src" / "mypkg" / "__init__.py")
    check = kinds_of(detect_checks(tmp_path), "typecheck")[0]
    assert check.command == ("mypy", "src")


def test_mypy_falls_back_to_dot_when_no_package_is_obvious(tmp_path):
    write(tmp_path / "mypy.ini", "[mypy]\n")
    write(tmp_path / "script.py", "x = 1\n")
    check = kinds_of(detect_checks(tmp_path), "typecheck")[0]
    assert check.command == ("mypy", ".")


def test_pyright_config_detects_pyright(tmp_path):
    write(tmp_path / "pyrightconfig.json", "{}")
    assert tools_for(detect_checks(tmp_path), "typecheck") == ["pyright"]


def test_mypy_outranks_pyright_when_both_are_configured(tmp_path):
    write(tmp_path / "pyproject.toml", "[tool.mypy]\n")
    write(tmp_path / "pyrightconfig.json", "{}")
    assert tools_for(detect_checks(tmp_path), "typecheck") == ["mypy", "pyright"]


# ── build, and kind filtering ─────────────────────────────────────────────


def test_build_is_detected_but_not_selected_by_default(tmp_path):
    write(tmp_path / "pyproject.toml", "[build-system]\nrequires=['setuptools']\n")
    assert tools_for(detect_checks(tmp_path), "build") == ["build"]
    # ...and asking for the default kinds leaves it out, because it writes dist/.
    from tars.verification.types import DEFAULT_KINDS

    default = detect_checks(tmp_path, kinds=DEFAULT_KINDS)
    assert kinds_of(default, "build") == []


def test_kinds_filter_narrows_detection(tmp_path):
    write(tmp_path / "pyproject.toml", "[tool.ruff]\n[tool.pytest.ini_options]\n")
    only_test = detect_checks(tmp_path, kinds=("test",))
    assert {c.kind for c in only_test} == {"test"}


def test_checks_come_back_cheapest_signal_first(tmp_path):
    write(
        tmp_path / "pyproject.toml",
        "[tool.ruff]\n[tool.mypy]\n[tool.pytest.ini_options]\n",
    )
    assert [c.kind for c in detect_checks(tmp_path)] == [
        "lint",
        "format",
        "typecheck",
        "test",
    ]


def test_detection_ignores_the_venv_directory_as_a_package(tmp_path):
    write(tmp_path / "pyproject.toml", "[tool.mypy]\n")
    write(tmp_path / ".venv" / "lib" / "__init__.py")
    write(tmp_path / "mypkg" / "__init__.py")
    check = kinds_of(detect_checks(tmp_path), "typecheck")[0]
    assert check.command == ("mypy", "mypkg")
