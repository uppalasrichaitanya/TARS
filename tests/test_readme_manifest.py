"""Reading a project's declared metadata without depending on a TOML library.

`tomllib` only exists on Python 3.11+, and this package supports 3.10 (see
`requires-python`). So there are two paths, and the fallback has to be honest
about being a subset parser rather than pretend to be TOML.

The other rule here matches the verification gate: a malformed manifest degrades
to "no metadata, plus a note", never a crash. Plenty of real repos have a broken
pyproject.toml on some branch, and a docs tool that dies on one is useless
exactly when you want to look at the repo.
"""

import pytest

from tars.readme.manifest import parse_toml_subset, read_manifest


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


PYPROJECT = """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "tars"
version = "0.1.0"
description = "A disciplined coding agent loop"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8"]

[project.scripts]
tars-verify = "tars.verification.cli:main"
"""


def test_reads_the_declared_identity_of_a_project(tmp_path):
    write(tmp_path / "pyproject.toml", PYPROJECT)
    manifest = read_manifest(tmp_path)
    assert manifest.name == "tars"
    assert manifest.version == "0.1.0"
    assert manifest.description == "A disciplined coding agent loop"
    assert manifest.requires_python == ">=3.10"
    assert manifest.source == "pyproject.toml"


def test_reads_console_scripts_as_entry_points(tmp_path):
    write(tmp_path / "pyproject.toml", PYPROJECT)
    manifest = read_manifest(tmp_path)
    assert manifest.scripts == {"tars-verify": "tars.verification.cli:main"}


def test_reads_extras_so_install_instructions_can_be_real(tmp_path):
    write(tmp_path / "pyproject.toml", PYPROJECT)
    assert read_manifest(tmp_path).extras == {"dev": ("pytest>=8",)}


def test_a_build_backend_means_the_project_is_installable(tmp_path):
    write(tmp_path / "pyproject.toml", PYPROJECT)
    assert read_manifest(tmp_path).installable is True


def test_a_pyproject_without_a_build_system_is_not_called_installable(tmp_path):
    # `pip install -e .` is a claim about the repo. Only make it when the repo
    # declares a backend that could honour it.
    write(tmp_path / "pyproject.toml", "[project]\nname = 'x'\n")
    assert read_manifest(tmp_path).installable is False


def test_a_directory_with_no_manifest_yields_empty_metadata(tmp_path):
    manifest = read_manifest(tmp_path)
    assert manifest.name == ""
    assert manifest.scripts == {}
    assert manifest.notes == ()


def test_a_broken_manifest_degrades_to_a_note_instead_of_an_exception(tmp_path):
    write(tmp_path / "pyproject.toml", "[project\nname = ")
    manifest = read_manifest(tmp_path)
    assert manifest.name == ""
    assert any("pyproject.toml" in note for note in manifest.notes)


def test_poetry_style_metadata_is_read_from_its_own_table(tmp_path):
    write(
        tmp_path / "pyproject.toml",
        '[tool.poetry]\nname = "widget"\nversion = "2.0"\ndescription = "Widgets"\n',
    )
    manifest = read_manifest(tmp_path)
    assert (manifest.name, manifest.version) == ("widget", "2.0")
    assert manifest.description == "Widgets"


def test_package_json_metadata_is_read_when_there_is_no_pyproject(tmp_path):
    write(
        tmp_path / "package.json",
        '{"name": "webthing", "version": "1.2.3", "description": "A web thing",'
        ' "scripts": {"build": "tsc"}}',
    )
    manifest = read_manifest(tmp_path)
    assert manifest.name == "webthing"
    assert manifest.description == "A web thing"
    assert manifest.scripts == {"npm run build": "tsc"}
    assert manifest.source == "package.json"


def test_pyproject_wins_over_package_json_in_a_polyglot_repo(tmp_path):
    write(tmp_path / "pyproject.toml", PYPROJECT)
    write(tmp_path / "package.json", '{"name": "webthing"}')
    assert read_manifest(tmp_path).name == "tars"


# ── the no-tomllib fallback ───────────────────────────────────────────────
#
# Tested directly rather than through read_manifest: on 3.11+ read_manifest
# never reaches it, so going through the front door would leave the fallback
# untested on exactly the interpreters anyone runs the tests on.


def test_the_subset_parser_reads_tables_and_string_values():
    parsed = parse_toml_subset('[project]\nname = "tars"\nversion = "0.1.0"\n')
    assert parsed["project"]["name"] == "tars"
    assert parsed["project"]["version"] == "0.1.0"


def test_the_subset_parser_nests_dotted_table_headers():
    parsed = parse_toml_subset('[project.scripts]\ntars-verify = "tars.cli:main"\n')
    assert parsed["project"]["scripts"]["tars-verify"] == "tars.cli:main"


def test_the_subset_parser_reads_inline_and_multiline_lists():
    parsed = parse_toml_subset(
        "[project]\n"
        'dependencies = ["httpx", "rich"]\n'
        "keywords = [\n"
        '  "agent",\n'
        '  "loop",\n'
        "]\n"
    )
    assert parsed["project"]["dependencies"] == ["httpx", "rich"]
    assert parsed["project"]["keywords"] == ["agent", "loop"]


def test_the_subset_parser_ignores_comments_and_blank_lines():
    parsed = parse_toml_subset('# a comment\n[project]\n\nname = "x"  # trailing\n')
    assert parsed["project"] == {"name": "x"}


def test_the_subset_parser_rejects_input_it_cannot_understand():
    # Silently returning {} would make a real manifest look like an absent one.
    with pytest.raises(ValueError):
        parse_toml_subset("[project\nname = ")
