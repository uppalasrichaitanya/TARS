"""The `tars-readme` command.

`main()` takes argv and streams, so almost everything here drives the real entry
point in-process. One test spawns `python -m tars.readme` for real, because
"runnable as a command" is a claim about the module being executable and only a
process proves it.

Two properties get pinned hardest, because both protect something a reader owns:

* **An existing README is never overwritten without `--force`.** This tool runs
  unattended. Silently replacing a hand-written README with a generated one
  destroys work that no test can restore.
* **`--check` never writes.** It is the CI mode; a checker with a side effect on
  the tree it is checking is a checker nobody can trust in a pipeline.
"""

import json
import subprocess
import sys
from io import StringIO

from tars.readme.cli import main

PYPROJECT = """\
[build-system]
requires = ["setuptools"]

[project]
name = "widget"
version = "2.1.0"
description = "Turns things into other things."
requires-python = ">=3.10"

[project.scripts]
widget = "widget.cli:main"
"""


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def make_repo(tmp_path):
    write(tmp_path / "pyproject.toml", PYPROJECT)
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
    write(
        tmp_path / "widget" / "cli.py",
        "import argparse\n\n\n"
        "def build_parser():\n"
        "    return argparse.ArgumentParser(prog='widget', description='Convert.')\n\n\n"
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


def run(*argv):
    out, err = StringIO(), StringIO()
    code = main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


# ── stdout is the default ─────────────────────────────────────────────────


def test_a_documentable_repo_prints_a_readme_to_stdout(tmp_path):
    repo = make_repo(tmp_path)
    code, out, _ = run(str(repo))
    assert code == 0
    assert out.startswith("# widget\n")
    assert "from widget import convert" in out


def test_nothing_is_written_to_the_tree_unless_output_is_asked_for(tmp_path):
    repo = make_repo(tmp_path)
    run(str(repo))
    assert not (repo / "README.md").exists()


def test_the_default_path_is_the_current_directory(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    monkeypatch.chdir(repo)
    code, out, _ = run()
    assert code == 0
    assert out.startswith("# widget\n")


def test_the_same_repo_renders_the_same_bytes_every_time(tmp_path):
    repo = make_repo(tmp_path)
    assert run(str(repo))[1] == run(str(repo))[1]


# ── writing a file ────────────────────────────────────────────────────────


def test_output_writes_the_readme_and_says_where(tmp_path):
    repo = make_repo(tmp_path)
    target = repo / "README.md"
    code, out, err = run(str(repo), "-o", str(target))
    assert code == 0
    assert target.read_text().startswith("# widget\n")
    # The markdown went to the file, so stdout stays clean for piping.
    assert out == ""
    assert "README.md" in err


def test_an_existing_readme_is_never_clobbered_without_force(tmp_path):
    repo = make_repo(tmp_path)
    target = write(repo / "README.md", "# Hand-written, do not lose\n")
    code, _, err = run(str(repo), "-o", str(target))
    assert code == 2
    assert target.read_text() == "# Hand-written, do not lose\n"
    assert "--force" in err


def test_force_overwrites_an_existing_readme(tmp_path):
    repo = make_repo(tmp_path)
    target = write(repo / "README.md", "# Old\n")
    code, _, _ = run(str(repo), "-o", str(target), "--force")
    assert code == 0
    assert target.read_text().startswith("# widget\n")


def test_writing_into_a_directory_that_does_not_exist_is_a_usage_error(tmp_path):
    repo = make_repo(tmp_path)
    code, _, err = run(str(repo), "-o", str(repo / "nope" / "README.md"))
    assert code == 2
    assert "Traceback" not in err


# ── --check ───────────────────────────────────────────────────────────────


def test_check_passes_when_the_file_on_disk_is_current(tmp_path):
    repo = make_repo(tmp_path)
    run(str(repo), "-o", str(repo / "README.md"))
    code, out, _ = run(str(repo), "--check")
    assert code == 0
    assert "up to date" in out.lower()


def test_check_fails_when_the_file_on_disk_is_stale(tmp_path):
    repo = make_repo(tmp_path)
    write(repo / "README.md", "# widget\n\nSomething else entirely.\n")
    code, out, _ = run(str(repo), "--check")
    assert code == 1
    assert "README.md" in out


def test_check_shows_what_differs_rather_than_only_that_it_differs(tmp_path):
    repo = make_repo(tmp_path)
    write(repo / "README.md", "# widget\n\nSomething else entirely.\n")
    code, out, _ = run(str(repo), "--check")
    assert code == 1
    assert "Something else entirely." in out


def test_a_missing_readme_counts_as_out_of_date(tmp_path):
    repo = make_repo(tmp_path)
    code, out, _ = run(str(repo), "--check")
    assert code == 1
    assert "does not exist" in out


def test_check_never_writes_the_file_it_is_checking(tmp_path):
    # A checker with a side effect on the tree it checks is one nobody can put
    # in a pipeline.
    repo = make_repo(tmp_path)
    run(str(repo), "--check")
    assert not (repo / "README.md").exists()


def test_check_looks_at_the_output_path_when_one_is_given(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "docs").mkdir()
    run(str(repo), "-o", str(repo / "docs" / "README.md"))
    assert run(str(repo), "--check", "-o", str(repo / "docs" / "README.md"))[0] == 0
    assert run(str(repo), "--check")[0] == 1


# ── exit codes ────────────────────────────────────────────────────────────


def test_a_repo_with_nothing_to_document_exits_three_not_zero(tmp_path):
    # 3, not 0: a CI step reading 0 as "documented" must not read an empty
    # directory as a documented one.
    code, _, err = run(str(tmp_path))
    assert code == 3
    assert "nothing to document" in err.lower()


def test_a_path_that_is_not_a_directory_is_a_usage_error(tmp_path):
    target = write(tmp_path / "file.txt", "x")
    code, _, err = run(str(target))
    assert code == 2
    assert "Traceback" not in err


def test_a_path_that_does_not_exist_is_a_usage_error(tmp_path):
    code, _, err = run(str(tmp_path / "nope"))
    assert code == 2
    assert "nope" in err
    assert "Traceback" not in err


def test_an_unknown_flag_is_a_usage_error_not_a_traceback(tmp_path):
    code, _, err = run(str(tmp_path), "--vibes")
    assert code == 2
    assert "Traceback" not in err


def test_check_and_json_together_are_refused_rather_than_guessed(tmp_path):
    repo = make_repo(tmp_path)
    code, _, err = run(str(repo), "--check", "--json")
    assert code == 2
    assert "--json" in err


# ── --json ────────────────────────────────────────────────────────────────


def test_json_emits_the_facts_with_their_provenance(tmp_path):
    repo = make_repo(tmp_path)
    code, out, _ = run(str(repo), "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["name"] == "widget"
    assert payload["examples"][0]["path"] == "widget/__init__.py"
    assert payload["examples"][0]["line"] > 0


def test_json_includes_the_exit_code_so_a_caller_reading_only_stdout_knows(tmp_path):
    repo = make_repo(tmp_path)
    payload = json.loads(run(str(repo), "--json")[1])
    assert payload["exit_code"] == 0


# ── selection flags ───────────────────────────────────────────────────────


def test_max_examples_caps_what_is_printed(tmp_path):
    repo = make_repo(tmp_path)
    text = run(str(repo), "--max-examples", "1")[1]
    assert text.count("```python") <= 1


def test_no_tests_leaves_test_derived_examples_out(tmp_path):
    repo = make_repo(tmp_path)
    assert "test_convert_returns_its_input" not in run(str(repo), "--no-tests")[1]
    assert "test_convert_returns_its_input" in run(str(repo))[1]


def test_notes_explain_what_the_tool_dropped_and_why(tmp_path):
    repo = make_repo(tmp_path)
    # A snippet that touches nothing in the project is dropped as irrelevant.
    # `--notes` is how a maintainer finds out that happened.
    write(
        repo / "widget" / "unrelated.py",
        '"""Unrelated.\n\n    import os\n\n    os.getcwd()\n"""\n',
    )
    loud = run(str(repo), "--notes")[1]
    assert "## Notes from generation" in loud
    assert "dropped" in loud
    assert "## Notes from generation" not in run(str(repo))[1]


def test_include_unverified_prints_stale_examples_labelled_as_such(tmp_path):
    repo = make_repo(tmp_path)
    write(
        repo / "widget" / "stale.py",
        '"""Stale.\n\n    from widget import gone\n\n    gone()\n"""\n',
    )
    plain = run(str(repo))[1]
    loud = run(str(repo), "--include-unverified")[1]
    assert "from widget import gone" not in plain
    assert "from widget import gone" in loud
    assert "unverified" in loud


# ── it really is a command ────────────────────────────────────────────────


def test_the_module_is_executable_as_a_command(tmp_path):
    repo = make_repo(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "tars.readme", str(repo)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("# widget\n")
