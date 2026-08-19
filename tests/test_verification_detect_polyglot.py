"""Go, Rust, and the Makefile fallback.

Three deliberate omissions are pinned here as tests, because they look like gaps
until you know why:

- No Go format check. `gofmt -l .` lists unformatted files and still exits 0, so
  a gate built on exit codes would report a permanent pass. A check that cannot
  fail is worse than no check.
- Clippy only with config evidence. `cargo clippy` on a toolchain without the
  component exits non-zero, which the gate would report as a lint *failure*
  rather than a missing tool.
- No pre-commit. Its hooks routinely rewrite the files they inspect, and the
  gate must not mutate the tree it's checking.
"""

from tars.verification.detect import detect_checks, detect_ecosystems


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def for_kind(checks, kind):
    return [c for c in checks if c.kind == kind]


def commands(checks, kind):
    return [c.command for c in for_kind(checks, kind)]


# ── go ────────────────────────────────────────────────────────────────────


def test_go_mod_marks_a_go_repo(tmp_path):
    write(tmp_path / "go.mod", "module example.com/x\n")
    assert "go" in detect_ecosystems(tmp_path)


def test_go_test_and_vet_are_detected(tmp_path):
    write(tmp_path / "go.mod", "module example.com/x\n")
    checks = detect_checks(tmp_path)
    assert commands(checks, "test") == [("go", "test", "./...")]
    assert commands(checks, "typecheck") == [("go", "vet", "./...")]


def test_go_has_no_format_check_because_gofmt_always_exits_zero(tmp_path):
    write(tmp_path / "go.mod", "module example.com/x\n")
    assert for_kind(detect_checks(tmp_path), "format") == []


def test_golangci_config_detects_the_linter(tmp_path):
    write(tmp_path / "go.mod", "module example.com/x\n")
    write(tmp_path / ".golangci.yml", "linters:\n")
    assert commands(detect_checks(tmp_path), "lint") == [("golangci-lint", "run")]


def test_go_without_a_linter_config_gets_no_lint_check(tmp_path):
    write(tmp_path / "go.mod", "module example.com/x\n")
    assert for_kind(detect_checks(tmp_path), "lint") == []


def test_go_build_is_a_build_kind(tmp_path):
    write(tmp_path / "go.mod", "module example.com/x\n")
    assert commands(detect_checks(tmp_path), "build") == [("go", "build", "./...")]


# ── rust ──────────────────────────────────────────────────────────────────


def test_cargo_toml_marks_a_rust_repo(tmp_path):
    write(tmp_path / "Cargo.toml", "[package]\nname='x'\n")
    assert "rust" in detect_ecosystems(tmp_path)


def test_cargo_test_check_and_fmt_are_detected(tmp_path):
    write(tmp_path / "Cargo.toml", "[package]\nname='x'\n")
    checks = detect_checks(tmp_path)
    assert commands(checks, "test") == [("cargo", "test")]
    assert commands(checks, "typecheck") == [("cargo", "check", "--all-targets")]
    assert commands(checks, "format") == [("cargo", "fmt", "--check")]


def test_clippy_needs_config_evidence(tmp_path):
    write(tmp_path / "Cargo.toml", "[package]\nname='x'\n")
    assert for_kind(detect_checks(tmp_path), "lint") == []

    write(tmp_path / "clippy.toml", "msrv='1.70'\n")
    lint = for_kind(detect_checks(tmp_path), "lint")[0]
    assert lint.command == ("cargo", "clippy", "--all-targets", "--", "-D", "warnings")


def test_cargo_lints_section_is_clippy_evidence(tmp_path):
    write(
        tmp_path / "Cargo.toml", "[package]\nname='x'\n\n[lints.clippy]\nall='deny'\n"
    )
    assert commands(detect_checks(tmp_path), "lint") == [
        ("cargo", "clippy", "--all-targets", "--", "-D", "warnings")
    ]


# ── makefile fallback ─────────────────────────────────────────────────────


def test_a_makefile_target_is_used_when_nothing_else_covers_the_kind(tmp_path):
    write(tmp_path / "Makefile", "test:\n\t./run-tests.sh\n")
    check = for_kind(detect_checks(tmp_path), "test")[0]
    assert check.command == ("make", "test")
    assert check.ecosystem == "make"
    assert "Makefile" in check.evidence


def test_make_yields_to_a_real_ecosystem_for_the_same_kind(tmp_path):
    # `make test` almost certainly just calls pytest. Running both would double
    # the wall clock for one signal.
    write(tmp_path / "pyproject.toml", "[tool.pytest.ini_options]\n")
    write(tmp_path / "Makefile", "test:\n\tpytest\n")
    assert commands(detect_checks(tmp_path), "test") == [("pytest",)]


def test_make_still_covers_kinds_the_ecosystem_missed(tmp_path):
    write(tmp_path / "pyproject.toml", "[tool.pytest.ini_options]\n")
    write(tmp_path / "Makefile", "test:\n\tpytest\nlint:\n\tshellcheck *.sh\n")
    checks = detect_checks(tmp_path)
    assert commands(checks, "test") == [("pytest",)]
    assert commands(checks, "lint") == [("make", "lint")]


def test_only_check_shaped_make_format_targets_are_used(tmp_path):
    write(tmp_path / "Makefile", "format:\n\tblack .\n")
    assert for_kind(detect_checks(tmp_path), "format") == []

    write(tmp_path / "Makefile", "fmt-check:\n\tblack --check .\n")
    assert commands(detect_checks(tmp_path), "format") == [("make", "fmt-check")]


def test_variable_assignment_is_not_mistaken_for_a_target(tmp_path):
    write(tmp_path / "Makefile", "test := notatarget\nlint:\n\truff check .\n")
    checks = detect_checks(tmp_path)
    assert for_kind(checks, "test") == []
    assert commands(checks, "lint") == [("make", "lint")]


def test_lowercase_makefile_is_recognized(tmp_path):
    write(tmp_path / "makefile", "test:\n\t./t.sh\n")
    assert commands(detect_checks(tmp_path), "test") == [("make", "test")]


def test_a_makefile_with_no_known_targets_yields_nothing(tmp_path):
    write(tmp_path / "Makefile", "deploy:\n\t./ship.sh\n")
    assert detect_checks(tmp_path) == ()


def test_pre_commit_config_is_not_turned_into_a_check(tmp_path):
    # Real case: a repo whose only check-shaped file is .pre-commit-config.yaml.
    # Running `pre-commit run --all-files` would rewrite files (black,
    # end-of-file-fixer, …), and the gate never mutates the tree it checks.
    write(tmp_path / ".pre-commit-config.yaml", "repos:\n- repo: local\n")
    write(tmp_path / "pyproject.toml", "[project]\nname='x'\n")
    assert for_kind(detect_checks(tmp_path), "lint") == []
