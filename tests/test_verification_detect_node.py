"""Detecting a JS/TS repo's check surface.

The subtle one here is format: `npm run format` usually *rewrites* files, so the
gate must only ever accept a check-shaped script (`format:check`) or a `--check`
invocation. A verification gate that reformats the repo has changed the thing it
was asked to measure.
"""

import json

from tars.verification.detect import detect_checks, detect_ecosystems


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def package_json(tmp_path, **payload):
    return write(tmp_path / "package.json", json.dumps(payload))


def for_kind(checks, kind):
    return [c for c in checks if c.kind == kind]


def commands(checks, kind):
    return [c.command for c in for_kind(checks, kind)]


def test_package_json_marks_a_node_repo(tmp_path):
    package_json(tmp_path, name="x")
    assert "node" in detect_ecosystems(tmp_path)


def test_test_script_runs_through_npm_by_default(tmp_path):
    package_json(tmp_path, scripts={"test": "vitest run"})
    check = for_kind(detect_checks(tmp_path), "test")[0]
    assert check.command == ("npm", "run", "test")
    assert check.ecosystem == "node"
    assert "scripts.test" in check.evidence


def test_pnpm_lockfile_selects_pnpm(tmp_path):
    package_json(tmp_path, scripts={"test": "vitest run"})
    write(tmp_path / "pnpm-lock.yaml", "lockfileVersion: 9\n")
    assert commands(detect_checks(tmp_path), "test") == [("pnpm", "run", "test")]


def test_yarn_lockfile_selects_yarn(tmp_path):
    package_json(tmp_path, scripts={"test": "jest"})
    write(tmp_path / "yarn.lock", "")
    assert commands(detect_checks(tmp_path), "test") == [("yarn", "run", "test")]


def test_bun_lockfile_selects_bun(tmp_path):
    package_json(tmp_path, scripts={"test": "bun test"})
    write(tmp_path / "bun.lockb", "")
    assert commands(detect_checks(tmp_path), "test") == [("bun", "run", "test")]


def test_package_manager_field_beats_the_lockfile(tmp_path):
    # A stale package-lock.json alongside an explicit packageManager field means
    # the field is the intent.
    package_json(tmp_path, packageManager="pnpm@9.1.0", scripts={"test": "vitest run"})
    write(tmp_path / "package-lock.json", "{}")
    assert commands(detect_checks(tmp_path), "test") == [("pnpm", "run", "test")]


def test_lint_and_typecheck_scripts_are_detected(tmp_path):
    package_json(tmp_path, scripts={"lint": "eslint .", "typecheck": "tsc --noEmit"})
    checks = detect_checks(tmp_path)
    assert commands(checks, "lint") == [("npm", "run", "lint")]
    assert commands(checks, "typecheck") == [("npm", "run", "typecheck")]


def test_hyphenated_type_check_script_is_detected(tmp_path):
    package_json(tmp_path, scripts={"type-check": "tsc --noEmit"})
    assert commands(detect_checks(tmp_path), "typecheck") == [("npm", "run", "type-check")]


def test_a_check_shaped_format_script_is_used(tmp_path):
    package_json(tmp_path, scripts={"format:check": "prettier --check ."})
    assert commands(detect_checks(tmp_path), "format") == [("npm", "run", "format:check")]


def test_a_rewriting_format_script_is_never_used(tmp_path):
    # `npm run format` would reformat the working tree. Detect nothing rather
    # than mutate the repo under inspection.
    package_json(tmp_path, scripts={"format": "prettier --write ."})
    assert for_kind(detect_checks(tmp_path), "format") == []


def test_prettier_dependency_gives_a_check_only_invocation(tmp_path):
    package_json(tmp_path, devDependencies={"prettier": "^3.0.0"})
    assert commands(detect_checks(tmp_path), "format") == [("prettier", "--check", ".")]


def test_tsconfig_gives_a_noemit_typecheck(tmp_path):
    # --noEmit matters: plain `tsc` would write .js files next to the sources.
    package_json(tmp_path, name="x")
    write(tmp_path / "tsconfig.json", "{}")
    check = for_kind(detect_checks(tmp_path), "typecheck")[0]
    assert check.command == ("tsc", "--noEmit")
    assert "tsconfig.json" in check.evidence


def test_eslint_flat_config_gives_a_direct_invocation(tmp_path):
    package_json(tmp_path, name="x")
    write(tmp_path / "eslint.config.js", "export default []\n")
    assert commands(detect_checks(tmp_path), "lint") == [("eslint", ".")]


def test_legacy_eslintrc_is_recognized(tmp_path):
    package_json(tmp_path, name="x")
    write(tmp_path / ".eslintrc.json", "{}")
    assert commands(detect_checks(tmp_path), "lint") == [("eslint", ".")]


def test_a_script_outranks_a_direct_binary_for_the_same_kind(tmp_path):
    # The repo's own script encodes flags the gate shouldn't have to guess.
    package_json(tmp_path, scripts={"lint": "eslint . --max-warnings 0"})
    write(tmp_path / "eslint.config.js", "export default []\n")
    assert commands(detect_checks(tmp_path), "lint") == [("npm", "run", "lint"), ("eslint", ".")]


def test_vitest_dependency_without_a_script_runs_once_not_in_watch_mode(tmp_path):
    package_json(tmp_path, devDependencies={"vitest": "^2.0.0"})
    assert commands(detect_checks(tmp_path), "test") == [("vitest", "run")]


def test_jest_dependency_without_a_script(tmp_path):
    package_json(tmp_path, devDependencies={"jest": "^29.0.0"})
    assert commands(detect_checks(tmp_path), "test") == [("jest",)]


def test_build_script_is_detected_as_build_kind(tmp_path):
    package_json(tmp_path, scripts={"build": "tsc -b"})
    assert commands(detect_checks(tmp_path), "build") == [("npm", "run", "build")]


def test_malformed_package_json_does_not_crash_detection(tmp_path):
    write(tmp_path / "package.json", "{ this is not json")
    write(tmp_path / "tsconfig.json", "{}")
    # Still a node repo, and the non-JSON-dependent signals still work.
    assert "node" in detect_ecosystems(tmp_path)
    assert commands(detect_checks(tmp_path), "typecheck") == [("tsc", "--noEmit")]


def test_a_polyglot_repo_gets_checks_from_both_ecosystems(tmp_path):
    # A monorepo with a Python backend and a JS frontend genuinely needs both
    # suites; picking one ecosystem would silently skip half the repo.
    write(tmp_path / "pyproject.toml", "[tool.pytest.ini_options]\n")
    package_json(tmp_path, scripts={"test": "vitest run"})
    ecosystems = {c.ecosystem for c in for_kind(detect_checks(tmp_path), "test")}
    assert ecosystems == {"python", "node"}
