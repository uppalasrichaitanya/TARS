"""The packaged `tars-verify` console script.

An entry point that names a function that doesn't exist installs fine and fails
at the first invocation, so this resolves the string in pyproject the same way
the installer will.
"""

import importlib

import pytest

tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+; the check needs a TOML parser")

from pathlib import Path  # noqa: E402  (after the importorskip guard, deliberately)

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


@pytest.fixture(scope="module")
def scripts():
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle).get("project", {}).get("scripts", {})


def test_tars_verify_is_declared_as_a_console_script(scripts):
    assert "tars-verify" in scripts


def test_the_entry_point_resolves_to_a_callable(scripts):
    module_name, _, attr = scripts["tars-verify"].partition(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attr))
