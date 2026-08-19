"""The package surface: what the rest of TARS is allowed to import.

Also pins one structural property worth keeping — importing the gate must not
drag in `tars.providers`. The gate is a standalone tool that needs no
credentials and no SDKs; a stray import would couple it to a layer that
currently has two competing implementations.
"""

import subprocess
import sys

import tars.verification as pkg


def test_verify_is_reachable_from_the_package_root():
    from tars.verification import verify

    assert callable(verify)


def test_the_documented_entry_points_are_exported():
    for name in ("verify", "select_checks", "detect_checks", "run_check", "GateReport"):
        assert name in pkg.__all__, name


def test_every_exported_name_actually_resolves():
    missing = [name for name in pkg.__all__ if not hasattr(pkg, name)]
    assert missing == []


def test_dunder_all_is_sorted():
    # Matches tars.providers, and keeps diffs on this list readable.
    assert list(pkg.__all__) == sorted(pkg.__all__)


def test_importing_the_gate_does_not_import_the_provider_layer():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, tars.verification; "
            "print(any(m.startswith('tars.providers') for m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"
