"""The release artifact must work outside the source tree that built it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_wheel_installs_and_runs_every_console_script(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    dist = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(dist.glob("tars_agent-*.whl"))
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    bin_dir = venv / "bin"
    subprocess.run(
        [str(bin_dir / "pip"), "install", "--no-deps", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    version = subprocess.run(
        [str(bin_dir / "tars"), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert version.stdout.startswith("tars ")
    for command in ("tars-verify", "tars-readme", "tars-memory"):
        assert (bin_dir / command).is_file()
    resource = subprocess.run(
        [
            str(bin_dir / "python"),
            "-c",
            (
                "from importlib.resources import files; "
                "print(files('tars.router').joinpath('data/labeled_tasks.json').is_file())"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert resource.stdout.strip() == "True"
