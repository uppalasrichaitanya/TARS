"""TARS — a disciplined, multi-session coding agent loop."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tars-agent")
except PackageNotFoundError:
    __version__ = "0.1.2"
