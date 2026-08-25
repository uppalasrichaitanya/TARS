"""Which executor does ACT's work, chosen by name rather than by import.

Factories are lazy and stay lazy. Registering a name must not import the module
behind it, because the point of this indirection is that an external harness —
another language's runtime, a subprocess, a pile of dependencies — can be *an*
option without becoming *a* requirement. A repo using only the built-in executor
pays nothing for the existence of the others.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import Executor

LOCAL = "local"

ExecutorFactory = Callable[[], Executor]

_FACTORIES: dict[str, ExecutorFactory] = {}
_CACHE: dict[str, Executor] = {}


class UnknownExecutorError(LookupError):
    """Asked for an executor nobody registered."""


def register_executor(name: str, factory: ExecutorFactory) -> None:
    """Register (or replace) an executor factory. Clears any cached instance."""
    _FACTORIES[name] = factory
    _CACHE.pop(name, None)


def get_executor(name: str) -> Executor:
    """Return the executor registered under `name`, constructing it once."""
    if name in _CACHE:
        return _CACHE[name]
    factory = _FACTORIES.get(name)
    if factory is None:
        raise UnknownExecutorError(
            f"no executor registered as {name!r}; "
            f"known executors: {', '.join(sorted(_FACTORIES)) or 'none'}"
        )
    executor = factory()
    _CACHE[name] = executor
    return executor


def known_executors() -> tuple[str, ...]:
    """Every registered name, for a report or a `--help` line."""
    return tuple(sorted(_FACTORIES))


def reset_executors() -> None:
    """Drop cached instances, keeping registrations (used by tests)."""
    _CACHE.clear()


def install_default_executors() -> None:
    """Register the built-in executor, without displacing a caller's override."""
    if LOCAL not in _FACTORIES:
        from .local import LocalExecutor

        register_executor(LOCAL, LocalExecutor)
